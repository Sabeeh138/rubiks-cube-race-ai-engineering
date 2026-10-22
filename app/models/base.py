"""Shared, transport-independent contract for cube move adapters."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from app.config import ConfigurationError
from app.cube.engine import LEGAL_MOVES, CubeState
from app.models.prompts import (
    RUBIKS_MOVE_OBJECTIVE_V1,
    RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
    SUPPORTED_PROMPT_VERSIONS,
)

TOKEN_USAGE_FIELDS = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
)


def safe_token_usage(value: object) -> dict[str, int] | None:
    """Keep only known non-negative integer token counters from provider metadata."""

    if not isinstance(value, Mapping):
        return None
    result: dict[str, int] = {}
    for field in TOKEN_USAGE_FIELDS:
        count = value.get(field)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result[field] = count
    return result or None


class AdapterError(RuntimeError):
    """Base class for failures a future benchmark can report per adapter turn."""


class InvalidMoveError(AdapterError):
    """Raised when an adapter proposes a token outside the current legal set."""

    def __init__(
        self,
        proposed_move: object,
        *,
        model_id: str,
        legal_moves: Sequence[str],
        token_usage: Mapping[str, int] | None = None,
    ) -> None:
        self.proposed_move = proposed_move
        self.model_id = model_id
        self.legal_moves = tuple(legal_moves)
        self.token_usage = safe_token_usage(token_usage)
        super().__init__(
            f"adapter {model_id!r} proposed invalid move {proposed_move!r}; "
            "the move must exactly match one of the legal move tokens"
        )


class AdapterTimeoutError(AdapterError):
    """Raised when an adapter turn exceeds its requested timeout."""

    def __init__(
        self,
        *,
        model_id: str,
        timeout_seconds: float | None,
        elapsed_seconds: float,
    ) -> None:
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"adapter {model_id!r} timed out"
            + (
                f" after {timeout_seconds:g} seconds"
                if timeout_seconds is not None
                else ""
            )
        )


class AdapterExecutionError(AdapterError):
    """Raised when the adapter implementation fails for a non-timeout reason."""

    def __init__(self, *, model_id: str, original_error: Exception) -> None:
        self.model_id = model_id
        self.original_error = original_error
        # Do not copy arbitrary exception text into benchmark output; it may
        # contain transport details that a later integration considers secret.
        super().__init__(
            f"adapter {model_id!r} failed with {type(original_error).__name__}"
        )


class MockScriptExhaustedError(AdapterError):
    """Raised when a scripted mock has no further outcomes to return."""


def validate_move(
    proposed_move: object,
    *,
    legal_moves: Sequence[str] = LEGAL_MOVES,
    model_id: str = "unknown",
    token_usage: Mapping[str, int] | None = None,
) -> str:
    """Validate an exact move token against a supplied legal-action set.

    No trimming, case folding, notation conversion, or other coercion is done.
    """

    if not isinstance(proposed_move, str) or proposed_move not in legal_moves:
        raise InvalidMoveError(
            proposed_move,
            model_id=model_id,
            legal_moves=legal_moves,
            token_usage=token_usage,
        )
    return proposed_move


def _move_tuple(value: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of move tokens, not a string")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of move tokens") from exc
    for move in result:
        if not isinstance(move, str) or move not in LEGAL_MOVES:
            raise ValueError(f"{field_name} contains invalid move token {move!r}")
    return result


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Canonical serialized cube state and context for one proposed turn."""

    cube_state: str
    recent_moves: tuple[str, ...] = ()
    legal_moves: tuple[str, ...] = LEGAL_MOVES
    timeout_seconds: float | None = None
    original_scramble: tuple[str, ...] = ()
    prompt_version: str = RUBIKS_MOVE_OBJECTIVE_V1
    action_previews_enabled: bool = False

    def __post_init__(self) -> None:
        # Reuse the engine's canonical parser so every adapter receives the
        # same validated 54-facelet URFDLB representation.
        CubeState.from_facelets(self.cube_state)
        recent_moves = _move_tuple(self.recent_moves, field_name="recent_moves")
        legal_moves = _move_tuple(self.legal_moves, field_name="legal_moves")
        original_scramble = _move_tuple(self.original_scramble, field_name="original_scramble")
        if not legal_moves:
            raise ValueError("legal_moves must contain at least one allowed move")
        if len(set(legal_moves)) != len(legal_moves):
            raise ValueError("legal_moves must not contain duplicate move tokens")
        object.__setattr__(self, "recent_moves", recent_moves)
        object.__setattr__(self, "legal_moves", legal_moves)
        object.__setattr__(self, "original_scramble", original_scramble)
        if self.prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            raise ValueError("prompt_version is not a supported frozen condition")
        if not isinstance(self.action_previews_enabled, bool):
            raise TypeError("action_previews_enabled must be a boolean")
        if self.action_previews_enabled and self.prompt_version != RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE:
            raise ValueError("action previews require the state-aware v2 prompt")

        timeout = self.timeout_seconds
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout_seconds must be a positive finite number or None")
            if not math.isfinite(float(timeout)) or timeout <= 0:
                raise ValueError("timeout_seconds must be a positive finite number")
            object.__setattr__(self, "timeout_seconds", float(timeout))


@dataclass(frozen=True, slots=True)
class AdapterReply:
    """Unvalidated adapter output before shared move-token validation."""

    proposed_move: object
    raw_response: object | None = None
    token_usage: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_usage", safe_token_usage(self.token_usage))


@dataclass(frozen=True, slots=True)
class MoveProposal:
    """Validated move and lightweight metadata from one adapter turn."""

    move: str
    model_id: str
    elapsed_seconds: float
    raw_response: object | None = None
    token_usage: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        validate_move(self.move, model_id=self.model_id)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "token_usage", safe_token_usage(self.token_usage))


@runtime_checkable
class ModelAdapter(Protocol):
    """Common async interface; transport and provider request formats stay private."""

    model_id: str

    async def choose_move(self, context: TurnContext) -> MoveProposal:
        """Propose exactly one standard face-turn move for ``context``."""


class BaseModelAdapter(ABC):
    """Shared timing, timeout, error normalization, and output validation.

    Provider-specific adapters can subclass this while implementing their own
    ``_request_move`` method and request/response parsing for their API style.
    """

    execution_provenance = "unknown"
    provenance_label: str | None = None

    def __init__(self, model_id: str) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id

    @abstractmethod
    async def _request_move(self, context: TurnContext) -> AdapterReply:
        """Perform adapter-specific work and return its unvalidated move output."""

    async def choose_move(self, context: TurnContext) -> MoveProposal:
        """Return a validated move or raise a typed adapter exception."""

        if not isinstance(context, TurnContext):
            raise TypeError("context must be a TurnContext")
        started = perf_counter()
        try:
            request = self._request_move(context)
            if context.timeout_seconds is None:
                reply = await request
            else:
                reply = await asyncio.wait_for(request, timeout=context.timeout_seconds)
        except TimeoutError as exc:
            elapsed = perf_counter() - started
            raise AdapterTimeoutError(
                model_id=self.model_id,
                timeout_seconds=context.timeout_seconds,
                elapsed_seconds=elapsed,
            ) from exc
        except (AdapterError, ConfigurationError):
            raise
        except Exception as exc:
            raise AdapterExecutionError(
                model_id=self.model_id,
                original_error=exc,
            ) from exc

        elapsed = perf_counter() - started
        if not isinstance(reply, AdapterReply):
            raise AdapterExecutionError(
                model_id=self.model_id,
                original_error=TypeError("_request_move must return AdapterReply"),
            )
        move = validate_move(
            reply.proposed_move,
            legal_moves=context.legal_moves,
            model_id=self.model_id,
            token_usage=reply.token_usage,
        )
        return MoveProposal(
            move=move,
            model_id=self.model_id,
            elapsed_seconds=elapsed,
            raw_response=reply.raw_response,
            token_usage=reply.token_usage,
        )
