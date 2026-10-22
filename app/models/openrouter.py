"""OpenRouter adapters for the assignment's exact model IDs.

The Decisions and Chat Completions APIs use separate endpoint and parsing
implementations. Both produce the shared ``AdapterReply`` consumed by the
Phase 4 validation and timing contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import httpx

from app.config import Settings
from app.models.base import (
    AdapterError,
    AdapterExecutionError,
    AdapterReply,
    AdapterTimeoutError,
    BaseModelAdapter,
    TurnContext,
    safe_token_usage,
)
from app.cube.engine import CubeState, FACE_ORDER, SOLVED_FACELETS
from app.models.prompts import RUBIKS_MOVE_OBJECTIVE_V1, instructions_for_prompt

DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
CHAT_COMPLETIONS_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

JEV_113_MODEL_ID = "typesafe/jev-1.13"
GPT_6_LUNA_DECISIONS_MODEL_ID = "openai/gpt-6-luna-decisions"
CLOUDFLARE_CLEF_MODEL_ID = "cloudflare/clef"
CLAUDE_OPUS_55_MODEL_ID = "anthropic/claude-opus-5.5"
CLAUDE_MAX_COMPLETION_TOKENS = 256
ASSIGNED_MODEL_IDS = (
    JEV_113_MODEL_ID,
    GPT_6_LUNA_DECISIONS_MODEL_ID,
    CLOUDFLARE_CLEF_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
)

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=25.0,
    write=10.0,
    pool=5.0,
)

_FACE_NAMES = {
    "U": "up",
    "R": "right",
    "F": "front",
    "D": "down",
    "L": "left",
    "B": "back",
}
_MOVE_INSTRUCTIONS = instructions_for_prompt(RUBIKS_MOVE_OBJECTIVE_V1)


class OpenRouterHTTPError(AdapterError):
    """An unsuccessful OpenRouter HTTP response, without exposing its body."""

    def __init__(
        self,
        *,
        model_id: str,
        status_code: int,
        retry_after: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"OpenRouter returned HTTP {status_code} for {model_id!r}")


class OpenRouterRateLimitError(OpenRouterHTTPError):
    """HTTP 429 response; callers may inspect ``retry_after`` before retrying."""


class AdapterResponseError(AdapterError):
    """A successful HTTP response did not contain a usable API answer.

    ``category``, ``finish_reason`` and the allowlisted token counts are safe
    diagnostics for benchmark result records. Response text and arbitrary
    provider error details are deliberately excluded.
    """

    def __init__(
        self,
        *,
        model_id: str,
        detail: str,
        category: str = "response_shape",
        finish_reason: object = None,
        usage: object = None,
    ) -> None:
        self.model_id = model_id
        self.detail = detail
        self.category = category if category in _RESPONSE_ERROR_CATEGORIES else "response_shape"
        self.finish_reason = _safe_finish_reason(finish_reason)
        self.usage = _safe_token_usage(usage)
        diagnostics = [f"category={self.category}"]
        if self.finish_reason is not None:
            diagnostics.append(f"finish_reason={self.finish_reason}")
        if self.usage:
            counts = ", ".join(f"{key}={value}" for key, value in self.usage.items())
            diagnostics.append(f"usage({counts})")
        super().__init__(
            f"invalid response from OpenRouter model {model_id!r}: {detail} "
            f"[{'; '.join(diagnostics)}]"
        )


class OpenRouterAdapter(BaseModelAdapter):
    execution_provenance = "live"

    """Common authenticated HTTP transport and lifecycle for OpenRouter calls."""

    endpoint: str

    def __init__(
        self,
        model_id: str,
        *,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        request_timeout: float | httpx.Timeout | None = None,
    ) -> None:
        super().__init__(model_id)
        self._settings = settings if settings is not None else Settings.from_env()
        self._http_client = http_client
        self._owns_http_client = http_client is None
        if request_timeout is None:
            self._request_timeout = DEFAULT_HTTP_TIMEOUT
        elif isinstance(request_timeout, httpx.Timeout):
            self._request_timeout = request_timeout
        else:
            self._request_timeout = httpx.Timeout(request_timeout)

    async def _post_json(
        self,
        payload: Mapping[str, Any],
    ) -> object:
        api_key = self._settings.require_openrouter_api_key()
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=self._request_timeout)
            self._http_client = client

        try:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._request_timeout,
            )
        except httpx.TimeoutException as exc:
            del exc
            raise TimeoutError("OpenRouter request timed out") from None
        except httpx.RequestError as exc:
            safe_error = RuntimeError(
                f"HTTP transport failure ({type(exc).__name__})"
            )
            raise AdapterExecutionError(
                model_id=self.model_id,
                original_error=safe_error,
            ) from None

        if response.status_code == 429:
            raise OpenRouterRateLimitError(
                model_id=self.model_id,
                status_code=429,
                retry_after=response.headers.get("Retry-After"),
            )
        if not response.is_success:
            raise OpenRouterHTTPError(
                model_id=self.model_id,
                status_code=response.status_code,
                retry_after=response.headers.get("Retry-After"),
            )

        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="response body was not valid JSON",
                category="malformed_response_json",
            ) from None
        return body

    async def aclose(self) -> None:
        """Close an internally-created client; injected clients remain caller-owned."""

        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> OpenRouterAdapter:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


class DecisionsAPIAdapter(OpenRouterAdapter):
    """OpenRouter Decisions API transport and ``answers.move.choice`` parser."""

    endpoint = DECISIONS_ENDPOINT

    async def _request_move(self, context: TurnContext) -> AdapterReply:
        definitions = _move_definitions(context)
        payload = {
            "model": self.model_id,
            "state": _turn_state(context, definitions),
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": instructions_for_prompt(context.prompt_version),
                    "criteria": definitions,
                }
            },
        }
        body = await self._post_json(payload)
        if not isinstance(body, dict):
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="Decisions response must be a JSON object",
            )
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="Decisions response is missing the answers object",
            )
        answer = answers.get("move")
        if not isinstance(answer, dict):
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="Decisions response is missing answers.move",
            )
        answer_type = answer.get("type")
        if answer_type is not None and answer_type != "choice":
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="answers.move.type must be 'choice'",
            )
        choice = answer.get("choice")
        if not isinstance(choice, str) or not choice:
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="answers.move.choice must be a non-empty string",
            )
        return AdapterReply(
            proposed_move=choice,
            raw_response=body,
            token_usage=_safe_token_usage(body.get("usage")),
        )


class ChatCompletionsAPIAdapter(OpenRouterAdapter):
    """OpenRouter Chat Completions transport and JSON-schema move parser."""

    endpoint = CHAT_COMPLETIONS_ENDPOINT

    async def _request_move(self, context: TurnContext) -> AdapterReply:
        definitions = _move_definitions(context)
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": instructions_for_prompt(context.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        _turn_state(context, definitions),
                        separators=(",", ":"),
                    ),
                },
            ],
            # Leave room for Claude's adaptive reasoning and the structured
            # move object while keeping a bounded per-request output cost.
            "max_completion_tokens": CLAUDE_MAX_COMPLETION_TOKENS,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "rubiks_cube_move",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "move": {
                                "type": "string",
                                "enum": list(context.legal_moves),
                            }
                        },
                        "required": ["move"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        body = await self._post_json(payload)
        if not isinstance(body, dict):
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="Chat Completions response must be a JSON object",
            )
        if "error" in body:
            raise _chat_response_error(
                self.model_id,
                "Chat Completions response contained a provider error",
                category="provider_error",
                body=body,
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _chat_response_error(
                self.model_id,
                "Chat Completions response must contain exactly one choice",
                category="response_shape",
                body=body,
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _chat_response_error(
                self.model_id,
                "choices[0] must be an object",
                category="response_shape",
                body=body,
            )
        finish_reason = _safe_finish_reason(choice.get("finish_reason"))
        if "error" in choice or finish_reason == "error":
            raise _chat_response_error(
                self.model_id,
                "Chat Completions choice reported a provider error",
                category="provider_error",
                body=body,
                choice=choice,
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise _chat_response_error(
                self.model_id,
                "choices[0].message must be an assistant message",
                category="response_shape",
                body=body,
                choice=choice,
            )

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise _chat_response_error(
                self.model_id,
                "assistant message reported a refusal",
                category="refusal",
                body=body,
                choice=choice,
            )
        if refusal is not None:
            raise _chat_response_error(
                self.model_id,
                "assistant message refusal field had an unexpected shape",
                category="response_shape",
                body=body,
                choice=choice,
            )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            if finish_reason == "length":
                category = "token_limit"
                detail = "completion ended at the token limit without usable assistant content"
            elif finish_reason == "content_filter":
                category = "content_filtered"
                detail = "completion ended due to a content filter without usable assistant content"
            elif "content" not in message:
                category = "response_shape"
                detail = "assistant message is missing its content field"
            elif content is None:
                category = "response_shape"
                detail = "assistant message content was null"
            elif isinstance(content, str):
                category = "empty_content"
                detail = "assistant message content was empty"
            else:
                category = "response_shape"
                detail = "assistant message content was not a string"
            raise _chat_response_error(
                self.model_id,
                detail,
                category=category,
                body=body,
                choice=choice,
            )
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            if finish_reason == "length":
                category = "token_limit"
                detail = "completion reached the token limit before valid JSON was produced"
            elif finish_reason == "content_filter":
                category = "content_filtered"
                detail = "completion was content-filtered before valid JSON was produced"
            else:
                category = "malformed_output"
                detail = "assistant content was not valid JSON"
            raise _chat_response_error(
                self.model_id,
                detail,
                category=category,
                body=body,
                choice=choice,
            ) from None
        if (
            not isinstance(decoded, dict)
            or set(decoded) != {"move"}
            or not isinstance(decoded.get("move"), str)
            or not decoded["move"]
        ):
            raise _chat_response_error(
                self.model_id,
                "assistant JSON must contain only a non-empty 'move' string",
                category="malformed_output",
                body=body,
                choice=choice,
            )
        return AdapterReply(
            proposed_move=decoded["move"],
            raw_response=body,
            token_usage=_safe_token_usage(body.get("usage")),
        )


class Jev113Adapter(DecisionsAPIAdapter):
    """Decisions API adapter for the assigned TypeSafe Jev 1.13 model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(JEV_113_MODEL_ID, **kwargs)


class GPT6LunaDecisionsAdapter(DecisionsAPIAdapter):
    """Decisions API adapter for the assigned GPT-6 Luna Decisions model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(GPT_6_LUNA_DECISIONS_MODEL_ID, **kwargs)


class CloudflareClefAdapter(DecisionsAPIAdapter):
    """Decisions API adapter for the assigned Cloudflare Clef model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(CLOUDFLARE_CLEF_MODEL_ID, **kwargs)


class ClaudeOpus55Adapter(ChatCompletionsAPIAdapter):
    """Chat Completions adapter for the assigned Claude Opus 5.5 model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(CLAUDE_OPUS_55_MODEL_ID, **kwargs)


def build_openrouter_adapters(
    *,
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
    request_timeout: float | httpx.Timeout | None = None,
) -> dict[str, OpenRouterAdapter]:
    """Build the four assigned adapters in assignment order.

    An injected client remains caller-owned and can be shared by all four
    adapters. Without one, each adapter lazily owns a client that callers can
    close with ``aclose`` or an async context manager.
    """

    options = {
        "settings": settings,
        "http_client": http_client,
        "request_timeout": request_timeout,
    }
    adapters: tuple[OpenRouterAdapter, ...] = (
        Jev113Adapter(**options),
        GPT6LunaDecisionsAdapter(**options),
        CloudflareClefAdapter(**options),
        ClaudeOpus55Adapter(**options),
    )
    return {adapter.model_id: adapter for adapter in adapters}


def _move_definitions(context: TurnContext) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for move in context.legal_moves:
        suffix = move[1:]
        direction = {
            "": "clockwise quarter-turn",
            "'": "counterclockwise quarter-turn",
            "2": "half-turn",
        }[suffix]
        definitions[move] = f"{direction} of the {_FACE_NAMES[move[0]]} face."
    return definitions


def _turn_state(
    context: TurnContext,
    definitions: Mapping[str, str],
) -> dict[str, object]:
    state: dict[str, object] = {
        "cube_state": context.cube_state,
        "recent_moves": list(context.recent_moves),
        "legal_moves": list(context.legal_moves),
        "legal_move_definitions": dict(definitions),
    }
    if context.prompt_version == RUBIKS_MOVE_OBJECTIVE_V1:
        return state

    facelets = context.cube_state
    state.update(
        {
            "prompt_version": context.prompt_version,
            "facelet_convention": (
                "The 54 characters use the cube engine's URFDLB order. Each face's nine-character "
                "block is a row-major 3x3 grid viewed from outside that face; its center is index 4. "
                "Face labels identify the fixed center colors."
            ),
            "face_grids": {
                face: [
                    list(facelets[face_index * 9 + row * 3 : face_index * 9 + row * 3 + 3])
                    for row in range(3)
                ]
                for face_index, face in enumerate(FACE_ORDER)
            },
            "solved_target": SOLVED_FACELETS,
            "original_scramble": list(context.original_scramble),
            "full_move_history": list(context.recent_moves),
        }
    )
    if context.action_previews_enabled:
        current = CubeState.from_facelets(facelets)
        successors = dict(current.successors())
        state["action_previews"] = [
            {"move": move, "resulting_cube_state": successors[move].serialize()}
            for move in context.legal_moves
        ]
    return state


_RESPONSE_ERROR_CATEGORIES = frozenset(
    {
        "empty_content",
        "refusal",
        "token_limit",
        "content_filtered",
        "malformed_output",
        "malformed_response_json",
        "response_shape",
        "provider_error",
    }
)
_SAFE_FINISH_REASON = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")
def _safe_finish_reason(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_FINISH_REASON.fullmatch(value):
        return value
    return None


def _safe_token_usage(value: object) -> dict[str, int]:
    safe = safe_token_usage(value)
    return safe or {}


def _chat_response_error(
    model_id: str,
    detail: str,
    *,
    category: str,
    body: Mapping[str, Any] | None = None,
    choice: Mapping[str, Any] | None = None,
) -> AdapterResponseError:
    finish_reason = choice.get("finish_reason") if choice is not None else None
    usage = body.get("usage") if body is not None else None
    return AdapterResponseError(
        model_id=model_id,
        detail=detail,
        category=category,
        finish_reason=finish_reason,
        usage=usage,
    )
