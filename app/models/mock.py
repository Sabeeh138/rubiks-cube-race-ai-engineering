"""Deterministic, network-free adapters for future workflow tests."""

from __future__ import annotations

from collections.abc import Sequence

from app.models.base import (
    AdapterReply,
    BaseModelAdapter,
    MockScriptExhaustedError,
    TurnContext,
)


class ScriptedMockAdapter(BaseModelAdapter):
    execution_provenance = "mock"

    """Return scripted outcomes one at a time, without network access.

    String outcomes are proposed as-is and validated by ``BaseModelAdapter``.
    An ``Exception`` outcome is raised and normalized to a typed adapter
    failure; ``TimeoutError`` outcomes become ``AdapterTimeoutError``. This
    mock is for deterministic plumbing tests only and says nothing about the
    move quality, latency, or performance of an AI model.
    """

    def __init__(
        self,
        outcomes: Sequence[object],
        *,
        model_id: str = "mock/scripted",
    ) -> None:
        super().__init__(model_id)
        if isinstance(outcomes, (str, bytes)):
            raise TypeError("outcomes must be a sequence of scripted values")
        try:
            self._outcomes = tuple(outcomes)
        except TypeError as exc:
            raise TypeError("outcomes must be a sequence of scripted values") from exc
        self._next_index = 0

    async def _request_move(self, context: TurnContext) -> AdapterReply:
        del context  # The script deliberately does not infer moves from cube state.
        if self._next_index >= len(self._outcomes):
            raise MockScriptExhaustedError(
                f"script for adapter {self.model_id!r} has no remaining outcomes"
            )
        index = self._next_index
        self._next_index += 1
        outcome = self._outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return AdapterReply(
            proposed_move=outcome,
            raw_response={
                "mock": True,
                "script_index": index,
                "scripted_outcome": outcome,
            },
        )
