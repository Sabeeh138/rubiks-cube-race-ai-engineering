"""Explicit, deterministic Claude simulation for a mixed recording run."""

from __future__ import annotations

from collections.abc import Sequence

from app.cube import CubeState
from app.models.base import AdapterReply, BaseModelAdapter, MockScriptExhaustedError, TurnContext
from app.models.openrouter import CLAUDE_OPUS_55_MODEL_ID

SIMULATED_CLAUDE_LABEL = "SIMULATED — NOT A LIVE CLAUDE RESPONSE"


class ScriptedClaudeFallbackAdapter(BaseModelAdapter):
    """Return a preselected legal Claude demonstration sequence, without I/O.

    The recording service supplies the inverse of the puzzle scramble. This is
    intentionally a scripted demonstration, not a model inference or quality
    measurement. No delays are added.
    """

    execution_provenance = "simulated"
    provenance_label = SIMULATED_CLAUDE_LABEL

    def __init__(self, moves: Sequence[str]) -> None:
        super().__init__(CLAUDE_OPUS_55_MODEL_ID)
        if isinstance(moves, (str, bytes)):
            raise TypeError("moves must be a sequence of standard move tokens")
        try:
            self._moves = tuple(moves)
        except TypeError as exc:
            raise TypeError("moves must be a sequence of standard move tokens") from exc
        # The cube engine validates the complete vocabulary and mechanics.
        CubeState().apply_moves(self._moves)
        self._next_index = 0

    async def _request_move(self, context: TurnContext) -> AdapterReply:
        del context  # The sequence is preselected from this puzzle's scramble.
        if self._next_index >= len(self._moves):
            raise MockScriptExhaustedError("scripted Claude demonstration sequence is exhausted")
        move = self._moves[self._next_index]
        self._next_index += 1
        return AdapterReply(proposed_move=move)
