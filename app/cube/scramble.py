"""Reproducible standard face-turn scrambles."""

from __future__ import annotations

import random
from dataclasses import dataclass

from app.cube.engine import CubeState, LEGAL_MOVES, invert_moves

DEFAULT_SCRAMBLE_LENGTH = 25


@dataclass(frozen=True, slots=True)
class Scramble:
    """A seeded scramble and its standard face-turn sequence."""

    seed: int
    moves: tuple[str, ...]

    @property
    def move_count_htm(self) -> int:
        """Half-turn metric count: every token, including a half turn, is one."""

        return len(self.moves)

    @property
    def notation(self) -> str:
        return " ".join(self.moves)

    def apply_to(self, cube: CubeState | None = None) -> CubeState:
        return (cube or CubeState()).apply_moves(self.moves)

    def undo_from(self, cube: CubeState) -> CubeState:
        return cube.apply_moves(invert_moves(self.moves))


def generate_scramble(seed: int, length: int = DEFAULT_SCRAMBLE_LENGTH) -> Scramble:
    """Generate a reproducible sequence without adjacent turns of one face."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 0:
        raise ValueError("length must be zero or greater")

    rng = random.Random(seed)
    result: list[str] = []
    previous_face: str | None = None
    for _ in range(length):
        choices = tuple(move for move in LEGAL_MOVES if move[0] != previous_face)
        move = rng.choice(choices)
        result.append(move)
        previous_face = move[0]

    return Scramble(seed=seed, moves=tuple(result))
