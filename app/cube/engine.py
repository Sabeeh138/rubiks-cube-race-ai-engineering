"""A stable facelet-oriented wrapper around the ``magiccube`` move engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from magiccube import Cube as _MagicCube

FACE_ORDER = "URFDLB"
SOLVED_FACELETS = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
LEGAL_MOVES = tuple(face + suffix for face in FACE_ORDER for suffix in ("", "'", "2"))
_CENTERS = {face: index * 9 + 4 for index, face in enumerate(FACE_ORDER)}

# magiccube uses W/R/G/Y/O/B colors for U/R/F/D/L/B, respectively.
_FACE_TO_COLOR = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}
_COLOR_TO_FACE = {color: face for face, color in _FACE_TO_COLOR.items()}
_MAGICCUBE_FACE_ORDER = "ULFRBD"


def validate_facelets(facelets: str) -> str:
    """Validate the canonical 54-character URFDLB facelet format.

    This validates shape, symbols, sticker counts, and fixed centers. It does
    not determine whether an arbitrary externally supplied arrangement is
    physically reachable (for example, a single flipped edge).
    """

    if not isinstance(facelets, str):
        raise TypeError("facelets must be a string")
    if len(facelets) != 54:
        raise ValueError(f"facelets must contain exactly 54 characters; got {len(facelets)}")

    invalid = set(facelets) - set(FACE_ORDER)
    if invalid:
        raise ValueError(f"facelets contain invalid symbols: {''.join(sorted(invalid))}")

    counts = {face: facelets.count(face) for face in FACE_ORDER}
    if any(count != 9 for count in counts.values()):
        raise ValueError("facelets must contain exactly nine stickers for each of URFDLB")

    for face, index in _CENTERS.items():
        if facelets[index] != face:
            raise ValueError(f"the center facelet at index {index} must be {face!r}")

    return facelets


def _to_magiccube_state(facelets: str) -> str:
    """Convert canonical URFDLB facelets into magiccube's U-L-F-R-B-D order."""

    start_by_face = {face: FACE_ORDER.index(face) * 9 for face in FACE_ORDER}
    return "".join(
        _FACE_TO_COLOR[facelets[start_by_face[face] : start_by_face[face] + 9][index]]
        for face in _MAGICCUBE_FACE_ORDER
        for index in range(9)
    )


def _from_magiccube_state(cube: _MagicCube) -> str:
    """Convert magiccube's Kociemba-order colors to canonical URFDLB letters."""

    return "".join(_COLOR_TO_FACE[color] for color in cube.get_kociemba_facelet_colors())


def _normalize_moves(moves: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(moves, str):
        normalized = tuple(moves.split())
    else:
        try:
            normalized = tuple(moves)
        except TypeError as exc:
            raise TypeError("moves must be a whitespace-delimited string or iterable of move tokens") from exc

    for move in normalized:
        if not isinstance(move, str) or move not in LEGAL_MOVES:
            raise ValueError(f"invalid face-turn move {move!r}; expected one of the 18 legal moves")
    return normalized


def invert_move(move: str) -> str:
    """Return the inverse of one legal face-turn move."""

    (token,) = _normalize_moves((move,))
    if token.endswith("2"):
        return token
    if token.endswith("'"):
        return token[:-1]
    return token + "'"


def invert_moves(moves: str | Iterable[str]) -> tuple[str, ...]:
    """Return the inverse move sequence in execution order."""

    return tuple(invert_move(move) for move in reversed(_normalize_moves(moves)))


@dataclass(frozen=True, slots=True)
class CubeState:
    """Immutable 3x3 cube state serialized in Kociemba URFDLB facelet order."""

    facelets: str = SOLVED_FACELETS

    def __post_init__(self) -> None:
        validate_facelets(self.facelets)

    @classmethod
    def from_facelets(cls, facelets: str) -> CubeState:
        """Create a state from a canonical URFDLB 54-character string."""

        validate_facelets(facelets)
        return cls(facelets)

    def serialize(self) -> str:
        """Return the stable 54-character state shared with model adapters."""

        return self.facelets

    @property
    def is_solved(self) -> bool:
        return self.facelets == SOLVED_FACELETS

    def apply_move(self, move: str) -> CubeState:
        """Apply one legal standard face turn and return a new state."""

        return self.apply_moves((move,))

    def apply_moves(self, moves: str | Iterable[str]) -> CubeState:
        """Apply legal standard face turns and return a new state.

        The accepted vocabulary is exactly U, R, F, D, L, B with optional
        prime or half-turn suffixes. Slice, wide, and cube-rotation moves are
        intentionally excluded.
        """

        sequence = _normalize_moves(moves)
        if not sequence:
            return self

        cube = _MagicCube(3, _to_magiccube_state(self.facelets), hist=False)
        cube.rotate(" ".join(sequence))
        return CubeState(_from_magiccube_state(cube))

    def successors(self) -> tuple[tuple[str, CubeState], ...]:
        """Return every legal one-move successor in the shared move order.

        The library cube is reused for all 18 turns. Each turn is undone before
        the next successor is computed, avoiding a fresh cube construction per
        candidate while keeping all move mechanics inside ``magiccube``.
        """

        cube = _MagicCube(3, _to_magiccube_state(self.facelets), hist=True)
        successors: list[tuple[str, CubeState]] = []
        for move in LEGAL_MOVES:
            cube.rotate(move)
            successors.append((move, CubeState(_from_magiccube_state(cube))))
            cube.undo()
        return tuple(successors)
