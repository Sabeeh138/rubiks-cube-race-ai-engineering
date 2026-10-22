"""Pinned, reproducible five-puzzle primary experiment dataset."""

from __future__ import annotations

from dataclasses import dataclass

from app.benchmark.runner import PuzzleSpec
from app.cube import exact_shortest_solution, generate_scramble

PRIMARY_DATASET_ID = "rubiks-cube-race-primary-v1"

# The generated notation and references are pinned so a future change to the
# PRNG or move ordering cannot silently change the experiment inputs.
_DEFINITIONS = (
    ("primary-01", 2026, 4, ("R", "F2", "D2", "F2"), 4),
    ("primary-02", 2027, 4, ("R", "B2", "F'", "U'"), 4),
    ("primary-03", 2028, 4, ("D2", "B'", "U2", "L2"), 4),
    ("primary-04", 2029, 5, ("B", "F'", "B2", "U2", "L2"), 4),
    ("primary-05", 2030, 5, ("D", "F", "R2", "B2", "F2"), 5),
)


@dataclass(frozen=True, slots=True)
class PrimaryPuzzle:
    """One immutable dataset row with engine-generated and solver-verified data."""

    puzzle_id: str
    seed: int
    scramble_length: int
    scramble_moves: tuple[str, ...]
    initial_state: str
    exact_shortest_distance: int

    @property
    def scramble_notation(self) -> str:
        return " ".join(self.scramble_moves)

    @property
    def puzzle_spec(self) -> PuzzleSpec:
        return PuzzleSpec(
            puzzle_id=self.puzzle_id,
            scramble_moves=self.scramble_moves,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "puzzle_id": self.puzzle_id,
            "seed": self.seed,
            "scramble_length": self.scramble_length,
            "scramble_moves": list(self.scramble_moves),
            "initial_state": self.initial_state,
            "exact_shortest_distance": self.exact_shortest_distance,
        }


def build_primary_dataset() -> tuple[PrimaryPuzzle, ...]:
    """Generate the pinned puzzles and verify their exact HTM references."""

    entries: list[PrimaryPuzzle] = []
    for puzzle_id, seed, length, expected_moves, expected_distance in _DEFINITIONS:
        scramble = generate_scramble(seed=seed, length=length)
        if scramble.moves != expected_moves:
            raise RuntimeError(
                f"seeded generator changed the pinned scramble for {puzzle_id}"
            )
        spec = PuzzleSpec(
            puzzle_id=puzzle_id,
            scramble_moves=scramble.moves,
            seed=seed,
        )
        initial = spec.initial_state
        reference = exact_shortest_solution(initial)
        if reference is None or reference.distance_htm != expected_distance:
            raise RuntimeError(
                f"exact HTM reference changed for {puzzle_id}; expected {expected_distance}"
            )
        if not initial.apply_moves(reference.moves).is_solved:
            raise RuntimeError(f"exact reference did not solve {puzzle_id}")
        entries.append(
            PrimaryPuzzle(
                puzzle_id=puzzle_id,
                seed=seed,
                scramble_length=length,
                scramble_moves=scramble.moves,
                initial_state=initial.serialize(),
                exact_shortest_distance=reference.distance_htm,
            )
        )
    return tuple(entries)


def primary_puzzle_specs() -> tuple[PuzzleSpec, ...]:
    """Return the exact five inputs in the runner's existing puzzle format."""

    return tuple(entry.puzzle_spec for entry in build_primary_dataset())
