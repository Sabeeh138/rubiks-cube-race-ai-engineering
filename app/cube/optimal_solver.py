"""Exact shortest-path reference for short benchmark states."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from app.cube.engine import (
    SOLVED_FACELETS,
    CubeState,
    invert_moves,
)

MAX_REFERENCE_DEPTH = 5


@dataclass(frozen=True, slots=True)
class ShortestSolution:
    """A shortest HTM solution found within the reference search limit."""

    distance_htm: int
    moves: tuple[str, ...]
    explored_states: int

    @property
    def notation(self) -> str:
        return " ".join(self.moves)


_Parent = tuple[str, str] | None


def _expand_layer(
    frontier: Iterable[str],
    depths: dict[str, int],
    parents: dict[str, _Parent],
) -> list[str]:
    """Expand one complete BFS layer through the shared engine move API."""

    next_frontier: list[str] = []
    for facelets in frontier:
        cube = CubeState.from_facelets(facelets)
        next_depth = depths[facelets] + 1
        for move, neighbor_cube in cube.successors():
            neighbor = neighbor_cube.serialize()
            if neighbor in depths:
                continue
            depths[neighbor] = next_depth
            parents[neighbor] = (facelets, move)
            next_frontier.append(neighbor)
    return next_frontier


def _path_from_root(facelets: str, parents: dict[str, _Parent]) -> tuple[str, ...]:
    reverse_path: list[str] = []
    current = facelets
    link = parents[current]
    while link is not None:
        parent, move = link
        reverse_path.append(move)
        current = parent
        link = parents[current]
    return tuple(reversed(reverse_path))


@lru_cache(maxsize=2048)
def _search_facelets(facelets: str, max_depth: int) -> ShortestSolution | None:
    if facelets == SOLVED_FACELETS:
        return ShortestSolution(distance_htm=0, moves=(), explored_states=1)
    if max_depth == 0:
        return None

    # Explore complete balls whose radii sum to max_depth. Any path within the
    # cap has a point no farther than each radius from its respective endpoint.
    # Every shared state therefore gives a valid candidate path; minimizing
    # the sum of BFS depths returns the shortest path in the searched range.
    start_radius = max_depth // 2
    goal_radius = max_depth - start_radius
    start_depths = {facelets: 0}
    goal_depths = {SOLVED_FACELETS: 0}
    start_parents: dict[str, _Parent] = {facelets: None}
    goal_parents: dict[str, _Parent] = {SOLVED_FACELETS: None}
    start_frontier = [facelets]
    goal_frontier = [SOLVED_FACELETS]

    for _ in range(start_radius):
        start_frontier = _expand_layer(start_frontier, start_depths, start_parents)
    for _ in range(goal_radius):
        goal_frontier = _expand_layer(goal_frontier, goal_depths, goal_parents)

    shared = start_depths.keys() & goal_depths.keys()
    if not shared:
        return None

    meeting = min(
        shared,
        key=lambda state: (start_depths[state] + goal_depths[state], state),
    )
    distance = start_depths[meeting] + goal_depths[meeting]
    if distance > max_depth:
        return None

    from_start = _path_from_root(meeting, start_parents)
    from_goal_root = _path_from_root(meeting, goal_parents)
    solution = from_start + invert_moves(from_goal_root)
    explored = len(start_depths.keys() | goal_depths.keys())
    return ShortestSolution(
        distance_htm=distance,
        moves=solution,
        explored_states=explored,
    )


def exact_shortest_solution(
    cube: CubeState,
    *,
    max_depth: int = MAX_REFERENCE_DEPTH,
) -> ShortestSolution | None:
    """Find an exact shortest HTM solution for a state within a small depth cap.

    The default and maximum supported search depth is five HTM moves, matching
    the primary benchmark's short scrambles. ``None`` means no solution was
    found within ``max_depth``; it does not estimate an arbitrary state's full
    distance. Exact results are cached by canonical state and depth cap.
    """

    if not isinstance(cube, CubeState):
        raise TypeError("cube must be a CubeState")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise TypeError("max_depth must be an integer")
    if not 0 <= max_depth <= MAX_REFERENCE_DEPTH:
        raise ValueError(f"max_depth must be between 0 and {MAX_REFERENCE_DEPTH}")

    return _search_facelets(cube.serialize(), max_depth)
