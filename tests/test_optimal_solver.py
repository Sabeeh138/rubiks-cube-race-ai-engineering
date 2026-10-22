import pytest

from app.cube import (
    LEGAL_MOVES,
    MAX_REFERENCE_DEPTH,
    SOLVED_FACELETS,
    CubeState,
    exact_shortest_solution,
    generate_scramble,
)


def _exhaustive_distance_at_most(target: CubeState, max_depth: int) -> int | None:
    """Small independent one-sided enumeration used to check known fixtures."""

    if target.is_solved:
        return 0
    visited = {SOLVED_FACELETS}
    frontier = [CubeState()]
    for depth in range(1, max_depth + 1):
        next_frontier: list[CubeState] = []
        for state in frontier:
            for move in LEGAL_MOVES:
                neighbor = state.apply_move(move)
                if neighbor.facelets == target.facelets:
                    return depth
                if neighbor.facelets not in visited:
                    visited.add(neighbor.facelets)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return None


def test_solved_cube_has_exact_distance_zero() -> None:
    result = exact_shortest_solution(CubeState())

    assert result is not None
    assert result.distance_htm == 0
    assert result.moves == ()
    assert result.notation == ""


def test_one_move_state_has_exact_distance_one() -> None:
    state = CubeState().apply_move("R")
    result = exact_shortest_solution(state)

    assert result is not None
    assert result.distance_htm == 1
    assert len(result.moves) == 1
    assert state.apply_moves(result.moves).is_solved
    assert _exhaustive_distance_at_most(state, 0) is None


def test_known_short_sequences_match_independent_exhaustive_lower_bounds() -> None:
    # Each sequence gives an upper bound; one-sided enumeration confirms no
    # shorter solution exists for these fixed, hand-checkable cases.
    fixtures = [
        ("R U", 2),
        ("R U R'", 3),
    ]
    for notation, expected_distance in fixtures:
        state = CubeState().apply_moves(notation)
        result = exact_shortest_solution(state)

        assert _exhaustive_distance_at_most(state, expected_distance - 1) is None
        assert result is not None
        assert result.distance_htm == expected_distance
        assert len(result.moves) == expected_distance
        assert state.apply_moves(result.moves).is_solved


def test_half_turn_is_one_htm_move_and_repeated_turns_share_its_state() -> None:
    half_turn = CubeState().apply_move("R2")
    two_quarter_turns = CubeState().apply_moves("R R")

    assert half_turn == two_quarter_turns
    result = exact_shortest_solution(half_turn)
    assert result is not None
    assert result.distance_htm == 1
    assert len(result.moves) == 1
    assert half_turn.apply_moves(result.moves).is_solved


def test_inverse_and_canceling_sequences_have_distance_zero() -> None:
    state = CubeState().apply_moves("R U U' R'")

    result = exact_shortest_solution(state)

    assert state.is_solved
    assert result is not None
    assert result.distance_htm == 0
    assert result.moves == ()


def test_search_handles_short_scrambles_from_existing_seeded_generator() -> None:
    for seed, length in ((4, 4), (19, 5)):
        scramble = generate_scramble(seed=seed, length=length)
        state = scramble.apply_to()
        result = exact_shortest_solution(state)

        assert result is not None
        assert result.distance_htm <= scramble.move_count_htm
        assert result.distance_htm == len(result.moves)
        assert state.apply_moves(result.moves).is_solved
        assert exact_shortest_solution(state) == result


def test_reference_search_reports_no_solution_beyond_requested_depth() -> None:
    state = CubeState().apply_moves("R U")

    assert exact_shortest_solution(state, max_depth=1) is None


def test_search_depth_is_capped_to_the_primary_benchmark_scope() -> None:
    assert MAX_REFERENCE_DEPTH == 5
    with pytest.raises(ValueError, match="between 0 and 5"):
        exact_shortest_solution(CubeState(), max_depth=MAX_REFERENCE_DEPTH + 1)


@pytest.mark.parametrize("max_depth", [True, 2.5, "5"])
def test_invalid_search_depth_type_is_rejected(max_depth: object) -> None:
    with pytest.raises(TypeError, match="max_depth must be an integer"):
        exact_shortest_solution(CubeState(), max_depth=max_depth)  # type: ignore[arg-type]


def test_non_cube_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="cube must be a CubeState"):
        exact_shortest_solution(None)  # type: ignore[arg-type]
