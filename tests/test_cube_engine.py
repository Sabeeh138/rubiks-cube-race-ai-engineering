import random

import pytest

from app.cube import (
    FACE_ORDER,
    LEGAL_MOVES,
    SOLVED_FACELETS,
    CubeState,
    invert_move,
    invert_moves,
)


def test_default_cube_is_solved_and_serializes_to_canonical_facelets() -> None:
    cube = CubeState()

    assert cube.is_solved
    assert cube.serialize() == SOLVED_FACELETS
    assert len(cube.serialize()) == 54
    assert cube.serialize() == "".join(face * 9 for face in FACE_ORDER)


def test_facelet_serialization_round_trips_scrambled_state() -> None:
    scrambled = CubeState().apply_moves("R U F2 D' L B")

    restored = CubeState.from_facelets(scrambled.serialize())

    assert restored == scrambled
    assert restored.serialize() == scrambled.facelets


def test_clockwise_right_turn_uses_standard_facelet_orientation() -> None:
    assert CubeState().apply_move("R").serialize() == (
        "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"
    )


@pytest.mark.parametrize("move", LEGAL_MOVES)
def test_each_legal_move_changes_the_solved_state_and_keeps_facelet_counts(move: str) -> None:
    result = CubeState().apply_move(move)

    assert not result.is_solved
    assert len(result.facelets) == 54
    assert {face: result.facelets.count(face) for face in FACE_ORDER} == {
        face: 9 for face in FACE_ORDER
    }
    assert all(
        result.facelets[face_index * 9 + 4] == face
        for face_index, face in enumerate(FACE_ORDER)
    )


@pytest.mark.parametrize("move", LEGAL_MOVES)
def test_every_legal_move_and_its_inverse_restore_solved_state(move: str) -> None:
    result = CubeState().apply_move(move)

    assert result.apply_move(invert_move(move)).is_solved


def test_inverse_sequence_restores_state_after_a_move_sequence() -> None:
    moves = ("R", "U", "F2", "L'", "D", "B2", "U'")
    scrambled = CubeState().apply_moves(moves)

    assert not scrambled.is_solved
    assert scrambled.apply_moves(invert_moves(moves)).is_solved


def test_applying_a_sequence_matches_applying_each_move_separately() -> None:
    moves = ("R", "U", "R'", "U'", "F2")
    one_at_a_time = CubeState()
    for move in moves:
        one_at_a_time = one_at_a_time.apply_move(move)

    assert CubeState().apply_moves(moves) == one_at_a_time
    assert CubeState().apply_moves("R U R' U' F2") == one_at_a_time
    assert one_at_a_time.apply_moves(()) is one_at_a_time


def test_successors_match_applying_each_shared_legal_move() -> None:
    cube = CubeState().apply_moves("R U")
    successors = cube.successors()

    assert tuple(move for move, _ in successors) == LEGAL_MOVES
    assert dict(successors) == {move: cube.apply_move(move) for move in LEGAL_MOVES}
    assert not cube.is_solved


@pytest.mark.parametrize("seed", [0, 1, 2026, -13])
def test_random_legal_move_sequences_preserve_facelet_invariants_and_invert(seed: int) -> None:
    rng = random.Random(seed)
    moves = tuple(rng.choice(LEGAL_MOVES) for _ in range(40))

    scrambled = CubeState().apply_moves(moves)

    assert not scrambled.is_solved
    assert len(scrambled.facelets) == 54
    assert all(scrambled.facelets.count(face) == 9 for face in FACE_ORDER)
    assert scrambled.apply_moves(invert_moves(moves)).is_solved


@pytest.mark.parametrize("invalid_move", ["X", "M", "r", "R3", "R 2", ""])
def test_nonstandard_or_malformed_moves_are_rejected(invalid_move: str) -> None:
    with pytest.raises(ValueError, match="invalid face-turn move"):
        CubeState().apply_move(invalid_move)


def test_non_iterable_move_sequence_is_rejected() -> None:
    with pytest.raises(TypeError, match="moves must be"):
        CubeState().apply_moves(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("facelets", "error"),
    [
        (SOLVED_FACELETS[:-1], "exactly 54"),
        (SOLVED_FACELETS[:-1] + "X", "invalid symbols"),
        ("R" + SOLVED_FACELETS[1:], "exactly nine"),
        (SOLVED_FACELETS[:4] + "R" + SOLVED_FACELETS[5:13] + "U" + SOLVED_FACELETS[14:], "center"),
    ],
)
def test_invalid_facelet_strings_are_rejected(facelets: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        CubeState.from_facelets(facelets)


def test_non_string_facelet_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="facelets must be a string"):
        CubeState.from_facelets(None)  # type: ignore[arg-type]
