import pytest

from app.cube import CubeState, generate_scramble, invert_moves


def test_scramble_is_reproducible_from_seed_and_length() -> None:
    first = generate_scramble(seed=2026, length=25)
    second = generate_scramble(seed=2026, length=25)

    assert first == second
    assert first.seed == 2026
    assert len(first.moves) == 25
    assert first.move_count_htm == 25
    assert first.notation == " ".join(first.moves)


def test_scramble_avoids_consecutive_turns_of_the_same_face() -> None:
    scramble = generate_scramble(seed=14, length=100)

    assert all(left[0] != right[0] for left, right in zip(scramble.moves, scramble.moves[1:]))


def test_different_seeds_generate_different_sequences_for_normal_length() -> None:
    assert generate_scramble(seed=1).moves != generate_scramble(seed=2).moves


def test_scramble_and_its_inverse_restore_the_starting_state() -> None:
    scramble = generate_scramble(seed=2026, length=30)
    scrambled = scramble.apply_to()

    assert not scrambled.is_solved
    assert scrambled.apply_moves(invert_moves(scramble.moves)).is_solved
    assert scramble.undo_from(scrambled).is_solved


def test_zero_length_scramble_is_a_reproducible_no_op() -> None:
    scramble = generate_scramble(seed=5, length=0)

    assert scramble.moves == ()
    assert scramble.move_count_htm == 0
    assert scramble.apply_to() == CubeState()


@pytest.mark.parametrize("seed", [None, True, 1.5, "42"])
def test_non_integer_seed_is_rejected(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        generate_scramble(seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("length", [True, 2.5, "10"])
def test_non_integer_length_is_rejected(length: object) -> None:
    with pytest.raises(TypeError, match="length must be an integer"):
        generate_scramble(seed=1, length=length)  # type: ignore[arg-type]


def test_negative_scramble_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="length must be zero or greater"):
        generate_scramble(seed=1, length=-1)
