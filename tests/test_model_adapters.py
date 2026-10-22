"""Offline contract and scripted mock tests for model adapters."""

import asyncio

import pytest

from app.cube import CubeState, generate_scramble, invert_moves
from app.models import (
    AdapterExecutionError,
    AdapterTimeoutError,
    BaseModelAdapter,
    InvalidMoveError,
    ModelAdapter,
    MockScriptExhaustedError,
    ScriptedMockAdapter,
    TurnContext,
)


def _context(
    *,
    state: CubeState | None = None,
    legal_moves: tuple[str, ...] | None = None,
    timeout_seconds: float | None = 1.0,
) -> TurnContext:
    kwargs: dict[str, object] = {
        "cube_state": (state or CubeState()).serialize(),
        "timeout_seconds": timeout_seconds,
    }
    if legal_moves is not None:
        kwargs["legal_moves"] = legal_moves
    return TurnContext(**kwargs)  # type: ignore[arg-type]


def _choose(adapter: ModelAdapter, context: TurnContext):
    return asyncio.run(adapter.choose_move(context))


def test_scripted_mock_satisfies_common_interface_and_returns_metadata() -> None:
    adapter = ScriptedMockAdapter(["R", "U2"], model_id="mock/fixed-fixture")
    assert isinstance(adapter, ModelAdapter)

    first = _choose(adapter, _context())
    second = _choose(adapter, _context())

    assert first.move == "R"
    assert second.move == "U2"
    assert first.model_id == "mock/fixed-fixture"
    assert first.elapsed_seconds >= 0
    assert first.raw_response == {
        "mock": True,
        "script_index": 0,
        "scripted_outcome": "R",
    }


def test_scripted_mock_can_replay_a_known_solution_for_a_seeded_scramble() -> None:
    scramble = generate_scramble(seed=2026, length=4)
    state = scramble.apply_to()
    solution = invert_moves(scramble.moves)
    adapter = ScriptedMockAdapter(solution)

    for _ in solution:
        result = _choose(
            adapter,
            _context(state=state),
        )
        state = state.apply_move(result.move)

    assert state.is_solved


@pytest.mark.parametrize("invalid", ["M", "R3", "R U", " R", "r", ""])
def test_mock_invalid_outputs_raise_structured_error(invalid: str) -> None:
    adapter = ScriptedMockAdapter([invalid], model_id="mock/invalid")

    with pytest.raises(InvalidMoveError) as caught:
        _choose(adapter, _context())

    assert caught.value.model_id == "mock/invalid"
    assert caught.value.proposed_move == invalid


def test_move_must_be_allowed_by_the_current_turn_context() -> None:
    adapter = ScriptedMockAdapter(["U"])

    with pytest.raises(InvalidMoveError) as caught:
        _choose(adapter, _context(legal_moves=("R",)))

    assert caught.value.legal_moves == ("R",)


def test_timeout_script_is_reported_as_typed_timeout() -> None:
    adapter = ScriptedMockAdapter([TimeoutError("simulated timeout")])

    with pytest.raises(AdapterTimeoutError) as caught:
        _choose(adapter, _context(timeout_seconds=0.5))

    assert caught.value.model_id == "mock/scripted"
    assert caught.value.timeout_seconds == 0.5
    assert caught.value.elapsed_seconds >= 0


def test_exception_script_is_wrapped_without_losing_original_error() -> None:
    cause = RuntimeError("simulated adapter failure")
    adapter = ScriptedMockAdapter([cause])

    with pytest.raises(AdapterExecutionError) as caught:
        _choose(adapter, _context())

    assert caught.value.model_id == "mock/scripted"
    assert caught.value.original_error is cause


def test_mock_exhaustion_is_a_clear_adapter_error() -> None:
    adapter = ScriptedMockAdapter(["R"])
    _choose(adapter, _context())

    with pytest.raises(MockScriptExhaustedError, match="no remaining outcomes"):
        _choose(adapter, _context())


@pytest.mark.parametrize("bad_moves", [("M",), ("R", "R"), ()])
def test_turn_context_rejects_invalid_legal_move_sets(bad_moves: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        _context(legal_moves=bad_moves)


def test_turn_context_rejects_invalid_serialized_cube_state() -> None:
    with pytest.raises(ValueError, match="exactly 54"):
        TurnContext(cube_state="bad")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), True])
def test_turn_context_rejects_invalid_timeouts(timeout: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TurnContext(cube_state=CubeState().serialize(), timeout_seconds=timeout)  # type: ignore[arg-type]


def test_adapter_timeout_context_cancels_a_slow_adapter() -> None:
    class SlowAdapter(BaseModelAdapter):
        async def _request_move(self, context: TurnContext):
            await asyncio.sleep(0.02)
            from app.models import AdapterReply

            return AdapterReply("R")

    with pytest.raises(AdapterTimeoutError):
        _choose(SlowAdapter("mock/slow"), _context(timeout_seconds=0.001))
