"""Offline benchmark runner tests using the shared cube and mock contracts."""

from __future__ import annotations

import asyncio

import pytest

from app.benchmark import BenchmarkRunner, PuzzleSpec, aggregate_metrics
from app.cube import CubeState, exact_shortest_solution, invert_moves
from app.models import (
    ASSIGNED_MODEL_IDS,
    AdapterReply,
    AdapterResponseError,
    BaseModelAdapter,
    ScriptedMockAdapter,
    TurnContext,
    RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
)


class SteppingClock:
    def __init__(self, step: float = 0.25) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.now
        self.now += self.step
        return current


class RecordingAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, outcomes: list[object]) -> None:
        super().__init__(model_id)
        self.outcomes = list(outcomes)
        self.contexts: list[TurnContext] = []

    async def _request_move(self, context: TurnContext) -> AdapterReply:
        self.contexts.append(context)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return AdapterReply(outcome, raw_response={"model": "mock-served-snapshot"})


def _run(coro):
    return asyncio.run(coro)


def test_single_mock_trial_solves_and_records_exact_reference() -> None:
    puzzle = PuzzleSpec("one-turn", ("R",), seed=42)
    adapter = ScriptedMockAdapter(invert_moves(puzzle.scramble_moves), model_id="mock/solver")

    trial = _run(BenchmarkRunner().run_trial(puzzle, adapter))

    assert trial.solved
    assert trial.termination_status == "solved"
    assert trial.move_history == ("R'",)
    assert trial.valid_move_count == 1
    assert trial.attempted_request_count == 1
    assert trial.failed_request_count == 0
    assert trial.exact_shortest_distance == 1
    assert trial.optimality_gap == 0
    assert trial.final_state == CubeState().serialize()
    assert trial.attempts[0].served_model is None  # The deterministic mock has no provider snapshot.


def test_already_solved_puzzle_finishes_without_an_adapter_request() -> None:
    puzzle = PuzzleSpec("already-solved", ())
    adapter = ScriptedMockAdapter(["R'"], model_id="mock/no-call")

    trial = _run(BenchmarkRunner().run_trial(puzzle, adapter))

    assert trial.solved
    assert trial.valid_move_count == 0
    assert trial.attempted_request_count == 0
    assert trial.exact_shortest_distance == 0
    assert trial.optimality_gap == 0


def test_reaching_the_move_cap_is_unsuccessful_and_no_extra_request_is_made() -> None:
    puzzle = PuzzleSpec("thirty-turn-cap", ("R",))
    adapter = ScriptedMockAdapter(["U"] * 30 + ["R'"], model_id="mock/keeps-going")

    trial = _run(BenchmarkRunner().run_trial(puzzle, adapter))

    assert trial.termination_status == "move_limit"
    assert not trial.solved
    assert trial.valid_move_count == 30
    assert trial.attempted_request_count == 30
    assert trial.failed_request_count == 0
    assert trial.optimality_gap is None


def test_move_cap_accepts_a_solution_on_the_last_allowed_move() -> None:
    puzzle = PuzzleSpec("last-move-solves", ("R",))
    adapter = ScriptedMockAdapter(["R'"], model_id="mock/last-move")

    trial = _run(BenchmarkRunner(max_valid_moves=1).run_trial(puzzle, adapter))

    assert trial.solved
    assert trial.valid_move_count == 1
    assert trial.termination_status == "solved"


def test_state_aware_runner_rebuilds_turn_context_after_each_applied_move() -> None:
    puzzle = PuzzleSpec("state-aware-loop", ("R",))
    adapter = RecordingAdapter("test/state-aware", ["U", "R'"])
    runner = BenchmarkRunner(
        max_valid_moves=2,
        prompt_version=RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
        action_previews_enabled=True,
    )

    trial = _run(runner.run_trial(puzzle, adapter))

    assert trial.termination_status == "move_limit"
    assert len(adapter.contexts) == 2
    assert adapter.contexts[0].cube_state == puzzle.initial_state.serialize()
    assert adapter.contexts[0].original_scramble == ("R",)
    assert adapter.contexts[0].recent_moves == ()
    after_u = puzzle.initial_state.apply_move("U")
    assert adapter.contexts[1].cube_state == after_u.serialize()
    assert adapter.contexts[1].original_scramble == ("R",)
    assert adapter.contexts[1].recent_moves == ("U",)
    assert adapter.contexts[1].prompt_version == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
    assert adapter.contexts[1].action_previews_enabled is True


def test_all_assigned_models_get_identical_start_and_failures_are_isolated() -> None:
    puzzle = PuzzleSpec("fairness", ("R", "U"), seed=11)
    adapters = [
        RecordingAdapter(ASSIGNED_MODEL_IDS[0], ["U'", "R'"]),
        RecordingAdapter(ASSIGNED_MODEL_IDS[1], ["M"]),
        RecordingAdapter(ASSIGNED_MODEL_IDS[2], [TimeoutError("private timeout detail")]),
        RecordingAdapter(ASSIGNED_MODEL_IDS[3], [RuntimeError("private provider detail")]),
    ]
    runner = BenchmarkRunner(max_valid_moves=3, clock=SteppingClock())

    run = _run(runner.run_suite([puzzle], adapters, run_id="fair-run"))

    assert [trial.model_id for trial in run.trials] == list(ASSIGNED_MODEL_IDS)
    assert all(trial.initial_state == puzzle.initial_state.serialize() for trial in run.trials)
    assert adapters[0].contexts[0].cube_state == puzzle.initial_state.serialize()
    assert adapters[1].contexts[0].cube_state == puzzle.initial_state.serialize()
    assert adapters[2].contexts[0].cube_state == puzzle.initial_state.serialize()
    assert adapters[3].contexts[0].cube_state == puzzle.initial_state.serialize()
    assert run.trials[0].solved
    assert run.trials[1].termination_status == "invalid_move"
    assert run.trials[2].termination_status == "timeout"
    assert run.trials[3].termination_status == "adapter_error"
    assert all(not trial.solved for trial in run.trials[1:])
    assert all(trial.failed_request_count == 1 for trial in run.trials[1:])
    assert run.trials[1].valid_move_count == 0
    assert run.trials[1].final_state == run.trials[1].initial_state
    assert "private" not in run.trials[2].attempts[0].error_message
    assert "private" not in run.trials[3].attempts[0].error_message


def test_invalid_adapter_result_is_recorded_without_applying_a_move() -> None:
    puzzle = PuzzleSpec("invalid", ("R",))
    adapter = ScriptedMockAdapter(["R3"], model_id="mock/invalid")

    trial = _run(BenchmarkRunner().run_trial(puzzle, adapter))

    assert trial.termination_status == "invalid_move"
    assert trial.valid_move_count == 0
    assert trial.attempted_request_count == 1
    assert trial.failed_request_count == 1
    assert trial.invalid_proposal_count == 1
    assert trial.attempts[0].rejected_move == "R3"
    assert trial.final_state == trial.initial_state
    assert trial.optimality_gap is None


def test_runner_clock_defines_latency_and_total_runtime_consistently() -> None:
    puzzle = PuzzleSpec("controlled-timing", ("R", "U"))
    adapter = ScriptedMockAdapter(["U", "M"], model_id="mock/timing")
    runner = BenchmarkRunner(clock=SteppingClock(step=0.25))

    trial = _run(runner.run_trial(puzzle, adapter))

    assert trial.attempted_request_count == 2
    assert trial.valid_move_count == 1
    assert trial.failed_request_count == 1
    assert trial.average_move_latency_seconds == pytest.approx(0.25)
    assert trial.average_request_latency_seconds == pytest.approx(0.25)
    assert trial.attempts[0].elapsed_trial_seconds == pytest.approx(0.5)
    assert trial.total_runtime_seconds == pytest.approx(1.25)


def test_runner_persists_only_safe_provider_token_usage_for_success_and_error() -> None:
    class UsageAdapter(BaseModelAdapter):
        async def _request_move(self, _context: TurnContext) -> AdapterReply:
            return AdapterReply(
                "R'",
                token_usage={
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "authorization": "must not persist",  # type: ignore[dict-item]
                    "output_tokens": True,  # type: ignore[dict-item]
                },
            )

    class UsageErrorAdapter(BaseModelAdapter):
        async def _request_move(self, _context: TurnContext) -> AdapterReply:
            raise AdapterResponseError(
                model_id=self.model_id,
                detail="safe fixture failure",
                usage={"prompt_tokens": 6, "provider_secret": 99},
            )

    valid = _run(BenchmarkRunner().run_trial(PuzzleSpec("usage", ("R",)), UsageAdapter("mock/usage")))
    failed = _run(
        BenchmarkRunner().run_trial(
            PuzzleSpec("usage-error", ("R",)),
            UsageErrorAdapter("mock/usage-error"),
        )
    )

    assert valid.solved
    assert valid.attempts[0].token_usage == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }
    assert failed.termination_status == "adapter_error"
    assert failed.attempts[0].token_usage == {"prompt_tokens": 6}


def test_aggregate_latency_and_runtime_means_use_controlled_trial_timings() -> None:
    puzzle = PuzzleSpec("aggregate-timing", ("R",))
    adapters = [RecordingAdapter(model_id, ["R'"]) for model_id in ASSIGNED_MODEL_IDS]
    runner = BenchmarkRunner(clock=SteppingClock(step=0.25))

    run = _run(runner.run_suite([puzzle], adapters))
    metrics = aggregate_metrics(run)

    for summary in metrics.values():
        assert summary.average_move_latency_seconds == pytest.approx(0.25)
        assert summary.average_request_latency_seconds == pytest.approx(0.25)
        assert summary.average_total_runtime_seconds == pytest.approx(0.75)
        assert summary.average_valid_move_count == pytest.approx(1.0)


def test_suite_rejects_missing_assigned_models_and_empty_explicit_run_id() -> None:
    puzzle = PuzzleSpec("incomplete", ("R",))
    adapters = [ScriptedMockAdapter(["R'"], model_id=ASSIGNED_MODEL_IDS[0])]
    runner = BenchmarkRunner()

    with pytest.raises(ValueError, match="exactly the four assigned model IDs"):
        _run(runner.run_suite([puzzle], adapters))

    complete = [ScriptedMockAdapter(["R'"], model_id=model_id) for model_id in ASSIGNED_MODEL_IDS]
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        _run(runner.run_suite([puzzle], complete, run_id=""))


def test_reference_search_is_skipped_for_extension_scrambles() -> None:
    calls: list[str] = []

    def forbidden_reference(state: CubeState):
        calls.append(state.serialize())
        return exact_shortest_solution(state)

    puzzle = PuzzleSpec("extension", ("R", "U", "F", "L", "B", "D"))
    runner = BenchmarkRunner(max_valid_moves=1, reference_solver=forbidden_reference)
    trial = _run(runner.run_trial(puzzle, ScriptedMockAdapter(["U"], model_id="mock/extension")))

    assert calls == []
    assert trial.exact_shortest_distance is None
    assert trial.optimality_gap is None


@pytest.mark.parametrize("limit", [0, 31, True])
def test_runner_rejects_invalid_move_caps(limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BenchmarkRunner(max_valid_moves=limit)  # type: ignore[arg-type]


def test_seeded_puzzle_specs_keep_scramble_and_seed_reproducible() -> None:
    first = PuzzleSpec.from_seed("seeded", seed=2026, length=5)
    second = PuzzleSpec.from_seed("seeded", seed=2026, length=5)

    assert first == second
    assert first.seed == 2026
    assert len(first.scramble_moves) == 5
    assert first.initial_state == second.initial_state


def test_derived_run_id_distinguishes_prompt_and_action_preview_conditions() -> None:
    puzzle = PuzzleSpec("condition-id", ())

    def run_for(runner: BenchmarkRunner):
        adapters = [ScriptedMockAdapter([], model_id=model_id) for model_id in ASSIGNED_MODEL_IDS]
        return _run(runner.run_suite((puzzle,), adapters))

    v1 = run_for(BenchmarkRunner())
    v2_without_previews = run_for(
        BenchmarkRunner(prompt_version=RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE)
    )
    v2_with_previews = run_for(
        BenchmarkRunner(
            prompt_version=RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
            action_previews_enabled=True,
        )
    )

    assert len({v1.run_id, v2_without_previews.run_id, v2_with_previews.run_id}) == 3


def test_five_puzzles_run_as_twenty_comparable_trials() -> None:
    puzzles = [
        PuzzleSpec.from_seed(f"primary-{index:02d}", seed=2026 + index, length=4)
        for index in range(1, 6)
    ]
    outcomes = [
        move
        for puzzle in puzzles
        for move in invert_moves(puzzle.scramble_moves)
    ]
    adapters = [
        ScriptedMockAdapter(outcomes, model_id=model_id)
        for model_id in ASSIGNED_MODEL_IDS
    ]

    run = _run(BenchmarkRunner().run_suite(puzzles, adapters, run_id="five-primary-puzzles"))

    assert len(run.trials) == 20
    assert all(trial.solved for trial in run.trials)
    for puzzle in puzzles:
        trials = [trial for trial in run.trials if trial.puzzle_id == puzzle.puzzle_id]
        assert {trial.model_id for trial in trials} == set(ASSIGNED_MODEL_IDS)
        assert len({trial.initial_state for trial in trials}) == 1
