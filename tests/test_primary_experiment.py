"""Offline checks for the pinned primary dataset and analysis path."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from app.benchmark import BenchmarkRunner, save_run
from app.benchmark.primary_dataset import (
    PRIMARY_DATASET_ID,
    build_primary_dataset,
    primary_puzzle_specs,
)
from app.benchmark.report import build_analysis_report, render_markdown
from app.cube import exact_shortest_solution
from app.models import ASSIGNED_MODEL_IDS, ScriptedMockAdapter


class _SteppingClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        current = self.now
        self.now += 0.25
        return current


def _synthetic_run(run_id: str, *, execution_mode: str | None):
    dataset = build_primary_dataset()
    outcomes: list[list[str]] = [[] for _ in ASSIGNED_MODEL_IDS]
    for puzzle in dataset:
        reference = exact_shortest_solution(puzzle.puzzle_spec.initial_state)
        assert reference is not None
        shortest = list(reference.moves)
        outcomes[0].extend(shortest)
        outcomes[1].extend(("U", "U'", *shortest))
        outcomes[2].extend(("R2", TimeoutError("synthetic offline timeout")))
        outcomes[3].append("M")
    adapters = [
        ScriptedMockAdapter(script, model_id=model_id)
        for model_id, script in zip(ASSIGNED_MODEL_IDS, outcomes, strict=True)
    ]
    generated = asyncio.run(
        BenchmarkRunner(clock=_SteppingClock()).run_suite(
            primary_puzzle_specs(),
            adapters,
            run_id=run_id,
        )
    )
    if execution_mode == "live":
        generated = replace(
            generated,
            trials=tuple(replace(trial, provenance="live") for trial in generated.trials),
        )
    return replace(generated, execution_mode=execution_mode)


def test_primary_dataset_is_pinned_seeded_and_exact() -> None:
    dataset = build_primary_dataset()
    assert PRIMARY_DATASET_ID == "rubiks-cube-race-primary-v1"
    assert [
        (item.puzzle_id, item.seed, item.scramble_length, item.scramble_notation, item.exact_shortest_distance)
        for item in dataset
    ] == [
        ("primary-01", 2026, 4, "R F2 D2 F2", 4),
        ("primary-02", 2027, 4, "R B2 F' U'", 4),
        ("primary-03", 2028, 4, "D2 B' U2 L2", 4),
        ("primary-04", 2029, 5, "B F' B2 U2 L2", 4),
        ("primary-05", 2030, 5, "D F R2 B2 F2", 5),
    ]
    assert len({item.initial_state for item in dataset}) == 5
    assert all(len(item.initial_state) == 54 for item in dataset)


def test_primary_runner_uses_same_state_for_all_models_and_controls() -> None:
    run = _synthetic_run("fair-primary", execution_mode="live")
    dataset = {item.puzzle_id: item for item in build_primary_dataset()}

    assert run.max_valid_moves == 30
    assert run.turn_timeout_seconds == 30.0
    assert run.execution_mode == "live"
    assert len(run.trials) == 20
    for puzzle_id, expected in dataset.items():
        trials = [trial for trial in run.trials if trial.puzzle_id == puzzle_id]
        assert {trial.model_id for trial in trials} == set(ASSIGNED_MODEL_IDS)
        assert len({trial.initial_state for trial in trials}) == 1
        assert {trial.initial_state for trial in trials} == {expected.initial_state}
        assert all(trial.seed == expected.seed for trial in trials)
        assert all(trial.exact_shortest_distance == expected.exact_shortest_distance for trial in trials)
        assert all(trial.max_valid_moves == 30 for trial in trials)


def test_report_uses_only_persisted_comparable_live_runs(tmp_path) -> None:
    live = _synthetic_run("synthetic-live-primary", execution_mode="live")
    mock = _synthetic_run("synthetic-mock-primary", execution_mode="mock")
    save_run(live, tmp_path / "live.json")
    save_run(mock, tmp_path / "mock.json")

    report = build_analysis_report(tmp_path)

    assert report["analysis_status"] == "ready"
    assert report["selected_run_id"] == live.run_id
    assert report["eligible_run_ids"] == [live.run_id]
    assert {row["classification"] for row in report["run_classifications"]} == {
        "eligible_live_primary",
        "excluded_mock",
    }
    comparison = report["per_model_comparison"]
    assert list(comparison) == list(ASSIGNED_MODEL_IDS)
    assert comparison[ASSIGNED_MODEL_IDS[0]]["solved_trial_count"] == 5
    assert comparison[ASSIGNED_MODEL_IDS[0]]["average_optimality_gap"] == 0
    assert comparison[ASSIGNED_MODEL_IDS[1]]["average_optimality_gap"] == 2
    assert comparison[ASSIGNED_MODEL_IDS[2]]["termination_counts"]["timeout"] == 5
    assert comparison[ASSIGNED_MODEL_IDS[3]]["termination_counts"]["invalid_move"] == 5
    assert comparison[ASSIGNED_MODEL_IDS[3]]["attempted_request_count"] == 5
    assert report["cost_monitoring"]["actual_spend"] is None
    assert report["assignment_analysis"]["fastest_also_best_gap"]["answer"] in {"yes", "no"}
    markdown = render_markdown(report)
    assert "Descriptive evidence only" in markdown
    assert "not an estimate" in markdown


def test_mock_only_and_legacy_unknown_mode_reports_remain_pending(tmp_path) -> None:
    mock = _synthetic_run("mock-only", execution_mode="mock")
    legacy = _synthetic_run("legacy-unknown", execution_mode=None)
    legacy_data = legacy.to_dict()
    legacy_data["schema_version"] = 1
    legacy_data.pop("execution_mode")
    legacy_data.pop("prompt_version")
    legacy_data.pop("action_previews_enabled")
    for trial in legacy_data["trials"]:
        for attempt in trial["attempts"]:
            attempt.pop("token_usage", None)
    (tmp_path / "legacy.json").write_text(json.dumps(legacy_data), encoding="utf-8")
    save_run(mock, tmp_path / "mock.json")

    report = build_analysis_report(tmp_path)

    assert report["analysis_status"] == "pending"
    assert report["pending_reason"] == "no_eligible_live_primary_run_is_persisted"
    assert report["per_model_comparison"] == {}
    classifications = {row["classification"] for row in report["run_classifications"]}
    assert classifications == {"excluded_mock", "excluded_unknown_mode"}
    markdown = render_markdown(report)
    assert "Assignment analysis is pending" in markdown
    assert "one-request diagnostics are not treated" in markdown


def test_live_smoke_and_multiple_primary_runs_are_not_silently_analyzed(tmp_path) -> None:
    full = _synthetic_run("full-primary", execution_mode="live")
    save_run(full, tmp_path / "full.json")
    first_puzzle_trials = tuple(trial for trial in full.trials if trial.puzzle_id == "primary-01")
    smoke = replace(
        full,
        run_id="live-smoke",
        max_valid_moves=30,
        trials=tuple(replace(trial, trial_id=f"live-smoke:{trial.puzzle_id}:{trial.model_id}") for trial in first_puzzle_trials),
    )
    save_run(smoke, tmp_path / "smoke.json")

    # Two otherwise eligible runs require explicit selection instead of pooling.
    second = _synthetic_run("second-primary", execution_mode="live")
    save_run(second, tmp_path / "second.json")
    ambiguous = build_analysis_report(tmp_path)
    assert ambiguous["analysis_status"] == "pending"
    assert ambiguous["pending_reason"] == "multiple_eligible_runs_require_explicit_run_id_selection"
    assert any(
        row.get("reason") == "puzzle_ids_do_not_match_pinned_primary_dataset"
        for row in ambiguous["run_classifications"]
    )

    selected = build_analysis_report(tmp_path, run_id=full.run_id)
    assert selected["analysis_status"] == "ready"
    assert selected["selected_run_id"] == full.run_id

