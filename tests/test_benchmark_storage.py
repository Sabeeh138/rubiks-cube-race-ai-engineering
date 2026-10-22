"""Offline integration and JSON persistence tests for benchmark results."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from app.benchmark import (
    BenchmarkRunner,
    PuzzleSpec,
    ResultsFormatError,
    ResultsStorageError,
    aggregate_metrics,
    load_run,
    save_run,
)
from app.cube import invert_moves
from app.models import ASSIGNED_MODEL_IDS, ScriptedMockAdapter


def _integration_run():
    puzzle = PuzzleSpec("primary-01", ("R", "U"), seed=90210)
    adapters = (
        ScriptedMockAdapter(invert_moves(puzzle.scramble_moves), model_id=ASSIGNED_MODEL_IDS[0]),
        ScriptedMockAdapter(("U", "U'", "U'", "R'"), model_id=ASSIGNED_MODEL_IDS[1]),
        ScriptedMockAdapter(("U", "U", "U", "U"), model_id=ASSIGNED_MODEL_IDS[2]),
        ScriptedMockAdapter(("M",), model_id=ASSIGNED_MODEL_IDS[3]),
    )
    return asyncio.run(
        BenchmarkRunner(max_valid_moves=4).run_suite((puzzle,), adapters, run_id="integration-01")
    )


def test_cube_mock_runner_and_json_storage_integrate_offline(tmp_path) -> None:
    run = _integration_run()
    destination = tmp_path / "nested" / "results" / "mock-run.json"

    saved_path = save_run(run, destination)
    loaded = load_run(saved_path)

    assert saved_path == destination
    assert loaded == run
    assert len(loaded.trials) == 4
    assert all(trial.initial_state == loaded.trials[0].initial_state for trial in loaded.trials)
    assert loaded.trials[0].solved and loaded.trials[0].optimality_gap == 0
    assert loaded.trials[1].solved and loaded.trials[1].optimality_gap == 2
    assert loaded.trials[2].termination_status == "move_limit"
    assert loaded.trials[2].optimality_gap is None
    assert loaded.trials[3].termination_status == "invalid_move"
    assert loaded.trials[3].optimality_gap is None
    assert loaded.trials[3].attempted_request_count == 1
    assert loaded.trials[3].valid_move_count == 0
    assert loaded.trials[3].attempts[0].rejected_move == "M"
    assert all(trial.retry_count == 0 for trial in loaded.trials)

    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 4
    assert persisted["execution_mode"] is None
    assert persisted["prompt_version"] == "rubiks-move-objective-v1"
    assert persisted["action_previews_enabled"] is False
    assert [trial["model_id"] for trial in persisted["trials"]] == list(ASSIGNED_MODEL_IDS)
    assert persisted["trials"][0]["scramble_moves"] == ["R", "U"]
    assert persisted["trials"][0]["seed"] == 90210
    assert persisted["trials"][0]["attempts"][0]["cube_state_before"] == persisted["trials"][0]["initial_state"]


def test_aggregate_metric_denominators_and_gap_exclusions_are_explicit() -> None:
    metrics = aggregate_metrics(_integration_run())

    first = metrics[ASSIGNED_MODEL_IDS[0]]
    second = metrics[ASSIGNED_MODEL_IDS[1]]
    move_limited = metrics[ASSIGNED_MODEL_IDS[2]]
    invalid = metrics[ASSIGNED_MODEL_IDS[3]]
    assert first.trial_count == 1
    assert first.solve_rate == 1.0
    assert first.average_valid_move_count == 2.0
    assert first.average_move_latency_seconds is not None
    assert first.optimality_gap_sample_count == 1
    assert first.average_optimality_gap == 0.0
    assert second.average_optimality_gap == 2.0
    assert move_limited.solve_rate == 0.0
    assert move_limited.unsuccessful_trial_count == 1
    assert move_limited.optimality_gap_sample_count == 0
    assert move_limited.average_optimality_gap is None
    assert invalid.invalid_proposal_count == 1
    assert invalid.failed_request_count == 1
    assert invalid.average_move_latency_seconds is None
    assert invalid.average_request_latency_seconds is not None


def test_malformed_json_and_inconsistent_derived_counts_are_rejected(tmp_path) -> None:
    malformed_json = tmp_path / "broken.json"
    malformed_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ResultsFormatError, match="not valid JSON"):
        load_run(malformed_json)

    inconsistent = tmp_path / "inconsistent.json"
    data = _integration_run().to_dict()
    data["trials"][0]["valid_move_count"] = 99  # type: ignore[index]
    inconsistent.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ResultsFormatError, match="supported schema"):
        load_run(inconsistent)


def test_missing_result_file_and_unwritable_parent_have_clear_errors(tmp_path) -> None:
    with pytest.raises(ResultsStorageError, match="could not read"):
        load_run(tmp_path / "does-not-exist.json")

    blocked_parent = tmp_path / "a-file"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ResultsStorageError, match="could not save"):
        save_run(_integration_run(), blocked_parent / "results.json")


def test_execution_mode_and_safe_token_usage_survive_json_round_trip(tmp_path) -> None:
    run = _integration_run()
    trial = run.trials[0]
    attempt = replace(
        trial.attempts[0],
        token_usage={"prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17},
    )
    updated_trial = replace(trial, attempts=(attempt, *trial.attempts[1:]))
    live_trials = tuple(replace(item, provenance="live") for item in run.trials)
    live_run = replace(
        run,
        trials=(replace(updated_trial, provenance="live"), *live_trials[1:]),
        execution_mode="live",
    )

    path = save_run(live_run, tmp_path / "live.json")
    loaded = load_run(path)

    assert loaded.execution_mode == "live"
    assert loaded.trials[0].attempts[0].token_usage == {
        "prompt_tokens": 14,
        "completion_tokens": 3,
        "total_tokens": 17,
    }


def test_legacy_schema_v1_loads_as_unknown_execution_mode(tmp_path) -> None:
    data = _integration_run().to_dict()
    data["schema_version"] = 1
    data.pop("execution_mode")
    data.pop("prompt_version")
    data.pop("action_previews_enabled")
    for trial in data["trials"]:
        trial.pop("provenance", None)
        trial.pop("provenance_label", None)
        for attempt in trial["attempts"]:
            attempt.pop("token_usage", None)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_run(path)

    assert loaded.schema_version == 1
    assert loaded.execution_mode is None


@pytest.mark.parametrize("mode, provenance", [("mock", "mock"), ("live", "live")])
def test_legacy_schema_v2_preserves_persisted_run_mode_and_infers_trial_provenance(
    mode: str, provenance: str, tmp_path
) -> None:
    data = _integration_run().to_dict()
    data["schema_version"] = 2
    data["execution_mode"] = mode
    data.pop("prompt_version")
    data.pop("action_previews_enabled")
    for trial in data["trials"]:
        trial.pop("provenance", None)
        trial.pop("provenance_label", None)
    path = tmp_path / f"legacy-v2-{mode}.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_run(path)

    assert loaded.schema_version == 2
    assert loaded.execution_mode == mode
    assert all(trial.provenance == provenance for trial in loaded.trials)


def test_legacy_schema_v3_loads_without_inventing_a_prompt_condition(tmp_path) -> None:
    data = _integration_run().to_dict()
    data["schema_version"] = 3
    data.pop("prompt_version")
    data.pop("action_previews_enabled")
    path = tmp_path / "legacy-v3.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_run(path)

    assert loaded.schema_version == 3
    assert loaded.prompt_version is None
    assert loaded.action_previews_enabled is False


def test_token_usage_rejects_untrusted_or_invalid_counters() -> None:
    run = _integration_run()
    trial = run.trials[0]
    with pytest.raises(ValueError, match="unsupported counter"):
        replace(trial.attempts[0], token_usage={"api_key": 1})
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(trial.attempts[0], token_usage={"prompt_tokens": True})
