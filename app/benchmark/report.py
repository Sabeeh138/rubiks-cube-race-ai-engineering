"""Offline primary-experiment analysis from persisted JSON benchmark runs.

This module never constructs adapters or performs network access. Mock runs,
legacy runs whose execution mode was not persisted, and runs outside the pinned
five-puzzle protocol are excluded from assignment analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from app.benchmark.metrics import aggregate_metrics
from app.benchmark.primary_dataset import (
    PRIMARY_DATASET_ID,
    PrimaryPuzzle,
    build_primary_dataset,
)
from app.benchmark.results import BenchmarkRun, TrialResult
from app.benchmark.storage import ResultsStorageError, load_run
from app.models import ASSIGNED_MODEL_IDS
from app.models.prompts import RUBIKS_MOVE_OBJECTIVE_V1

PRIMARY_MAX_VALID_MOVES = 30
PRIMARY_TIMEOUT_SECONDS = 30.0
_FAILED_TERMINATIONS = {"invalid_move", "timeout", "adapter_error"}
_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)


def build_analysis_report(
    results_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Read result JSON files and analyze one eligible live primary run.

    If no run ID is supplied, analysis proceeds only when exactly one eligible
    run exists. Multiple eligible runs require explicit selection to prevent
    accidental pooling of repeats or unrelated experiment sessions.
    """

    directory = Path(results_dir)
    dataset = build_primary_dataset()
    classifications: list[dict[str, str]] = []
    eligible: dict[str, BenchmarkRun] = {}
    try:
        paths = sorted(directory.glob("*.json"), key=lambda path: path.name)
    except OSError:
        paths = []

    for path in paths:
        try:
            run = load_run(path)
        except ResultsStorageError:
            classifications.append(
                {"file": path.name, "classification": "unreadable", "reason": "invalid_result_file"}
            )
            continue
        if run.execution_mode == "mock":
            classifications.append(
                {"file": path.name, "run_id": run.run_id, "classification": "excluded_mock"}
            )
            continue
        if run.execution_mode == "recording":
            classifications.append(
                {
                    "file": path.name,
                    "run_id": run.run_id,
                    "classification": "excluded_recording_demo",
                    "reason": "contains_scripted_claude_trial_and_is_not_an_official_live_run",
                }
            )
            continue
        if run.execution_mode == "diagnostic":
            classifications.append(
                {
                    "file": path.name,
                    "run_id": run.run_id,
                    "classification": "excluded_exploratory_diagnostic",
                    "reason": "one_move_exploratory_run_is_separate_from_the_official_primary_dataset",
                }
            )
            continue
        if run.execution_mode == "state_aware_experiment":
            classifications.append(
                {
                    "file": path.name,
                    "run_id": run.run_id,
                    "classification": "excluded_exploratory_state_aware_experiment",
                    "reason": "v2_state_aware_prompt_and_action_previews_are_separate_from_the_frozen_official_v1_protocol",
                }
            )
            continue
        if run.execution_mode != "live":
            classifications.append(
                {
                    "file": path.name,
                    "run_id": run.run_id,
                    "classification": "excluded_unknown_mode",
                    "reason": "execution_mode_was_not_persisted",
                }
            )
            continue
        reason = _primary_ineligibility_reason(run, dataset)
        if reason is not None:
            classifications.append(
                {
                    "file": path.name,
                    "run_id": run.run_id,
                    "classification": "excluded_noncomparable_live_run",
                    "reason": reason,
                }
            )
            continue
        eligible[run.run_id] = run
        classifications.append(
            {"file": path.name, "run_id": run.run_id, "classification": "eligible_live_primary"}
        )

    selected: BenchmarkRun | None = None
    pending_reason: str | None = None
    if run_id is not None:
        selected = eligible.get(run_id)
        if selected is None:
            pending_reason = "requested_run_id_is_not_an_eligible_live_primary_run"
    elif len(eligible) == 1:
        selected = next(iter(eligible.values()))
    elif not eligible:
        pending_reason = "no_eligible_live_primary_run_is_persisted"
    else:
        pending_reason = "multiple_eligible_runs_require_explicit_run_id_selection"

    report: dict[str, Any] = {
        "report_version": 1,
        "dataset_id": PRIMARY_DATASET_ID,
        "analysis_status": "ready" if selected is not None else "pending",
        "selected_run_id": selected.run_id if selected is not None else None,
        "eligible_run_ids": sorted(eligible),
        "pending_reason": pending_reason,
        "puzzle_dataset": [entry.to_dict() for entry in dataset],
        "files_scanned": len(paths),
        "run_classifications": classifications,
        "per_model_comparison": {},
        "assignment_analysis": {},
        "cost_monitoring": {
            "actual_spend": None,
            "status": "not_recorded_in_benchmark_results",
            "source_of_truth": "OpenRouter account usage/activity for the experiment time window",
            "note": "No cost is estimated from catalog rates or token counts.",
        },
    }
    if selected is not None:
        report["per_model_comparison"] = _model_comparison(selected)
        report["assignment_analysis"] = _assignment_answers(report["per_model_comparison"])
    return report


def _primary_ineligibility_reason(
    run: BenchmarkRun,
    dataset: Sequence[PrimaryPuzzle],
) -> str | None:
    if run.execution_mode != "live":
        return "execution_mode_is_not_live"
    if run.prompt_version != RUBIKS_MOVE_OBJECTIVE_V1 or run.action_previews_enabled:
        return "prompt_condition_does_not_match_the_frozen_official_v1_protocol"
    if any(trial.provenance != "live" for trial in run.trials):
        return "nonlive_trial_provenance_is_present"
    if run.max_valid_moves != PRIMARY_MAX_VALID_MOVES:
        return "max_valid_moves_does_not_match_primary_protocol"
    if run.turn_timeout_seconds is None or not math.isclose(
        run.turn_timeout_seconds,
        PRIMARY_TIMEOUT_SECONDS,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        return "turn_timeout_does_not_match_primary_protocol"

    expected = {entry.puzzle_id: entry for entry in dataset}
    if {trial.puzzle_id for trial in run.trials} != set(expected):
        return "puzzle_ids_do_not_match_pinned_primary_dataset"
    if len(run.trials) != len(dataset) * len(ASSIGNED_MODEL_IDS):
        return "trial_count_does_not_match_five_puzzles_by_four_models"
    for entry in dataset:
        trials = [trial for trial in run.trials if trial.puzzle_id == entry.puzzle_id]
        if {trial.model_id for trial in trials} != set(ASSIGNED_MODEL_IDS) or len(trials) != 4:
            return f"model_coverage_is_incomplete_for_{entry.puzzle_id}"
        for trial in trials:
            if (
                trial.seed != entry.seed
                or trial.scramble_moves != entry.scramble_moves
                or trial.initial_state != entry.initial_state
                or trial.exact_shortest_distance != entry.exact_shortest_distance
                or trial.max_valid_moves != PRIMARY_MAX_VALID_MOVES
            ):
                return f"puzzle_or_reference_mismatch_for_{entry.puzzle_id}"
    return None


def _model_comparison(run: BenchmarkRun) -> dict[str, dict[str, Any]]:
    summaries = aggregate_metrics(run)
    output: dict[str, dict[str, Any]] = {}
    for model_id in ASSIGNED_MODEL_IDS:
        trials = [trial for trial in run.trials if trial.model_id == model_id]
        summary = summaries[model_id]
        accepted = [
            attempt
            for trial in trials
            for attempt in trial.attempts
            if attempt.outcome == "valid"
        ]
        requests = [attempt for trial in trials for attempt in trial.attempts]
        solved = [trial for trial in trials if trial.solved]
        terminations = Counter(trial.termination_status for trial in trials)
        token_values: dict[str, list[int]] = {field: [] for field in _TOKEN_FIELDS}
        for attempt in requests:
            if attempt.token_usage:
                for field in _TOKEN_FIELDS:
                    if field in attempt.token_usage:
                        token_values[field].append(attempt.token_usage[field])
        output[model_id] = {
            "trial_count": summary.trial_count,
            "solved_trial_count": summary.solved_trial_count,
            "solve_rate": summary.solve_rate,
            "average_valid_move_latency_seconds": summary.average_move_latency_seconds,
            "valid_move_latency_sample_count": len(accepted),
            "average_request_latency_seconds": summary.average_request_latency_seconds,
            "attempted_request_count": len(requests),
            "failed_request_count": summary.failed_request_count,
            "invalid_proposal_count": summary.invalid_proposal_count,
            "timeout_count": summary.timeout_count,
            "adapter_error_count": summary.adapter_error_count,
            "termination_counts": {
                status: terminations.get(status, 0)
                for status in ("solved", "move_limit", "invalid_move", "timeout", "adapter_error")
            },
            "total_valid_move_count": sum(trial.valid_move_count for trial in trials),
            "average_valid_move_count": summary.average_valid_move_count,
            "average_solved_move_count": (
                sum(trial.valid_move_count for trial in solved) / len(solved) if solved else None
            ),
            "solved_move_count_sample_count": len(solved),
            "average_total_runtime_seconds": summary.average_total_runtime_seconds,
            "optimality_gap_sample_count": summary.optimality_gap_sample_count,
            "average_optimality_gap": summary.average_optimality_gap,
            "reported_token_usage": {
                field: {
                    "response_count": len(values),
                    "sum_of_reported_counts": sum(values),
                }
                for field, values in token_values.items()
                if values
            },
        }
    return output


def _assignment_answers(comparison: dict[str, dict[str, Any]]) -> dict[str, Any]:
    latency = {
        model_id: row["average_valid_move_latency_seconds"]
        for model_id, row in comparison.items()
        if row["average_valid_move_latency_seconds"] is not None
    }
    fastest = _minimum_ids(latency)
    solve_rates = {model_id: row["solve_rate"] for model_id, row in comparison.items()}
    highest_solve_rate = _maximum_ids(solve_rates)
    gaps = {
        model_id: row["average_optimality_gap"]
        for model_id, row in comparison.items()
        if row["average_optimality_gap"] is not None
    }
    best_gap = _minimum_ids(gaps)
    solved_lengths = {
        model_id: row["average_solved_move_count"]
        for model_id, row in comparison.items()
        if row["average_solved_move_count"] is not None
    }
    shortest_solved = _minimum_ids(solved_lengths)
    overlap = [model_id for model_id in fastest if model_id in set(best_gap)]

    return {
        "fastest_per_decision": {
            "model_ids": fastest,
            "basis": "mean latency of accepted legal-move responses only",
            "latency_seconds": latency,
            "valid_response_samples": {
                model_id: comparison[model_id]["valid_move_latency_sample_count"]
                for model_id in comparison
            },
        },
        "highest_solve_rate": {
            "model_ids": highest_solve_rate,
            "solve_rate": max(solve_rates.values()) if solve_rates else None,
            "denominator": "five fixed puzzles per model",
        },
        "solution_quality": {
            "lowest_mean_optimality_gap_model_ids": best_gap,
            "mean_optimality_gap": gaps,
            "shortest_mean_solved_move_count_model_ids": shortest_solved,
            "mean_solved_move_count": solved_lengths,
        },
        "fastest_also_best_gap": {
            "answer": "yes" if overlap else ("no" if fastest and best_gap else "pending_no_solved_reference_samples"),
            "overlapping_model_ids": overlap,
            "fastest_model_ids": fastest,
            "best_gap_model_ids": best_gap,
        },
        "strategy_guidance": (
            "Descriptive evidence only: a single five-puzzle set is too small to establish a stable model ranking. "
            "Use the reported speed, solve-rate, and solved-trial gap trade-offs as preliminary evidence."
        ),
    }


def _minimum_ids(values: dict[str, float | None]) -> list[str]:
    available = {key: value for key, value in values.items() if value is not None}
    if not available:
        return []
    minimum = min(available.values())
    return [key for key, value in available.items() if math.isclose(value, minimum, rel_tol=1e-9, abs_tol=1e-9)]


def _maximum_ids(values: dict[str, float]) -> list[str]:
    if not values:
        return []
    maximum = max(values.values())
    return [key for key, value in values.items() if math.isclose(value, maximum, rel_tol=1e-9, abs_tol=1e-9)]


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise report while making pending/live status explicit."""

    lines = [
        "# Rubik’s Cube Race: Primary Experiment Analysis",
        "",
        f"- Dataset: `{report['dataset_id']}`",
        f"- Analysis status: **{report['analysis_status'].upper()}**",
        f"- Selected run: `{report['selected_run_id'] or 'none'}`",
        f"- Persisted result files scanned: {report['files_scanned']}",
        "",
    ]
    if report["analysis_status"] != "ready":
        lines.extend(
            [
                "Assignment analysis is pending. Only a completed live run matching all five pinned puzzles, all four exact model IDs, a 30-valid-move cap, and a 30-second turn timeout is eligible.",
                f"Reason: `{report['pending_reason']}`.",
                "Mock demonstrations, legacy runs without a saved execution mode, and one-request diagnostics are not treated as official v1 model measurements. Full state-aware exploratory runs are excluded as well.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Per-model comparison",
                "",
                "Latency is measured around the complete adapter decision call. Valid-response latency includes accepted legal moves only; request counts include failed attempts. Move-count means include all trials, while optimality gaps include only solved trials with an exact reference.",
                "",
                "| Exact model ID | Solved | Solve rate | Valid-response latency (s, n) | Request latency (s) | Mean runtime (s) | Mean valid moves | Solved-only moves | Mean HTM gap (n) | Requests / failures |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_id, row in report["per_model_comparison"].items():
            avg_gap = _format_optional(row["average_optimality_gap"])
            lines.append(
                "| `{id}` | {solved}/5 | {rate:.0%} | {latency} ({samples}) | {request_latency} | {runtime:.3f} | {moves:.2f} | {solved_moves} ({solved_n}) | {gap} ({gap_n}) | {requests} / {failures} |".format(
                    id=model_id,
                    solved=row["solved_trial_count"],
                    rate=row["solve_rate"],
                    latency=_format_optional(row["average_valid_move_latency_seconds"]),
                    samples=row["valid_move_latency_sample_count"],
                    request_latency=_format_optional(row["average_request_latency_seconds"]),
                    runtime=row["average_total_runtime_seconds"],
                    moves=row["average_valid_move_count"],
                    solved_moves=_format_optional(row["average_solved_move_count"]),
                    solved_n=row["solved_move_count_sample_count"],
                    gap=avg_gap,
                    gap_n=row["optimality_gap_sample_count"],
                    requests=row["attempted_request_count"],
                    failures=row["failed_request_count"],
                )
            )
        lines.extend(["", "## Assignment questions", ""])
        answers = report["assignment_analysis"]
        lines.append(
            f"- Fastest per decision among accepted legal responses: {_ids(answers['fastest_per_decision']['model_ids'])}."
        )
        lines.append(
            f"- Highest observed solve rate: {_ids(answers['highest_solve_rate']['model_ids'])} ({_format_optional(answers['highest_solve_rate']['solve_rate'])})."
        )
        lines.append(
            f"- Lowest mean optimality gap among solved trials: {_ids(answers['solution_quality']['lowest_mean_optimality_gap_model_ids'])}."
        )
        lines.append(
            f"- Was the fastest also best by optimality gap? {answers['fastest_also_best_gap']['answer']} ({_ids(answers['fastest_also_best_gap']['overlapping_model_ids'])})."
        )
        lines.extend(["", answers["strategy_guidance"], ""])

    lines.extend(
        [
            "## Usage and cost audit",
            "",
            "Per-attempt request counts and failure outcomes come from persisted trial records. Provider-reported token counters are retained only when the response supplies safe non-negative integer fields; absent usage is missing data, not zero. Different counter names are kept separate and are not inferred or combined.",
            "",
            "Actual spend is not calculated from token counts or catalog prices. Verify it in OpenRouter account usage/activity for the experiment time window. The $1 API key limit is a safeguard, not an estimate that the full experiment will cost $1 or less.",
            "",
            "The report excludes mocks and unknown-mode records. A legacy result cannot be classified as live from its puzzle name or model IDs alone.",
            "",
        ]
    )
    if report["run_classifications"]:
        lines.extend(["## Result files considered", "", "| File | Classification | Reason |", "| --- | --- | --- |"])
        for item in report["run_classifications"]:
            lines.append(
                f"| `{item['file']}` | {item['classification']} | {item.get('reason', '—')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _ids(values: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in values) if values else "pending / no sample"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Directory containing persisted run JSON files")
    parser.add_argument("--output", default="reports/primary-analysis.md", help="Markdown output path")
    parser.add_argument("--run-id", help="Explicitly select one eligible run if multiple exist")
    args = parser.parse_args(argv)

    report = build_analysis_report(args.results_dir, run_id=args.run_id)
    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"Could not write analysis report to {output}: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(f"Wrote {output} (analysis {report['analysis_status']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
