"""Derived aggregate metrics for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

from app.models import ASSIGNED_MODEL_IDS

from app.benchmark.results import BenchmarkRun


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Per-model aggregates with explicit denominators for every mean."""

    model_id: str
    trial_count: int
    solved_trial_count: int
    solve_rate: float
    average_move_latency_seconds: float | None
    average_request_latency_seconds: float | None
    average_total_runtime_seconds: float
    average_valid_move_count: float
    optimality_gap_sample_count: int
    average_optimality_gap: float | None
    unsuccessful_trial_count: int
    failed_request_count: int
    invalid_proposal_count: int
    timeout_count: int
    adapter_error_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "trial_count": self.trial_count,
            "solved_trial_count": self.solved_trial_count,
            "solve_rate": self.solve_rate,
            "average_move_latency_seconds": self.average_move_latency_seconds,
            "average_request_latency_seconds": self.average_request_latency_seconds,
            "average_total_runtime_seconds": self.average_total_runtime_seconds,
            "average_valid_move_count": self.average_valid_move_count,
            "optimality_gap_sample_count": self.optimality_gap_sample_count,
            "average_optimality_gap": self.average_optimality_gap,
            "unsuccessful_trial_count": self.unsuccessful_trial_count,
            "failed_request_count": self.failed_request_count,
            "invalid_proposal_count": self.invalid_proposal_count,
            "timeout_count": self.timeout_count,
            "adapter_error_count": self.adapter_error_count,
        }


def aggregate_metrics(run: BenchmarkRun) -> dict[str, ModelMetrics]:
    """Calculate means from raw trials without treating failures as success.

    Valid-move latency pools only accepted move responses. Request latency
    includes every attempted request, including invalid output and failures.
    Total runtime and move-count means include every trial. Optimality gap
    includes only solved trials whose short-state exact reference is known.
    """

    result: dict[str, ModelMetrics] = {}
    for model_id in ASSIGNED_MODEL_IDS:
        trials = [
            trial
            for trial in run.trials
            if trial.model_id == model_id and trial.provenance != "simulated"
        ]
        if not trials:
            continue
        solved = [trial for trial in trials if trial.solved]
        valid_latencies = [
            attempt.latency_seconds
            for trial in trials
            for attempt in trial.attempts
            if attempt.outcome == "valid"
        ]
        request_latencies = [
            attempt.latency_seconds
            for trial in trials
            for attempt in trial.attempts
        ]
        gaps = [trial.optimality_gap for trial in trials if trial.optimality_gap is not None]
        result[model_id] = ModelMetrics(
            model_id=model_id,
            trial_count=len(trials),
            solved_trial_count=len(solved),
            solve_rate=len(solved) / len(trials),
            average_move_latency_seconds=(
                sum(valid_latencies) / len(valid_latencies) if valid_latencies else None
            ),
            average_request_latency_seconds=(
                sum(request_latencies) / len(request_latencies) if request_latencies else None
            ),
            average_total_runtime_seconds=sum(trial.total_runtime_seconds for trial in trials) / len(trials),
            average_valid_move_count=sum(trial.valid_move_count for trial in trials) / len(trials),
            optimality_gap_sample_count=len(gaps),
            average_optimality_gap=sum(gaps) / len(gaps) if gaps else None,
            unsuccessful_trial_count=sum(not trial.solved for trial in trials),
            failed_request_count=sum(trial.failed_request_count for trial in trials),
            invalid_proposal_count=sum(trial.invalid_proposal_count for trial in trials),
            timeout_count=sum(trial.timeout_count for trial in trials),
            adapter_error_count=sum(
                attempt.outcome == "adapter_error"
                for trial in trials
                for attempt in trial.attempts
            ),
        )
    return result
