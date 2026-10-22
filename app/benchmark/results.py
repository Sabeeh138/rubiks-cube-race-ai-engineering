"""Validated, JSON-serializable benchmark result records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.cube import CubeState, LEGAL_MOVES
from app.models import (
    ASSIGNED_MODEL_IDS,
    CLAUDE_OPUS_55_MODEL_ID,
    SIMULATED_CLAUDE_LABEL,
)
from app.models.base import TOKEN_USAGE_FIELDS
from app.models.prompts import (
    RUBIKS_MOVE_OBJECTIVE_V1,
    RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
    SUPPORTED_PROMPT_VERSIONS,
)

RESULTS_SCHEMA_VERSION = 4
TRIAL_PROVENANCES = frozenset({"live", "mock", "simulated", "unknown"})
TRIAL_TERMINATIONS = frozenset(
    {"solved", "move_limit", "invalid_move", "timeout", "adapter_error"}
)
ATTEMPT_OUTCOMES = frozenset({"valid", "invalid_move", "timeout", "adapter_error"})


def _object(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        extra = sorted(set(value) - fields)
        raise ValueError(f"{name} fields do not match schema (missing={missing}, extra={extra})")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: object, *, name: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a non-negative finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _moves(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of legal move tokens")
    result = tuple(value)
    if any(not isinstance(move, str) or move not in LEGAL_MOVES for move in result):
        raise ValueError(f"{name} contains an invalid move token")
    return result


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name=name)


def _token_usage(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("token_usage must be an object or None")
    if set(value) - TOKEN_USAGE_FIELDS:
        raise ValueError("token_usage contains an unsupported counter")
    output: dict[str, int] = {}
    for field, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"token_usage.{field} must be a non-negative integer")
        output[field] = count
    return output or None


@dataclass(frozen=True, slots=True)
class TurnAttempt:
    """One adapter request, including failed requests and the exact input state."""

    attempt_index: int
    cube_state_before: str
    recent_moves: tuple[str, ...]
    latency_seconds: float
    elapsed_trial_seconds: float
    outcome: str
    move: str | None = None
    served_model: str | None = None
    token_usage: dict[str, int] | None = None
    error_type: str | None = None
    error_message: str | None = None
    rejected_move: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt_index, bool) or not isinstance(self.attempt_index, int) or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        CubeState.from_facelets(self.cube_state_before)
        recent = _moves(self.recent_moves, name="recent_moves")
        object.__setattr__(self, "recent_moves", recent)
        latency = _number(self.latency_seconds, name="latency_seconds")
        elapsed = _number(self.elapsed_trial_seconds, name="elapsed_trial_seconds")
        object.__setattr__(self, "latency_seconds", latency)
        object.__setattr__(self, "elapsed_trial_seconds", elapsed)
        if self.outcome not in ATTEMPT_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(ATTEMPT_OUTCOMES)}")
        if self.outcome == "valid":
            if self.move not in LEGAL_MOVES:
                raise ValueError("a valid attempt must contain one legal move")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("a valid attempt cannot contain error details")
            if self.rejected_move is not None:
                raise ValueError("a valid attempt cannot contain a rejected move")
        else:
            if self.move is not None:
                raise ValueError("a failed attempt cannot contain an applied move")
            _text(self.error_type, name="error_type")
            _text(self.error_message, name="error_message")
            if self.outcome != "invalid_move" and self.rejected_move is not None:
                raise ValueError("only an invalid-move attempt can contain rejected_move")
        if self.rejected_move is not None:
            if not isinstance(self.rejected_move, str) or len(self.rejected_move) > 160:
                raise ValueError("rejected_move must be a string of at most 160 characters")
        if self.served_model is not None:
            _text(self.served_model, name="served_model")
        object.__setattr__(self, "token_usage", _token_usage(self.token_usage))

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "cube_state_before": self.cube_state_before,
            "recent_moves": list(self.recent_moves),
            "latency_seconds": self.latency_seconds,
            "elapsed_trial_seconds": self.elapsed_trial_seconds,
            "outcome": self.outcome,
            "move": self.move,
            "served_model": self.served_model,
            "token_usage": self.token_usage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "rejected_move": self.rejected_move,
        }

    @classmethod
    def from_dict(cls, value: object) -> TurnAttempt:
        fields = {
            "attempt_index", "cube_state_before", "recent_moves", "latency_seconds",
            "elapsed_trial_seconds", "outcome", "move", "served_model", "token_usage",
            "error_type", "error_message", "rejected_move",
        }
        if isinstance(value, dict) and set(value) == fields - {"token_usage"}:
            # Phase 6 schema-v1 records predate safe token-usage retention.
            data = value
        else:
            data = _object(value, name="turn attempt", fields=fields)
        if isinstance(data["attempt_index"], bool) or not isinstance(data["attempt_index"], int):
            raise ValueError("attempt_index must be an integer")
        rejected_move = data["rejected_move"]
        if rejected_move is not None and not isinstance(rejected_move, str):
            raise ValueError("rejected_move must be a string or None")
        return cls(
            attempt_index=data["attempt_index"],
            cube_state_before=_text(data["cube_state_before"], name="cube_state_before"),
            recent_moves=_moves(data["recent_moves"], name="recent_moves"),
            latency_seconds=_number(data["latency_seconds"], name="latency_seconds"),
            elapsed_trial_seconds=_number(data["elapsed_trial_seconds"], name="elapsed_trial_seconds"),
            outcome=_text(data["outcome"], name="outcome"),
            move=_optional_text(data["move"], name="move"),
            served_model=_optional_text(data["served_model"], name="served_model"),
            token_usage=_token_usage(data.get("token_usage")),
            error_type=_optional_text(data["error_type"], name="error_type"),
            error_message=_optional_text(data["error_message"], name="error_message"),
            rejected_move=rejected_move,
        )


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Auditable record for one puzzle/model attempt."""

    trial_id: str
    puzzle_id: str
    model_id: str
    seed: int | None
    scramble_moves: tuple[str, ...]
    initial_state: str
    attempts: tuple[TurnAttempt, ...]
    final_state: str
    termination_status: str
    max_valid_moves: int
    total_runtime_seconds: float
    exact_shortest_distance: int | None
    retry_count: int = 0
    provenance: str = "unknown"
    provenance_label: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("trial_id", "puzzle_id", "model_id"):
            _text(getattr(self, field_name), name=field_name)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        scramble = _moves(self.scramble_moves, name="scramble_moves")
        object.__setattr__(self, "scramble_moves", scramble)
        initial = CubeState.from_facelets(self.initial_state)
        expected_initial = CubeState().apply_moves(scramble)
        if initial != expected_initial:
            raise ValueError("initial_state does not match the recorded scramble")
        final = CubeState.from_facelets(self.final_state)
        if isinstance(self.max_valid_moves, bool) or not isinstance(self.max_valid_moves, int):
            raise ValueError("max_valid_moves must be an integer")
        if not 1 <= self.max_valid_moves <= 30:
            raise ValueError("max_valid_moves must be between 1 and 30")
        runtime = _number(self.total_runtime_seconds, name="total_runtime_seconds")
        object.__setattr__(self, "total_runtime_seconds", runtime)
        if self.provenance not in TRIAL_PROVENANCES:
            raise ValueError(f"provenance must be one of {sorted(TRIAL_PROVENANCES)}")
        if self.provenance == "simulated":
            if self.model_id != CLAUDE_OPUS_55_MODEL_ID:
                raise ValueError("only the assigned Claude trial may use simulated provenance")
            if self.provenance_label != SIMULATED_CLAUDE_LABEL:
                raise ValueError("simulated Claude trials must use the required provenance label")
            if runtime != 0.0:
                raise ValueError("simulated trials must not record model runtime")
        elif self.provenance_label is not None:
            raise ValueError("only a simulated trial may contain provenance_label")
        if self.exact_shortest_distance is not None and (
            isinstance(self.exact_shortest_distance, bool)
            or not isinstance(self.exact_shortest_distance, int)
            or self.exact_shortest_distance < 0
        ):
            raise ValueError("exact_shortest_distance must be a non-negative integer or None")
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if self.termination_status not in TRIAL_TERMINATIONS:
            raise ValueError(f"termination_status must be one of {sorted(TRIAL_TERMINATIONS)}")

        attempts = tuple(self.attempts)
        if any(not isinstance(attempt, TurnAttempt) for attempt in attempts):
            raise ValueError("attempts must contain TurnAttempt records")
        object.__setattr__(self, "attempts", attempts)
        if self.provenance == "simulated" and any(
            attempt.latency_seconds != 0.0
            or attempt.elapsed_trial_seconds != 0.0
            or attempt.token_usage is not None
            for attempt in attempts
        ):
            raise ValueError("simulated trials cannot record model latency or token usage")
        replay = initial
        history: list[str] = []
        failed_outcome: str | None = None
        previous_elapsed = 0.0
        for index, attempt in enumerate(attempts):
            if attempt.attempt_index != index:
                raise ValueError("attempt indexes must be contiguous and start at zero")
            if attempt.elapsed_trial_seconds < previous_elapsed:
                raise ValueError("attempt timestamps must be non-decreasing")
            if attempt.elapsed_trial_seconds > runtime + 1e-9:
                raise ValueError("attempt timestamp cannot exceed total runtime")
            previous_elapsed = attempt.elapsed_trial_seconds
            if failed_outcome is not None:
                raise ValueError("no request can follow a failed attempt")
            if attempt.cube_state_before != replay.serialize():
                raise ValueError("attempt cube_state_before does not match replayed state")
            if attempt.recent_moves != tuple(history):
                raise ValueError("attempt recent_moves does not match replayed move history")
            if attempt.outcome == "valid":
                assert attempt.move is not None
                replay = replay.apply_move(attempt.move)
                history.append(attempt.move)
            else:
                failed_outcome = attempt.outcome
        if len(history) > self.max_valid_moves:
            raise ValueError("valid move count exceeds max_valid_moves")
        if replay.serialize() != final.serialize():
            raise ValueError("final_state does not match replayed moves")

        status = self.termination_status
        if status == "solved":
            if failed_outcome is not None or not final.is_solved:
                raise ValueError("a solved trial must finish solved without a failed request")
        elif status == "move_limit":
            if failed_outcome is not None or final.is_solved or len(history) != self.max_valid_moves:
                raise ValueError("move_limit requires an unsolved cube at the valid-move cap")
        else:
            if failed_outcome != status or final.is_solved:
                raise ValueError("failure status must match the final failed request on an unsolved cube")
        if (
            status == "solved"
            and self.exact_shortest_distance is not None
            and len(history) < self.exact_shortest_distance
        ):
            raise ValueError("solved move count cannot be shorter than the exact reference")

    @property
    def solved(self) -> bool:
        return self.termination_status == "solved" and CubeState.from_facelets(self.final_state).is_solved

    @property
    def move_history(self) -> tuple[str, ...]:
        return tuple(attempt.move for attempt in self.attempts if attempt.outcome == "valid" and attempt.move)

    @property
    def valid_move_count(self) -> int:
        return len(self.move_history)

    @property
    def attempted_request_count(self) -> int:
        return len(self.attempts)

    @property
    def failed_request_count(self) -> int:
        return sum(attempt.outcome != "valid" for attempt in self.attempts)

    @property
    def invalid_proposal_count(self) -> int:
        return sum(attempt.outcome == "invalid_move" for attempt in self.attempts)

    @property
    def timeout_count(self) -> int:
        return sum(attempt.outcome == "timeout" for attempt in self.attempts)

    @property
    def average_move_latency_seconds(self) -> float | None:
        if self.provenance == "simulated":
            return None
        latencies = [attempt.latency_seconds for attempt in self.attempts if attempt.outcome == "valid"]
        return sum(latencies) / len(latencies) if latencies else None

    @property
    def average_request_latency_seconds(self) -> float | None:
        if self.provenance == "simulated":
            return None
        latencies = [attempt.latency_seconds for attempt in self.attempts]
        return sum(latencies) / len(latencies) if latencies else None

    @property
    def optimality_gap(self) -> int | None:
        if self.provenance == "simulated" or not self.solved or self.exact_shortest_distance is None:
            return None
        return self.valid_move_count - self.exact_shortest_distance

    def to_dict(self, *, include_provenance: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "trial_id": self.trial_id,
            "puzzle_id": self.puzzle_id,
            "model_id": self.model_id,
            "seed": self.seed,
            "scramble_moves": list(self.scramble_moves),
            "initial_state": self.initial_state,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "final_state": self.final_state,
            "termination_status": self.termination_status,
            "solved": self.solved,
            "max_valid_moves": self.max_valid_moves,
            "valid_move_count": self.valid_move_count,
            "attempted_request_count": self.attempted_request_count,
            "move_history": list(self.move_history),
            "failed_request_count": self.failed_request_count,
            "invalid_proposal_count": self.invalid_proposal_count,
            "timeout_count": self.timeout_count,
            "average_move_latency_seconds": self.average_move_latency_seconds,
            "average_request_latency_seconds": self.average_request_latency_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "exact_shortest_distance": self.exact_shortest_distance,
            "optimality_gap": self.optimality_gap,
            "retry_count": self.retry_count,
        }
        if include_provenance:
            result["provenance"] = self.provenance
            result["provenance_label"] = self.provenance_label
        return result

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        legacy_provenance: str = "unknown",
        require_provenance: bool = False,
    ) -> TrialResult:
        fields = {
            "trial_id", "puzzle_id", "model_id", "seed", "scramble_moves", "initial_state",
            "attempts", "final_state", "termination_status", "solved", "max_valid_moves",
            "valid_move_count", "attempted_request_count", "move_history", "failed_request_count",
            "invalid_proposal_count", "timeout_count", "average_move_latency_seconds",
            "average_request_latency_seconds", "total_runtime_seconds", "exact_shortest_distance",
            "optimality_gap", "retry_count",
        }
        provenance_fields = {"provenance", "provenance_label"}
        if require_provenance:
            data = _object(value, name="trial", fields=fields | provenance_fields)
        elif isinstance(value, dict) and set(value) == fields:
            data = value
        else:
            data = _object(value, name="trial", fields=fields | provenance_fields)
        if data["seed"] is not None and (isinstance(data["seed"], bool) or not isinstance(data["seed"], int)):
            raise ValueError("seed must be an integer or None")
        if isinstance(data["max_valid_moves"], bool) or not isinstance(data["max_valid_moves"], int):
            raise ValueError("max_valid_moves must be an integer")
        exact = data["exact_shortest_distance"]
        if exact is not None and (isinstance(exact, bool) or not isinstance(exact, int)):
            raise ValueError("exact_shortest_distance must be an integer or None")
        retry_count = data["retry_count"]
        if isinstance(retry_count, bool) or not isinstance(retry_count, int):
            raise ValueError("retry_count must be an integer")
        if not isinstance(data["solved"], bool):
            raise ValueError("solved must be a boolean")
        for count_field in (
            "valid_move_count", "attempted_request_count", "failed_request_count",
            "invalid_proposal_count", "timeout_count",
        ):
            count = data[count_field]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{count_field} must be a non-negative integer")
        move_history = _moves(data["move_history"], name="move_history")
        if data["optimality_gap"] is not None and (
            isinstance(data["optimality_gap"], bool) or not isinstance(data["optimality_gap"], int)
        ):
            raise ValueError("optimality_gap must be an integer or None")
        for latency_field in ("average_move_latency_seconds", "average_request_latency_seconds"):
            _number(data[latency_field], name=latency_field, allow_none=True)
        attempts_data = data["attempts"]
        if not isinstance(attempts_data, list):
            raise ValueError("attempts must be a JSON list")
        result = cls(
            trial_id=_text(data["trial_id"], name="trial_id"),
            puzzle_id=_text(data["puzzle_id"], name="puzzle_id"),
            model_id=_text(data["model_id"], name="model_id"),
            seed=data["seed"],
            scramble_moves=_moves(data["scramble_moves"], name="scramble_moves"),
            initial_state=_text(data["initial_state"], name="initial_state"),
            attempts=tuple(TurnAttempt.from_dict(item) for item in attempts_data),
            final_state=_text(data["final_state"], name="final_state"),
            termination_status=_text(data["termination_status"], name="termination_status"),
            max_valid_moves=data["max_valid_moves"],
            total_runtime_seconds=_number(data["total_runtime_seconds"], name="total_runtime_seconds"),
            exact_shortest_distance=exact,
            retry_count=retry_count,
            provenance=(
                _text(data["provenance"], name="provenance")
                if "provenance" in data
                else legacy_provenance
            ),
            provenance_label=_optional_text(data.get("provenance_label"), name="provenance_label"),
        )
        if move_history != result.move_history:
            raise ValueError("move_history does not match raw attempts")
        _verify_derived(data, result)
        return result


def _verify_derived(data: dict[str, Any], trial: TrialResult) -> None:
    derived: dict[str, object] = {
        "solved": trial.solved,
        "valid_move_count": trial.valid_move_count,
        "attempted_request_count": trial.attempted_request_count,
        "move_history": list(trial.move_history),
        "failed_request_count": trial.failed_request_count,
        "invalid_proposal_count": trial.invalid_proposal_count,
        "timeout_count": trial.timeout_count,
        "average_move_latency_seconds": trial.average_move_latency_seconds,
        "average_request_latency_seconds": trial.average_request_latency_seconds,
        "optimality_gap": trial.optimality_gap,
    }
    for field, expected in derived.items():
        observed = data[field]
        if isinstance(expected, float):
            valid = isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9)
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(f"trial {field} does not match its raw attempt records")


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One fair comparison run containing all four assigned models per puzzle."""

    run_id: str
    max_valid_moves: int
    turn_timeout_seconds: float | None
    trials: tuple[TrialResult, ...]
    execution_mode: str | None = None
    schema_version: int = RESULTS_SCHEMA_VERSION
    prompt_version: str | None = RUBIKS_MOVE_OBJECTIVE_V1
    action_previews_enabled: bool = False

    def __post_init__(self) -> None:
        _text(self.run_id, name="run_id")
        if self.schema_version not in {1, 2, 3, RESULTS_SCHEMA_VERSION}:
            raise ValueError(f"unsupported result schema version {self.schema_version!r}")
        if self.execution_mode not in {None, "mock", "live", "recording", "diagnostic", "state_aware_experiment"}:
            raise ValueError("execution_mode must be 'mock', 'live', 'recording', 'diagnostic', 'state_aware_experiment', or None")
        if self.schema_version == 1 and self.execution_mode is not None:
            raise ValueError("schema-v1 runs cannot record execution_mode")
        if self.schema_version < 3 and self.execution_mode in {"recording", "diagnostic", "state_aware_experiment"}:
            raise ValueError("recording and exploratory modes require result schema version 3")
        if not isinstance(self.action_previews_enabled, bool):
            raise ValueError("action_previews_enabled must be a boolean")
        if self.schema_version < 4:
            if self.prompt_version is not None or self.action_previews_enabled:
                raise ValueError("legacy result schemas cannot record prompt-condition metadata")
        else:
            if self.prompt_version not in SUPPORTED_PROMPT_VERSIONS:
                raise ValueError("schema-v4 runs must record a supported prompt_version")
            if self.prompt_version == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE:
                if self.execution_mode not in {None, "diagnostic", "state_aware_experiment"}:
                    raise ValueError("the state-aware v2 condition is reserved for exploratory runs")
            elif self.action_previews_enabled:
                raise ValueError("action previews require the state-aware v2 condition")
            if self.execution_mode == "diagnostic" and self.prompt_version != RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE:
                raise ValueError("new diagnostic runs must record the state-aware v2 prompt")
            if self.execution_mode == "state_aware_experiment":
                if self.prompt_version != RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE or not self.action_previews_enabled:
                    raise ValueError("the full state-aware experiment requires v2 and action previews")
                if self.max_valid_moves != 30 or self.turn_timeout_seconds is None or not math.isclose(
                    self.turn_timeout_seconds, 30.0, rel_tol=0, abs_tol=1e-9
                ):
                    raise ValueError("the full state-aware experiment requires the 30-move and 30-second settings")
        if isinstance(self.max_valid_moves, bool) or not isinstance(self.max_valid_moves, int):
            raise ValueError("max_valid_moves must be an integer")
        if not 1 <= self.max_valid_moves <= 30:
            raise ValueError("max_valid_moves must be between 1 and 30")
        timeout = _number(self.turn_timeout_seconds, name="turn_timeout_seconds", allow_none=True)
        if timeout == 0:
            raise ValueError("turn_timeout_seconds must be positive or None")
        object.__setattr__(self, "turn_timeout_seconds", timeout)
        trials = tuple(self.trials)
        if not trials or any(not isinstance(trial, TrialResult) for trial in trials):
            raise ValueError("trials must contain at least one TrialResult")
        object.__setattr__(self, "trials", trials)
        ids = [trial.trial_id for trial in trials]
        if len(set(ids)) != len(ids):
            raise ValueError("trial_id values must be unique within a run")
        groups: dict[str, list[TrialResult]] = {}
        for trial in trials:
            if trial.max_valid_moves != self.max_valid_moves:
                raise ValueError("trial move limit does not match the run configuration")
            expected_trial_id = f"{self.run_id}:{trial.puzzle_id}:{trial.model_id}"
            if trial.trial_id != expected_trial_id:
                raise ValueError("trial_id does not match deterministic run/puzzle/model identity")
            groups.setdefault(trial.puzzle_id, []).append(trial)
        if self.execution_mode == "diagnostic" and len(groups) != 1:
            raise ValueError("diagnostic runs contain exactly one puzzle")
        pinned_dataset = None
        if self.execution_mode == "state_aware_experiment":
            if self.schema_version < 4:
                raise ValueError("the full state-aware experiment requires prompt metadata in result schema v4")
            from app.benchmark.primary_dataset import build_primary_dataset

            pinned_dataset = {entry.puzzle_id: entry for entry in build_primary_dataset()}
            if set(groups) != set(pinned_dataset) or len(trials) != len(pinned_dataset) * len(ASSIGNED_MODEL_IDS):
                raise ValueError("the full state-aware experiment must contain all five pinned puzzles and four model trials each")
        for puzzle_id, puzzle_trials in groups.items():
            if {trial.model_id for trial in puzzle_trials} != set(ASSIGNED_MODEL_IDS) or len(puzzle_trials) != 4:
                raise ValueError(f"puzzle {puzzle_id!r} must have exactly one trial for each assigned model")
            exemplar = puzzle_trials[0]
            for trial in puzzle_trials[1:]:
                if (
                    trial.seed != exemplar.seed
                    or trial.scramble_moves != exemplar.scramble_moves
                    or trial.initial_state != exemplar.initial_state
                    or trial.exact_shortest_distance != exemplar.exact_shortest_distance
                ):
                    raise ValueError(f"models for puzzle {puzzle_id!r} do not share the same puzzle inputs")
            by_model = {trial.model_id: trial for trial in puzzle_trials}
            if self.execution_mode == "recording":
                for model_id, trial in by_model.items():
                    expected = "simulated" if model_id == CLAUDE_OPUS_55_MODEL_ID else "live"
                    if trial.provenance != expected:
                        raise ValueError("recording runs require three live trials and simulated Claude")
            elif self.execution_mode == "diagnostic":
                if self.max_valid_moves != 3 or exemplar.scramble_moves != ("R",):
                    raise ValueError("diagnostic runs require the fixed R scramble and three-move limit")
                for model_id, trial in by_model.items():
                    expected = "simulated" if model_id == CLAUDE_OPUS_55_MODEL_ID else "live"
                    if trial.provenance != expected:
                        raise ValueError("diagnostic runs require three live trials and simulated Claude")
            elif self.execution_mode == "state_aware_experiment":
                assert pinned_dataset is not None
                entry = pinned_dataset[puzzle_id]
                if (
                    exemplar.seed != entry.seed
                    or exemplar.scramble_moves != entry.scramble_moves
                    or exemplar.initial_state != entry.initial_state
                    or exemplar.exact_shortest_distance != entry.exact_shortest_distance
                ):
                    raise ValueError(f"puzzle {puzzle_id!r} does not match the pinned primary dataset")
                for model_id, trial in by_model.items():
                    expected = "simulated" if model_id == CLAUDE_OPUS_55_MODEL_ID else "live"
                    if trial.provenance != expected:
                        raise ValueError("the full state-aware experiment requires three live trials and simulated Claude")
            elif self.execution_mode == "mock" and any(
                trial.provenance == "simulated" for trial in puzzle_trials
            ):
                raise ValueError("simulated trials are allowed only in recording or exploratory modes")
            elif self.execution_mode == "live" and any(trial.provenance != "live" for trial in puzzle_trials):
                raise ValueError("live runs may contain only live-provenance trials")

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "max_valid_moves": self.max_valid_moves,
            "turn_timeout_seconds": self.turn_timeout_seconds,
            "trials": [
                trial.to_dict(include_provenance=self.schema_version >= 3)
                for trial in self.trials
            ],
        }
        if self.schema_version >= 2:
            result["execution_mode"] = self.execution_mode
        if self.schema_version >= 4:
            result["prompt_version"] = self.prompt_version
            result["action_previews_enabled"] = self.action_previews_enabled
        return result

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkRun:
        if not isinstance(value, dict):
            raise ValueError("benchmark run must be a JSON object")
        version = value.get("schema_version")
        legacy_fields = {"schema_version", "run_id", "max_valid_moves", "turn_timeout_seconds", "trials"}
        if version == 1:
            data = _object(value, name="benchmark run", fields=legacy_fields)
        elif version == 2:
            data = _object(value, name="benchmark run", fields=legacy_fields | {"execution_mode"})
        elif version == 3:
            data = _object(value, name="benchmark run", fields=legacy_fields | {"execution_mode"})
        elif version == RESULTS_SCHEMA_VERSION:
            data = _object(
                value,
                name="benchmark run",
                fields=legacy_fields | {"execution_mode", "prompt_version", "action_previews_enabled"},
            )
        else:
            raise ValueError(f"unsupported result schema version {version!r}")
        limit = data["max_valid_moves"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("schema_version must be an integer")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("max_valid_moves must be an integer")
        trials_data = data["trials"]
        if not isinstance(trials_data, list):
            raise ValueError("trials must be a JSON list")
        execution_mode = data.get("execution_mode")
        legacy_provenance = "live" if execution_mode == "live" else (
            "mock" if execution_mode == "mock" else "unknown"
        )
        return cls(
            run_id=_text(data["run_id"], name="run_id"),
            max_valid_moves=limit,
            turn_timeout_seconds=_number(data["turn_timeout_seconds"], name="turn_timeout_seconds", allow_none=True),
            trials=tuple(
                TrialResult.from_dict(
                    item,
                    legacy_provenance=legacy_provenance,
                    require_provenance=version >= 3,
                )
                for item in trials_data
            ),
            execution_mode=execution_mode,
            schema_version=version,
            prompt_version=data.get("prompt_version"),
            action_previews_enabled=data.get("action_previews_enabled", False),
        )
