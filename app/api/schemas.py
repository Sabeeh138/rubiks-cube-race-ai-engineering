"""Typed public request and response schemas for FastAPI."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.benchmark import BenchmarkRun, PuzzleSpec, aggregate_metrics
from app.benchmark.primary_dataset import build_primary_dataset, primary_puzzle_specs
from app.cube import LEGAL_MOVES

ExecutionModeSchema = Literal["mock", "live", "recording", "diagnostic", "state_aware_experiment"]


class PuzzleRequest(BaseModel):
    """A puzzle from either an explicit sequence or a seed and length."""

    model_config = ConfigDict(extra="forbid")

    puzzle_id: str = Field(min_length=1, max_length=80, description="Stable caller-selected puzzle identifier.")
    scramble_moves: list[str] | None = Field(
        default=None,
        max_length=25,
        description="Explicit legal HTM move tokens. Mutually exclusive with seed/scramble_length.",
    )
    seed: int | None = Field(
        default=None,
        strict=True,
        description="Seed used by the existing deterministic generator when scramble_moves is omitted.",
    )
    scramble_length: int | None = Field(
        default=None,
        ge=0,
        le=25,
        strict=True,
        description="Generated scramble length; defaults to 4 when seed is supplied.",
    )

    @field_validator("puzzle_id")
    @classmethod
    def puzzle_id_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("puzzle_id must contain a non-whitespace character")
        return value

    @field_validator("scramble_moves")
    @classmethod
    def tokens_must_be_legal(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            for token in value:
                if token not in LEGAL_MOVES:
                    raise ValueError(f"invalid face-turn move {token!r}; use one of the 18 legal moves")
        return value

    @model_validator(mode="after")
    def validate_scramble_source(self) -> PuzzleRequest:
        if self.scramble_moves is not None:
            if self.seed is not None or self.scramble_length is not None:
                raise ValueError("provide scramble_moves, or seed with optional scramble_length, but not both")
        elif self.seed is None:
            raise ValueError("seed is required when scramble_moves is omitted")
        return self

    def to_puzzle_spec(self) -> PuzzleSpec:
        if self.scramble_moves is not None:
            return PuzzleSpec(self.puzzle_id, tuple(self.scramble_moves))
        assert self.seed is not None
        length = self.scramble_length if self.scramble_length is not None else 4
        return PuzzleSpec.from_seed(self.puzzle_id, seed=self.seed, length=length)


class CubePreviewResponse(BaseModel):
    """Actual backend-generated scramble and canonical initial facelets."""

    model_config = ConfigDict(extra="forbid")

    puzzle_id: str
    seed: int | None
    scramble_moves: list[str]
    initial_state: str = Field(description="54 facelets in URFDLB order.")


class PrimaryPuzzleResponse(CubePreviewResponse):
    """One fixed primary puzzle, including its verified reference distance."""

    seed: int
    scramble_length: int
    exact_shortest_distance: int


class PrimaryDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    puzzles: list[PrimaryPuzzleResponse]


class ReferenceSolutionStep(BaseModel):
    """A move and the real engine state immediately after applying it."""

    model_config = ConfigDict(extra="forbid")

    move: str
    state_after: str = Field(description="54 facelets after this legal move, in URFDLB order.")


class ReferenceSolutionResponse(BaseModel):
    """An exact short-state solution replayed and verified by the cube engine."""

    model_config = ConfigDict(extra="forbid")

    puzzle_id: str
    scramble_moves: list[str]
    initial_state: str
    solution_moves: list[str]
    move_count: int
    steps: list[ReferenceSolutionStep]
    final_state: str
    solved: Literal[True]


class StudentDisplayConfig(BaseModel):
    """Public, non-secret student labels used by the recording banner."""

    model_config = ConfigDict(extra="forbid")

    student_name: str | None
    student_roll_number: str | None


class BenchmarkRequest(BaseModel):
    """Request a mock, live, recording, or exploratory state-aware run."""

    model_config = ConfigDict(extra="forbid")

    puzzles: list[PuzzleRequest] = Field(min_length=1, max_length=20)
    execution_mode: ExecutionModeSchema = Field(
        default="mock",
        description=(
            "Defaults to offline deterministic mocks. 'live' opts all four models into OpenRouter. "
            "'recording' uses live OpenRouter adapters for the first three assigned models and a "
            "clearly labeled scripted Claude demonstration; it still requires an API key. "
            "'diagnostic' uses the three live adapters and scripted Claude against only the fixed R "
            "scramble, with exactly three valid moves allowed. It uses the separate state-aware v2 "
            "instruction and optionally includes engine-generated candidate-state previews; it requires an API key. "
            "'state_aware_experiment' runs the five pinned primary puzzles with v2 and previews enabled, "
            "the three assigned live adapters, and scripted Claude; it requires explicit credit acknowledgement and an API key."
        ),
    )
    max_valid_moves: int = Field(default=30, ge=1, le=30, strict=True)
    turn_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    action_previews_enabled: bool = Field(
        default=False,
        strict=True,
        description=(
            "For state-aware exploratory modes, include the engine-computed resulting "
            "54-facelet state for each legal next move. This is context, not a solver recommendation."
        ),
    )
    live_usage_confirmed: bool = Field(
        default=False,
        strict=True,
        description="Required explicit acknowledgement before starting the exploratory five-puzzle live run.",
    )

    @model_validator(mode="after")
    def puzzle_ids_are_unique(self) -> BenchmarkRequest:
        ids = [puzzle.puzzle_id for puzzle in self.puzzles]
        if len(ids) != len(set(ids)):
            raise ValueError("puzzle_id values must be unique within a benchmark request")
        if self.turn_timeout_seconds < 0.1:
            raise ValueError("turn_timeout_seconds must be at least 0.1 seconds")
        if self.live_usage_confirmed and self.execution_mode != "state_aware_experiment":
            raise ValueError("live_usage_confirmed is available only for the full state-aware experiment")
        if self.action_previews_enabled and self.execution_mode not in {"diagnostic", "state_aware_experiment"}:
            raise ValueError("action_previews_enabled is available only in state-aware exploratory modes")
        if self.execution_mode == "diagnostic":
            if len(self.puzzles) != 1 or self.puzzles[0].scramble_moves != ["R"]:
                raise ValueError("diagnostic mode requires exactly one puzzle with the fixed scramble ['R']")
            if self.max_valid_moves != 3:
                raise ValueError("diagnostic mode requires max_valid_moves=3")
        if self.execution_mode == "state_aware_experiment":
            expected = build_primary_dataset()
            if len(self.puzzles) != len(expected):
                raise ValueError("the full state-aware experiment requires all five pinned primary puzzles")
            for request, puzzle in zip(self.puzzles, expected, strict=True):
                if (
                    request.puzzle_id != puzzle.puzzle_id
                    or request.seed != puzzle.seed
                    or request.scramble_length != puzzle.scramble_length
                    or request.scramble_moves is not None
                    or request.to_puzzle_spec().scramble_moves != puzzle.scramble_moves
                ):
                    raise ValueError("the full state-aware experiment must use the exact pinned primary puzzle definitions")
            if self.max_valid_moves != 30 or self.turn_timeout_seconds != 30.0:
                raise ValueError("the full state-aware experiment requires the existing 30-move and 30-second limits")
            if not self.action_previews_enabled:
                raise ValueError("the full state-aware experiment requires one-step action previews")
            if not self.live_usage_confirmed:
                raise ValueError("explicit live-usage acknowledgement is required before starting the exploratory experiment")
        return self

    def to_puzzle_specs(self) -> tuple[PuzzleSpec, ...]:
        if self.execution_mode == "state_aware_experiment":
            return primary_puzzle_specs()
        return tuple(puzzle.to_puzzle_spec() for puzzle in self.puzzles)


class BenchmarkSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["running", "completed", "failed"]
    execution_mode: ExecutionModeSchema
    status_url: str
    results_url: str


class RunProgressResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed_trials: int
    total_trials: int
    completed_puzzles: int
    total_puzzles: int
    current_puzzle_id: str | None


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["running", "completed", "failed"]
    execution_mode: ExecutionModeSchema | None
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    progress: RunProgressResponse
    error: str | None


class TurnAttemptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_index: int
    cube_state_before: str
    recent_moves: list[str]
    latency_seconds: float
    elapsed_trial_seconds: float
    outcome: Literal["valid", "invalid_move", "timeout", "adapter_error"]
    move: str | None
    served_model: str | None
    token_usage: dict[str, int] | None
    error_type: str | None
    error_message: str | None
    rejected_move: str | None


class TrialResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    puzzle_id: str
    model_id: str
    seed: int | None
    scramble_moves: list[str]
    initial_state: str
    attempts: list[TurnAttemptResponse]
    final_state: str
    termination_status: Literal["solved", "move_limit", "invalid_move", "timeout", "adapter_error"]
    solved: bool
    max_valid_moves: int
    valid_move_count: int
    attempted_request_count: int
    move_history: list[str]
    failed_request_count: int
    invalid_proposal_count: int
    timeout_count: int
    average_move_latency_seconds: float | None
    average_request_latency_seconds: float | None
    total_runtime_seconds: float
    exact_shortest_distance: int | None
    optimality_gap: int | None
    retry_count: int
    provenance: Literal["live", "mock", "simulated", "unknown"] = "unknown"
    provenance_label: str | None = None


class ModelMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class BenchmarkResultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    run_id: str
    execution_mode: ExecutionModeSchema | None
    prompt_version: Literal[
        "rubiks-move-objective-v1",
        "rubiks-move-objective-v2-state-aware",
    ] | None = None
    action_previews_enabled: bool = False
    max_valid_moves: int
    turn_timeout_seconds: float | None
    trials: list[TrialResultResponse]
    metrics: dict[str, ModelMetricsResponse]


class RunSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    execution_mode: ExecutionModeSchema | None
    prompt_version: Literal[
        "rubiks-move-objective-v1",
        "rubiks-move-objective-v2-state-aware",
    ] | None = None
    action_previews_enabled: bool = False
    puzzle_count: int
    trial_count: int
    max_valid_moves: int
    solved_trials_by_model: dict[str, int]
    solve_rate_by_model: dict[str, float]


def results_response(
    run: BenchmarkRun,
    *,
    execution_mode: ExecutionModeSchema | None,
) -> BenchmarkResultsResponse:
    metrics = aggregate_metrics(run)
    data = run.to_dict()
    data["trials"] = [trial.to_dict() for trial in run.trials]
    return BenchmarkResultsResponse.model_validate(
        {
            **data,
            "execution_mode": execution_mode,
            "metrics": {model_id: metric.to_dict() for model_id, metric in metrics.items()},
        }
    )
