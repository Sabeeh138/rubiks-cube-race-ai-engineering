"""Provider-neutral orchestration for comparable cube trials."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from app.config import ConfigurationError
from app.cube import (
    MAX_REFERENCE_DEPTH,
    CubeState,
    LEGAL_MOVES,
    ShortestSolution,
    exact_shortest_solution,
    generate_scramble,
)
from app.models import (
    ASSIGNED_MODEL_IDS,
    AdapterError,
    AdapterResponseError,
    AdapterTimeoutError,
    InvalidMoveError,
    ModelAdapter,
    MoveProposal,
    TurnContext,
    validate_move,
)
from app.models.prompts import RUBIKS_MOVE_OBJECTIVE_V1

from app.benchmark.results import BenchmarkRun, TrialResult, TurnAttempt

MAX_TRIAL_VALID_MOVES = 30
ReferenceSolver = Callable[[CubeState], ShortestSolution | None]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class PuzzleSpec:
    """A named, reproducible puzzle defined by its scramble and optional seed."""

    puzzle_id: str
    scramble_moves: tuple[str, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.puzzle_id, str) or not self.puzzle_id.strip():
            raise ValueError("puzzle_id must be a non-empty string")
        if isinstance(self.scramble_moves, str):
            raise TypeError("scramble_moves must be a sequence of move tokens")
        try:
            moves = tuple(self.scramble_moves)
        except TypeError as exc:
            raise TypeError("scramble_moves must be a sequence of move tokens") from exc
        # Let the shared engine validate the project-wide move vocabulary.
        CubeState().apply_moves(moves)
        object.__setattr__(self, "scramble_moves", moves)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")

    @classmethod
    def from_seed(
        cls,
        puzzle_id: str,
        *,
        seed: int,
        length: int = 5,
    ) -> PuzzleSpec:
        """Build a puzzle from the engine's deterministic seeded generator."""

        scramble = generate_scramble(seed=seed, length=length)
        return cls(puzzle_id=puzzle_id, scramble_moves=scramble.moves, seed=scramble.seed)

    @property
    def scramble_notation(self) -> str:
        return " ".join(self.scramble_moves)

    @property
    def initial_state(self) -> CubeState:
        return CubeState().apply_moves(self.scramble_moves)


class _AdapterContractViolation(Exception):
    """An adapter returned a value inconsistent with the shared contract."""


class BenchmarkRunner:
    """Run sequential, isolated trials with shared puzzle and timing policies.

    The injected monotonic clock measures each complete ``choose_move`` call
    (including response parsing and the runner's move validation) consistently
    for every adapter. The exact reference search runs before trial timing.
    """

    def __init__(
        self,
        *,
        max_valid_moves: int = MAX_TRIAL_VALID_MOVES,
        turn_timeout_seconds: float | None = 30.0,
        clock: Clock = perf_counter,
        reference_solver: ReferenceSolver = exact_shortest_solution,
        prompt_version: str = RUBIKS_MOVE_OBJECTIVE_V1,
        action_previews_enabled: bool = False,
    ) -> None:
        if isinstance(max_valid_moves, bool) or not isinstance(max_valid_moves, int):
            raise TypeError("max_valid_moves must be an integer")
        if not 1 <= max_valid_moves <= MAX_TRIAL_VALID_MOVES:
            raise ValueError(f"max_valid_moves must be between 1 and {MAX_TRIAL_VALID_MOVES}")
        if turn_timeout_seconds is not None and (
            isinstance(turn_timeout_seconds, bool)
            or not isinstance(turn_timeout_seconds, (int, float))
            or not math.isfinite(float(turn_timeout_seconds))
            or turn_timeout_seconds <= 0
        ):
            raise ValueError("turn_timeout_seconds must be a positive finite number or None")
        self.max_valid_moves = max_valid_moves
        self.turn_timeout_seconds = (
            None if turn_timeout_seconds is None else float(turn_timeout_seconds)
        )
        self._clock = clock
        self._reference_solver = reference_solver
        self.prompt_version = prompt_version
        self.action_previews_enabled = action_previews_enabled
        # Validate prompt metadata before any adapter receives a request.
        TurnContext(
            cube_state=CubeState().serialize(),
            prompt_version=prompt_version,
            action_previews_enabled=action_previews_enabled,
        )

    async def run_trial(
        self,
        puzzle: PuzzleSpec,
        adapter: ModelAdapter,
        *,
        run_id: str = "single-trial",
    ) -> TrialResult:
        """Run one adapter against one puzzle; useful for offline unit tests."""

        _validate_adapter(adapter)
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        initial = puzzle.initial_state
        reference_distance = self._reference_distance(puzzle, initial)
        return await self._run_one(
            puzzle,
            adapter,
            initial,
            reference_distance,
            run_id=run_id,
        )

    async def run_suite(
        self,
        puzzles: Iterable[PuzzleSpec],
        adapters: Iterable[ModelAdapter] | Mapping[str, ModelAdapter],
        *,
        run_id: str | None = None,
    ) -> BenchmarkRun:
        """Run every puzzle against exactly the four assigned model IDs.

        Every model receives a new immutable ``CubeState`` reconstructed from
        the same canonical initial serialization. A failure ends only that
        model's trial; remaining adapters and puzzles still run.
        """

        puzzle_list = tuple(puzzles)
        if not puzzle_list or any(not isinstance(puzzle, PuzzleSpec) for puzzle in puzzle_list):
            raise ValueError("puzzles must contain at least one PuzzleSpec")
        puzzle_ids = [puzzle.puzzle_id for puzzle in puzzle_list]
        if len(set(puzzle_ids)) != len(puzzle_ids):
            raise ValueError("puzzle_id values must be unique within a run")

        adapter_list = tuple(adapters.values()) if isinstance(adapters, Mapping) else tuple(adapters)
        for adapter in adapter_list:
            _validate_adapter(adapter)
        adapter_by_id = {adapter.model_id: adapter for adapter in adapter_list}
        if len(adapter_by_id) != len(adapter_list):
            raise ValueError("adapter model_id values must be unique")
        if set(adapter_by_id) != set(ASSIGNED_MODEL_IDS):
            raise ValueError(
                "run_suite requires exactly the four assigned model IDs: "
                + ", ".join(ASSIGNED_MODEL_IDS)
            )
        ordered_adapters = tuple(adapter_by_id[model_id] for model_id in ASSIGNED_MODEL_IDS)
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("run_id must be a non-empty string")
        resolved_run_id = run_id if run_id is not None else _deterministic_run_id(
            puzzle_list,
            max_valid_moves=self.max_valid_moves,
            timeout=self.turn_timeout_seconds,
            prompt_version=self.prompt_version,
            action_previews_enabled=self.action_previews_enabled,
        )

        trials: list[TrialResult] = []
        for puzzle in puzzle_list:
            shared_initial_state = puzzle.initial_state
            reference_distance = self._reference_distance(puzzle, shared_initial_state)
            for adapter in ordered_adapters:
                trials.append(
                    await self._run_one(
                        puzzle,
                        adapter,
                        shared_initial_state,
                        reference_distance,
                        run_id=resolved_run_id,
                    )
                )
        return BenchmarkRun(
            run_id=resolved_run_id,
            max_valid_moves=self.max_valid_moves,
            turn_timeout_seconds=self.turn_timeout_seconds,
            trials=tuple(trials),
            prompt_version=self.prompt_version,
            action_previews_enabled=self.action_previews_enabled,
        )

    def _reference_distance(self, puzzle: PuzzleSpec, initial: CubeState) -> int | None:
        # Keep the exact search in the documented primary benchmark scope.
        if len(puzzle.scramble_moves) > MAX_REFERENCE_DEPTH:
            return None
        reference = self._reference_solver(initial)
        if reference is None:
            return None
        if not isinstance(reference, ShortestSolution):
            raise TypeError("reference_solver must return ShortestSolution or None")
        return reference.distance_htm

    async def _run_one(
        self,
        puzzle: PuzzleSpec,
        adapter: ModelAdapter,
        common_initial_state: CubeState,
        exact_shortest_distance: int | None,
        *,
        run_id: str,
    ) -> TrialResult:
        model_id = adapter.model_id
        provenance = getattr(adapter, "execution_provenance", "unknown")
        provenance_label = getattr(adapter, "provenance_label", None)
        if provenance not in {"live", "mock", "simulated", "unknown"}:
            raise ValueError("adapter execution_provenance is not recognized")
        simulated = provenance == "simulated"
        trial_started = self._clock()
        state = CubeState.from_facelets(common_initial_state.serialize())
        attempts: list[TurnAttempt] = []
        history: list[str] = []
        termination_status: str | None = None

        while termination_status is None:
            if state.is_solved:
                termination_status = "solved"
                break
            if len(history) >= self.max_valid_moves:
                termination_status = "move_limit"
                break

            context = TurnContext(
                cube_state=state.serialize(),
                recent_moves=tuple(history),
                legal_moves=LEGAL_MOVES,
                timeout_seconds=self.turn_timeout_seconds,
                original_scramble=puzzle.scramble_moves,
                prompt_version=self.prompt_version,
                action_previews_enabled=self.action_previews_enabled,
            )
            request_started = self._clock()
            try:
                proposal = await adapter.choose_move(context)
                if not isinstance(proposal, MoveProposal):
                    raise _AdapterContractViolation("adapter did not return MoveProposal")
                if proposal.model_id != model_id:
                    raise _AdapterContractViolation("proposal model_id differs from adapter model_id")
                move = validate_move(
                    proposal.move,
                    legal_moves=context.legal_moves,
                    model_id=model_id,
                    token_usage=proposal.token_usage,
                )
            except InvalidMoveError as exc:
                ended = self._clock()
                latency = 0.0 if simulated else _duration(request_started, ended)
                attempts.append(
                    self._failure_attempt(
                        index=len(attempts),
                        state=state,
                        history=history,
                        latency=latency,
                        trial_elapsed=0.0 if simulated else _duration(trial_started, ended),
                        outcome="invalid_move",
                        error=exc,
                        rejected_move=_rejected_move(exc),
                        token_usage=exc.token_usage,
                    )
                )
                termination_status = "invalid_move"
                break
            except (AdapterTimeoutError, TimeoutError) as exc:
                ended = self._clock()
                latency = 0.0 if simulated else _duration(request_started, ended)
                attempts.append(
                    self._failure_attempt(
                        index=len(attempts),
                        state=state,
                        history=history,
                        latency=latency,
                        trial_elapsed=0.0 if simulated else _duration(trial_started, ended),
                        outcome="timeout",
                        error=exc,
                    )
                )
                termination_status = "timeout"
                break
            except Exception as exc:
                ended = self._clock()
                latency = 0.0 if simulated else _duration(request_started, ended)
                attempts.append(
                    self._failure_attempt(
                        index=len(attempts),
                        state=state,
                        history=history,
                        latency=latency,
                        trial_elapsed=0.0 if simulated else _duration(trial_started, ended),
                        outcome="adapter_error",
                        error=exc,
                        token_usage=exc.usage if isinstance(exc, AdapterResponseError) else None,
                    )
                )
                termination_status = "adapter_error"
                break

            ended = self._clock()
            latency = 0.0 if simulated else _duration(request_started, ended)
            attempts.append(
                TurnAttempt(
                    attempt_index=len(attempts),
                    cube_state_before=state.serialize(),
                    recent_moves=tuple(history),
                    latency_seconds=latency,
                    elapsed_trial_seconds=0.0 if simulated else _duration(trial_started, ended),
                    outcome="valid",
                    move=move,
                    served_model=_served_model(proposal),
                    token_usage=(
                        dict(proposal.token_usage) if proposal.token_usage is not None else None
                    ),
                )
            )
            state = state.apply_move(move)
            history.append(move)

        ended = self._clock()
        total_runtime = 0.0 if simulated else _duration(trial_started, ended)
        return TrialResult(
            trial_id=f"{run_id}:{puzzle.puzzle_id}:{model_id}",
            puzzle_id=puzzle.puzzle_id,
            model_id=model_id,
            seed=puzzle.seed,
            scramble_moves=puzzle.scramble_moves,
            initial_state=common_initial_state.serialize(),
            attempts=tuple(attempts),
            final_state=state.serialize(),
            termination_status=termination_status,
            max_valid_moves=self.max_valid_moves,
            total_runtime_seconds=total_runtime,
            exact_shortest_distance=exact_shortest_distance,
            provenance=provenance,
            provenance_label=provenance_label,
        )

    @staticmethod
    def _failure_attempt(
        *,
        index: int,
        state: CubeState,
        history: list[str],
        latency: float,
        trial_elapsed: float,
        outcome: str,
        error: Exception,
        rejected_move: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> TurnAttempt:
        return TurnAttempt(
            attempt_index=index,
            cube_state_before=state.serialize(),
            recent_moves=tuple(history),
            latency_seconds=latency,
            elapsed_trial_seconds=trial_elapsed,
            outcome=outcome,
            error_type=type(error).__name__,
            error_message=_safe_error_message(error),
            rejected_move=rejected_move,
            token_usage=token_usage,
        )


def _validate_adapter(adapter: ModelAdapter) -> None:
    model_id = getattr(adapter, "model_id", None)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("each adapter must expose a non-empty model_id")
    if not callable(getattr(adapter, "choose_move", None)):
        raise TypeError("each adapter must implement async choose_move(context)")


def _duration(start: float, end: float) -> float:
    duration = end - start
    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError("benchmark clock must be monotonic and return finite values")
    return duration


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, InvalidMoveError):
        return "adapter proposal did not match the legal move vocabulary"
    if isinstance(error, ConfigurationError):
        return str(error)[:300]
    if isinstance(error, (AdapterError, _AdapterContractViolation)):
        return str(error)[:300]
    return f"adapter raised {type(error).__name__}; details omitted"


def _rejected_move(error: InvalidMoveError) -> str:
    proposed = error.proposed_move
    if isinstance(proposed, str):
        return proposed[:160]
    return f"<{type(proposed).__name__}>"


def _served_model(proposal: MoveProposal) -> str | None:
    response = proposal.raw_response
    if isinstance(response, Mapping):
        model = response.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None


def _deterministic_run_id(
    puzzles: tuple[PuzzleSpec, ...],
    *,
    max_valid_moves: int,
    timeout: float | None,
    prompt_version: str,
    action_previews_enabled: bool,
) -> str:
    configuration = {
        "models": list(ASSIGNED_MODEL_IDS),
        "max_valid_moves": max_valid_moves,
        "turn_timeout_seconds": timeout,
        "prompt_version": prompt_version,
        "action_previews_enabled": action_previews_enabled,
        "puzzles": [
            {"puzzle_id": puzzle.puzzle_id, "seed": puzzle.seed, "scramble_moves": puzzle.scramble_moves}
            for puzzle in puzzles
        ],
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return "run-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
