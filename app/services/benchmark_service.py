"""Thread-safe service for background benchmark runs and result access."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.benchmark import BenchmarkRun, BenchmarkRunner, PuzzleSpec
from app.benchmark.primary_dataset import primary_puzzle_specs
from app.benchmark.storage import ResultsStorageError, load_run, save_run
from app.config import ConfigurationError, PROJECT_ROOT, Settings
from app.cube import invert_moves
from app.models import (
    ASSIGNED_MODEL_IDS,
    CLAUDE_OPUS_55_MODEL_ID,
    ModelAdapter,
    ScriptedClaudeFallbackAdapter,
    ScriptedMockAdapter,
    build_openrouter_adapters,
)
from app.models.prompts import (
    RUBIKS_MOVE_OBJECTIVE_V1,
    RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
)

ExecutionMode = Literal["mock", "live", "recording", "diagnostic", "state_aware_experiment"]
AdapterFactory = Callable[[ExecutionMode, Settings | None, PuzzleSpec], Mapping[str, ModelAdapter]]
RunnerFactory = Callable[..., BenchmarkRunner]
SettingsProvider = Callable[[], Settings]
SaveRun = Callable[[BenchmarkRun, str | Path], Path]
LoadRun = Callable[[str | Path], BenchmarkRun]


class BenchmarkAlreadyRunning(RuntimeError):
    def __init__(self, active_run_id: str) -> None:
        self.active_run_id = active_run_id
        super().__init__(f"benchmark run {active_run_id} is already running")


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused for a different benchmark request."""


class UnknownRunError(LookupError):
    pass


class RunNotReadyError(RuntimeError):
    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"benchmark run is {status}; results are not available")


class BenchmarkServiceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunProgress:
    completed_trials: int
    total_trials: int
    completed_puzzles: int
    total_puzzles: int
    current_puzzle_id: str | None


@dataclass(frozen=True, slots=True)
class RunStatusSnapshot:
    run_id: str
    status: Literal["running", "completed", "failed"]
    execution_mode: ExecutionMode | None
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    progress: RunProgress
    error: str | None


@dataclass(slots=True)
class _RunRecord:
    run_id: str
    execution_mode: ExecutionMode
    status: Literal["running", "completed", "failed"]
    created_at: datetime
    started_at: datetime
    completed_at: datetime | None
    total_puzzles: int
    total_trials: int
    completed_puzzles: int = 0
    completed_trials: int = 0
    current_puzzle_id: str | None = None
    error: str | None = None


class BenchmarkService:
    """Admit one benchmark at a time and run it on a dedicated worker thread.

    A process-local lock serializes admissions and state updates. Each run gets
    a cryptographically unique UUID filename, while an optional
    ``Idempotency-Key`` lets a client safely retry a submission. This service
    intentionally supports one active run per process; multi-process locking
    is outside the current local-backend scope.
    """

    def __init__(
        self,
        *,
        results_dir: str | Path = PROJECT_ROOT / "results",
        settings_provider: SettingsProvider = Settings.from_env,
        adapter_factory: AdapterFactory | None = None,
        runner_factory: RunnerFactory = BenchmarkRunner,
        save_run_fn: SaveRun = save_run,
        load_run_fn: LoadRun = load_run,
    ) -> None:
        self.results_dir = Path(results_dir)
        self._settings_provider = settings_provider
        self._adapter_factory = adapter_factory or _default_adapter_factory
        self._runner_factory = runner_factory
        self._save_run = save_run_fn
        self._load_run = load_run_fn
        self._lock = threading.RLock()
        self._records: dict[str, _RunRecord] = {}
        self._active_run_id: str | None = None
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._executor: ThreadPoolExecutor | None = None

    def start_benchmark(
        self,
        puzzles: Sequence[PuzzleSpec],
        *,
        execution_mode: ExecutionMode = "mock",
        max_valid_moves: int = 30,
        turn_timeout_seconds: float = 30.0,
        action_previews_enabled: bool = False,
        idempotency_key: str | None = None,
    ) -> RunStatusSnapshot:
        puzzle_list = tuple(puzzles)
        _validate_submission(
            puzzle_list,
            execution_mode=execution_mode,
            max_valid_moves=max_valid_moves,
            turn_timeout_seconds=turn_timeout_seconds,
            action_previews_enabled=action_previews_enabled,
        )
        fingerprint = _request_fingerprint(
            puzzle_list,
            execution_mode=execution_mode,
            max_valid_moves=max_valid_moves,
            turn_timeout_seconds=turn_timeout_seconds,
            action_previews_enabled=action_previews_enabled,
        )
        settings: Settings | None = None
        if execution_mode in {"live", "recording", "diagnostic", "state_aware_experiment"}:
            settings = self._settings_provider()
            settings.require_openrouter_api_key()

        normalized_key = _normalize_idempotency_key(idempotency_key)
        with self._lock:
            if normalized_key is not None and normalized_key in self._idempotency:
                prior_fingerprint, prior_run_id = self._idempotency[normalized_key]
                if prior_fingerprint != fingerprint:
                    raise IdempotencyConflict("Idempotency-Key was already used with a different request")
                record = self._records.get(prior_run_id)
                if record is not None:
                    return _snapshot(record)
                persisted = self._load_persisted(prior_run_id)
                if persisted is not None:
                    return _completed_snapshot(
                        persisted,
                        execution_mode=persisted.execution_mode,
                    )
                raise UnknownRunError(prior_run_id)

            if self._active_run_id is not None:
                raise BenchmarkAlreadyRunning(self._active_run_id)

            run_id = str(uuid.uuid4())
            now = _utc_now()
            record = _RunRecord(
                run_id=run_id,
                execution_mode=execution_mode,
                status="running",
                created_at=now,
                started_at=now,
                completed_at=None,
                total_puzzles=len(puzzle_list),
                total_trials=len(puzzle_list) * len(ASSIGNED_MODEL_IDS),
                current_puzzle_id=puzzle_list[0].puzzle_id,
            )
            self._records[run_id] = record
            self._active_run_id = run_id
            if normalized_key is not None:
                self._idempotency[normalized_key] = (fingerprint, run_id)
            try:
                executor = self._get_executor()
                executor.submit(
                    self._execute,
                    run_id,
                    puzzle_list,
                    execution_mode,
                    max_valid_moves,
                    float(turn_timeout_seconds),
                    action_previews_enabled,
                    settings,
                )
            except Exception as exc:
                record.status = "failed"
                record.completed_at = _utc_now()
                record.current_puzzle_id = None
                record.error = f"{type(exc).__name__}: benchmark worker could not be started"
                self._active_run_id = None
                raise BenchmarkServiceError("benchmark worker could not be started") from exc
            return _snapshot(record)

    def get_status(self, run_id: str) -> RunStatusSnapshot | None:
        normalized = _valid_run_id(run_id)
        if normalized is None:
            return None
        with self._lock:
            record = self._records.get(normalized)
            if record is not None:
                return _snapshot(record)
        persisted = self._load_persisted(normalized)
        return (
            _completed_snapshot(persisted, execution_mode=persisted.execution_mode)
            if persisted is not None
            else None
        )

    def get_results(
        self,
        run_id: str,
    ) -> tuple[BenchmarkRun, ExecutionMode | None]:
        normalized = _valid_run_id(run_id)
        if normalized is None:
            raise UnknownRunError(run_id)
        with self._lock:
            record = self._records.get(normalized)
            if record is not None:
                if record.status != "completed":
                    raise RunNotReadyError(record.status)
                execution_mode: ExecutionMode | None = record.execution_mode
            else:
                execution_mode = None
        path = self._result_path(normalized)
        try:
            exists = path.is_file()
        except OSError as exc:
            raise ResultsStorageError(f"could not inspect result file for run {normalized}") from exc
        if not exists:
            if record is None:
                raise UnknownRunError(normalized)
            raise ResultsStorageError(f"completed result file is missing for run {normalized}")
        run = self._load_run(path)
        return run, execution_mode if execution_mode is not None else run.execution_mode

    def list_results(self, *, limit: int = 20) -> list[tuple[BenchmarkRun, ExecutionMode | None]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            paths = sorted(
                (path for path in self.results_dir.glob("*.json") if _valid_run_id(path.stem)),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:limit]
        except OSError as exc:
            raise ResultsStorageError("could not list benchmark result files") from exc
        output: list[tuple[BenchmarkRun, ExecutionMode | None]] = []
        for path in paths:
            run = self._load_run(path)
            with self._lock:
                record = self._records.get(run.run_id)
                mode = record.execution_mode if record is not None else run.execution_mode
            output.append((run, mode))
        return output

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work through the executor and wait during app shutdown."""

        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cube-benchmark")
        return self._executor

    def _execute(
        self,
        run_id: str,
        puzzles: tuple[PuzzleSpec, ...],
        execution_mode: ExecutionMode,
        max_valid_moves: int,
        turn_timeout_seconds: float,
        action_previews_enabled: bool,
        settings: Settings | None,
    ) -> None:
        try:
            run = asyncio.run(
                self._execute_async(
                    run_id,
                    puzzles,
                    execution_mode,
                    max_valid_moves,
                    turn_timeout_seconds,
                    action_previews_enabled,
                    settings,
                )
            )
            self._save_run(run, self._result_path(run_id))
        except Exception as exc:
            with self._lock:
                record = self._records[run_id]
                record.status = "failed"
                record.completed_at = _utc_now()
                record.current_puzzle_id = None
                record.error = _safe_service_error(exc)
                if self._active_run_id == run_id:
                    self._active_run_id = None
            return
        with self._lock:
            record = self._records[run_id]
            record.status = "completed"
            record.completed_at = _utc_now()
            record.current_puzzle_id = None
            record.completed_trials = len(run.trials)
            record.completed_puzzles = record.total_puzzles
            if self._active_run_id == run_id:
                self._active_run_id = None

    async def _execute_async(
        self,
        run_id: str,
        puzzles: tuple[PuzzleSpec, ...],
        execution_mode: ExecutionMode,
        max_valid_moves: int,
        turn_timeout_seconds: float,
        action_previews_enabled: bool,
        settings: Settings | None,
    ) -> BenchmarkRun:
        trials = []
        for puzzle in puzzles:
            with self._lock:
                self._records[run_id].current_puzzle_id = puzzle.puzzle_id
            adapters = self._adapter_factory(execution_mode, settings, puzzle)
            try:
                runner = self._runner_factory(
                    max_valid_moves=max_valid_moves,
                    turn_timeout_seconds=turn_timeout_seconds,
                    prompt_version=(
                        RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
                        if execution_mode in {"diagnostic", "state_aware_experiment"}
                        else RUBIKS_MOVE_OBJECTIVE_V1
                    ),
                    action_previews_enabled=action_previews_enabled,
                )
                partial_run = await runner.run_suite((puzzle,), adapters, run_id=run_id)
                trials.extend(partial_run.trials)
            finally:
                await _close_adapters(adapters)
            with self._lock:
                record = self._records[run_id]
                record.completed_puzzles += 1
                record.completed_trials += len(partial_run.trials)
                record.current_puzzle_id = (
                    puzzles[record.completed_puzzles].puzzle_id
                    if record.completed_puzzles < len(puzzles)
                    else None
                )
        return BenchmarkRun(
            run_id=run_id,
            max_valid_moves=max_valid_moves,
            turn_timeout_seconds=turn_timeout_seconds,
            trials=tuple(trials),
            execution_mode=execution_mode,
            prompt_version=(
                RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
                if execution_mode in {"diagnostic", "state_aware_experiment"}
                else RUBIKS_MOVE_OBJECTIVE_V1
            ),
            action_previews_enabled=action_previews_enabled,
        )

    def _load_persisted(self, run_id: str) -> BenchmarkRun | None:
        path = self._result_path(run_id)
        try:
            if not path.is_file():
                return None
        except OSError as exc:
            raise ResultsStorageError(f"could not inspect result file for run {run_id}") from exc
        return self._load_run(path)

    def _result_path(self, run_id: str) -> Path:
        normalized = _valid_run_id(run_id)
        if normalized is None:
            raise UnknownRunError(run_id)
        return self.results_dir / f"{normalized}.json"


def _default_adapter_factory(
    execution_mode: ExecutionMode,
    settings: Settings | None,
    puzzle: PuzzleSpec,
) -> Mapping[str, ModelAdapter]:
    if execution_mode == "mock":
        outcomes = invert_moves(puzzle.scramble_moves)
        return {
            model_id: ScriptedMockAdapter(outcomes, model_id=model_id)
            for model_id in ASSIGNED_MODEL_IDS
        }
    if settings is None:
        raise ConfigurationError("OpenRouter settings were not provided for live execution")
    settings.require_openrouter_api_key()
    adapters = dict(build_openrouter_adapters(settings=settings))
    if execution_mode in {"recording", "diagnostic", "state_aware_experiment"}:
        adapters[CLAUDE_OPUS_55_MODEL_ID] = ScriptedClaudeFallbackAdapter(
            invert_moves(puzzle.scramble_moves)
        )
    return adapters


async def _close_adapters(adapters: Mapping[str, ModelAdapter]) -> None:
    for adapter in adapters.values():
        close = getattr(adapter, "aclose", None)
        if callable(close):
            await close()


def _validate_submission(
    puzzles: tuple[PuzzleSpec, ...],
    *,
    execution_mode: str,
    max_valid_moves: int,
    turn_timeout_seconds: float,
    action_previews_enabled: bool,
) -> None:
    if not puzzles or any(not isinstance(puzzle, PuzzleSpec) for puzzle in puzzles):
        raise ValueError("puzzles must contain at least one PuzzleSpec")
    if len({puzzle.puzzle_id for puzzle in puzzles}) != len(puzzles):
        raise ValueError("puzzle_id values must be unique")
    if execution_mode not in {"mock", "live", "recording", "diagnostic", "state_aware_experiment"}:
        raise ValueError("execution_mode must be 'mock', 'live', 'recording', 'diagnostic', or 'state_aware_experiment'")
    if not isinstance(action_previews_enabled, bool):
        raise TypeError("action_previews_enabled must be a boolean")
    if action_previews_enabled and execution_mode not in {"diagnostic", "state_aware_experiment"}:
        raise ValueError("action previews are available only in state-aware exploratory modes")
    if isinstance(max_valid_moves, bool) or not isinstance(max_valid_moves, int) or not 1 <= max_valid_moves <= 30:
        raise ValueError("max_valid_moves must be between 1 and 30")
    if (
        isinstance(turn_timeout_seconds, bool)
        or not isinstance(turn_timeout_seconds, (int, float))
        or not math.isfinite(float(turn_timeout_seconds))
        or not 0.1 <= turn_timeout_seconds <= 120
    ):
        raise ValueError("turn_timeout_seconds must be between 0.1 and 120 seconds")
    if execution_mode == "diagnostic":
        if len(puzzles) != 1 or puzzles[0].scramble_moves != ("R",):
            raise ValueError("diagnostic mode requires exactly one puzzle with the fixed scramble ['R']")
        if max_valid_moves != 3:
            raise ValueError("diagnostic mode requires max_valid_moves=3")
    if execution_mode == "state_aware_experiment":
        expected = primary_puzzle_specs()
        if len(puzzles) != len(expected) or any(
            actual.puzzle_id != pinned.puzzle_id
            or actual.seed != pinned.seed
            or actual.scramble_moves != pinned.scramble_moves
            for actual, pinned in zip(puzzles, expected, strict=True)
        ):
            raise ValueError("the full state-aware experiment requires the exact five pinned primary puzzles")
        if max_valid_moves != 30 or not math.isclose(
            float(turn_timeout_seconds), 30.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("the full state-aware experiment requires the existing 30-move and 30-second limits")
        if not action_previews_enabled:
            raise ValueError("the full state-aware experiment requires one-step action previews")


def _normalize_idempotency_key(key: str | None) -> str | None:
    if key is None:
        return None
    normalized = key.strip()
    if not normalized or len(normalized) > 128:
        raise ValueError("Idempotency-Key must contain 1 to 128 non-whitespace characters")
    return normalized


def _request_fingerprint(
    puzzles: tuple[PuzzleSpec, ...],
    *,
    execution_mode: ExecutionMode,
    max_valid_moves: int,
    turn_timeout_seconds: float,
    action_previews_enabled: bool,
) -> str:
    request = {
        "execution_mode": execution_mode,
        "max_valid_moves": max_valid_moves,
        "turn_timeout_seconds": float(turn_timeout_seconds),
        "prompt_version": (
            RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
            if execution_mode in {"diagnostic", "state_aware_experiment"}
            else RUBIKS_MOVE_OBJECTIVE_V1
        ),
        "action_previews_enabled": action_previews_enabled,
        "puzzles": [
            {
                "puzzle_id": puzzle.puzzle_id,
                "seed": puzzle.seed,
                "scramble_moves": puzzle.scramble_moves,
            }
            for puzzle in puzzles
        ],
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_run_id(run_id: str) -> str | None:
    try:
        parsed = uuid.UUID(run_id)
    except (ValueError, AttributeError, TypeError):
        return None
    normalized = str(parsed)
    return normalized if normalized == run_id.lower() else None


def _snapshot(record: _RunRecord) -> RunStatusSnapshot:
    return RunStatusSnapshot(
        run_id=record.run_id,
        status=record.status,
        execution_mode=record.execution_mode,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        progress=RunProgress(
            completed_trials=record.completed_trials,
            total_trials=record.total_trials,
            completed_puzzles=record.completed_puzzles,
            total_puzzles=record.total_puzzles,
            current_puzzle_id=record.current_puzzle_id,
        ),
        error=record.error,
    )


def _completed_snapshot(
    run: BenchmarkRun | None,
    *,
    execution_mode: ExecutionMode | None,
) -> RunStatusSnapshot:
    if run is None:
        raise UnknownRunError("run was not found")
    puzzle_count = len({trial.puzzle_id for trial in run.trials})
    return RunStatusSnapshot(
        run_id=run.run_id,
        status="completed",
        execution_mode=execution_mode,
        created_at=None,
        started_at=None,
        completed_at=None,
        progress=RunProgress(
            completed_trials=len(run.trials),
            total_trials=len(run.trials),
            completed_puzzles=puzzle_count,
            total_puzzles=puzzle_count,
            current_puzzle_id=None,
        ),
        error=None,
    )


def _safe_service_error(error: Exception) -> str:
    if isinstance(error, ConfigurationError):
        return str(error)[:300]
    if isinstance(error, ResultsStorageError):
        return "ResultsStorageError: benchmark results could not be persisted"
    return f"{type(error).__name__}: benchmark execution failed; details omitted"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
