"""HTTP routes for health checks and asynchronous benchmark execution."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status

from app.api.schemas import (
    BenchmarkRequest,
    BenchmarkResultsResponse,
    BenchmarkSubmissionResponse,
    CubePreviewResponse,
    ReferenceSolutionResponse,
    RunProgressResponse,
    RunStatusResponse,
    RunSummaryResponse,
    TrialResultResponse,
    PuzzleRequest,
    PrimaryDatasetResponse,
    PrimaryPuzzleResponse,
    StudentDisplayConfig,
    results_response,
)
from app.benchmark import aggregate_metrics
from app.benchmark.primary_dataset import PRIMARY_DATASET_ID, build_primary_dataset
from app.benchmark.storage import ResultsStorageError
from app.config import ConfigurationError, Settings
from app.cube import exact_shortest_solution
from app.services.benchmark_service import (
    BenchmarkAlreadyRunning,
    BenchmarkService,
    BenchmarkServiceError,
    IdempotencyConflict,
    RunNotReadyError,
    UnknownRunError,
)

router = APIRouter()


def get_benchmark_service(request: Request) -> BenchmarkService:
    return request.app.state.benchmark_service


@router.get(
    "/api/ui-config",
    response_model=StudentDisplayConfig,
    tags=["frontend"],
    summary="Get public student-identification labels",
)
def get_ui_config() -> StudentDisplayConfig:
    """Return only the optional public labels; never serialize API settings."""

    settings = Settings.from_env()
    return StudentDisplayConfig(
        student_name=settings.student_name,
        student_roll_number=settings.student_roll_number,
    )


@router.post(
    "/api/cube/preview",
    response_model=CubePreviewResponse,
    tags=["frontend"],
    summary="Preview a puzzle using the shared cube engine",
)
def preview_cube(body: PuzzleRequest) -> CubePreviewResponse:
    """Generate facelets with the same PuzzleSpec and cube engine as trials."""

    puzzle = body.to_puzzle_spec()
    return CubePreviewResponse(
        puzzle_id=puzzle.puzzle_id,
        seed=puzzle.seed,
        scramble_moves=list(puzzle.scramble_moves),
        initial_state=puzzle.initial_state.serialize(),
    )


@router.get(
    "/api/benchmark/primary-dataset",
    response_model=PrimaryDatasetResponse,
    tags=["benchmarks"],
    summary="Get the pinned five-puzzle primary dataset",
)
def get_primary_dataset() -> PrimaryDatasetResponse:
    """Return the canonical pinned inputs for the exploratory dashboard mode."""

    return PrimaryDatasetResponse(
        dataset_id=PRIMARY_DATASET_ID,
        puzzles=[PrimaryPuzzleResponse(**entry.to_dict()) for entry in build_primary_dataset()],
    )


@router.post(
    "/api/cube/reference-solution",
    response_model=ReferenceSolutionResponse,
    tags=["frontend"],
    summary="Find and verify an exact short-state reference solution",
    responses={422: {"description": "No exact solution is available within the five-move reference scope."}},
)
def reference_solution(body: PuzzleRequest) -> ReferenceSolutionResponse:
    """Return a solver path and its cube-engine replay, without creating a trial."""

    puzzle = body.to_puzzle_spec()
    initial = puzzle.initial_state
    reference = exact_shortest_solution(initial)
    if reference is None:
        raise HTTPException(
            status_code=422,
            detail="No exact reference solution is available within the supported five-move search scope.",
        )

    current = initial
    steps = []
    for move in reference.moves:
        current = current.apply_move(move)
        steps.append({"move": move, "state_after": current.serialize()})
    if not current.is_solved or len(steps) != reference.distance_htm:
        raise HTTPException(status_code=500, detail="reference solver path failed cube-engine verification")

    return ReferenceSolutionResponse(
        puzzle_id=puzzle.puzzle_id,
        scramble_moves=list(puzzle.scramble_moves),
        initial_state=initial.serialize(),
        solution_moves=list(reference.moves),
        move_count=reference.distance_htm,
        steps=steps,
        final_state=current.serialize(),
        solved=True,
    )


@router.post(
    "/api/benchmark",
    response_model=BenchmarkSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["benchmarks"],
    summary="Start a benchmark run",
    responses={
        409: {"description": "Another run is active, or an idempotency key conflicts."},
        422: {"description": "The request contains invalid puzzle or run settings."},
        503: {"description": "Live execution is not configured or the worker cannot start."},
    },
)
def submit_benchmark(
    body: BenchmarkRequest,
    response: Response,
    service: BenchmarkService = Depends(get_benchmark_service),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> BenchmarkSubmissionResponse:
    try:
        snapshot = service.start_benchmark(
            body.to_puzzle_specs(),
            execution_mode=body.execution_mode,
            max_valid_moves=body.max_valid_moves,
            turn_timeout_seconds=body.turn_timeout_seconds,
            action_previews_enabled=body.action_previews_enabled,
            idempotency_key=idempotency_key,
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BenchmarkAlreadyRunning as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": "a benchmark is already running", "active_run_id": exc.active_run_id},
        ) from exc
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BenchmarkServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    status_url = f"/api/benchmark/{snapshot.run_id}"
    results_url = f"{status_url}/results"
    response.headers["Location"] = status_url
    return BenchmarkSubmissionResponse(
        run_id=snapshot.run_id,
        status=snapshot.status,
        execution_mode=body.execution_mode,
        status_url=status_url,
        results_url=results_url,
    )


@router.get(
    "/api/benchmark/{run_id}",
    response_model=RunStatusResponse,
    tags=["benchmarks"],
    summary="Get run status and progress",
    responses={404: {"description": "The run ID was not found."}, 503: {"description": "Stored status data is unreadable."}},
)
def get_benchmark_status(
    run_id: str,
    service: BenchmarkService = Depends(get_benchmark_service),
) -> RunStatusResponse:
    try:
        snapshot = service.get_status(run_id)
    except ResultsStorageError as exc:
        raise HTTPException(status_code=503, detail="stored benchmark results could not be read") from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail="benchmark run not found")
    return RunStatusResponse(
        run_id=snapshot.run_id,
        status=snapshot.status,
        execution_mode=snapshot.execution_mode,
        created_at=snapshot.created_at,
        started_at=snapshot.started_at,
        completed_at=snapshot.completed_at,
        progress=RunProgressResponse(
            completed_trials=snapshot.progress.completed_trials,
            total_trials=snapshot.progress.total_trials,
            completed_puzzles=snapshot.progress.completed_puzzles,
            total_puzzles=snapshot.progress.total_puzzles,
            current_puzzle_id=snapshot.progress.current_puzzle_id,
        ),
        error=snapshot.error,
    )


@router.get(
    "/api/benchmark/{run_id}/results",
    response_model=BenchmarkResultsResponse,
    tags=["benchmarks"],
    summary="Get completed results and aggregate metrics",
    responses={
        404: {"description": "The run ID was not found."},
        409: {"description": "The run is still running or failed."},
        503: {"description": "The result file could not be read."},
    },
)
def get_benchmark_results(
    run_id: str,
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResultsResponse:
    run, execution_mode = _read_run(service, run_id)
    return results_response(run, execution_mode=execution_mode)


@router.get(
    "/api/benchmark/{run_id}/trials/{trial_id:path}",
    response_model=TrialResultResponse,
    tags=["benchmarks"],
    summary="Get one completed puzzle/model trial",
    responses={
        404: {"description": "The run or trial ID was not found."},
        409: {"description": "The run is still running or failed."},
        503: {"description": "The result file could not be read."},
    },
)
def get_benchmark_trial(
    run_id: str,
    trial_id: str,
    service: BenchmarkService = Depends(get_benchmark_service),
) -> TrialResultResponse:
    run, _execution_mode = _read_run(service, run_id)
    for trial in run.trials:
        if trial.trial_id == trial_id:
            return TrialResultResponse.model_validate(trial.to_dict())
    raise HTTPException(status_code=404, detail="trial not found")


@router.get(
    "/api/results",
    response_model=list[RunSummaryResponse],
    tags=["results"],
    summary="List persisted benchmark summaries",
    responses={422: {"description": "The requested result limit is invalid."}, 503: {"description": "Result files could not be listed or read."}},
)
def list_benchmark_results(
    limit: int = Query(default=20, ge=1, le=100),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> list[RunSummaryResponse]:
    try:
        runs = service.list_results(limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResultsStorageError as exc:
        raise HTTPException(status_code=503, detail="benchmark result files could not be listed or read") from exc
    summaries: list[RunSummaryResponse] = []
    for run, execution_mode in runs:
        metrics = aggregate_metrics(run)
        summaries.append(
            RunSummaryResponse(
                run_id=run.run_id,
                execution_mode=execution_mode,
                prompt_version=run.prompt_version,
                action_previews_enabled=run.action_previews_enabled,
                puzzle_count=len({trial.puzzle_id for trial in run.trials}),
                trial_count=len(run.trials),
                max_valid_moves=run.max_valid_moves,
                solved_trials_by_model={model_id: metric.solved_trial_count for model_id, metric in metrics.items()},
                solve_rate_by_model={model_id: metric.solve_rate for model_id, metric in metrics.items()},
            )
        )
    return summaries


def _read_run(service: BenchmarkService, run_id: str):
    try:
        return service.get_results(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="benchmark run not found") from exc
    except RunNotReadyError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "status": exc.status},
        ) from exc
    except ResultsStorageError as exc:
        raise HTTPException(status_code=503, detail="stored benchmark results could not be read") from exc
