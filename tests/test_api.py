"""Offline API integration tests for benchmark submission and retrieval."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from app.benchmark import BenchmarkRunner, PuzzleSpec, load_run
from app.benchmark.primary_dataset import build_primary_dataset
from app.benchmark.storage import ResultsStorageError
from app.config import Settings
from app.cube import invert_moves
from app.models import (
    ASSIGNED_MODEL_IDS,
    CLAUDE_OPUS_55_MODEL_ID,
    AdapterReply,
    BaseModelAdapter,
    ScriptedClaudeFallbackAdapter,
    ScriptedMockAdapter,
    SIMULATED_CLAUDE_LABEL,
    TurnContext,
)
from app.services.benchmark_service import BenchmarkService
from app.main import create_app


def _service(tmp_path: Path, **kwargs) -> BenchmarkService:
    return BenchmarkService(results_dir=tmp_path / "results", **kwargs)


def _wait_for_status(client: TestClient, run_id: str, terminal: str) -> dict:
    deadline = time.monotonic() + 5
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/benchmark/{run_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] == terminal:
            return last
        time.sleep(0.01)
    pytest.fail(f"run did not reach {terminal!r}; last status was {last!r}")


def test_health_is_available_without_openrouter_client(monkeypatch, tmp_path: Path) -> None:
    def forbidden_client(*_args, **_kwargs):
        raise AssertionError("health must not create an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    with TestClient(create_app(benchmark_service=_service(tmp_path))) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_interactive_docs_and_openapi_include_backend_routes(tmp_path: Path) -> None:
    with TestClient(create_app(benchmark_service=_service(tmp_path))) as client:
        docs = client.get("/docs")
        schema_response = client.get("/openapi.json")
    assert docs.status_code == 200
    assert "SwaggerUIBundle" in docs.text
    assert schema_response.status_code == 200
    paths = schema_response.json()["paths"]
    assert "/api/health" in paths
    assert "/api/benchmark" in paths
    assert "/api/benchmark/{run_id}" in paths
    assert "/api/benchmark/{run_id}/results" in paths
    assert "/api/benchmark/{run_id}/trials/{trial_id}" in paths
    assert "/api/results" in paths


def test_mock_submission_completes_and_returns_metrics_trials_and_summary(
    monkeypatch, tmp_path: Path
) -> None:
    def forbidden_client(*_args, **_kwargs):
        raise AssertionError("mock API tests must never construct an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    service = _service(tmp_path)
    with TestClient(create_app(benchmark_service=service)) as client:
        submitted = client.post(
            "/api/benchmark",
            json={"puzzles": [{"puzzle_id": "api-puzzle", "scramble_moves": ["R", "U"]}]},
        )
        assert submitted.status_code == 202
        body = submitted.json()
        run_id = body["run_id"]
        assert body["status"] == "running"
        assert body["execution_mode"] == "mock"
        assert submitted.headers["location"] == f"/api/benchmark/{run_id}"

        status = _wait_for_status(client, run_id, "completed")
        assert status["progress"]["completed_trials"] == 4
        assert status["progress"]["total_trials"] == 4
        assert status["error"] is None

        results_response = client.get(f"/api/benchmark/{run_id}/results")
        assert results_response.status_code == 200
        results = results_response.json()
        assert results["run_id"] == run_id
        assert results["execution_mode"] == "mock"
        assert len(results["trials"]) == 4
        assert {trial["model_id"] for trial in results["trials"]} == set(ASSIGNED_MODEL_IDS)
        assert all(trial["solved"] for trial in results["trials"])
        assert all(trial["optimality_gap"] == 0 for trial in results["trials"])
        assert set(results["metrics"]) == set(ASSIGNED_MODEL_IDS)

        trial_id = results["trials"][0]["trial_id"]
        one_trial = client.get(f"/api/benchmark/{run_id}/trials/{trial_id}")
        assert one_trial.status_code == 200
        assert one_trial.json()["trial_id"] == trial_id

        summaries = client.get("/api/results")
        assert summaries.status_code == 200
        assert summaries.json()[0]["run_id"] == run_id
        assert summaries.json()[0]["trial_count"] == 4

    persisted = load_run(tmp_path / "results" / f"{run_id}.json")
    assert persisted.execution_mode == "mock"
    restarted_service = _service(tmp_path)
    recovered_status = restarted_service.get_status(run_id)
    recovered_run, recovered_mode = restarted_service.get_results(run_id)
    assert recovered_status is not None
    assert recovered_status.execution_mode == "mock"
    assert recovered_run.execution_mode == "mock"
    assert recovered_mode == "mock"


def test_recording_submission_persists_simulated_claude_separately_without_network(
    monkeypatch, tmp_path: Path
) -> None:
    def forbidden_client(*_args, **_kwargs):
        raise AssertionError("recording test must not construct an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)

    class OfflineLiveFixture(ScriptedMockAdapter):
        execution_provenance = "live"

    def factory(mode, _settings, puzzle):
        assert mode == "recording"
        moves = invert_moves(puzzle.scramble_moves)
        return {
            **{
                model_id: OfflineLiveFixture(moves, model_id=model_id)
                for model_id in ASSIGNED_MODEL_IDS[:3]
            },
            CLAUDE_OPUS_55_MODEL_ID: ScriptedClaudeFallbackAdapter(moves),
        }

    service = _service(
        tmp_path,
        settings_provider=lambda: Settings(openrouter_api_key="offline-test-key"),
        adapter_factory=factory,
    )
    with TestClient(create_app(benchmark_service=service)) as client:
        submitted = client.post(
            "/api/benchmark",
            json={
                "execution_mode": "recording",
                "puzzles": [{"puzzle_id": "record-api", "scramble_moves": ["R", "U"]}],
            },
        )
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]
        assert submitted.json()["execution_mode"] == "recording"
        assert _wait_for_status(client, run_id, "completed")["status"] == "completed"
        response = client.get(f"/api/benchmark/{run_id}/results")

    assert response.status_code == 200
    results = response.json()
    assert results["execution_mode"] == "recording"
    assert results["schema_version"] == 4
    assert results["prompt_version"] == "rubiks-move-objective-v1"
    assert results["action_previews_enabled"] is False
    assert set(results["metrics"]) == set(ASSIGNED_MODEL_IDS[:3])
    claude = next(trial for trial in results["trials"] if trial["model_id"] == CLAUDE_OPUS_55_MODEL_ID)
    assert claude["provenance"] == "simulated"
    assert claude["provenance_label"] == SIMULATED_CLAUDE_LABEL
    assert claude["solved"] is True
    assert claude["average_move_latency_seconds"] is None
    persisted = load_run(tmp_path / "results" / f"{run_id}.json")
    assert persisted.execution_mode == "recording"
    assert next(t for t in persisted.trials if t.model_id == CLAUDE_OPUS_55_MODEL_ID).provenance == "simulated"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"puzzles": []},
        {"puzzles": [{"puzzle_id": "bad", "scramble_moves": ["M"]}]},
        {"puzzles": [{"puzzle_id": "p", "seed": 7, "scramble_moves": ["R"]}]},
        {
            "puzzles": [
                {"puzzle_id": "duplicate", "scramble_moves": []},
                {"puzzle_id": "duplicate", "scramble_moves": ["R"]},
            ]
        },
        {"puzzles": [{"puzzle_id": "p", "seed": 7}], "max_valid_moves": 31},
        {"puzzles": [{"puzzle_id": "p", "seed": True}]},
        {"puzzles": [{"puzzle_id": "p", "seed": 7}], "max_valid_moves": True},
        {"puzzles": [{"puzzle_id": "p", "seed": 7}], "unexpected": True},
    ],
)
def test_invalid_benchmark_requests_return_422(payload: dict, tmp_path: Path) -> None:
    with TestClient(create_app(benchmark_service=_service(tmp_path))) as client:
        response = client.post("/api/benchmark", json=payload)
    assert response.status_code == 422


def test_unknown_runs_and_trials_return_404(tmp_path: Path) -> None:
    with TestClient(create_app(benchmark_service=_service(tmp_path))) as client:
        assert client.get("/api/benchmark/not-a-uuid").status_code == 404
        assert client.get("/api/benchmark/00000000-0000-0000-0000-000000000000/results").status_code == 404
        assert (
            client.get(
                "/api/benchmark/00000000-0000-0000-0000-000000000000/trials/missing"
            ).status_code
            == 404
        )


class _BlockingAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, entered: Event, release: Event) -> None:
        super().__init__(model_id)
        self.entered = entered
        self.release = release

    async def _request_move(self, context: TurnContext) -> AdapterReply:
        self.entered.set()
        await asyncio.to_thread(self.release.wait, 3)
        return AdapterReply("R'")


def test_running_status_progress_locking_and_idempotency(tmp_path: Path) -> None:
    entered = Event()
    release = Event()

    def factory(_mode, _settings, _puzzle):
        return {
            model_id: _BlockingAdapter(model_id, entered, release)
            for model_id in ASSIGNED_MODEL_IDS
        }

    # The test exercises worker admission/progress/idempotency, not the
    # comparatively expensive exact-reference BFS. Use the runner's supported
    # solver injection so adapter readiness is independent of a cold solver
    # cache and machine speed.
    def runner_factory(**kwargs) -> BenchmarkRunner:
        return BenchmarkRunner(reference_solver=lambda _cube: None, **kwargs)

    service = _service(
        tmp_path,
        adapter_factory=factory,
        runner_factory=runner_factory,
    )
    with TestClient(create_app(benchmark_service=service)) as client:
        try:
            request = {
                "puzzles": [{"puzzle_id": "blocked", "scramble_moves": ["R"]}],
                "execution_mode": "mock",
            }
            first = client.post("/api/benchmark", json=request, headers={"Idempotency-Key": "same"})
            assert first.status_code == 202
            run_id = first.json()["run_id"]
            assert entered.wait(1), (
                "background worker did not reach its adapter; "
                f"run status: {client.get(f'/api/benchmark/{run_id}').json()}"
            )

            status = client.get(f"/api/benchmark/{run_id}").json()
            assert status["status"] == "running"
            assert status["progress"]["total_trials"] == 4
            assert status["progress"]["completed_trials"] == 0

            retry = client.post("/api/benchmark", json=request, headers={"Idempotency-Key": "same"})
            assert retry.status_code == 202
            assert retry.json()["run_id"] == run_id

            conflict = client.post(
                "/api/benchmark",
                json={"puzzles": [{"puzzle_id": "other", "scramble_moves": ["U"]}]},
            )
            assert conflict.status_code == 409
            assert conflict.json()["detail"]["active_run_id"] == run_id

            conflicting_key = client.post(
                "/api/benchmark",
                json={"puzzles": [{"puzzle_id": "other", "scramble_moves": ["U"]}]},
                headers={"Idempotency-Key": "same"},
            )
            assert conflicting_key.status_code == 409
        finally:
            release.set()
        assert _wait_for_status(client, run_id, "completed")["progress"]["completed_trials"] == 4


def test_worker_exception_is_visible_as_failed_status_and_not_result(tmp_path: Path) -> None:
    def broken_runner_factory(**_kwargs):
        raise RuntimeError("private diagnostic")

    service = _service(tmp_path, runner_factory=broken_runner_factory)
    with TestClient(create_app(benchmark_service=service)) as client:
        response = client.post(
            "/api/benchmark",
            json={"puzzles": [{"puzzle_id": "failure", "scramble_moves": ["R"]}]},
        )
        run_id = response.json()["run_id"]
        failed = _wait_for_status(client, run_id, "failed")
        assert "private diagnostic" not in failed["error"]
        assert failed["error"] == "RuntimeError: benchmark execution failed; details omitted"
        assert client.get(f"/api/benchmark/{run_id}/results").status_code == 409


def test_storage_failure_marks_run_failed_and_returns_no_partial_results(tmp_path: Path) -> None:
    def fail_save(_run, _path):
        raise ResultsStorageError("sensitive disk detail")

    service = _service(tmp_path, save_run_fn=fail_save)
    with TestClient(create_app(benchmark_service=service)) as client:
        response = client.post(
            "/api/benchmark",
            json={"puzzles": [{"puzzle_id": "storage", "scramble_moves": []}]},
        )
        run_id = response.json()["run_id"]
        failed = _wait_for_status(client, run_id, "failed")
        assert failed["error"] == "ResultsStorageError: benchmark results could not be persisted"
        assert "sensitive" not in failed["error"]
        assert client.get(f"/api/benchmark/{run_id}/results").status_code == 409


@pytest.mark.parametrize("execution_mode", ["live", "recording", "diagnostic", "state_aware_experiment"])
def test_live_modes_require_key_before_creating_adapters_or_making_requests(
    execution_mode: str, tmp_path: Path
) -> None:
    factory_called = False

    def factory(_mode, _settings, _puzzle):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("adapter factory must not be called without a configured key")

    service = _service(
        tmp_path,
        settings_provider=lambda: Settings(openrouter_api_key=None),
        adapter_factory=factory,
    )
    if execution_mode == "state_aware_experiment":
        payload = {
            "puzzles": [
                {
                    "puzzle_id": entry.puzzle_id,
                    "seed": entry.seed,
                    "scramble_length": entry.scramble_length,
                }
                for entry in build_primary_dataset()
            ],
            "action_previews_enabled": True,
            "live_usage_confirmed": True,
        }
    else:
        payload = {
            "puzzles": [{"puzzle_id": "live", "scramble_moves": ["R"]}],
            **({"max_valid_moves": 3} if execution_mode == "diagnostic" else {}),
        }
    payload["execution_mode"] = execution_mode
    with TestClient(create_app(benchmark_service=service)) as client:
        response = client.post("/api/benchmark", json=payload)
    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]
    assert factory_called is False
