"""Offline coverage for the exploratory diagnostic and solver showcase."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.benchmark import load_run
from app.benchmark.report import build_analysis_report
from app.config import Settings
from app.cube import CubeState, SOLVED_FACELETS, exact_shortest_solution, invert_moves
from app.main import create_app
from app.models import (
    ASSIGNED_MODEL_IDS,
    CLAUDE_OPUS_55_MODEL_ID,
    ScriptedClaudeFallbackAdapter,
    ScriptedMockAdapter,
)
from app.services.benchmark_service import BenchmarkService


class _OfflineLiveAdapter(ScriptedMockAdapter):
    """A scripted test fixture with live provenance and no transport."""

    execution_provenance = "live"

    def __init__(self, outcomes, *, model_id: str) -> None:
        super().__init__(outcomes, model_id=model_id)
        self.contexts = []

    async def choose_move(self, context):
        self.contexts.append(context)
        return await super().choose_move(context)


def _wait_for_status(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(f"/api/benchmark/{run_id}").json()
        if status["status"] in {"completed", "failed"}:
            return status
        time.sleep(0.01)
    raise AssertionError("diagnostic worker did not finish")


def test_one_move_r_diagnostic_has_verified_inverse_solution() -> None:
    initial = CubeState().apply_move("R")
    solved = initial.apply_move("R'")
    assert initial.serialize() != SOLVED_FACELETS
    assert solved.serialize() == SOLVED_FACELETS
    assert solved.is_solved


def test_one_move_diagnostic_preserves_real_outcomes_and_is_excluded_from_official_report(
    monkeypatch, tmp_path: Path
) -> None:
    def forbidden_http_client(*_args, **_kwargs):
        raise AssertionError("offline diagnostic test must not construct a provider HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_http_client)

    created = {}

    def factory(mode, _settings, puzzle):
        assert mode == "diagnostic"
        assert puzzle.scramble_moves == ("R",)
        created.update({
            ASSIGNED_MODEL_IDS[0]: _OfflineLiveAdapter(("R'",), model_id=ASSIGNED_MODEL_IDS[0]),
            ASSIGNED_MODEL_IDS[1]: _OfflineLiveAdapter(("U", "U", "U"), model_id=ASSIGNED_MODEL_IDS[1]),
            ASSIGNED_MODEL_IDS[2]: _OfflineLiveAdapter((RuntimeError("provider test failure"),), model_id=ASSIGNED_MODEL_IDS[2]),
            CLAUDE_OPUS_55_MODEL_ID: ScriptedClaudeFallbackAdapter(invert_moves(puzzle.scramble_moves)),
        })
        return created

    service = BenchmarkService(
        results_dir=tmp_path / "results",
        settings_provider=lambda: Settings(openrouter_api_key="offline-only-test-key"),
        adapter_factory=factory,
    )
    with TestClient(create_app(benchmark_service=service)) as client:
        response = client.post(
            "/api/benchmark",
            json={
                "execution_mode": "diagnostic",
                "puzzles": [{"puzzle_id": "exploratory-diagnostic-r", "scramble_moves": ["R"]}],
                "max_valid_moves": 3,
                "action_previews_enabled": True,
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        assert _wait_for_status(client, run_id)["status"] == "completed"
        result_response = client.get(f"/api/benchmark/{run_id}/results")

    assert result_response.status_code == 200
    data = result_response.json()
    assert data["execution_mode"] == "diagnostic"
    assert data["prompt_version"] == "rubiks-move-objective-v2-state-aware"
    assert data["action_previews_enabled"] is True
    assert data["max_valid_moves"] == 3
    cube = CubeState().apply_move("R")
    reference = exact_shortest_solution(cube)
    assert reference is not None
    assert reference.distance_htm == 1
    assert len(data["trials"]) == 4
    assert all(trial["initial_state"] == cube.serialize() for trial in data["trials"])
    assert created[ASSIGNED_MODEL_IDS[0]].contexts[0].prompt_version == data["prompt_version"]
    assert created[ASSIGNED_MODEL_IDS[0]].contexts[0].action_previews_enabled is True
    assert created[ASSIGNED_MODEL_IDS[0]].contexts[0].original_scramble == ("R",)

    by_model = {trial["model_id"]: trial for trial in data["trials"]}
    solved_live = by_model[ASSIGNED_MODEL_IDS[0]]
    assert solved_live["provenance"] == "live"
    assert solved_live["solved"] is True
    assert solved_live["final_state"] == SOLVED_FACELETS
    assert solved_live["move_history"] == ["R'"]
    assert solved_live["exact_shortest_distance"] == 1
    assert solved_live["optimality_gap"] == 0

    unsolved_live = by_model[ASSIGNED_MODEL_IDS[1]]
    assert unsolved_live["provenance"] == "live"
    assert unsolved_live["termination_status"] == "move_limit"
    assert unsolved_live["move_history"] == ["U", "U", "U"]
    assert unsolved_live["solved"] is False
    assert unsolved_live["optimality_gap"] is None

    failed_live = by_model[ASSIGNED_MODEL_IDS[2]]
    assert failed_live["provenance"] == "live"
    assert failed_live["termination_status"] == "adapter_error"
    assert failed_live["attempts"][0]["outcome"] == "adapter_error"
    assert "AdapterExecutionError" in failed_live["attempts"][0]["error_type"]
    assert failed_live["move_history"] == []

    simulated = by_model[CLAUDE_OPUS_55_MODEL_ID]
    assert simulated["provenance"] == "simulated"
    assert "SIMULATED" in simulated["provenance_label"]
    assert simulated["solved"] is True
    assert set(data["metrics"]) == set(ASSIGNED_MODEL_IDS[:3])

    persisted = load_run(tmp_path / "results" / f"{run_id}.json")
    assert persisted.execution_mode == "diagnostic"
    assert persisted.prompt_version == "rubiks-move-objective-v2-state-aware"
    assert persisted.action_previews_enabled is True
    report = build_analysis_report(tmp_path / "results")
    assert report["analysis_status"] == "pending"
    assert report["per_model_comparison"] == {}
    assert report["run_classifications"][0]["classification"] == "excluded_exploratory_diagnostic"


def test_diagnostic_request_rejects_other_puzzles_and_move_limits(tmp_path: Path) -> None:
    with TestClient(create_app(benchmark_service=BenchmarkService(results_dir=tmp_path / "results"))) as client:
        wrong_scramble = client.post(
            "/api/benchmark",
            json={"execution_mode": "diagnostic", "puzzles": [{"puzzle_id": "bad", "scramble_moves": ["U"]}], "max_valid_moves": 3},
        )
        wrong_limit = client.post(
            "/api/benchmark",
            json={"execution_mode": "diagnostic", "puzzles": [{"puzzle_id": "bad", "scramble_moves": ["R"]}], "max_valid_moves": 2},
        )
    assert wrong_scramble.status_code == 422
    assert wrong_limit.status_code == 422


def test_action_previews_are_rejected_outside_the_exploratory_diagnostic(tmp_path: Path) -> None:
    with TestClient(create_app(benchmark_service=BenchmarkService(results_dir=tmp_path / "results"))) as client:
        response = client.post(
            "/api/benchmark",
            json={
                "execution_mode": "mock",
                "puzzles": [{"puzzle_id": "no-v2-in-mock", "scramble_moves": ["R"]}],
                "action_previews_enabled": True,
            },
        )
    assert response.status_code == 422


def test_reference_solution_endpoint_replays_and_verifies_one_move_without_trial_storage(
    monkeypatch, tmp_path: Path
) -> None:
    def forbidden_http_client(*_args, **_kwargs):
        raise AssertionError("reference solver must not construct a provider HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_http_client)
    results_dir = tmp_path / "results"
    with TestClient(create_app(benchmark_service=BenchmarkService(results_dir=results_dir))) as client:
        response = client.post(
            "/api/cube/reference-solution",
            json={"puzzle_id": "reference-r", "scramble_moves": ["R"]},
        )
        openapi = client.get("/openapi.json").json()

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["solution_moves"] == ["R'"]
    assert data["move_count"] == 1
    assert data["solved"] is True
    assert data["final_state"] == SOLVED_FACELETS
    current = CubeState.from_facelets(data["initial_state"])
    for step in data["steps"]:
        current = current.apply_move(step["move"])
        assert step["state_after"] == current.serialize()
    assert current.is_solved
    assert "/api/cube/reference-solution" in openapi["paths"]
    assert not results_dir.exists()


def test_frontend_contains_named_diagnostic_and_reference_showcase_controls(tmp_path: Path) -> None:
    with TestClient(create_app(benchmark_service=BenchmarkService(results_dir=tmp_path / "results"))) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "State-aware diagnostic · R scramble" in response.text
    assert "Include one-step action previews" in response.text
    assert "Play Verified Reference Solution" in response.text
    assert "REFERENCE SOLVER — ALGORITHMIC BASELINE, NOT AN LLM RESULT" in response.text
