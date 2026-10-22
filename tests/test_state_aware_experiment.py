"""Offline coverage for the five-puzzle state-aware exploratory experiment."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.benchmark import load_run
from app.benchmark.primary_dataset import PRIMARY_DATASET_ID, build_primary_dataset
from app.benchmark.report import build_analysis_report
from app.config import Settings
from app.cube import CubeState, SOLVED_FACELETS, invert_moves
from app.main import create_app
from app.models import ASSIGNED_MODEL_IDS, CLAUDE_OPUS_55_MODEL_ID, ScriptedClaudeFallbackAdapter, ScriptedMockAdapter
from app.models.openrouter import _move_definitions, _turn_state
from app.models.prompts import RUBIKS_MOVE_OBJECTIVE_V1, RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
from app.services.benchmark_service import BenchmarkService
import app.services.benchmark_service as benchmark_service_module


class _OfflineLiveAdapter(ScriptedMockAdapter):
    """A no-network scripted adapter marked live only to test result handling."""

    execution_provenance = "live"

    def __init__(self, outcomes, *, model_id: str) -> None:
        super().__init__(outcomes, model_id=model_id)
        self.contexts = []

    async def choose_move(self, context):
        self.contexts.append(context)
        return await super().choose_move(context)


def _wait(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/benchmark/{run_id}")
        assert response.status_code == 200, response.text
        status = response.json()
        if status["status"] in {"completed", "failed"}:
            return status
        time.sleep(0.01)
    raise AssertionError("offline state-aware run did not finish")


def _pinned_request() -> list[dict[str, object]]:
    return [
        {
            "puzzle_id": entry.puzzle_id,
            "seed": entry.seed,
            "scramble_length": entry.scramble_length,
        }
        for entry in build_primary_dataset()
    ]


def _full_request(**overrides) -> dict[str, object]:
    request: dict[str, object] = {
        "execution_mode": "state_aware_experiment",
        "puzzles": _pinned_request(),
        "max_valid_moves": 30,
        "turn_timeout_seconds": 30,
        "action_previews_enabled": True,
        "live_usage_confirmed": True,
    }
    request.update(overrides)
    return request


def _service(tmp_path: Path, factory) -> BenchmarkService:
    return BenchmarkService(
        results_dir=tmp_path / "results",
        settings_provider=lambda: Settings(openrouter_api_key="offline-only-test-key"),
        adapter_factory=factory,
    )


def test_full_state_aware_run_uses_all_pinned_puzzles_and_rebuilds_contexts(
    monkeypatch, tmp_path: Path
) -> None:
    def forbidden_http_client(*_args, **_kwargs):
        raise AssertionError("offline exploratory test must not construct a provider HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_http_client)
    dataset = build_primary_dataset()
    adapters_by_puzzle = {}

    def factory(mode, _settings, puzzle):
        assert mode == "state_aware_experiment"
        matching = next(entry for entry in dataset if entry.puzzle_id == puzzle.puzzle_id)
        assert puzzle.seed == matching.seed
        assert puzzle.scramble_moves == matching.scramble_moves
        adapters = {
            ASSIGNED_MODEL_IDS[0]: _OfflineLiveAdapter(
                invert_moves(puzzle.scramble_moves), model_id=ASSIGNED_MODEL_IDS[0]
            ),
            ASSIGNED_MODEL_IDS[1]: _OfflineLiveAdapter(
                (RuntimeError("synthetic offline adapter failure"),), model_id=ASSIGNED_MODEL_IDS[1]
            ),
            ASSIGNED_MODEL_IDS[2]: _OfflineLiveAdapter(
                ("NOT_A_MOVE",), model_id=ASSIGNED_MODEL_IDS[2]
            ),
            CLAUDE_OPUS_55_MODEL_ID: ScriptedClaudeFallbackAdapter(invert_moves(puzzle.scramble_moves)),
        }
        adapters_by_puzzle[puzzle.puzzle_id] = adapters
        return adapters

    service = _service(tmp_path, factory)
    with TestClient(create_app(benchmark_service=service)) as client:
        dataset_response = client.get("/api/benchmark/primary-dataset")
        submitted = client.post("/api/benchmark", json=_full_request())
        assert submitted.status_code == 202, submitted.text
        run_id = submitted.json()["run_id"]
        assert _wait(client, run_id)["status"] == "completed"
        results_response = client.get(f"/api/benchmark/{run_id}/results")

    assert dataset_response.status_code == 200
    assert dataset_response.json()["dataset_id"] == PRIMARY_DATASET_ID
    assert dataset_response.json()["puzzles"] == [entry.to_dict() for entry in dataset]
    assert results_response.status_code == 200, results_response.text
    result = results_response.json()
    assert result["execution_mode"] == "state_aware_experiment"
    assert result["prompt_version"] == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
    assert result["action_previews_enabled"] is True
    assert result["max_valid_moves"] == 30
    assert result["turn_timeout_seconds"] == 30
    assert len(result["trials"]) == 20
    assert set(result["metrics"]) == set(ASSIGNED_MODEL_IDS[:3])

    by_puzzle_model = {(trial["puzzle_id"], trial["model_id"]): trial for trial in result["trials"]}
    for puzzle in dataset:
        initial = CubeState.from_facelets(puzzle.initial_state)
        expected_moves = invert_moves(puzzle.scramble_moves)
        for model_id in ASSIGNED_MODEL_IDS:
            same_puzzle = by_puzzle_model[(puzzle.puzzle_id, model_id)]
            assert same_puzzle["initial_state"] == puzzle.initial_state
            assert same_puzzle["seed"] == puzzle.seed
            assert same_puzzle["scramble_moves"] == list(puzzle.scramble_moves)
        solving = by_puzzle_model[(puzzle.puzzle_id, ASSIGNED_MODEL_IDS[0])]
        assert solving["seed"] == puzzle.seed
        assert solving["scramble_moves"] == list(puzzle.scramble_moves)
        assert solving["initial_state"] == puzzle.initial_state
        assert solving["exact_shortest_distance"] == puzzle.exact_shortest_distance
        assert solving["solved"] is True
        assert solving["final_state"] == SOLVED_FACELETS
        assert solving["move_history"] == list(expected_moves)
        assert solving["optimality_gap"] == len(expected_moves) - puzzle.exact_shortest_distance

        failed = by_puzzle_model[(puzzle.puzzle_id, ASSIGNED_MODEL_IDS[1])]
        assert failed["provenance"] == "live"
        assert failed["termination_status"] == "adapter_error"
        invalid = by_puzzle_model[(puzzle.puzzle_id, ASSIGNED_MODEL_IDS[2])]
        assert invalid["provenance"] == "live"
        assert invalid["termination_status"] == "invalid_move"
        assert invalid["valid_move_count"] == 0

        claude = by_puzzle_model[(puzzle.puzzle_id, CLAUDE_OPUS_55_MODEL_ID)]
        assert claude["provenance"] == "simulated"
        assert claude["provenance_label"] == "SIMULATED — NOT A LIVE CLAUDE RESPONSE"
        assert claude["solved"] is True
        assert claude["final_state"] == SOLVED_FACELETS
        assert claude["average_move_latency_seconds"] is None
        assert claude["total_runtime_seconds"] == 0

        for model_id in ASSIGNED_MODEL_IDS[:3]:
            adapter = adapters_by_puzzle[puzzle.puzzle_id][model_id]
            assert adapter.contexts
            for context in adapter.contexts:
                assert context.prompt_version == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
                assert context.action_previews_enabled is True
                assert context.original_scramble == puzzle.scramble_moves
                assert context.legal_moves
                replayed = initial.apply_moves(context.recent_moves)
                assert context.cube_state == replayed.serialize()
                state = _turn_state(context, _move_definitions(context))
                expected_previews = [
                    {"move": move, "resulting_cube_state": successor.serialize()}
                    for move, successor in replayed.successors()
                ]
                assert state["action_previews"] == expected_previews

        solve_contexts = adapters_by_puzzle[puzzle.puzzle_id][ASSIGNED_MODEL_IDS[0]].contexts
        assert [context.recent_moves for context in solve_contexts] == [
            tuple(expected_moves[:index]) for index in range(len(expected_moves))
        ]

    persisted = load_run(tmp_path / "results" / f"{run_id}.json")
    assert persisted.execution_mode == "state_aware_experiment"
    assert persisted.prompt_version == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
    assert persisted.action_previews_enabled is True
    report = build_analysis_report(tmp_path / "results")
    assert report["analysis_status"] == "pending"
    assert report["per_model_comparison"] == {}
    assert report["run_classifications"][0]["classification"] == "excluded_exploratory_state_aware_experiment"


def test_full_experiment_requires_pinned_data_settings_previews_and_acknowledgement(tmp_path: Path) -> None:
    service = _service(tmp_path, lambda *_args: {})
    with TestClient(create_app(benchmark_service=service)) as client:
        missing_ack = client.post("/api/benchmark", json=_full_request(live_usage_confirmed=False))
        missing_previews = client.post("/api/benchmark", json=_full_request(action_previews_enabled=False))
        changed_limit = client.post("/api/benchmark", json=_full_request(max_valid_moves=3))
        altered_puzzle = _pinned_request()
        altered_puzzle[0] = {"puzzle_id": "primary-01", "scramble_moves": ["U"]}
        changed_dataset = client.post("/api/benchmark", json=_full_request(puzzles=altered_puzzle))

    assert all(response.status_code == 422 for response in (missing_ack, missing_previews, changed_limit, changed_dataset))


def test_default_full_experiment_factory_keeps_three_live_adapters_and_scripts_only_claude(monkeypatch) -> None:
    original = {
        model_id: _OfflineLiveAdapter(("U",), model_id=model_id)
        for model_id in ASSIGNED_MODEL_IDS
    }

    def forbidden_http_client(*_args, **_kwargs):
        raise AssertionError("the state-aware factory must not construct a Claude HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_http_client)
    monkeypatch.setattr(benchmark_service_module, "build_openrouter_adapters", lambda **_kwargs: original)
    puzzle = build_primary_dataset()[0].puzzle_spec
    adapters = benchmark_service_module._default_adapter_factory(
        "state_aware_experiment",
        Settings(openrouter_api_key="offline-only-test-key"),
        puzzle,
    )

    for model_id in ASSIGNED_MODEL_IDS[:3]:
        assert adapters[model_id] is original[model_id]
    claude = adapters[CLAUDE_OPUS_55_MODEL_ID]
    assert isinstance(claude, ScriptedClaudeFallbackAdapter)
    assert claude.execution_provenance == "simulated"
    assert tuple(adapters) == ASSIGNED_MODEL_IDS


def test_existing_live_mode_stays_on_frozen_v1_without_action_previews(tmp_path: Path) -> None:
    def factory(mode, _settings, puzzle):
        assert mode == "live"
        return {
            model_id: _OfflineLiveAdapter(invert_moves(puzzle.scramble_moves), model_id=model_id)
            for model_id in ASSIGNED_MODEL_IDS
        }

    service = _service(tmp_path, factory)
    with TestClient(create_app(benchmark_service=service)) as client:
        submitted = client.post(
            "/api/benchmark",
            json={"execution_mode": "live", "puzzles": [{"puzzle_id": "v1-check", "scramble_moves": ["R"]}]},
        )
        assert submitted.status_code == 202, submitted.text
        run_id = submitted.json()["run_id"]
        assert _wait(client, run_id)["status"] == "completed"
        result = client.get(f"/api/benchmark/{run_id}/results").json()

    assert result["execution_mode"] == "live"
    assert result["prompt_version"] == RUBIKS_MOVE_OBJECTIVE_V1
    assert result["action_previews_enabled"] is False
    assert all(trial["solved"] for trial in result["trials"])
