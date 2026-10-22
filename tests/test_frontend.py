"""Frontend serving and API-backed cube preview checks."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.benchmark import PuzzleSpec
from app.main import create_app
from app.services.benchmark_service import BenchmarkService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _client(tmp_path: Path) -> TestClient:
    service = BenchmarkService(results_dir=tmp_path / "results")
    return TestClient(create_app(benchmark_service=service))


def test_frontend_and_assets_are_served_by_fastapi(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
        cube_module = client.get("/static/cube.mjs")
        playback_module = client.get("/static/playback.mjs")
        stylesheet = client.get("/static/styles.css")
        favicon = client.get("/static/favicon.svg")

    assert page.status_code == script.status_code == cube_module.status_code == playback_module.status_code == stylesheet.status_code == favicon.status_code == 200
    assert "Cube Race" in page.text
    assert "Run mock benchmark" in page.text
    assert "Scramble length" in page.text
    assert "student-name" in page.text
    assert 'value="recording"' in page.text
    assert 'value="diagnostic"' in page.text
    assert 'value="state_aware_experiment"' in page.text
    assert "Full State-Aware Experiment — Exploratory" in page.text
    assert 'id="experiment-warning"' in page.text
    assert 'id="live-usage-confirmed"' in page.text
    assert 'id="action-previews-input"' in page.text
    assert "javascript" in script.headers["content-type"]
    assert "text/css" in stylesheet.headers["content-type"]


def test_preview_endpoint_uses_existing_seeded_puzzle_and_cube_engine(tmp_path: Path) -> None:
    puzzle_request = {"puzzle_id": "preview-42", "seed": 42, "scramble_length": 5}
    expected = PuzzleSpec.from_seed("preview-42", seed=42, length=5)
    with _client(tmp_path) as client:
        response = client.post("/api/cube/preview", json=puzzle_request)

    assert response.status_code == 200
    preview = response.json()
    assert preview["puzzle_id"] == expected.puzzle_id
    assert preview["seed"] == expected.seed
    assert preview["scramble_moves"] == list(expected.scramble_moves)
    assert preview["initial_state"] == expected.initial_state.serialize()
    assert len(preview["initial_state"]) == 54


def test_public_student_config_never_returns_the_openrouter_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-value")
    monkeypatch.setenv("STUDENT_NAME", "Example Student")
    monkeypatch.setenv("STUDENT_ROLL_NUMBER", "ID-000")
    with _client(tmp_path) as client:
        response = client.get("/api/ui-config")

    assert response.status_code == 200
    assert response.json() == {
        "student_name": "Example Student",
        "student_roll_number": "ID-000",
    }
    assert "private-test-value" not in response.text
    assert "OPENROUTER_API_KEY" not in response.text


def test_browser_assets_use_only_same_origin_backend_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    script = (PROJECT_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    page = (PROJECT_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    with _client(tmp_path) as client:
        assert client.get("/api/health").status_code == 200

    assert '"/api/benchmark"' in script
    assert '"/api/cube/preview"' in script
    assert '"/api/ui-config"' in script
    assert 'execution_mode: elements.executionMode.value' in script
    assert 'action_previews_enabled: fullStateAware || (diagnostic && elements.actionPreviews.checked)' in script
    assert '"/api/benchmark/primary-dataset"' in script
    assert "trialPlaybackSteps" in script
    assert "SIMULATED — NOT A LIVE CLAUDE RESPONSE" in script
    assert "openrouter.ai" not in script.lower()
    assert "OPENROUTER_API_KEY" not in script
    assert "OPENROUTER_API_KEY" not in page


def test_primary_dataset_endpoint_returns_the_pinned_scrambles_and_references(tmp_path: Path) -> None:
    from app.benchmark.primary_dataset import PRIMARY_DATASET_ID, build_primary_dataset

    with _client(tmp_path) as client:
        response = client.get("/api/benchmark/primary-dataset")

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == PRIMARY_DATASET_ID
    assert data["puzzles"] == [entry.to_dict() for entry in build_primary_dataset()]
