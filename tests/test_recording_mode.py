"""Offline checks for the explicit Claude-only scripted recording mode."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx
import pytest

import app.services.benchmark_service as benchmark_service
from app.benchmark import BenchmarkRun, BenchmarkRunner, PuzzleSpec, aggregate_metrics, load_run, save_run
from app.benchmark.report import build_analysis_report
from app.config import Settings
from app.cube import CubeState, invert_moves
from app.models import (
    ASSIGNED_MODEL_IDS,
    CLAUDE_OPUS_55_MODEL_ID,
    ClaudeOpus55Adapter,
    ScriptedClaudeFallbackAdapter,
    ScriptedMockAdapter,
    SIMULATED_CLAUDE_LABEL,
    TurnContext,
    build_openrouter_adapters,
)


def _run(coro):
    return asyncio.run(coro)


def test_recording_factory_keeps_three_live_adapters_and_replaces_only_claude(monkeypatch) -> None:
    settings = Settings(openrouter_api_key="offline-test-key")
    puzzle = PuzzleSpec("factory-check", ("R", "U"), seed=17)
    original = build_openrouter_adapters(settings=settings)
    monkeypatch.setattr(benchmark_service, "build_openrouter_adapters", lambda **_kwargs: original)

    live = benchmark_service._default_adapter_factory("live", settings, puzzle)
    assert all(live[model_id] is original[model_id] for model_id in ASSIGNED_MODEL_IDS)
    assert isinstance(live[CLAUDE_OPUS_55_MODEL_ID], ClaudeOpus55Adapter)

    def forbidden_client(*_args, **_kwargs):
        raise AssertionError("recording Claude must not construct an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    recording = benchmark_service._default_adapter_factory("recording", settings, puzzle)
    for model_id in ASSIGNED_MODEL_IDS[:3]:
        assert recording[model_id] is original[model_id]
    claude = recording[CLAUDE_OPUS_55_MODEL_ID]
    assert isinstance(claude, ScriptedClaudeFallbackAdapter)
    assert claude.model_id == CLAUDE_OPUS_55_MODEL_ID
    assert claude.execution_provenance == "simulated"
    assert claude.provenance_label == SIMULATED_CLAUDE_LABEL

    proposal = _run(
        claude.choose_move(
            TurnContext(cube_state=puzzle.initial_state.serialize(), timeout_seconds=2)
        )
    )
    assert proposal.move == "U'"
    assert proposal.model_id == CLAUDE_OPUS_55_MODEL_ID


def test_diagnostic_factory_preserves_three_live_adapters_and_scripts_only_claude(monkeypatch) -> None:
    settings = Settings(openrouter_api_key="offline-test-key")
    puzzle = PuzzleSpec("diagnostic-factory-check", ("R",))
    original = build_openrouter_adapters(settings=settings)
    monkeypatch.setattr(benchmark_service, "build_openrouter_adapters", lambda **_kwargs: original)

    def forbidden_client(*_args, **_kwargs):
        raise AssertionError("diagnostic mode must not construct a Claude HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    adapters = benchmark_service._default_adapter_factory("diagnostic", settings, puzzle)

    for model_id in ASSIGNED_MODEL_IDS[:3]:
        assert adapters[model_id] is original[model_id]
    claude = adapters[CLAUDE_OPUS_55_MODEL_ID]
    assert isinstance(claude, ScriptedClaudeFallbackAdapter)
    assert claude.execution_provenance == "simulated"
    assert claude.provenance_label == SIMULATED_CLAUDE_LABEL
    proposal = _run(
        claude.choose_move(
            TurnContext(cube_state=puzzle.initial_state.serialize(), timeout_seconds=2)
        )
    )
    assert proposal.move == "R'"


class _OfflineLiveAdapter(ScriptedMockAdapter):
    """Network-free test fixture carrying the live provenance expected by recording mode."""

    execution_provenance = "live"


def test_scripted_claude_moves_replay_to_cube_engine_states_and_stay_out_of_live_metrics(tmp_path) -> None:
    puzzle = PuzzleSpec("recording-puzzle", ("R", "U"), seed=902)
    adapters = [
        *[
            _OfflineLiveAdapter(invert_moves(puzzle.scramble_moves), model_id=model_id)
            for model_id in ASSIGNED_MODEL_IDS[:3]
        ],
        ScriptedClaudeFallbackAdapter(invert_moves(puzzle.scramble_moves)),
    ]
    generated = _run(BenchmarkRunner().run_suite((puzzle,), adapters, run_id="recording-run"))
    run = BenchmarkRun(
        run_id=generated.run_id,
        max_valid_moves=generated.max_valid_moves,
        turn_timeout_seconds=generated.turn_timeout_seconds,
        trials=generated.trials,
        execution_mode="recording",
    )

    claude = next(trial for trial in run.trials if trial.model_id == CLAUDE_OPUS_55_MODEL_ID)
    assert claude.solved
    assert claude.move_history == ("U'", "R'")
    assert claude.provenance == "simulated"
    assert claude.provenance_label == SIMULATED_CLAUDE_LABEL
    assert claude.total_runtime_seconds == 0
    assert claude.average_move_latency_seconds is None
    assert claude.average_request_latency_seconds is None
    assert claude.optimality_gap is None
    assert all(attempt.latency_seconds == attempt.elapsed_trial_seconds == 0 for attempt in claude.attempts)
    replay = CubeState.from_facelets(claude.initial_state)
    for attempt in claude.attempts:
        assert attempt.outcome == "valid"
        assert attempt.cube_state_before == replay.serialize()
        replay = replay.apply_move(attempt.move)
    assert replay.serialize() == claude.final_state
    assert replay.is_solved

    metrics = aggregate_metrics(run)
    assert set(metrics) == set(ASSIGNED_MODEL_IDS[:3])
    assert all(metric.trial_count == 1 for metric in metrics.values())
    with pytest.raises(ValueError, match="live runs may contain only live-provenance trials"):
        replace(run, execution_mode="live")

    result_path = save_run(run, tmp_path / "recording.json")
    persisted = load_run(result_path)
    persisted_claude = next(trial for trial in persisted.trials if trial.model_id == CLAUDE_OPUS_55_MODEL_ID)
    assert persisted.execution_mode == "recording"
    assert persisted_claude.provenance_label == "SIMULATED — NOT A LIVE CLAUDE RESPONSE"
    report = build_analysis_report(tmp_path)
    assert report["analysis_status"] == "pending"
    assert report["per_model_comparison"] == {}
    assert report["run_classifications"][0]["classification"] == "excluded_recording_demo"


def test_live_claude_error_remains_a_genuine_failed_trial_without_scripted_substitution() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url == httpx.URL("https://openrouter.ai/api/v1/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
                "usage": {"prompt_tokens": 832, "completion_tokens": 64, "total_tokens": 896},
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = ClaudeOpus55Adapter(
                settings=Settings(openrouter_api_key="offline-test-key"),
                http_client=client,
            )
            return await BenchmarkRunner().run_trial(
                PuzzleSpec("live-claude-error", ("R",)), adapter, run_id="live-error-run"
            )

    trial = _run(exercise())
    assert calls == 1
    assert trial.provenance == "live"
    assert trial.provenance_label is None
    assert trial.termination_status == "adapter_error"
    assert trial.move_history == ()
    assert trial.attempts[0].error_type == "AdapterResponseError"
    assert "finish_reason=length" in trial.attempts[0].error_message
    assert trial.attempts[0].token_usage == {
        "prompt_tokens": 832,
        "completion_tokens": 64,
        "total_tokens": 896,
    }


def test_recording_result_rejects_simulated_claude_without_required_provenance() -> None:
    puzzle = PuzzleSpec("guarded", ("R",))
    broken = ScriptedClaudeFallbackAdapter(("R'",))
    trial = _run(BenchmarkRunner().run_trial(puzzle, broken, run_id="guarded-run"))

    with pytest.raises(ValueError, match="required provenance label"):
        replace(trial, provenance_label="Claude")
