"""Offline tests for the manually invoked Claude diagnostic CLI."""

from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.cube import CubeState, LEGAL_MOVES
from app.models import CHAT_COMPLETIONS_ENDPOINT, CLAUDE_OPUS_55_MODEL_ID
from scripts.claude_diagnostic import _diagnostic_context, main

TEST_API_KEY = "offline-diagnostic-test-key"


def _response(
    content: object,
    *,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {"content": content, "role": "assistant"},
            }
        ],
        "model": CLAUDE_OPUS_55_MODEL_ID,
        **({"usage": usage} if usage is not None else {}),
    }


def _run_mocked(
    response_body: dict[str, object],
) -> tuple[int, list[str], list[httpx.Request]]:
    requests: list[httpx.Request] = []
    output: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body)

    code = main(
        ["--confirm-live"],
        settings_loader=lambda: Settings(openrouter_api_key=TEST_API_KEY),
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
        output=output.append,
    )
    return code, output, requests


def test_opt_in_is_required_before_configuration_or_client_access() -> None:
    output: list[str] = []
    settings_loaded = False
    client_created = False

    def settings_loader() -> Settings:
        nonlocal settings_loaded
        settings_loaded = True
        raise AssertionError("settings must not load before explicit opt-in")

    def client_factory() -> httpx.AsyncClient:
        nonlocal client_created
        client_created = True
        raise AssertionError("client must not be created before explicit opt-in")

    code = main(
        [],
        settings_loader=settings_loader,
        client_factory=client_factory,
        output=output.append,
    )

    assert code == 2
    assert not settings_loaded
    assert not client_created
    assert any("--confirm-live" in line for line in output)


def test_missing_api_key_refuses_before_creating_http_client() -> None:
    output: list[str] = []
    client_created = False

    def client_factory() -> httpx.AsyncClient:
        nonlocal client_created
        client_created = True
        raise AssertionError("client must not be created without a key")

    code = main(
        ["--confirm-live"],
        settings_loader=lambda: Settings(),
        client_factory=client_factory,
        output=output.append,
    )

    assert code == 2
    assert not client_created
    assert any("OPENROUTER_API_KEY is required" in line for line in output)
    assert any("no request was sent" in line for line in output)


def test_diagnostic_context_is_a_valid_known_one_move_cube_state() -> None:
    context = _diagnostic_context()

    assert context.cube_state == CubeState().apply_move("R").serialize()
    assert CubeState.from_facelets(context.cube_state) == CubeState().apply_move("R")
    assert context.recent_moves == ("R",)
    assert context.legal_moves == LEGAL_MOVES
    assert context.timeout_seconds == 30.0


def test_opted_in_success_sends_one_claude_request_and_prints_only_valid_move() -> None:
    sensitive_response_text = json.dumps({"move": "R'"})
    code, output, requests = _run_mocked(_response(sensitive_response_text))

    assert code == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == CHAT_COMPLETIONS_ENDPOINT
    assert request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
    payload = json.loads(request.content)
    assert payload["model"] == CLAUDE_OPUS_55_MODEL_ID
    assert payload["provider"] == {"require_parameters": True}
    assert payload["max_completion_tokens"] == 256
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["move"]["enum"] == list(LEGAL_MOVES)
    state_context = json.loads(payload["messages"][1]["content"])
    assert state_context["cube_state"] == _diagnostic_context().cube_state
    timeout = request.extensions["timeout"]
    assert timeout["connect"] == 5.0
    assert timeout["read"] == 25.0

    rendered = "\n".join(output)
    assert "status=success" in rendered
    assert "valid_move=R'" in rendered
    assert TEST_API_KEY not in rendered
    assert sensitive_response_text not in rendered


def test_failed_response_shows_safe_diagnostics_without_raw_output_or_retry() -> None:
    raw_content = "private provider response that was truncated"
    code, output, requests = _run_mocked(
        _response(
            raw_content,
            finish_reason="length",
            usage={"prompt_tokens": 42, "completion_tokens": 256, "total_tokens": 298},
        )
    )

    assert code == 1
    assert len(requests) == 1
    rendered = "\n".join(output)
    assert "status=failure" in rendered
    assert "category=token_limit" in rendered
    assert "finish_reason=length" in rendered
    assert "prompt_tokens=42" in rendered
    assert "completion_tokens=256" in rendered
    assert raw_content not in rendered
    assert TEST_API_KEY not in rendered


def test_invalid_move_failure_does_not_print_model_content() -> None:
    raw_content = '{"move":"private invalid output"}'
    code, output, requests = _run_mocked(_response(raw_content))

    assert code == 1
    assert len(requests) == 1
    rendered = "\n".join(output)
    assert "category=invalid_move" in rendered
    assert raw_content not in rendered
    assert "private invalid output" not in rendered


def test_http_failure_prints_only_safe_status_and_sends_no_retry() -> None:
    response_body = "private provider error body"
    requests: list[httpx.Request] = []
    output: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, text=response_body)

    code = main(
        ["--confirm-live"],
        settings_loader=lambda: Settings(openrouter_api_key=TEST_API_KEY),
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
        output=output.append,
    )

    assert code == 1
    assert len(requests) == 1
    rendered = "\n".join(output)
    assert "category=http_error" in rendered
    assert "http_status=503" in rendered
    assert response_body not in rendered
    assert TEST_API_KEY not in rendered
