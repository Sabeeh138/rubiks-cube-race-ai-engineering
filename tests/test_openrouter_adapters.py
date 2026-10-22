"""Offline OpenRouter adapter tests using HTTPX's in-memory transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.config import ConfigurationError, Settings
from app.benchmark.runner import BenchmarkRunner, PuzzleSpec
from app.cube import CubeState
from app.cube import FACE_ORDER, LEGAL_MOVES, SOLVED_FACELETS
from app.models import (
    ASSIGNED_MODEL_IDS,
    CHAT_COMPLETIONS_ENDPOINT,
    CLAUDE_OPUS_55_MODEL_ID,
    CLOUDFLARE_CLEF_MODEL_ID,
    DECISIONS_ENDPOINT,
    GPT_6_LUNA_DECISIONS_MODEL_ID,
    JEV_113_MODEL_ID,
    AdapterResponseError,
    AdapterTimeoutError,
    ClaudeOpus55Adapter,
    CloudflareClefAdapter,
    GPT6LunaDecisionsAdapter,
    InvalidMoveError,
    Jev113Adapter,
    OpenRouterHTTPError,
    OpenRouterRateLimitError,
    TurnContext,
    RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
    PROMPT_V2_INSTRUCTIONS,
    build_openrouter_adapters,
)
from app.models.openrouter import _move_definitions, _turn_state

TEST_SETTINGS = Settings(openrouter_api_key="offline-test-token")
EXPECTED_MOVE_INSTRUCTIONS = (
    "Given the current cube state and recent moves, choose exactly one move "
    "from the listed legal moves that best advances the cube toward the fully "
    "solved state. Return only that move in the required format."
)
TEST_CONTEXT = TurnContext(
    cube_state=CubeState().apply_moves("R U").serialize(),
    recent_moves=("R", "U"),
    legal_moves=("U", "R2", "F'"),
)
DECISIONS_ADAPTERS = (
    (Jev113Adapter, JEV_113_MODEL_ID),
    (GPT6LunaDecisionsAdapter, GPT_6_LUNA_DECISIONS_MODEL_ID),
    (CloudflareClefAdapter, CLOUDFLARE_CLEF_MODEL_ID),
)
EXPECTED_SERIALIZED_TURN = {
    "cube_state": TEST_CONTEXT.cube_state,
    "recent_moves": ["R", "U"],
    "legal_moves": ["U", "R2", "F'"],
    "legal_move_definitions": {
        "U": "clockwise quarter-turn of the up face.",
        "R2": "half-turn of the right face.",
        "F'": "counterclockwise quarter-turn of the front face.",
    },
}


def _decisions_response(move: str = "R2") -> dict[str, object]:
    return {
        "answers": {
            "move": {
                "choice": move,
                "confidence": 0.82,
                "probabilities": {"U": 0.08, "R2": 0.82, "F'": 0.10},
                "type": "choice",
            }
        },
        "id": "offline-generation-id",
        "model": "assigned-model-snapshot",
        "provider": "mock-provider",
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


def _chat_response(
    content: object,
    *,
    finish_reason: str = "stop",
    usage: object = None,
    message_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"content": content, "role": "assistant"}
    if message_fields:
        message.update(message_fields)
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": message,
            }
        ],
        "id": "offline-chat-id",
        "model": "assigned-model-snapshot",
        "object": "chat.completion",
        **({"usage": usage} if usage is not None else {}),
    }


def _run_adapter(
    adapter_type: type,
    handler: Callable[[httpx.Request], httpx.Response] | Any,
    *,
    context: TurnContext = TEST_CONTEXT,
    settings: Settings | None = TEST_SETTINGS,
    request_timeout: float | httpx.Timeout | None = None,
) -> object:
    async def invoke() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = adapter_type(
                settings=settings,
                http_client=client,
                request_timeout=request_timeout,
            )
            return await adapter.choose_move(context)

    return asyncio.run(invoke())


def test_assigned_registry_contains_only_the_exact_model_ids() -> None:
    adapters = build_openrouter_adapters(settings=TEST_SETTINGS)

    assert tuple(adapters) == ASSIGNED_MODEL_IDS
    assert tuple(adapter.model_id for adapter in adapters.values()) == ASSIGNED_MODEL_IDS


def test_v2_diagnostic_payloads_share_state_history_and_engine_action_previews() -> None:
    cube = CubeState().apply_moves(("R", "U"))
    context = TurnContext(
        cube_state=cube.serialize(),
        recent_moves=("R", "U"),
        original_scramble=("R", "U"),
        legal_moves=LEGAL_MOVES,
        prompt_version=RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
        action_previews_enabled=True,
    )
    captured: list[tuple[type, dict[str, object]]] = []
    adapter_cases = (
        (Jev113Adapter, JEV_113_MODEL_ID),
        (GPT6LunaDecisionsAdapter, GPT_6_LUNA_DECISIONS_MODEL_ID),
        (CloudflareClefAdapter, CLOUDFLARE_CLEF_MODEL_ID),
        (ClaudeOpus55Adapter, CLAUDE_OPUS_55_MODEL_ID),
    )
    expected_model_ids = dict(adapter_cases)

    for adapter_type, model_id in adapter_cases:
        def handler(request: httpx.Request, *, adapter_type=adapter_type) -> httpx.Response:
            payload = json.loads(request.content)
            captured.append((adapter_type, payload))
            if adapter_type is ClaudeOpus55Adapter:
                return httpx.Response(200, json=_chat_response(json.dumps({"move": "R'"})))
            return httpx.Response(200, json=_decisions_response("R'"))

        proposal = _run_adapter(adapter_type, handler, context=context)
        assert proposal.move == "R'"  # type: ignore[attr-defined]
        assert proposal.model_id == model_id  # type: ignore[attr-defined]

    decoded_states: list[dict[str, object]] = []
    instructions: list[str] = []
    for adapter_type, payload in captured:
        assert payload["model"] == expected_model_ids[adapter_type]
        if adapter_type is ClaudeOpus55Adapter:
            messages = payload["messages"]
            assert isinstance(messages, list)
            instructions.append(messages[0]["content"])
            state = json.loads(messages[1]["content"])
        else:
            question = payload["questions"]["move"]
            instructions.append(question["instructions"])
            state = payload["state"]
        decoded_states.append(state)

    assert instructions == [PROMPT_V2_INSTRUCTIONS] * 4
    assert all(state["prompt_version"] == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE for state in decoded_states)
    assert all(state["cube_state"] == cube.serialize() for state in decoded_states)
    assert all(state["solved_target"] == SOLVED_FACELETS for state in decoded_states)
    assert all(state["original_scramble"] == ["R", "U"] for state in decoded_states)
    assert all(state["full_move_history"] == ["R", "U"] for state in decoded_states)
    assert all(state["legal_moves"] == list(LEGAL_MOVES) for state in decoded_states)
    assert all(set(state["legal_move_definitions"]) == set(LEGAL_MOVES) for state in decoded_states)

    canonical_faces = cube.serialize()
    first = decoded_states[0]
    expected_grids = {
        face: [
            list(canonical_faces[index * 9 + row * 3 : index * 9 + row * 3 + 3])
            for row in range(3)
        ]
        for index, face in enumerate(FACE_ORDER)
    }
    assert first["face_grids"] == expected_grids
    assert all(state["face_grids"] == expected_grids for state in decoded_states)

    expected_previews = [
        {"move": move, "resulting_cube_state": next_state.serialize()}
        for move, next_state in cube.successors()
    ]
    assert all(state["action_previews"] == expected_previews for state in decoded_states)
    for state in decoded_states:
        previews = state["action_previews"]
        assert all(set(preview) == {"move", "resulting_cube_state"} for preview in previews)
        assert not {"is_solved", "recommended_move", "solution_distance", "answer"}.intersection(state)
        assert all(len(preview["resulting_cube_state"]) == 54 for preview in previews)


def test_v2_without_optional_action_previews_does_not_add_preview_data() -> None:
    context = TurnContext(
        cube_state=CubeState().apply_move("R").serialize(),
        original_scramble=("R",),
        prompt_version=RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE,
    )
    state = _turn_state(context, _move_definitions(context))
    assert state["prompt_version"] == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE
    assert "action_previews" not in state


@pytest.mark.parametrize(("adapter_type", "model_id"), DECISIONS_ADAPTERS)
def test_decisions_adapters_use_documented_endpoint_payload_and_choice_parser(
    adapter_type: type,
    model_id: str,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_decisions_response())

    proposal = _run_adapter(adapter_type, handler)

    assert proposal.move == "R2"  # type: ignore[attr-defined]
    assert proposal.model_id == model_id  # type: ignore[attr-defined]
    assert proposal.token_usage == {"input_tokens": 10, "output_tokens": 0}  # type: ignore[attr-defined]
    assert proposal.raw_response["answers"]["move"]["choice"] == "R2"  # type: ignore[attr-defined,index]
    request = seen[0]
    payload = json.loads(request.content)
    assert str(request.url) == DECISIONS_ENDPOINT
    assert request.headers["Authorization"] == "Bearer offline-test-token"
    assert payload["model"] == model_id
    assert payload["state"] == EXPECTED_SERIALIZED_TURN
    assert payload["questions"]["move"] == {
        "type": "choice",
        "instructions": EXPECTED_MOVE_INSTRUCTIONS,
        "criteria": payload["state"]["legal_move_definitions"],
    }
    assert "offline-test-token" not in json.dumps(payload)
    assert proposal.elapsed_seconds >= 0  # type: ignore[attr-defined]


def test_chat_completion_uses_its_own_endpoint_messages_and_json_schema() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=_chat_response(
                '{"move":"F\u0027"}',
                usage={"prompt_tokens": 18, "completion_tokens": 5, "total_tokens": 23},
            ),
        )

    proposal = _run_adapter(ClaudeOpus55Adapter, handler)

    assert proposal.move == "F'"  # type: ignore[attr-defined]
    assert proposal.model_id == "anthropic/claude-opus-5.5"  # type: ignore[attr-defined]
    assert proposal.token_usage == {"prompt_tokens": 18, "completion_tokens": 5, "total_tokens": 23}  # type: ignore[attr-defined]
    request = seen[0]
    payload = json.loads(request.content)
    assert str(request.url) == CHAT_COMPLETIONS_ENDPOINT
    assert payload["model"] == "anthropic/claude-opus-5.5"
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert payload["messages"][0]["content"] == EXPECTED_MOVE_INSTRUCTIONS
    serialized_state = json.loads(payload["messages"][1]["content"])
    assert serialized_state == EXPECTED_SERIALIZED_TURN
    assert payload["provider"] == {"require_parameters": True}
    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["move"]["enum"] == ["U", "R2", "F'"]
    assert request.headers["Authorization"] == "Bearer offline-test-token"


@pytest.mark.parametrize(("adapter_type", "_model_id"), DECISIONS_ADAPTERS)
def test_decisions_illegal_choice_is_rejected_by_shared_validator(
    adapter_type: type,
    _model_id: str,
) -> None:
    with pytest.raises(InvalidMoveError) as caught:
        _run_adapter(
            adapter_type,
            lambda _request: httpx.Response(200, json=_decisions_response("M")),
        )

    assert caught.value.proposed_move == "M"


def test_chat_illegal_move_is_rejected_by_shared_validator() -> None:
    with pytest.raises(InvalidMoveError) as caught:
        _run_adapter(
            ClaudeOpus55Adapter,
            lambda _request: httpx.Response(200, json=_chat_response('{"move":"M"}')),
        )

    assert caught.value.proposed_move == "M"


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"answers": []},
        {"answers": {"move": None}},
        {"answers": {"move": {"choice": ""}}},
        {"answers": {"move": {"type": "noul", "choice": "U"}}},
    ],
)
def test_decisions_malformed_or_empty_answers_raise_response_error(body: object) -> None:
    with pytest.raises(AdapterResponseError):
        _run_adapter(
            Jev113Adapter,
            lambda _request: httpx.Response(200, json=body),
        )


@pytest.mark.parametrize(
    "chat_content",
    [
        "",
        "R2",
        "{not-json}",
        "[]",
        '{"move":"R2","extra":true}',
        '{"move":""}',
    ],
)
def test_chat_malformed_or_empty_answer_raises_response_error(chat_content: str) -> None:
    with pytest.raises(AdapterResponseError):
        _run_adapter(
            ClaudeOpus55Adapter,
            lambda _request: httpx.Response(200, json=_chat_response(chat_content)),
        )


def test_chat_missing_choices_raises_response_error() -> None:
    with pytest.raises(AdapterResponseError, match="exactly one choice"):
        _run_adapter(
            ClaudeOpus55Adapter,
            lambda _request: httpx.Response(200, json={"choices": []}),
        )


def test_missing_api_key_fails_before_transport_is_used() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY is required"):
        _run_adapter(Jev113Adapter, handler, settings=Settings())
    assert called is False


@pytest.mark.parametrize("status_code", [400, 401, 402, 403, 500, 503])
def test_http_errors_surface_without_parsing_or_leaking_error_bodies(status_code: int) -> None:
    with pytest.raises(OpenRouterHTTPError) as caught:
        _run_adapter(
            Jev113Adapter,
            lambda _request: httpx.Response(
                status_code,
                json={"error": {"message": "provider detail"}},
            ),
        )

    assert caught.value.status_code == status_code
    assert "provider detail" not in str(caught.value)
    assert "offline-test-token" not in str(caught.value)


def test_rate_limits_have_specific_error_and_retry_after_metadata() -> None:
    with pytest.raises(OpenRouterRateLimitError) as caught:
        _run_adapter(
            Jev113Adapter,
            lambda _request: httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": {"message": "slow down"}},
            ),
        )

    assert caught.value.status_code == 429
    assert caught.value.retry_after == "2"


def test_success_response_with_invalid_json_is_rejected() -> None:
    with pytest.raises(AdapterResponseError, match="not valid JSON") as caught:
        _run_adapter(
            Jev113Adapter,
            lambda _request: httpx.Response(200, content=b"not-json"),
        )
    assert caught.value.category == "malformed_response_json"


def test_request_uses_explicit_http_connect_and_read_timeouts() -> None:
    captured: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.extensions["timeout"])
        return httpx.Response(200, json=_decisions_response())

    _run_adapter(Jev113Adapter, handler)

    assert captured[0]["connect"] == 5.0
    assert captured[0]["read"] == 25.0
    assert captured[0]["write"] == 10.0
    assert captured[0]["pool"] == 5.0


def test_per_turn_timeout_cancels_slow_mock_transport() -> None:
    async def slow_handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_decisions_response())

    context = TurnContext(
        cube_state=TEST_CONTEXT.cube_state,
        recent_moves=TEST_CONTEXT.recent_moves,
        legal_moves=TEST_CONTEXT.legal_moves,
        timeout_seconds=0.001,
    )
    with pytest.raises(AdapterTimeoutError):
        _run_adapter(Jev113Adapter, slow_handler, context=context)


def test_chat_empty_content_error_keeps_safe_finish_reason_and_usage() -> None:
    body = _chat_response(
        " \n",
        usage={"prompt_tokens": 31, "completion_tokens": 0, "total_tokens": 31},
    )
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(ClaudeOpus55Adapter, lambda _request: httpx.Response(200, json=body))

    error = caught.value
    assert error.category == "empty_content"
    assert error.finish_reason == "stop"
    assert error.usage == {"prompt_tokens": 31, "completion_tokens": 0, "total_tokens": 31}
    assert "category=empty_content" in str(error)
    assert "finish_reason=stop" in str(error)
    assert "completion_tokens=0" in str(error)


def test_chat_length_finish_reason_identifies_token_limit_without_raw_content() -> None:
    raw_content = '{"move":'
    body = _chat_response(
        raw_content,
        finish_reason="length",
        usage={"prompt_tokens": 40, "completion_tokens": 256, "total_tokens": 296},
    )
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(ClaudeOpus55Adapter, lambda _request: httpx.Response(200, json=body))

    error = caught.value
    assert error.category == "token_limit"
    assert error.finish_reason == "length"
    assert error.usage["completion_tokens"] == 256
    assert raw_content not in str(error)


def test_chat_refusal_is_distinguished_without_persisting_refusal_text() -> None:
    sensitive_refusal = "private refusal explanation from upstream"
    body = _chat_response(
        None,
        message_fields={"refusal": sensitive_refusal},
        usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    )
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(ClaudeOpus55Adapter, lambda _request: httpx.Response(200, json=body))

    error = caught.value
    assert error.category == "refusal"
    assert error.usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert sensitive_refusal not in str(error)


def test_chat_malformed_json_is_distinguished_and_raw_output_is_not_kept() -> None:
    raw_content = "provider output with private data: {oops"
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(
            ClaudeOpus55Adapter,
            lambda _request: httpx.Response(
                200,
                json=_chat_response(
                    raw_content,
                    usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                ),
            ),
        )

    assert caught.value.category == "malformed_output"
    assert raw_content not in str(caught.value)
    assert "private data" not in str(caught.value)


def test_chat_missing_content_is_reported_as_response_shape_issue() -> None:
    body = _chat_response(None)
    message = body["choices"][0]["message"]  # type: ignore[index]
    del message["content"]  # type: ignore[index]

    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(ClaudeOpus55Adapter, lambda _request: httpx.Response(200, json=body))

    assert caught.value.category == "response_shape"
    assert "missing its content field" in str(caught.value)


@pytest.mark.parametrize("content", [None, [], {"unexpected": "object"}])
def test_chat_non_string_content_is_reported_as_response_shape_issue(content: object) -> None:
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(
            ClaudeOpus55Adapter,
            lambda _request: httpx.Response(200, json=_chat_response(content)),
        )

    assert caught.value.category == "response_shape"
    assert "content was" in str(caught.value)


def test_chat_content_filter_termination_is_distinguished_from_refusal() -> None:
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(
            ClaudeOpus55Adapter,
            lambda _request: httpx.Response(
                200,
                json=_chat_response(None, finish_reason="content_filter"),
            ),
        )

    assert caught.value.category == "content_filtered"
    assert caught.value.finish_reason == "content_filter"


def test_chat_provider_completion_error_shape_is_distinguished_safely() -> None:
    body = _chat_response("", finish_reason="error")
    body["choices"][0]["error"] = {  # type: ignore[index]
        "message": "sensitive provider diagnostic",
        "code": "upstream_failure",
    }

    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(ClaudeOpus55Adapter, lambda _request: httpx.Response(200, json=body))

    assert caught.value.category == "provider_error"
    assert caught.value.finish_reason == "error"
    assert "sensitive provider diagnostic" not in str(caught.value)


def test_chat_token_usage_diagnostics_ignore_untrusted_non_integer_fields() -> None:
    body = _chat_response(
        "",
        usage={
            "prompt_tokens": 4,
            "completion_tokens": True,
            "total_tokens": -1,
            "secret": "should not be persisted",
        },
    )
    with pytest.raises(AdapterResponseError) as caught:
        _run_adapter(ClaudeOpus55Adapter, lambda _request: httpx.Response(200, json=body))

    assert caught.value.usage == {"prompt_tokens": 4}
    assert "should not be persisted" not in str(caught.value)


def test_chat_safe_diagnostics_are_preserved_in_failed_trial_record() -> None:
    raw_content = "upstream private text without JSON"
    body = _chat_response(
        raw_content,
        finish_reason="stop",
        usage={"prompt_tokens": 18, "completion_tokens": 5, "total_tokens": 23},
    )

    async def run_trial():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=body)
            )
        ) as client:
            adapter = ClaudeOpus55Adapter(settings=TEST_SETTINGS, http_client=client)
            return await BenchmarkRunner(max_valid_moves=1).run_trial(
                PuzzleSpec("safe-error-diagnostics", ("R",)),
                adapter,
            )

    trial = asyncio.run(run_trial())
    diagnostic = trial.attempts[0].error_message
    assert trial.termination_status == "adapter_error"
    assert diagnostic is not None
    assert "category=malformed_output" in diagnostic
    assert "finish_reason=stop" in diagnostic
    assert "prompt_tokens=18" in diagnostic
    assert raw_content not in diagnostic


def test_httpx_transport_timeout_uses_adapter_timeout_error() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("mock timeout", request=request)

    with pytest.raises(AdapterTimeoutError):
        _run_adapter(Jev113Adapter, timeout_handler)
