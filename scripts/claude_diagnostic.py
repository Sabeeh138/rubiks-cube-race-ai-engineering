"""Send exactly one explicitly authorized Claude adapter diagnostic request.

This script is intentionally independent of the benchmark service and never
persists the provider response. It uses the existing adapter contract, request
schema, move validator, and timeout policy.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence

import httpx

from app.config import ConfigurationError, Settings
from app.cube import CubeState, LEGAL_MOVES
from app.models import (
    AdapterError,
    AdapterResponseError,
    AdapterTimeoutError,
    CLAUDE_OPUS_55_MODEL_ID,
    ClaudeOpus55Adapter,
    InvalidMoveError,
    OpenRouterHTTPError,
    TurnContext,
)
from app.models.openrouter import DEFAULT_HTTP_TIMEOUT

OVERALL_TURN_TIMEOUT_SECONDS = 30.0
ClientFactory = Callable[[], httpx.AsyncClient]
SettingsLoader = Callable[[], Settings]
Output = Callable[[str], object]


def _default_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT)


def _diagnostic_context() -> TurnContext:
    """Use a valid one-move state and the shared legal-move vocabulary."""

    state = CubeState().apply_move("R")
    return TurnContext(
        cube_state=state.serialize(),
        recent_moves=("R",),
        legal_moves=LEGAL_MOVES,
        timeout_seconds=OVERALL_TURN_TIMEOUT_SECONDS,
    )


async def _request_once(
    *,
    settings: Settings,
    client_factory: ClientFactory,
    output: Output,
) -> int:
    """Make one Claude call and print only the validated move or safe errors."""

    try:
        async with client_factory() as client:
            adapter = ClaudeOpus55Adapter(
                settings=settings,
                http_client=client,
                request_timeout=DEFAULT_HTTP_TIMEOUT,
            )
            proposal = await adapter.choose_move(_diagnostic_context())
    except AdapterResponseError as exc:
        output("status=failure")
        output(f"model={exc.model_id}")
        output(f"category={exc.category}")
        output(f"detail={exc.detail}")
        if exc.finish_reason is not None:
            output(f"finish_reason={exc.finish_reason}")
        if exc.usage:
            output(
                "usage="
                + ", ".join(f"{name}={count}" for name, count in exc.usage.items())
            )
        return 1
    except OpenRouterHTTPError as exc:
        output("status=failure")
        output(f"model={exc.model_id}")
        output(f"category=http_error")
        output(f"http_status={exc.status_code}")
        return 1
    except AdapterTimeoutError as exc:
        output("status=failure")
        output(f"model={exc.model_id}")
        output("category=timeout")
        output(f"timeout_seconds={exc.timeout_seconds}")
        output(f"elapsed_seconds={exc.elapsed_seconds:.3f}")
        return 1
    except InvalidMoveError:
        output("status=failure")
        output(f"model={CLAUDE_OPUS_55_MODEL_ID}")
        output("category=invalid_move")
        output("detail=adapter did not return a legal move token")
        return 1
    except AdapterError as exc:
        # Avoid printing arbitrary model output (for example, an invalid move
        # candidate) or transport exception text.
        output("status=failure")
        output(f"model={CLAUDE_OPUS_55_MODEL_ID}")
        output(f"category=adapter_error")
        output(f"error_type={type(exc).__name__}")
        return 1
    except Exception as exc:
        # Keep configuration/client failures useful without printing arbitrary
        # exception text that may contain response or transport details.
        output("status=failure")
        output("category=diagnostic_error")
        output(f"error_type={type(exc).__name__}")
        return 1

    output("status=success")
    output(f"model={proposal.model_id}")
    output(f"valid_move={proposal.move}")
    output(f"elapsed_seconds={proposal.elapsed_seconds:.3f}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader: SettingsLoader = Settings.from_env,
    client_factory: ClientFactory | None = None,
    output: Output = print,
) -> int:
    """Run the guarded one-request CLI; dependencies are injectable offline."""

    parser = argparse.ArgumentParser(
        description=(
            "Send exactly one live inference request to the assigned Claude "
            "model. This request may incur a charge."
        )
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="confirm authorization for one Claude request that may incur a charge",
    )
    args = parser.parse_args(argv)
    if not args.confirm_live:
        output("Not run: pass --confirm-live to authorize exactly one Claude request.")
        return 2

    try:
        settings = settings_loader()
        settings.require_openrouter_api_key()
    except ConfigurationError:
        output("Not run: OPENROUTER_API_KEY is required; no request was sent.")
        return 2
    except Exception as exc:
        output(f"Not run: configuration error ({type(exc).__name__}); no request was sent.")
        return 2

    output("warning=This single live request may incur a provider charge.")
    selected_client_factory = (
        _default_client_factory if client_factory is None else client_factory
    )
    try:
        return asyncio.run(
            _request_once(
                settings=settings,
                client_factory=selected_client_factory,
                output=output,
            )
        )
    except RuntimeError as exc:
        # A CLI should normally own its event loop. Report a safe explanation
        # if called from an already-running loop instead of leaking a traceback.
        output(f"Diagnostic could not start ({type(exc).__name__}); no result recorded.")
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by manual invocation
    raise SystemExit(main())
