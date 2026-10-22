"""Application configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(RuntimeError):
    """Raised when a required application setting is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings needed by the application.

    The health-only Phase 1 app does not require an API key. Code that is about
    to make an OpenRouter request must call ``require_openrouter_api_key``.
    """

    openrouter_api_key: str | None = None
    student_name: str | None = None
    student_roll_number: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        require_openrouter_api_key: bool = False,
        dotenv_path: str | Path | None = None,
        load_dotenv_file: bool = True,
    ) -> "Settings":
        """Load settings from the environment and, by default, the project .env.

        Existing process environment values take precedence over values in
        ``.env``. ``load_dotenv_file=False`` is useful for isolated tests.
        """

        if load_dotenv_file:
            load_dotenv(
                dotenv_path=Path(dotenv_path) if dotenv_path else PROJECT_ROOT / ".env",
                override=False,
            )

        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        student_name = os.getenv("STUDENT_NAME", "").strip()
        student_roll_number = os.getenv("STUDENT_ROLL_NUMBER", "").strip()
        settings = cls(
            openrouter_api_key=key or None,
            student_name=student_name or None,
            student_roll_number=student_roll_number or None,
        )
        if require_openrouter_api_key:
            settings.require_openrouter_api_key()
        return settings

    def require_openrouter_api_key(self) -> str:
        """Return the configured API key or raise a clear setup error."""

        if not self.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for OpenRouter requests. "
                "Set it in your process environment or in the project .env file."
            )
        return self.openrouter_api_key

