import pytest

from app.config import ConfigurationError, Settings


def test_api_key_is_optional_until_a_request_requires_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    settings = Settings.from_env(load_dotenv_file=False)

    assert settings.openrouter_api_key is None


def test_missing_required_api_key_has_setup_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY is required"):
        Settings.from_env(
            require_openrouter_api_key=True,
            load_dotenv_file=False,
        )


def test_api_key_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "  test-key  ")

    settings = Settings.from_env(load_dotenv_file=False)

    assert settings.require_openrouter_api_key() == "test-key"


def test_student_display_settings_are_optional_and_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDENT_NAME", "  Ada Example ")
    monkeypatch.setenv("STUDENT_ROLL_NUMBER", " A-42  ")

    settings = Settings.from_env(load_dotenv_file=False)

    assert settings.student_name == "Ada Example"
    assert settings.student_roll_number == "A-42"

