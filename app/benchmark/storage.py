"""Atomic JSON persistence for auditable benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from app.benchmark.results import BenchmarkRun


class ResultsStorageError(OSError):
    """A benchmark result could not be read or written."""


class ResultsFormatError(ResultsStorageError):
    """A result file is not valid JSON for the supported schema."""


def save_run(run: BenchmarkRun, path: str | Path) -> Path:
    """Write a validated run as UTF-8 JSON, creating parent directories.

    A temporary file in the destination directory and atomic replacement keep
    interrupted writes from leaving a partial result at the requested path.
    """

    if not isinstance(run, BenchmarkRun):
        raise TypeError("run must be a BenchmarkRun")
    target = Path(path)
    temporary_path: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(run.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, target)
        temporary_path = None
    except (OSError, TypeError, ValueError) as exc:
        raise ResultsStorageError(f"could not save benchmark results to {target}") from exc
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return target


def load_run(path: str | Path) -> BenchmarkRun:
    """Load and validate one complete benchmark run from JSON."""

    target = Path(path)
    try:
        contents = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ResultsStorageError(f"could not read benchmark results from {target}") from exc
    try:
        decoded = json.loads(contents)
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ResultsFormatError(f"benchmark results file is not valid JSON: {target}") from exc
    try:
        return BenchmarkRun.from_dict(decoded)
    except (TypeError, ValueError, KeyError) as exc:
        raise ResultsFormatError(f"benchmark results do not match the supported schema: {target}") from exc
