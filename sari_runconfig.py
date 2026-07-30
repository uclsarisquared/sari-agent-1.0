"""Shared, strict loader for standalone-agent and Sari Bench TOML run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli

from overhaul.agent_core.context_policy import CONTEXT_POLICY_NAMES


class RunConfigError(ValueError):
    """The run configuration is malformed or contains an unsupported option."""


_SCHEMA: dict[str, dict[str, object]] = {
    "agent": {
        "task": str,
        "arm": {"vlm", "graph", "graph-advised"},
        "context_policy": set(CONTEXT_POLICY_NAMES),
        "resolver_backend": {"qwen", "claude-cli"},
        "completion_guard": {"deterministic", "vlm"},
        "leg_retries": int,
    },
    "limits": {
        "max_steps": int,
        "max_minutes": (int, float),
    },
    "environment": {
        "map_dir": str,
        "reset_start": bool,
        "restart_env": bool,
        "ws_uri": str,
        "ocr_url": str,
    },
    "output": {
        "run_dir": str,
        "summary": str,
    },
    "bench": {
        "prompts": str,
        "tries": int,
        "time_limit": (int, float),
        "per_leg_minutes": (int, float),
        "coordinator": str,
        "sandbox_startup_timeout": (int, float),
        "output_dir": str,
        "name": str,
        "resume": bool,
        "concurrency": int,
        "capture_interval": (int, float),
        "only": str,
        "max_steps": int,
        "arm": {"vlm", "graph", "graph-advised"},
        "context_policy": set(CONTEXT_POLICY_NAMES),
        "map_dir": str,
        "ocr_url": str,
        "leg_retries": int,
        "completion_guard": {"deterministic", "vlm"},
    },
}

_PATH_OPTIONS = {
    ("environment", "map_dir"),
    ("output", "run_dir"),
    ("output", "summary"),
    ("bench", "prompts"),
    ("bench", "output_dir"),
    ("bench", "map_dir"),
}


def _is_instance(value: object, expected: object) -> bool:
    # bool is an int subclass, but accepting `true` for max_steps is never useful.
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, tuple):
        return isinstance(value, expected) and not isinstance(value, bool)
    return isinstance(value, expected)


def _validate_value(path: Path, section: str, key: str, value: object, expected: object) -> None:
    label = f"{section}.{key}"
    if isinstance(expected, set):
        if not isinstance(value, str) or value not in expected:
            choices = ", ".join(repr(choice) for choice in sorted(expected))
            raise RunConfigError(f"{path}: {label} must be one of {choices}, got {value!r}")
        return
    if not _is_instance(value, expected):
        if isinstance(expected, tuple):
            name = "number"
        elif expected is int:
            name = "integer"
        else:
            name = expected.__name__
        article = "an" if name == "integer" else "a"
        raise RunConfigError(
            f"{path}: {label} must be {article} {name}, got {type(value).__name__}"
        )

    if key in {"max_steps", "tries", "concurrency"} and value < 1:
        raise RunConfigError(f"{path}: {label} must be at least 1")
    if key == "leg_retries" and value < 0:
        raise RunConfigError(f"{path}: {label} cannot be negative")
    if key in {
        "max_minutes",
        "time_limit",
        "per_leg_minutes",
        "sandbox_startup_timeout",
        "capture_interval",
    } and value < 0:
        raise RunConfigError(f"{path}: {label} cannot be negative")


class RunConfig:
    """Validated TOML values, with config-relative filesystem paths made absolute."""

    def __init__(self, path: Path, values: dict[str, dict[str, Any]]) -> None:
        self.path = path
        self._values = values

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._values.get(section, {}).get(key, default)

    def has(self, section: str, key: str) -> bool:
        return key in self._values.get(section, {})


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run config.

    Relative filesystem values are resolved from the TOML file, making the same file safe to use
    from both the repository root and ``overhaul/``.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            raw = tomli.load(handle)
    except FileNotFoundError as error:
        raise RunConfigError(f"run config does not exist: {config_path}") from error
    except (OSError, tomli.TOMLDecodeError) as error:
        raise RunConfigError(f"could not load run config {config_path}: {error}") from error

    unknown_sections = sorted(set(raw) - set(_SCHEMA))
    if unknown_sections:
        raise RunConfigError(
            f"{config_path}: unknown section(s): {', '.join(unknown_sections)}"
        )

    values: dict[str, dict[str, Any]] = {}
    for section, section_values in raw.items():
        if not isinstance(section_values, dict):
            raise RunConfigError(f"{config_path}: [{section}] must be a TOML table")
        unknown_keys = sorted(set(section_values) - set(_SCHEMA[section]))
        if unknown_keys:
            names = ", ".join(f"{section}.{key}" for key in unknown_keys)
            raise RunConfigError(f"{config_path}: unknown option(s): {names}")

        values[section] = {}
        for key, value in section_values.items():
            _validate_value(config_path, section, key, value, _SCHEMA[section][key])
            if (section, key) in _PATH_OPTIONS:
                if not value.strip():
                    raise RunConfigError(f"{config_path}: {section}.{key} cannot be empty")
                candidate = Path(value).expanduser()
                value = str(
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (config_path.parent / candidate).resolve()
                )
            values[section][key] = value

    return RunConfig(config_path, values)
