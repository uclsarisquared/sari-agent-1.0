"""Load version-controlled prompt assets from ``agent/prompts``.

Prompt text belongs in Markdown files. Python modules may continue to expose named
constants when that is convenient for callers, but those constants should be loaded
from this module rather than containing the prompt itself.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path, PurePosixPath


PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def _prompt_path(name: str) -> Path:
    """Resolve a root-relative prompt name without permitting path traversal."""
    relative = PurePosixPath(str(name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"prompt name must stay below {PROMPT_ROOT}: {name!r}")
    if relative.suffix != ".md":
        relative = relative.with_suffix(".md")
    path = PROMPT_ROOT.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(f"prompt asset not found: {relative.as_posix()}")
    return path


@lru_cache(maxsize=None)
def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def load_prompt(name: str) -> str:
    """Return a UTF-8 Markdown prompt, cached by its canonical asset path."""
    return _read_prompt(_prompt_path(name))


def render_prompt(name: str, /, **values: object) -> str:
    """Load a prompt and replace its explicit ``{{UPPER_CASE}}`` tokens.

    The deliberately small renderer avoids ``str.format`` because prompts commonly
    contain literal JSON braces. Missing and unused values are errors so prompt/code
    drift fails during import instead of silently reaching a model call.
    """
    template = load_prompt(name)
    required = set(_TOKEN_RE.findall(template))
    supplied = set(values)
    missing = required - supplied
    unused = supplied - required
    if missing or unused:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unused:
            details.append(f"unused: {', '.join(sorted(unused))}")
        raise ValueError(f"invalid values for prompt {name!r} ({'; '.join(details)})")
    return _TOKEN_RE.sub(lambda match: str(values[match.group(1)]), template)
