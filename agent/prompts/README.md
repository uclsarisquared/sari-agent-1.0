# Prompt assets

This directory is the canonical home for reusable production LLM instructions.
Keep runtime data (task text, screenshots, state, retrieved memory, and similar
per-call context) in Python; keep stable instruction text here.

Load a static prompt with `load_prompt("path/name")`. Use `render_prompt` only for
the explicit `{{UPPER_CASE}}` tokens found in an asset. The loader rejects missing
and unused substitutions and caches file reads.

Python modules may re-export loaded prompts under their existing constant names to
avoid coupling callers to filenames. JSON schemas, parsers, thresholds, and other
executable contracts remain in Python next to the code that enforces them.

Tests, experiments, deprecated code, generated memory, and one-off user messages
are intentionally not centralized here: they are fixtures, historical snapshots,
or runtime data rather than shared production prompt assets.
