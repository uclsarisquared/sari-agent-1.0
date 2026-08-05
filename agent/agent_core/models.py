"""Model ids for the agent runtime, declared in config.env rather than in code.

Every reasoner in this repo talks to an **OpenAI API compatible endpoint** (`$OPENAI_API_URL`,
bearer `$OPENAI_API_KEY`) — nothing here is tied to a particular vendor or host. Which model that
endpoint should be asked for is configuration, not a code constant, so it lives in the repo-root
`config.env`:

    SARI_MODEL            the model every agent-runtime reasoner uses (orchestrator, actor,
                          associative memory, perception, item resolver, planners, probes)
    SARI_ANNOTATOR_MODEL  the offline annotator's model when it runs on the OpenAI-compatible
                          endpoint (`--backend qwen`); falls back to SARI_MODEL when unset.
                          The `claude-cli` annotator backend is unaffected — it carries its own
                          `--model` (sonnet, the frozen quality baseline; see CLAUDE.md).

`DEFAULT_MODEL` below is the fallback for a checkout with no `config.env` entry, and is the ONLY
model id hardcoded in the tree. Point the endpoint somewhere else by setting the env var; no code
change is needed.

Deliberately dependency-light (dotenv + stdlib) so leaf scripts and the slamtest tools can import
it without pulling in the whole agent runtime.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo-root config.env (agent/agent_core/ -> repo root is three parents up), resolved from
# __file__ so it loads regardless of CWD or checkout location. Mirrors agent_core.agent.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "config.env")

# The one hardcoded model id in the tree; SARI_MODEL in config.env overrides it.
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


def agent_model() -> str:
    """The model every agent-runtime reasoner asks the endpoint for ($SARI_MODEL)."""
    return os.getenv("SARI_MODEL") or DEFAULT_MODEL


def annotator_model() -> str:
    """The offline annotator's model on the OpenAI-compatible endpoint ($SARI_ANNOTATOR_MODEL,
    defaulting to the agent model)."""
    return os.getenv("SARI_ANNOTATOR_MODEL") or agent_model()
