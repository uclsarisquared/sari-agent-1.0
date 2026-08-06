"""Compatibility entrypoint for the long-horizon typed-subtask orchestrator.

The implementation is split by responsibility across cli, orchestration,
orchestrator_llm, leg_runner, action_dispatch, and held_item_inspection.
Imports below preserve the historical API while callers migrate.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_AGENT_DIR)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orchestrator.action_dispatch import (  # noqa: E402,F401
    MANIPULATION_ACTIONS_REF,
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
    _GRAB_ACTIONS,
    _INSPECT_APPROACH_ACTIONS,
    _INSPECT_HELD_ACTIONS,
    _INSPECT_MACRO_ACTIONS,
    _INSPECT_MOVE_BUDGET_STEPS,
    _INSPECT_VISUAL_ACTIONS,
    _MACRO_ACTIONS,
    _crouched_grab,
    _grab_ready,
    _last_reach_line,
    _salvage_actions_times,
    dispatch_action,
    parse_actor_response,
)
from orchestrator.cli import main  # noqa: E402,F401
from orchestrator.held_item_inspection import (  # noqa: E402,F401
    _INSPECTION_PASS_RESET_DELTA,
    _inspection_action_batch,
    _inspection_macro_summary,
    _inspection_rotation_delta,
    _run_held_item_inspection_macro,
)
from orchestrator.leg_runner import (  # noqa: E402,F401
    _deterministic_guard_details,
    _fresh_agent_state,
    _model_facing_state,
    _off_target,
    _run_leg_impl,
    run_leg,
    write_step_output,
)
from orchestrator.orchestration import (  # noqa: E402,F401
    OrchestrationConfig,
    _current_nearest_cp,
    _load_store_map,
    _resolve_run_dir,
    orchestrate,
)
from orchestrator.orchestrator_llm import (  # noqa: E402,F401
    ASSOCIATIVE_CONFIG,
    ORCHESTRATOR_MODEL,
    VLM_CONFIG,
    _generate_findings_if_enabled,
    _llm_call,
    _llm_client,
    decompose_task,
    generate_findings_summary,
)

if __name__ == "__main__":
    main()
