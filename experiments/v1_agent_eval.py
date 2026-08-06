import os
import requests
import base64
import re
import ast
import sys
import json
from datetime import datetime

# This dormant V1-style entry point lives outside agent/, so add the live agent package root.
_AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from sim.hand_reset import reset_hands_in_front2

# Startup hand-centering DISABLED (user, 2026-07-21): the hand's default pose is now centered in
# Unity itself, so the Python side no longer poses it. Re-enable if Unity's default hand pose
# regresses to the giant-ghost-hands-in-frame default.
# reset_hands_in_front2(extra_elevation=-0.1, hand="left")

from sim.env import _REQUEST_SCREENSHOT_
from sim.env import *
from toolset.actions import (
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
    MANIPULATION_ACTIONS_REF,
)

MAIN_TASK = sys.argv[1]
RUN_ENTRY = sys.argv[2] if len(sys.argv) > 2 else None
# Phase 4.2 A/B arm: 'vlm' (control - the old VLM-driven navigation) or 'graph' (navigation
# dispatched to the mapping resolver + goto_checkpoint; VLM never sees nav actions).
NAV_MODE = sys.argv[3] if len(sys.argv) > 3 else 'vlm'
assert NAV_MODE in ('vlm', 'graph'), f"nav mode must be vlm|graph, got {NAV_MODE!r}"
ON_PLAY = True

at_init_state = True

time_step = 1
timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
run_name = "agent-state-" + (RUN_ENTRY or "") + "-" + timestamp if RUN_ENTRY else timestamp
init_logger(run_name=run_name)

CURRENT_AGENT_STATE = {
    "translation": (0, 0, 0),
    "rotation": (0, 0, 0),
    "isColliding": False,
    "leftTranslation": (0, 0, 0),
    "leftRotation": (0, 0, 0),
    "rightTranslation": (0, 0, 0),
    "rightRotation": (0, 0, 0),
    "leftHoveredObject": "None",
    "leftGrippedState": False,
    "rightHoveredObject": "None",
    "rightGrippedState": False,
    "last_action_blocked": False,
    "mode": "perception"
}

initial_agent_position = TransformAgent((0, 0, 0), (0, 0, 0))
initial_hands_position = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

initial_state = {**initial_agent_position, **initial_hands_position}

for k, v in initial_state.items():
    CURRENT_AGENT_STATE[k] = v

print(" INITIAL STATE ".center(100, "="))
print(CURRENT_AGENT_STATE)
print("=" * 100)

# ====== Initialize Embodied Agent ======
from agent_core.llm import agent_vlm_config
from agent_core.runtime import EmbodiedAgent

# Agent runtime = the OpenAI API compatible endpoint from config.env
# (user directive 2026-07-19; OpenRouter retired on 402).
# The ANNOTATOR is unaffected: always `claude -p` sonnet/medium (see CLAUDE.md).
vlm_config = agent_vlm_config(temperature=0.5)
associative_config = agent_vlm_config(temperature=0.3)

embodied_agent = EmbodiedAgent(
    vlm_config=vlm_config,
    associative_config=associative_config,
    mode='lean',
    nav_mode=NAV_MODE,
)
print(f"[env_simulation] navigation arm: {NAV_MODE}")
# ========================================

while ON_PLAY:
    if RUN_ENTRY:
        directory_path = os.path.join('screenshots', 'SIM_RUNS2/' + RUN_ENTRY)
        os.makedirs(directory_path, exist_ok=True)
        existing = [fn for fn in os.listdir(directory_path) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
        prefix = str(len(existing) + 1).zfill(6)
        suffix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        imagebytes = _REQUEST_SCREENSHOT_(prefix=prefix, suffix=suffix,
                                          folder_name=directory_path, save_image=True)['image']
    else:
        imagebytes = _REQUEST_SCREENSHOT_()['image']

    imageb64 = base64.b64encode(imagebytes).decode('utf-8')

    if at_init_state:
        request = {
            'task': MAIN_TASK,
            'image': imageb64,
            'state': CURRENT_AGENT_STATE,
        }
        at_init_state = False
    else:
        request = {
            'task': MAIN_TASK,
            'image': imageb64,
            'state': CURRENT_AGENT_STATE,
        }

    response = embodied_agent.execute_lean(request, time_step)
    print(f"Response @ Timestep {time_step}:\n{response}")

    if response['halt'] == True:
        print("Halting the simulation as per the response.")
        ON_PLAY = False
        break
    else:
        main_response = response['text']
        agent_mode = response['agent_mode']

        CURRENT_AGENT_STATE['mode'] = agent_mode

        extracted_main_response = re.search(embodied_agent.vlm_agent.extractable_json_structured_output, main_response)[1]
        main_response = ast.literal_eval(extracted_main_response)

        reasoning = main_response['reasoning']
        actions = main_response['actions']
        times = main_response['times']
        notes = main_response['notes']
        main_goal = notes['main_goal']
        sub_goals = notes['sub_goal']
        key_info = notes['key_info']
        status = notes['status']
        checklist = notes['checklist']

        time_step += 1

        blocked_reason = False
        for action, time in zip(actions, times):
            action = action.strip()
            time = int(time)

            # Parse inline argument: action_name('arg') or action_name("arg")
            inline_arg = None
            inline_match = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', action)
            if inline_match:
                action, inline_arg = inline_match.group(1), inline_match.group(2)

            # Manipulation actions are gated to manipulation mode. Pre-6.1 they no-op'd elsewhere
            # (hands inactive); as of Phase 6.1 hands stay ACTIVE at REST in every mode, so firing one
            # outside manipulation would perturb the hand/carried item - the gate stands (mode coherence).
            if action in MANIPULATION_ACTIONS_REF and agent_mode != "manipulation":
                print(f"[BLOCKED] '{action}' only works in *manipulation* mode "
                      f"(current mode: {agent_mode}); the hands are inactive otherwise. Skipped.")
                blocked_reason = f"{action} requires manipulation mode (was {agent_mode})"
                continue

            if action in NAVIGATION_ACTIONS_REF:
                action_ref = NAVIGATION_ACTIONS_REF[action]
            elif action in PERCEPTION_ACTIONS_REF:
                action_ref = PERCEPTION_ACTIONS_REF[action]
            elif action in MANIPULATION_ACTIONS_REF:
                action_ref = MANIPULATION_ACTIONS_REF[action]
            else:
                print(f"Unknown action: {action}")
                continue

            if action == "center_object_on_screen":
                target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
                action_ref(target_info)
            elif action in ("retrieve_item", "approach_object"):
                target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
                action_ref(main_goal)
            else:
                action_ref(time)

        # Refresh state from the environment after all actions have been executed
        updated_agent = TransformAgent((0, 0, 0), (0, 0, 0))
        updated_hands = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        for k, v in {**updated_agent, **updated_hands}.items():
            CURRENT_AGENT_STATE[k] = v
        CURRENT_AGENT_STATE['last_action_blocked'] = blocked_reason
        # mode is kept as set by the agent above
