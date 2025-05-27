import os
import requests
import base64
import re
import ast
import sys
import json
from datetime import datetime

from hand_reset import reset_hands

reset_hands()

from env import _REQUEST_SCREENSHOT_
from env import *
from actions import (
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
)

MAIN_TASK = sys.argv[1]
RUN_ENTRY = sys.argv[2] or None
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
from agent import EmbodiedAgent, GeminiConfig

vlm_config = GeminiConfig(
    model_id='gemini-2.5-pro-preview-05-06',
    max_thinking_tokens=3072,
    temperature=0,
    mode='lean'
)
associative_config = GeminiConfig(
    model_id='gemini-2.5-pro-preview-05-06',
    max_thinking_tokens=1024,
    temperature=0.35,
    mode='lean'
)

embodied_agent = EmbodiedAgent(
    vlm_config=vlm_config,
    associative_config=associative_config,
    mode='lean'
)
# ========================================

while ON_PLAY:
    if RUN_ENTRY:
        directory_path = os.path.join('screenshots', 'SIM_RUNS/' + RUN_ENTRY)
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

        print("=" * 100)
        print(f"Current Agent State @ Timestep {time_step}:\n{CURRENT_AGENT_STATE}")
        print("=" * 100)

        time_step += 1

        for action, time in zip(actions, times):
            action = action.strip()
            time = int(time)

            if action in NAVIGATION_ACTIONS_REF:
                action_ref = NAVIGATION_ACTIONS_REF[action]
            elif action in PERCEPTION_ACTIONS_REF:
                action_ref = PERCEPTION_ACTIONS_REF[action]
            else:
                print(f"Unknown action: {action}")
                continue

            if action == "center_object_on_screen":
                target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
                action_ref(target_info)  # Center the object based on the target info
            else:
                action_ref(time) # Execute the action with the specified time
