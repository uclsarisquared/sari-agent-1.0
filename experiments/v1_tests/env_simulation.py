import os
import requests
import base64
import re
import ast
import sys
import json
from datetime import datetime

from hand_reset import reset_hands_in_front2
from functions import call_agent_action

from env import _REQUEST_SCREENSHOT_, TransformAgent, TransformHands
from actions import (navigation_actions_ref, perception_actions_ref, manipulation_actions_ref, actions_ref)
from env import init_logger
from loguru import logger
from comprehension import determine_mode

INFERENCE_API = "http://localhost:8005/predict"
MODE_API = "http://202.92.159.242:8000/seek-mode"
ACTIONS_API = "http://202.92.159.242:8000/decide_action"
EXTRACT_JSON_PATTERN = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

MAIN_TASK = sys.argv[1]
RUN_ENTRY = sys.argv[2] or None
ON_PLAY = True

at_init_state = True

reset_hands_in_front2(extra_elevation=-0.1, hand="left")
time_step = 1
timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
run_name = "agent-state-" + (RUN_ENTRY or "") + "-" + timestamp if RUN_ENTRY else timestamp 
init_logger(run_name=run_name)
CURRENT_AGENT_STATE = {
    "translation": (0,0,0),
    "rotation": (0,0,0),
    "isColliding": False,
    "leftTranslation": (0,0,0),
    "leftRotation": (0,0,0),   
    "rightTranslation": (0,0,0),
    "rightRotation": (0,0,0),
    "leftHoveredObject": "None",
    "leftGrippedState": False,
    "rightHoveredObject": "None",
    "rightGrippedState": False,
    "mode": "perception"
}
initial_agent_position = TransformAgent((0,0,0),(0,0,0))
initial_hands_position = TransformHands((0,0,0),(0,0,0),(0,0,0),(0,0,0))

initial_state = {**initial_agent_position, **initial_hands_position}
for k, v in initial_state.items():
    CURRENT_AGENT_STATE[k] = v
print("Initial State: ", CURRENT_AGENT_STATE)

ACTIONS_HISTORY = []
mode = "INITIAL"
CURRENT_BOUNDING_BOX = {
    "xmin": 0,
    "xmax": 0,
    "ymin": 0,
    "ymax": 0
}

while ON_PLAY:
    # Prepare for state logging
    CURRENT_TIMESTEP_ACTIONS = []
    current_state_log = f"State @ timestep {time_step}\n"

    if RUN_ENTRY:
        # 1) Dynamic folder name: SIM_RUNS + RUN_ENTRY
        folder_name = os.path.join("screenshots", "SIM_RUNS/" + RUN_ENTRY)
        os.makedirs(folder_name, exist_ok=True)

        # 2) Count existing images in folder (you can adjust extensions as needed)
        existing = [
            fn for fn in os.listdir(folder_name)
            if fn.lower ().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # 3) Use only the current date and time for prefix
        prefix = str(len(existing) + 1).zfill(6)

        # 4) Use only the current date and time for suffix
        suffix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        # 5) Call your screenshot function
        imagebytes = _REQUEST_SCREENSHOT_(prefix=prefix, suffix=suffix, folder_name=folder_name, save_image=True)['image']
    else:
        imagebytes = _REQUEST_SCREENSHOT_()['image']

    if time_step > 1:
        actions_log = actions_log + f"Actions @ Time Step {time_step - 1}\n" + "\n".join([a["action"] for a in ACTIONS_HISTORY[time_step-2]])
        print("Actions Log:\n", actions_log)
    else:
        actions_log = "START of Execution\n"

    imageb64 = base64.b64encode(imagebytes).decode('utf-8')

    if at_init_state:
        post = {
            'task': MAIN_TASK,
            'image': imageb64,
            'state': CURRENT_AGENT_STATE,
            'actions': actions_log
        }
        at_init_state = False
    else:
        post = {
            'image': imageb64,
            'state': CURRENT_AGENT_STATE,
            'actions': str(ACTIONS_HISTORY)
        }
    logger.info(f"Time step: {time_step}")

    response = requests.post(
        INFERENCE_API,
        data=post,
    )
    if response.status_code != 200:
        break
    response = response.json()['response']
    text = response['text']
    history = response['history']
    extracted = re.search(EXTRACT_JSON_PATTERN, text)[1]
    extracted = ast.literal_eval(extracted)
    reasoning = extracted.get('reasoning', 'No Reasoning.')
    notes = extracted.get('notes', 'No Notes.')
    mode_payload = {
        "timestep": f"Time step: {time_step}",
        "current_state_log": current_state_log,
        "plan": reasoning,
        "previous_mode": mode,
        "bounding_box": CURRENT_BOUNDING_BOX,
        "bounding_box_area": (CURRENT_BOUNDING_BOX['xmax']-CURRENT_BOUNDING_BOX['xmin']) * (CURRENT_BOUNDING_BOX['ymax']-CURRENT_BOUNDING_BOX['ymin'])
    }
    # TODO: Add exponential backoff 
    mode_response = requests.post(
        MODE_API,
        data=mode_payload
    )
    print("Mode Response: ", mode_response.content)
    try:
        mode = json.loads(mode_response.json()["response"])["mode"]
    except json.decoder.JSONDecodeError as e:
        mode = "perception"
    CURRENT_AGENT_STATE["mode"] = mode
    print("Current Mode: ", mode)

    actions_payload = {
        "timestep": f"Time step: {time_step}",
        "current_state_log": current_state_log,
        "plan": reasoning,
        "mode": mode,
        "bounding_box": CURRENT_BOUNDING_BOX,
        "bounding_box_area": (CURRENT_BOUNDING_BOX['xmax']-CURRENT_BOUNDING_BOX['xmin']) * (CURRENT_BOUNDING_BOX['ymax']-CURRENT_BOUNDING_BOX['ymin'])
    }
    response_json = json.loads(
        requests.post(ACTIONS_API, data=actions_payload).content
    )
    actions_list = response_json.get("response", [])

    # Logging reasoning and notes
    print('#' * 50)
    print(f'Reasoning: {reasoning}')
    print('-' * 50)
    print(f'Notes: {notes}')
    print('#' * 50)

    logger.info(f'Reasoning: {reasoning}')
    logger.info(f'Notes: {notes}')
    print("GPT Actions: ", actions_list)

    # Dispatch map
    mode_ref_map = {
        "perception": perception_actions_ref,
        "navigation": navigation_actions_ref,
        "manipulation": manipulation_actions_ref
    }
    ref = mode_ref_map.get(mode, actions_ref)

    # Iterate over actions
    for action_item in actions_list:
        func_info = action_item.get("function", {})
        action_name = func_info.get("name")
        arguments = json.loads(func_info.get("arguments", "{}"))

        # Handle special actions
        if action_name == "STOP" or action_name == "stop":
            ON_PLAY = False
            print(f"'STOP' action detected. Stopping the loop...")
            logger.info(f"'STOP' action detected. Stopping the loop...")
            break

        # Determine number of executions (if any)
        exec_count = int(arguments.get("units", 1))

        try:
            action_func = ref[action_name]
        except KeyError:
            raise KeyError(f"Action '{action_name}' not found in `actions_ref`.")

        current_state_log += f"Executing action: {action_name} for {exec_count} times\n"
        print("Arguments", arguments)
        CURRENT_TIMESTEP_ACTIONS.append({
            "action": f"Executing action: {action_name} for {exec_count} times\n",
            "arguments": str(arguments)
        })

        # Execute function with full arguments if no "units"
        state_vars = action_func(**{k: v for k, v in arguments.items()}) or {}
        if action_name == "center_object_on_screen":
            CURRENT_BOUNDING_BOX = state_vars[1]
            state_vars = state_vars[0]

        for k, v in state_vars.items():
            CURRENT_AGENT_STATE[k] = v
        for k, v in CURRENT_AGENT_STATE.items():
            current_state_log += f"{k}: {v}\n"
        current_state_log += "\n"

        if action_name == "grab_and_read_item":
            ON_PLAY = False
            print(f"'STOP' action detected. Stopping the loop...")
            logger.info(f"'STOP' action detected. Stopping the loop...")
            break

        if ON_PLAY:
            print(f"ACTION: {action_name} executed for {exec_count} times.")
            logger.info(f"ACTION: {action_name} executed for {exec_count} times.")

    ACTIONS_HISTORY.append(CURRENT_TIMESTEP_ACTIONS)
    # Write state log to file
    os.makedirs("agent_states", exist_ok=True)
    with open(f"agent_states/{run_name}.txt", "a") as f:
        f.write(current_state_log)

    print('#' * 50)
    time_step += 1
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
