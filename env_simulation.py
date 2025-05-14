import os
import requests
import base64
import re
import ast
import sys
import json
from datetime import datetime

from env import _REQUEST_SCREENSHOT_, _PROPER_HAND_POSITIONING_, TransformAgent, TransformHands
from actions import actions_ref
from env import init_logger
from loguru import logger

INFERENCE_API = "http://localhost:8005/predict"
EXTRACT_JSON_PATTERN = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

MAIN_TASK = sys.argv[1]
RUN_ENTRY = sys.argv[2] or None
ON_PLAY = True

at_init_state = True

_PROPER_HAND_POSITIONING_()
time_step = 1
timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
run_name = "agent-state-" + (RUN_ENTRY or "") + timestamp if RUN_ENTRY else timestamp 
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
}
initial_agent_position = TransformAgent((0,0,0),(0,0,0))
initial_hands_position = TransformHands((0,0,0),(0,0,0),(0,0,0),(0,0,0))

initial_state = {**initial_agent_position, **initial_hands_position}
for k, v in initial_state.items():
    CURRENT_AGENT_STATE[k] = v
print("Initial State: ", CURRENT_AGENT_STATE)

while ON_PLAY:
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

    imageb64 = base64.b64encode(imagebytes).decode('utf-8')

    if at_init_state:
        post = {
            'task': MAIN_TASK,
            'image': imageb64,
            'state': CURRENT_AGENT_STATE,
        }
        at_init_state = False
    else:
        post = {
            'image': imageb64,
            'state': CURRENT_AGENT_STATE,
        }
    logger.info(f"Time step: {time_step}")

    response = requests.post(
        INFERENCE_API,
        data=post,
    )

    if response.status_code != 200:
        break

    response = response.json()['response']
    extracted = re.search(EXTRACT_JSON_PATTERN, response)[1]
    extracted = ast.literal_eval(extracted)

    reasoning = extracted.get('reasoning', 'No Reasoning.')
    actions = extracted.get('actions', 'No Actions.')
    times = extracted.get('times', 'No Times.')
    notes = extracted.get('notes', 'No Notes.')
    box2d = extracted.get('box2d', 'No Box 2D data.')

    actions_times = zip(actions, times)

    print('#' * 50)
    print(f'Reasoning: {reasoning}')
    print('-' * 50)
    print(f'Notes: {notes}')
    print('-' * 50)
    print(f'Object of Interest ([ymin, xmin, ymax, xmax]): {box2d}')
    print('-' * 50)
    logger.info(f'Reasoning: {reasoning}')
    logger.info(f'Notes: {notes}')
    logger.info(f'Object of Interest ([ymin, xmin, ymax, xmax]): {box2d}')

    for action, time in actions_times:
        action_name = action
        if action == 'STOP':
            ON_PLAY = False

        try:
            if not ON_PLAY:
                print(f"'STOP' action detected. Stopping the loop...")
                logger.info(f"'STOP' action detected. Stopping the loop...")
                break
            action_func = actions_ref[action]
            for cnt, t in enumerate(range(int(time))):
                current_state_log = f"State @ timestep {time_step}\n"
                state_vars = action_func()
                for k,v in state_vars.items():
                    CURRENT_AGENT_STATE[k] = v
                for k,v in CURRENT_AGENT_STATE.items():
                    current_state_log += f"{k}: {v}\n"
                current_state_log += "\n"
                with open(f"agent_states_{run_name}.txt", "w") as f:
                    f.write(current_state_log)
                    
        except KeyError:
            raise KeyError(f"Action '{action_name}' not found in `actions_ref`.")
        finally:
            if ON_PLAY:
                print(f"ACTION: {action_name} executed for {time} times.")
                logger.info(f"ACTION: {action_name} executed for {time} times.")

    print('#' * 50)
    time_step += 1
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
