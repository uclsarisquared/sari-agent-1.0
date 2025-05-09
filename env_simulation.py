import requests
import base64
import re
import ast
import sys

from env import _REQUEST_SCREENSHOT_
from actions import actions_ref

INFERECE_API = "http://localhost:8005/predict"
EXTRACT_JSON_PATTERN = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

MAIN_TASK = sys.argv[1]
ON_PLAY = True

at_init_state = True

while ON_PLAY:
    imagebytes = _REQUEST_SCREENSHOT_()['image']
    imageb64 = base64.b64encode(imagebytes).decode('utf-8')

    if at_init_state:
        post = {
            'task': MAIN_TASK,
            'image': imageb64,
        }
        at_init_state = False
    else:
        post = {
            'image': imageb64,
        }

    response = requests.post(
        INFERECE_API,
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

    for action, time in actions_times:
        action_name = action
        if action == 'STOP':
            ON_PLAY = False

        try:
            if not ON_PLAY:
                print(f"'STOP' action detected. Stopping the loop...")
                break
            action = actions_ref[action]
            for cnt, t in enumerate(range(int(time))):
                action()
        except KeyError:
            raise KeyError(f"Action '{action_name}' not found in `actions_ref`.")
        finally:
            if ON_PLAY:
                print(f"ACTION: {action_name} executed for {time} times.")

    print('#' * 50)
