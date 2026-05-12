import base64
import re
import ast
import sys
import time  # Added time module
import winsound # Added winsound module for beep at the end
import requests
import env

SERVER_URL = "http://localhost:8005/predict"

ACTION_MAP = {
    "MOVE_FWD":     lambda t: env.move_forward(t),
    "MOVE_BACK":    lambda t: env.move_backward(t),
    "MOVE_LEFT":    lambda t: env.move_left(t),
    "MOVE_RIGHT":   lambda t: env.move_right(t),
    "PAN_LEFT":     lambda t: env.pan_left(t),
    "PAN_RIGHT":    lambda t: env.pan_right(t),
    "TILT_UP":      lambda t: env.pan_up(t),
    "TILT_DOWN":    lambda t: env.pan_down(t),
    "EXTEND_LEFT":  lambda t: env.extend_left_hand_forward(t),
    "PULL_LEFT":    lambda t: env.pull_left_hand_backward(t),
    "EXTEND_RIGHT": lambda t: env.extend_right_hand_forward(t),
    "PULL_RIGHT":   lambda t: env.pull_right_hand_backward(t),
    "GRIP_LEFT":    lambda t: env.ToggleLeftGrip(),
    "GRIP_RIGHT":   lambda t: env.ToggleRightGrip(),
}

EXTRACTABLE_JSON = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)


def get_state():
    state = env.RequestJson()
    screenshot = env.RequestScreenshot()
    image_b64 = base64.b64encode(screenshot['image']).decode('utf-8')
    return state, image_b64


def dispatch(actions, times):
    for act, t in zip(actions, times):
        if act == "STOP":
            return True
        if act in ACTION_MAP:
            ACTION_MAP[act](t)
        else:
            print(f"[WARN] Unknown action skipped: {act}")
    return False


def run(task):
    actions_history = []
    start_time = time.time()  # Start the clock
    print("Starting at:", time.ctime(start_time))  # Print start time
    print(f"--- Starting Task: {task} ---")

    try:
        while True:
            state, image_b64 = get_state()

            payload = {
                "task": task,
                "state": state,
                "image": image_b64,
                "actions": actions_history,
            }

            resp = requests.post(SERVER_URL, json=payload, timeout=180)
            resp.raise_for_status()

            response = resp.json()['response']

            if isinstance(response, str):
                print("[DONE] Agent returned STOP.")
                break

            text = response['text']

            match = re.search(EXTRACTABLE_JSON, text)
            if not match:
                print("[ERROR] No action JSON in response. Stopping.")
                break

            action_json = ast.literal_eval(match.group(1))
            actions = action_json['actions']
            times = action_json['times']

            print(f"[STEP] {list(zip(actions, times))}")

            stopped = dispatch(actions, times)
            actions_history.append({'actions': actions, 'times': times})

            if stopped:
                print("[DONE] STOP action dispatched.")
                break
    finally:
        # End the clock and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        print("-" * 30)
        print(f"Runtime: {duration:.2f} seconds")
        print("-" * 30)
        winsound.Beep(392, 1000)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Task: ")
    run(task)