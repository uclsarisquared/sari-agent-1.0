"""
Multi-step task orchestrator for sari-agent-1.0.

Decomposes a complex task into ordered subtasks, runs each as a separate
agent loop, generates a handoff summary after each subtask, and passes that
context into the next subtask so the next agent instance knows what happened.

Usage:
    python subagent_run.py "pick up the milk and bring it to the counter"
    python subagent_run.py "grab a can of tuna and read the label at the counter"

Note: The inference server (server.py) maintains a global timestep and
conversation history that persists across subtasks. Each subtask's steps
append to the server's running context. To fully isolate memory between
subtasks, restart the inference server between orchestration runs or add a
reset endpoint to server.py.
"""

import ast
import base64
import json
import os
import re
import sys
import time
import winsound

import requests
from dotenv import load_dotenv
from openai import OpenAI

import env

load_dotenv('api.env')

SERVER_URL = "http://localhost:8005/predict"
ORCHESTRATOR_MODEL = "google/gemini-3.1-pro-preview"
EXTRACTABLE_JSON = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

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


# ---------------------------------------------------------------------------
# Orchestrator LLM helpers
# ---------------------------------------------------------------------------

def _llm_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )


def _llm_call(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def decompose_task(client: OpenAI, task: str) -> list:
    """
    Ask the LLM to split a complex task into an ordered list of simple
    subtasks, each completable by a single agent loop ending in STOP.
    """
    system = (
        "You are a task planner for an Embodied AI Agent in a 3D convenience "
        "store simulation. The agent can navigate, locate items on shelves, "
        "pick them up, carry them, and bring them to locations like the counter. "
        "Given a complex multi-step task, decompose it into a short ordered list "
        "of simple, self-contained subtasks. Each subtask should:\n"
        "  - Be completable in a single continuous agent run.\n"
        "  - End in a clear, verifiable physical state change.\n"
        "  - Reference what the agent is currently holding when relevant.\n"
        "Return ONLY a JSON array of subtask strings — no other text.\n\n"
        "Example input: \"pick up the milk and bring it to the counter\"\n"
        "Example output: "
        "[\"Pick up the milk from Shelf 9.\", "
        "\"Carry the held milk to the counter near the cash register and place it down.\"]"
        "\n\n If a task is already simple (e.g. 'pick up the milk'), just return it as a single-item array."
    )
    raw = _llm_call(client, system, f"Task: {task}")
    array_match = re.search(r'\[[\s\S]*\]', raw)
    if not array_match:
        print("[WARN] Decomposition returned no array — treating as a single subtask.")
        return [task]
    try:
        return json.loads(array_match.group(0))
    except json.JSONDecodeError:
        print("[WARN] Could not parse decomposition JSON — treating as a single subtask.")
        return [task]


def verify_subtask(
    client: OpenAI,
    subtask: str,
    final_state: dict,
    step_notes: list,
) -> dict:
    """
    Ask the orchestrator LLM whether the subtask was actually completed.
    Returns {"completed": bool, "reason": str}.
    """
    system = (
        "You are a task verifier for an Embodied AI Agent in a 3D convenience "
        "store simulation. Given a subtask, the agent's final physical state, "
        "and its step notes, determine whether the subtask was fully completed.\n\n"
        "Use the state data as ground truth:\n"
        "  - leftGrippedState / rightGrippedState: whether each hand is gripping\n"
        "  - leftHoveredObject / rightHoveredObject: item nearest each hand\n"
        "  - translation: agent position\n\n"
        "Return ONLY a JSON object with two fields:\n"
        "  \"completed\": true or false\n"
        "  \"reason\": one sentence explaining why"
    )
    try:
        state_display = json.dumps(json.loads(final_state) if isinstance(final_state, str) else final_state, indent=2)
    except (json.JSONDecodeError, TypeError):
        state_display = str(final_state)

    user = (
        f"Subtask: {subtask}\n\n"
        f"Final agent state:\n{state_display}\n\n"
        f"Step notes (last 5 steps):\n{json.dumps(step_notes[-5:], indent=2)}"
    )
    raw = _llm_call(client, system, user)
    obj_match = re.search(r'\{[\s\S]*\}', raw)
    if not obj_match:
        return {"completed": True, "reason": "Could not parse verification response — assuming complete."}
    try:
        return json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return {"completed": True, "reason": "Could not parse verification response — assuming complete."}


def generate_handoff_summary(
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    step_notes: list,
) -> str:
    """
    After a subtask completes, ask the LLM to produce a concise handoff
    summary that the next agent instance will receive as context.
    """
    system = (
        "You are a handoff reporter for an Embodied AI Agent in a 3D "
        "convenience store simulation. Given a completed subtask, the agent's "
        "final physical state, and a log of its step-by-step notes, write a "
        "concise handoff summary for the NEXT agent instance. Include:\n"
        "  - Current agent position (translate the coordinates into plain English "
        "    if possible, e.g. 'near Shelf 9').\n"
        "  - What each hand is holding (gripped items, or empty).\n"
        "  - Key environmental observations relevant to upcoming tasks.\n"
        "  - Anything unusual or important the next agent should know.\n"
        "Keep it under 150 words. Be specific and factual."
    )
    try:
        state_display = json.dumps(json.loads(final_state) if isinstance(final_state, str) else final_state, indent=2)
    except (json.JSONDecodeError, TypeError):
        state_display = str(final_state)

    user = (
        f"Completed subtask: {completed_subtask}\n\n"
        f"Final agent state:\n{state_display}\n\n"
        f"Step notes (last 5 steps):\n{json.dumps(step_notes[-5:], indent=2)}"
    )
    return _llm_call(client, system, user)


# ---------------------------------------------------------------------------
# Subtask runner
# ---------------------------------------------------------------------------

def get_state():
    state = env.RequestJson()
    screenshot = env.RequestScreenshot()
    image_b64 = base64.b64encode(screenshot['image']).decode('utf-8')
    return state, image_b64


def dispatch(actions, times) -> bool:
    """Execute actions. Returns True if STOP was encountered."""
    for act, t in zip(actions, times):
        if act == "STOP":
            return True
        if act in ACTION_MAP:
            ACTION_MAP[act](t)
        else:
            print(f"[WARN] Unknown action skipped: {act}")
    return False


def run_subtask(subtask: str, context: str = "", next_task_hint: str = "") -> dict:
    """
    Run a single subtask as a self-contained agent loop.

    subtask:       The immediate goal this agent instance should accomplish.
    context:       Handoff summary from the previous subtask (empty on first).
    next_task_hint: The next subtask, given for awareness only — the agent is
                   explicitly told not to pursue it yet. Empty on the last step.

    Returns a dict with:
        final_state  — the env.RequestJson() dict at the moment STOP was issued
        step_notes   — list of 'notes' dicts collected from every step response
    """
    parts = [f"CURRENT GOAL: {subtask}"]

    if context:
        parts.append(f"CONTEXT FROM PREVIOUS STEP:\n{context}")
    else:
        parts.append("CONTEXT FROM PREVIOUS STEP: None — this is the first subtask.")

    if next_task_hint:
        parts.append(
            "UPCOMING GOAL (for your awareness only — do NOT pursue this yet, "
            f"just keep it in mind when deciding where to stop):\n{next_task_hint}"
        )
    else:
        parts.append("UPCOMING GOAL: None — this is the final subtask.")

    augmented_task = "\n\n".join(parts)

    actions_history = []
    step_notes = []
    final_state = {}
    first_step = True

    print(f"\n[SUBTASK] {subtask}")
    if context:
        print(f"[CONTEXT] {context[:120]}{'...' if len(context) > 120 else ''}")

    while True:
        state, image_b64 = get_state()
        final_state = state

        payload = {
            "task": augmented_task,
            "state": state,
            "image": image_b64,
            "actions": actions_history,
            "reset_episode": first_step,
        }
        first_step = False

        resp = requests.post(SERVER_URL, json=payload, timeout=180)
        resp.raise_for_status()
        response = resp.json()['response']

        # The server returns a plain string (not a dict) on STOP
        if isinstance(response, str):
            print(f"[SUBTASK DONE] Agent returned STOP.")
            break

        text = response['text']
        match = re.search(EXTRACTABLE_JSON, text)
        if not match:
            print("[ERROR] No action JSON in response. Ending subtask.")
            break

        action_json = ast.literal_eval(match.group(1))
        actions = action_json['actions']
        times   = action_json['times']
        notes   = action_json.get('notes', {})

        step_notes.append(notes)
        print(f"[STEP] {list(zip(actions, times))}")

        stopped = dispatch(actions, times)
        actions_history.append({'actions': actions, 'times': times})

        if stopped:
            # Re-read state after actions have executed (state was captured before dispatch)
            post_state_raw, _ = get_state()
            final_state = post_state_raw

            try:
                post_state = json.loads(post_state_raw) if isinstance(post_state_raw, str) else post_state_raw
            except (json.JSONDecodeError, TypeError):
                post_state = {}

            grip_active = post_state.get('leftGrippedState', False) or post_state.get('rightGrippedState', False)
            is_pickup = any(kw in subtask.lower() for kw in ['pick up', 'grab', 'get', 'take', 'lift'])
            is_drop   = any(kw in subtask.lower() for kw in ['drop', 'place', 'put down', 'set down', 'release', 'leave'])

            if is_pickup and not grip_active:
                print("[GUARD] STOP blocked — pick-up task but nothing is gripped. Continuing...")
                continue
            if is_drop and grip_active:
                print("[GUARD] STOP blocked — drop task but hand is still gripping. Continuing...")
                continue

            print(f"[SUBTASK DONE] STOP accepted.")
            break

    return {"final_state": final_state, "step_notes": step_notes}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrate(task: str):
    client = _llm_client()

    start_time = time.time()
    print(f"Starting at: {time.ctime(start_time)}")
    print(f"\n[ORCHESTRATOR] Task: {task}")
    print("[ORCHESTRATOR] Decomposing into subtasks...")
    subtasks = decompose_task(client, task)

    print(f"[ORCHESTRATOR] {len(subtasks)} subtask(s):")
    for i, st in enumerate(subtasks, 1):
        print(f"  {i}. {st}")

    context = ""
    max_attempts = 3
    try:
        for i, subtask in enumerate(subtasks):
            next_hint = subtasks[i + 1] if i + 1 < len(subtasks) else ""
            print(f"\n[ORCHESTRATOR] ── Subtask {i + 1}/{len(subtasks)} ──")

            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    print(f"[ORCHESTRATOR] Retry {attempt}/{max_attempts} for: {subtask}")

                result = run_subtask(subtask, context=context, next_task_hint=next_hint)

                print("[ORCHESTRATOR] Verifying subtask completion...")
                verdict = verify_subtask(
                    client,
                    subtask=subtask,
                    final_state=result['final_state'],
                    step_notes=result['step_notes'],
                )
                print(f"[VERIFICATION] completed={verdict['completed']} — {verdict['reason']}")

                if verdict['completed']:
                    break

                if attempt < max_attempts:
                    # Feed the failure reason back as context for the retry
                    context = (
                        f"RETRY CONTEXT: The previous attempt at this subtask did not complete it. "
                        f"Reason: {verdict['reason']} — try a different approach."
                    )
                else:
                    print(f"[ORCHESTRATOR] Max attempts reached for subtask. Moving on.")

            # Generate handoff summary for all but the last subtask
            if i + 1 < len(subtasks):
                print("[ORCHESTRATOR] Generating handoff summary...")
                context = generate_handoff_summary(
                    client,
                    completed_subtask=subtask,
                    final_state=result['final_state'],
                    step_notes=result['step_notes'],
                )
                print(f"[HANDOFF SUMMARY]\n{context}\n")

        print("\n[ORCHESTRATOR] All subtasks complete.")
    finally:
        duration = time.time() - start_time
        print("-" * 30)
        print(f"Runtime: {duration:.2f} seconds")
        print("-" * 30)
        winsound.Beep(392, 1000)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Task: ")
    orchestrate(task)
