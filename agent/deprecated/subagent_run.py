"""
Multi-step task orchestrator for the agent agent.

Decomposes a complex task into ordered subtasks, runs each as a separate
EmbodiedAgent loop, generates a handoff summary after each subtask, and
passes that context into the next subtask.

Usage:
    python subagent_run.py "pick up the milk and bring it to the counter"
    python subagent_run.py "pick up the milk and bring it to the counter" soda_layout2

Each subtask creates a fresh EmbodiedAgent (resetting VLM history and memories).
Context is passed forward via the augmented task string.
"""

import ast
import base64
import json
import os
import re
import sys
import time
from sim import chime  # cross-platform run-completion beep (was winsound: Windows-only)
from datetime import datetime

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Repo-root config.env, resolved from __file__ so it loads regardless of CWD.
# (this file lives in agent/deprecated/, so the repo root is three levels up)
load_dotenv(Path(__file__).resolve().parent.parent.parent / 'config.env')

# agent/ on sys.path so the runtime modules below still import from deprecated/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.hand_reset import reset_hands_in_front2
from sim.env import (
    _REQUEST_SCREENSHOT_,
    TransformAgent,
    TransformHands,
    init_logger,
)
from toolset.actions import (
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
    MANIPULATION_ACTIONS_REF,
)
from agent_core.agent import EmbodiedAgent, OpenRouterConfig

ORCHESTRATOR_MODEL = "google/gemini-3.1-pro-preview"
EXTRACTABLE_JSON = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

VLM_CONFIG = OpenRouterConfig(
    model_id='google/gemini-3.1-pro-preview',
    temperature=0.5,
    mode='lean',
)
ASSOCIATIVE_CONFIG = OpenRouterConfig(
    model_id='google/gemini-3.1-pro-preview',
    temperature=0.3,
    mode='lean',
)


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
    system = (
        "You are a task planner for an Embodied AI Agent in a 3D convenience "
        "store simulation. The agent can navigate, locate items on shelves, "
        "pick them up, carry them, and bring them to locations like the checkout counter. "
        "Given a complex multi-step task, decompose it into a short ordered list "
        "of simple, self-contained subtasks. Each subtask should:\n"
        "  - Be completable in a single continuous agent run.\n"
        "  - End in a clear, verifiable physical state change.\n"
        "  - Reference what the agent is currently holding when relevant.\n"
        "  - Name locations only as the task or the store memory names them (e.g. 'Checkpoint 32', "
        "'the checkout counter'). Never invent shelf numbers or location names.\n"
        "Return ONLY a JSON array of subtask strings — no other text.\n\n"
        "Example input: \"pick up the milk and bring it to the counter\"\n"
        "Example output: "
        "[\"Pick up the milk.\", "
        "\"Carry the held milk to the checkout counter and place it down.\"]"
        "\n\nIf a task is already simple (e.g. 'pick up the milk'), just return it as a single-item array."
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


def verify_subtask(client: OpenAI, subtask: str, final_state: dict, step_notes: list) -> dict:
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
    user = (
        f"Subtask: {subtask}\n\n"
        f"Final agent state:\n{json.dumps(final_state, indent=2, default=str)}\n\n"
        f"Step notes (last 5 steps):\n{json.dumps(step_notes[-5:], indent=2)}"
    )
    raw = _llm_call(client, system, user)
    obj_match = re.search(r'\{[\s\S]*\}', raw)
    if not obj_match:
        return {"completed": False, "reason": "Could not parse verification response — treating as NOT complete (fail-closed)."}
    try:
        return json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return {"completed": False, "reason": "Could not parse verification response — treating as NOT complete (fail-closed)."}


def generate_handoff_summary(client: OpenAI, completed_subtask: str, final_state: dict, step_notes: list) -> str:
    system = (
        "You are a handoff reporter for an Embodied AI Agent in a 3D "
        "convenience store simulation. Given a completed subtask, the agent's "
        "final physical state, and a log of its step-by-step notes, write a "
        "concise handoff summary for the NEXT agent instance. Include:\n"
        "  - Current agent position (translate the coordinates into plain English "
        "    if possible, e.g. 'near Checkpoint 32').\n"
        "  - What each hand is holding (gripped items, or empty).\n"
        "  - Key environmental observations relevant to upcoming tasks.\n"
        "  - Anything unusual or important the next agent should know.\n"
        "Keep it under 150 words. Be specific and factual."
    )
    user = (
        f"Completed subtask: {completed_subtask}\n\n"
        f"Final agent state:\n{json.dumps(final_state, indent=2, default=str)}\n\n"
        f"Step notes (last 5 steps):\n{json.dumps(step_notes[-5:], indent=2)}"
    )
    return _llm_call(client, system, user)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

def dispatch_action(action: str, time_units: int, notes: dict, mode: str = None) -> None:
    action = action.strip()

    # Manipulation actions are gated to manipulation mode. Pre-6.1 they no-op'd elsewhere (hands
    # inactive); as of Phase 6.1 hands stay ACTIVE at REST in every mode, so firing one outside
    # manipulation would perturb the hand/carried item - the gate stands for mode coherence.
    if action in MANIPULATION_ACTIONS_REF and mode is not None and mode != "manipulation":
        print(f"[BLOCKED] '{action}' only works in *manipulation* mode (current mode: {mode}); "
              f"the hands are inactive otherwise. Skipped.")
        return

    if action in NAVIGATION_ACTIONS_REF:
        action_ref = NAVIGATION_ACTIONS_REF[action]
    elif action in PERCEPTION_ACTIONS_REF:
        action_ref = PERCEPTION_ACTIONS_REF[action]
    elif action in MANIPULATION_ACTIONS_REF:
        action_ref = MANIPULATION_ACTIONS_REF[action]
    else:
        print(f"[WARN] Unknown action skipped: {action}")
        return

    main_goal  = notes.get('main_goal', '')
    sub_goals  = notes.get('sub_goal', '')
    key_info   = notes.get('key_info', '')
    checklist  = notes.get('checklist', '')

    if action == "center_object_on_screen":
        target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
        action_ref(target_info)
    elif action in ("retrieve_item", "approach_object"):
        action_ref(main_goal)
    else:
        action_ref(time_units)


# ---------------------------------------------------------------------------
# Subtask runner
# ---------------------------------------------------------------------------

def _get_screenshot(run_entry: str, time_step: int) -> bytes:
    if run_entry:
        directory_path = os.path.join('screenshots', 'SIM_RUNS', run_entry)
        os.makedirs(directory_path, exist_ok=True)
        existing = [fn for fn in os.listdir(directory_path) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
        prefix = str(len(existing) + 1).zfill(6)
        suffix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        return _REQUEST_SCREENSHOT_(prefix=prefix, suffix=suffix,
                                    folder_name=directory_path, save_image=True)['image']
    return _REQUEST_SCREENSHOT_()['image']


def _fresh_agent_state() -> dict:
    agent_pos  = TransformAgent((0, 0, 0), (0, 0, 0))
    hands_pos  = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    state = {
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
        "mode": "perception",
    }
    for k, v in {**agent_pos, **hands_pos}.items():
        state[k] = v
    return state


def run_subtask(subtask: str, agent: EmbodiedAgent, context: str = "",
                future_subtasks: list = None, run_entry: str = "") -> dict:
    """
    Run a single subtask as a self-contained EmbodiedAgent loop.

    Returns a dict with:
        final_state  — CURRENT_AGENT_STATE dict at the moment the agent halted
        step_notes   — list of notes dicts collected from every step
    """
    parts = [f"CURRENT GOAL: {subtask}"]
    if context:
        parts.append(f"CONTEXT FROM PREVIOUS STEP:\n{context}")
    else:
        parts.append("CONTEXT FROM PREVIOUS STEP: None — this is the first subtask.")
    if future_subtasks:
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(future_subtasks))
        parts.append(
            "FUTURE GOALS (for awareness only — do NOT pursue these yet; "
            "store any observations that would help future agents accomplish them):\n"
            + numbered
        )
    else:
        parts.append("FUTURE GOALS: None — this is the final subtask.")

    augmented_task = "\n\n".join(parts)

    print(f"\n[SUBTASK] {subtask}")
    if context:
        print(f"[CONTEXT] {context[:120]}{'...' if len(context) > 120 else ''}")

    # Reset only conversation history — semantic/episodic memory carry over
    agent.vlm_agent.reset_history()

    current_state = _fresh_agent_state()
    step_notes: list = []
    time_step = 1

    while True:
        imagebytes = _get_screenshot(run_entry, time_step)
        imageb64 = base64.b64encode(imagebytes).decode('utf-8')

        request = {
            'task': augmented_task,
            'image': imageb64,
            'state': current_state,
        }

        response = agent.execute_lean(request, time_step)
        print(f"[STEP {time_step}] Response: {response}")

        if response['halt']:
            grip_active = (current_state.get('leftGrippedState', False) or
                           current_state.get('rightGrippedState', False))
            is_pickup = any(kw in subtask.lower() for kw in ['pick up', 'grab', 'get', 'take', 'lift'])
            is_drop   = any(kw in subtask.lower() for kw in ['drop', 'place', 'put down', 'set down', 'release', 'leave'])

            if is_pickup and not grip_active:
                print("[GUARD] STOP blocked — pick-up task but nothing is gripped. Continuing...")
                # Un-halt: patch response to keep going by continuing the loop
                # The agent flagged halt on its own; force another step
                time_step += 1
                continue

            if is_drop and grip_active:
                print("[GUARD] STOP blocked — drop task but hand is still gripping. Continuing...")
                time_step += 1
                continue

            print("[SUBTASK DONE] Agent halted.")
            break

        main_response_text = response['text']
        agent_mode = response['agent_mode']
        current_state['mode'] = agent_mode

        match = re.search(agent.vlm_agent.extractable_json_structured_output, main_response_text)
        if not match:
            print("[ERROR] No action JSON in response. Ending subtask.")
            break

        main_response = ast.literal_eval(match.group(1))

        reasoning = main_response['reasoning']
        actions   = main_response['actions']
        times     = main_response['times']
        notes     = main_response['notes']
        main_goal = notes['main_goal']
        sub_goals = notes['sub_goal']
        key_info  = notes['key_info']
        status    = notes['status']
        checklist = notes['checklist']
        step_notes.append(notes)

        print(f"[STEP {time_step}] mode={agent_mode} actions={list(zip(actions, times))}")

        time_step += 1

        for action, t in zip(actions, times):
            dispatch_action(action, int(t), notes, mode=agent_mode)

        # Refresh state from the environment
        updated_agent = TransformAgent((0, 0, 0), (0, 0, 0))
        updated_hands = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        for k, v in {**updated_agent, **updated_hands}.items():
            current_state[k] = v

    return {"final_state": current_state, "step_notes": step_notes}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrate(task: str, run_entry: str = ""):
    client = _llm_client()

    # Startup hand-centering DISABLED (user, 2026-07-21): Unity now owns the default hand pose.
    # reset_hands_in_front2(extra_elevation=-0.1, hand="left")

    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_name = f"subagent-{run_entry}-{timestamp}" if run_entry else f"subagent-{timestamp}"
    init_logger(run_name=log_name)

    agent = EmbodiedAgent(
        vlm_config=VLM_CONFIG,
        associative_config=ASSOCIATIVE_CONFIG,
        mode='lean',
    )

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
            future = subtasks[i + 1:] if i + 1 < len(subtasks) else []
            print(f"\n[ORCHESTRATOR] ── Subtask {i + 1}/{len(subtasks)} ──")

            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    print(f"[ORCHESTRATOR] Retry {attempt}/{max_attempts} for: {subtask}")

                result = run_subtask(subtask, agent, context=context,
                                     future_subtasks=future, run_entry=run_entry)

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
                    context = (
                        f"RETRY CONTEXT: The previous attempt at this subtask did not complete it. "
                        f"Reason: {verdict['reason']} — try a different approach."
                    )
                else:
                    print("[ORCHESTRATOR] Max attempts reached for subtask. Moving on.")

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
        chime.beep()


if __name__ == "__main__":
    _task = sys.argv[1] if len(sys.argv) > 1 else input("Task: ")
    _run_entry = sys.argv[2] if len(sys.argv) > 2 else ""
    orchestrate(_task, run_entry=_run_entry)
