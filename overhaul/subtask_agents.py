"""
subtask_agents.py

Multi-step orchestrator structured like env_simulation.py per subtask, with
shared semantic memory across all subtask agents.

Design:
  - One EmbodiedAgent is created for the entire task run.
  - Between subtasks, only VLM conversation history is reset; semantic and
    episodic memory carry over so each agent builds on what prior ones learned.
  - After each subtask, a comprehensive findings summary is generated and
    appended to a cumulative context passed to all subsequent subtask agents.

Usage:
    python subtask_agents.py "pick up the milk and bring it to the counter"
    python subtask_agents.py "pick up the milk and bring it to the counter" soda_layout2
"""

import ast
import base64
import json
import os
import re
import sys
import time
import winsound
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('../api.env')

from hand_reset import reset_hands_in_front2
from env import (
    _REQUEST_SCREENSHOT_,
    TransformAgent,
    TransformHands,
    init_logger,
)
from actions import (
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
    MANIPULATION_ACTIONS_REF,
)
from agent import EmbodiedAgent, OpenRouterConfig

ORCHESTRATOR_MODEL = "Qwen/Qwen3.6-27B"  # UCL qwen (OpenRouter retired 2026-07-19)
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
    from agent import _ucl_creds
    host, key = _ucl_creds()
    return OpenAI(base_url=f"http://{host}:8000/v1", api_key=key)


def _llm_call(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
        timeout=120,
    )
    return resp.choices[0].message.content


def decompose_task(client: OpenAI, task: str) -> list:
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


def generate_findings_summary(
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    new_semantic_entries: str,
) -> str:
    """
    Comprehensive summary of everything the agent found/learned during a subtask.
    Passed to the orchestrator so all future subtask agents receive accumulated context.
    """
    system = (
        "You are a findings reporter for an Embodied AI Agent in a 3D convenience "
        "store simulation. After a subtask completes, produce a comprehensive findings "
        "summary for future agent instances. Include ALL of the following:\n"
        "  1. POSITION: Current agent position in plain English (near which shelf/counter).\n"
        "  2. HANDS: What each hand is holding (gripped items, or empty).\n"
        "  3. OBJECTS LOCATED: Every object/item seen and its approximate shelf or position.\n"
        "  4. NAVIGATION INSIGHTS: Which paths/routes worked; where the agent got stuck or lost.\n"
        "  5. SEMANTIC LEARNINGS: Key facts about the store environment learned this subtask.\n"
        "  6. WHAT TO AVOID: Any approaches that failed or cost unnecessary time.\n"
        "  7. UPCOMING TASK PREP: Specific observations that will help with future subtasks.\n"
        "Be comprehensive and factual. Future agents cannot re-explore what you already found, "
        "so document every useful detail."
    )
    user = (
        f"Completed subtask: {completed_subtask}\n\n"
        f"Final agent state:\n{json.dumps(final_state, indent=2, default=str)}\n\n"
        f"New semantic memory entries learned during this subtask:\n{new_semantic_entries}"
    )
    return _llm_call(client, system, user)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

def dispatch_action(action: str, time_units: int, notes: dict, inline_arg: str = None) -> dict:
    """Execute one action. Returns a result dict; grab actions include a 'gripped' key."""
    if action in NAVIGATION_ACTIONS_REF:
        action_ref = NAVIGATION_ACTIONS_REF[action]
    elif action in PERCEPTION_ACTIONS_REF:
        action_ref = PERCEPTION_ACTIONS_REF[action]
    elif action in MANIPULATION_ACTIONS_REF:
        action_ref = MANIPULATION_ACTIONS_REF[action]
    else:
        print(f"[WARN] Unknown action skipped: {action}")
        return {}

    main_goal = notes.get('main_goal', '')
    sub_goals = notes.get('sub_goal', '')
    key_info  = notes.get('key_info', '')
    checklist = notes.get('checklist', '')

    if action == "center_object_on_screen":
        target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
        return action_ref(target_info) or {}
    elif action in ("retrieve_item", "approach_object"):
        return action_ref(main_goal) or {}
    elif action in ("grab_item_in_view_right", "grab_item_in_view_left"):
        item_name = notes.get('item_name', '') or inline_arg or main_goal
        result = action_ref(item_name) or {}
        if not result.get('gripped', False):
            print(f"[GRAB] Grab failed for '{item_name}' — agent should reposition.")
        return result
    else:
        return action_ref(time_units) or {}


# ---------------------------------------------------------------------------
# Subtask runner
# ---------------------------------------------------------------------------

def _get_screenshot(run_entry: str, subtask_idx: int = 0) -> bytes:
    if run_entry:
        directory_path = os.path.join('screenshots', 'SIM_RUNS2', run_entry, f'subtask_{subtask_idx}')
        os.makedirs(directory_path, exist_ok=True)
        existing = [fn for fn in os.listdir(directory_path) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]
        prefix = str(len(existing) + 1).zfill(6)
        suffix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        return _REQUEST_SCREENSHOT_(prefix=prefix, suffix=suffix,
                                    folder_name=directory_path, save_image=True)['image']
    return _REQUEST_SCREENSHOT_()['image']


def _fresh_agent_state() -> dict:
    agent_pos = TransformAgent((0, 0, 0), (0, 0, 0))
    hands_pos = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
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
        "last_grab_failed": False,
        "mode": "perception",
    }
    for k, v in {**agent_pos, **hands_pos}.items():
        state[k] = v
    return state


def run_subtask(
    subtask: str,
    agent: EmbodiedAgent,
    context: str = "",
    future_subtasks: list = None,
    run_entry: str = "",
    subtask_idx: int = 0,
) -> dict:
    """
    Run a single subtask as a self-contained EmbodiedAgent loop.

    Semantic memory on the agent is preserved from prior subtasks.
    Only VLM conversation history is reset between subtasks.

    Returns:
        final_state          — CURRENT_AGENT_STATE dict at the moment the agent halted
        new_semantic_entries — semantic memory text appended during this subtask
    """
    parts = [f"CURRENT GOAL: {subtask}"]
    if context:
        parts.append(f"CONTEXT FROM PREVIOUS SUBTASKS:\n{context}")
    else:
        parts.append("CONTEXT FROM PREVIOUS SUBTASKS: None — this is the first subtask.")
    if future_subtasks:
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(future_subtasks))
        parts.append(
            "FUTURE GOALS (for awareness only — do NOT pursue these yet; "
            "record any observations that would help future agents accomplish them):\n"
            + numbered
        )
    else:
        parts.append("FUTURE GOALS: None — this is the final subtask.")

    augmented_task = "\n\n".join(parts)

    print(f"\n[SUBTASK] {subtask}")
    if context:
        print(f"[CONTEXT] {context[:120]}{'...' if len(context) > 120 else ''}")

    # Reset conversation history only; semantic and episodic memory persist across subtasks
    agent.vlm_agent.reset_history()
    semantic_memory_before = agent.vlm_agent.base_semantic_memory

    current_state = _fresh_agent_state()
    time_step = 1

    while True:
        imagebytes = _get_screenshot(run_entry, subtask_idx=subtask_idx)
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

        actions = main_response['actions']
        times   = main_response['times']
        notes   = main_response['notes']

        print(f"[STEP {time_step}] mode={agent_mode} actions={list(zip(actions, times))}")
        time_step += 1

        grab_failed = False
        for action, t in zip(actions, times):
            raw_action = action.strip()
            inline_arg = None
            inline_match = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', raw_action)
            if inline_match:
                raw_action, inline_arg = inline_match.group(1), inline_match.group(2)
            result = dispatch_action(raw_action, int(t), notes, inline_arg=inline_arg)
            if raw_action in ("grab_item_in_view_right", "grab_item_in_view_left"):
                if not result.get('gripped', False):
                    grab_failed = True
        current_state['last_grab_failed'] = grab_failed

        # Refresh state from the environment after all actions have executed
        updated_agent = TransformAgent((0, 0, 0), (0, 0, 0))
        updated_hands = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        for k, v in {**updated_agent, **updated_hands}.items():
            current_state[k] = v

    new_semantic_entries = agent.vlm_agent.base_semantic_memory[len(semantic_memory_before):]

    return {
        "final_state": current_state,
        "new_semantic_entries": new_semantic_entries,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrate(task: str, run_entry: str = ""):
    client = _llm_client()

    reset_hands_in_front2(extra_elevation=-0.1, hand="left")

    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_name = f"subtask-{run_entry}-{timestamp}" if run_entry else f"subtask-{timestamp}"
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

    # cumulative_context grows with each subtask's findings summary
    cumulative_context = ""

    try:
        for i, subtask in enumerate(subtasks):
            future = subtasks[i + 1:] if i + 1 < len(subtasks) else []
            print(f"\n[ORCHESTRATOR] ── Subtask {i + 1}/{len(subtasks)} ──")

            result = run_subtask(
                subtask, agent,
                context=cumulative_context,
                future_subtasks=future,
                run_entry=run_entry,
                subtask_idx=i + 1,
            )

            if i + 1 < len(subtasks):
                print("[ORCHESTRATOR] Generating findings summary...")
                findings = generate_findings_summary(
                    client,
                    completed_subtask=subtask,
                    final_state=result['final_state'],
                    new_semantic_entries=result['new_semantic_entries'],
                )
                print(f"[FINDINGS SUMMARY]\n{findings}\n")
                cumulative_context += f"\n\n--- SUBTASK {i + 1} FINDINGS ---\n{findings}"

        print("\n[ORCHESTRATOR] All subtasks complete.")
    finally:
        duration = time.time() - start_time
        print("-" * 30)
        print(f"Runtime: {duration:.2f} seconds")
        print("-" * 30)
        winsound.Beep(392, 1000)


if __name__ == "__main__":
    _task = sys.argv[1] if len(sys.argv) > 1 else input("Task: ")
    _run_entry = sys.argv[2] if len(sys.argv) > 2 else ""
    orchestrate(_task, run_entry=_run_entry)
