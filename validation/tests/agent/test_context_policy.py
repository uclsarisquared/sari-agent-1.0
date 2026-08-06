"""Offline tests for named context-window policies and semantic retention."""

from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError
from io import BytesIO
import os
import re
import sys

import pytest
from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_core import agent as agent_module
from agent_core.agent import VLMAgent
from agent_core.context import SemanticLog
from agent_core.context_policy import (
    CONTEXT_POLICIES,
    CONTEXT_POLICY_NAMES,
    ContextPolicy,
    resolve_context_policy,
    validate_context_policy,
)


def test_registry_defaults_and_named_arms() -> None:
    assert CONTEXT_POLICY_NAMES == (
        "baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4"
    )
    assert CONTEXT_POLICIES["baseline"] == ContextPolicy()
    assert CONTEXT_POLICIES["a2c"].semantic_dedupe == 0.80
    assert CONTEXT_POLICIES["a2c"].semantic_dedupe_window == 8
    assert CONTEXT_POLICIES["a2c"].semantic_keep_last == 12
    assert CONTEXT_POLICIES["a4"].findings_max_chars == 600
    with pytest.raises(FrozenInstanceError):
        CONTEXT_POLICIES["baseline"].findings_enabled = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "policy, message",
    [
        (ContextPolicy(semantic_dedupe=-0.01), "semantic_dedupe"),
        (ContextPolicy(semantic_dedupe=1.01), "semantic_dedupe"),
        (ContextPolicy(semantic_dedupe_window=0), "semantic_dedupe_window"),
        (ContextPolicy(semantic_keep_last=-1), "semantic_keep_last"),
        (ContextPolicy(findings_max_chars=0), "findings_max_chars"),
        (ContextPolicy(actor_image_history=0), "actor_image_history"),
    ],
)
def test_invalid_policy_values_are_rejected(policy: ContextPolicy, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_context_policy(policy)
    with pytest.raises(ValueError, match="unknown context policy"):
        resolve_context_policy("not-an-arm")


def test_baseline_render_and_delta_preserve_the_old_byte_contract() -> None:
    base = "BASE\n"
    log = SemanticLog(base, resolve_context_policy("baseline"))
    mark = log.mark()
    assert log.append("@ leg 1 step 1", "first fact")
    assert log.append("@ leg 1 step 2", "")
    expected_delta = "@ leg 1 step 1: first fact\n@ leg 1 step 2: \n"
    assert log.render().encode() == (base + expected_delta).encode()
    assert log.since(mark).encode() == expected_delta.encode()


def test_dedupe_is_strict_and_uses_only_eight_survivors() -> None:
    policy = ContextPolicy(semantic_dedupe=0.80, semantic_dedupe_window=8)
    log = SemanticLog("", policy)
    assert log.append("t0", "abcdefghij")
    # Ratio is exactly 0.80, and the rule is strictly greater than the threshold.
    assert log.append("t1", "abcdefghXY")
    assert not log.append("t2", "abcdefghij!")

    for index in range(8):
        assert log.append(f"fresh{index}", chr(ord("k") + index) * 20)
    # The first exact fact has now fallen outside the last-eight survivor window.
    assert log.append("again", "abcdefghij")


def test_a2c_renders_last_twelve_but_since_survives_eviction() -> None:
    log = SemanticLog("BASE\n", resolve_context_policy("a2c"))
    mark = log.mark()
    for index in range(15):
        assert log.append(f"t{index}", chr(ord("a") + index) * 20)
    rendered = log.render()
    assert "t2:" not in rendered
    assert f"t3: {'d' * 20}" in rendered
    assert f"t14: {'o' * 20}" in rendered
    delta = log.since(mark)
    assert delta.startswith(f"t0: {'a' * 20}\n")
    assert delta.count("\n") == 15


def test_a1_preserves_only_the_immutable_base_when_rendered() -> None:
    log = SemanticLog("BASE BYTES\n", resolve_context_policy("a1"))
    mark = log.mark()
    log.append("t1", "learned")
    assert log.render() == "BASE BYTES\n"
    assert log.since(mark) == "t1: learned\n"


def test_a6_filters_only_old_outbound_user_images() -> None:
    agent = object.__new__(VLMAgent)
    agent.context_policy = resolve_context_policy("a6-2")
    agent.history = []
    for index in range(3):
        agent.history.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"image-{index}"}},
                        {"type": "text", "text": f"user-{index}"},
                    ],
                },
                {"role": "assistant", "content": f"assistant-{index}"},
            ]
        )

    retained_before = repr(agent.history)
    outbound = agent._outbound_history()
    assert [part["type"] for part in outbound[0]["content"]] == ["text"]
    assert [part["type"] for part in outbound[2]["content"]] == ["image_url", "text"]
    assert [part["type"] for part in outbound[4]["content"]] == ["image_url", "text"]
    assert outbound[1] is agent.history[1]
    assert repr(agent.history) == retained_before
    assert "USER: user-0" in agent.get_history_text(n=8)


class _ActorCapture:
    def __init__(self, policy: ContextPolicy) -> None:
        self.semantic_log = SemanticLog("BASE\n", policy)
        self.semantic_log.append("@ leg 1 step 1", "old fact")
        self.episodic_memory = "prior reflection"
        self.actor_content: list[dict] | None = None

    def send_message(self, content: list[dict]) -> str:
        self.actor_content = content
        return "{'actions': ['look_down'], 'times': [1], 'notes': {}}"

    def get_history_text(self, n: int = 8) -> str:
        return "unchanged episodic learner history"


class _AssociativeShape:
    extractable_json_structured_output = re.compile(
        r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL
    )


def _one_pixel_png() -> str:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _execute_step_two(tmp_path, policy_name: str):
    policy = resolve_context_policy(policy_name)
    embodied = object.__new__(agent_module.EmbodiedAgent)
    embodied.context_policy = policy
    embodied.nav_mode = "vlm"
    embodied._run_dir = str(tmp_path)
    embodied._mem_leg = 2
    embodied.vlm_agent = _ActorCapture(policy)
    embodied.associative_learner = _AssociativeShape()
    semantic_prompts: list[str] = []
    episodic_prompts: list[str] = []
    semantic_reply = (
        "{'new_semantic_memory': 'new fact', 'recall': 'old fact', "
        "'next_action': 'look down', 'reported_answer': '', 'mode': 'perception'}"
    )

    def semantic_call(_system, _image, text):
        semantic_prompts.append(text)
        return semantic_reply

    embodied._call_associative = semantic_call
    embodied._call_episodic = lambda history: (
        episodic_prompts.append(history)
        or "{'dense_summary': 'dense', 'what_worked': 'worked', 'what_to_avoid': 'avoid'}"
    )
    embodied._set_hand_pose = lambda _pose: None
    embodied._invalidate_hand_pose = lambda: None
    state = {"leftGrippedState": False, "rightGrippedState": False}
    embodied.execute_lean(
        {"task": "Inspect the shelf", "state": state, "image": _one_pixel_png()},
        2,
    )
    actor_text = next(
        part["text"] for part in embodied.vlm_agent.actor_content
        if part.get("type") == "text"
    )
    return embodied, semantic_prompts, episodic_prompts, actor_text, state


@pytest.mark.parametrize("timestep", [1, 2])
def test_stop_records_and_persists_final_semantic_observation_consistently(
    tmp_path, timestep
) -> None:
    run_dir = tmp_path / f"step-{timestep}"
    run_dir.mkdir()
    policy = resolve_context_policy("baseline")
    embodied = object.__new__(agent_module.EmbodiedAgent)
    embodied.context_policy = policy
    embodied.nav_mode = "vlm"
    embodied._run_dir = str(run_dir)
    embodied._mem_leg = None
    embodied.vlm_agent = _ActorCapture(policy)
    embodied.associative_learner = _AssociativeShape()
    embodied._call_associative = lambda *_args: (
        "{'new_semantic_memory': 'final fact', 'recall': 'done', "
        "'next_action': 'stop', 'reported_answer': 'finished', 'mode': 'STOP'}"
    )
    pose_calls = []
    embodied._set_hand_pose = pose_calls.append
    embodied._invalidate_hand_pose = lambda: pose_calls.append("invalidated")

    response = embodied.execute_lean(
        {
            "task": "Finish",
            "state": {"leftGrippedState": False, "rightGrippedState": False},
            "image": _one_pixel_png(),
        },
        timestep,
    )

    assert response["halt"] is True
    assert response["reported_answer"] == "finished"
    assert pose_calls == ["rest"]
    assert "final fact" in (run_dir / "semantic_memory.txt").read_text(encoding="utf-8")
    assert (run_dir / "episodic_memory.txt").read_text(encoding="utf-8") == "prior reflection"


def test_baseline_semantic_prompt_actor_message_and_artifact_are_byte_stable(tmp_path) -> None:
    embodied, semantic_prompts, episodic_prompts, actor_text, state = _execute_step_two(
        tmp_path, "baseline"
    )
    expected_semantic = (
        "## MAIN TASK: Inspect the shelf\n"
        "## CURRENT TIMESTEP: 2\n"
        "## SEMANTIC MEMORY: BASE\n@ leg 1 step 1: old fact\n\n"
        "## EXISTING EPISODIC MEMORY: prior reflection\n"
        f"## STATE: {state}\n"
    )
    assert semantic_prompts[0].encode() == expected_semantic.encode()
    assert actor_text.startswith(
        "## CURRENT OBSERVATION\n"
        "## CURRENT TIMESTEP: 2\n"
        "## RECALL FROM SEMANTIC MEMORY: old fact\n"
        "## THIS STEP'S INTENDED ACTION: look down\n"
        "## EXISTING EPISODIC MEMORY: prior reflection\n"
        f"## STATE: {state}\n"
        "## AGENT MODE: perception\n"
    )
    expected_memory = "BASE\n@ leg 1 step 1: old fact\n@ leg 2 step 2: new fact\n"
    assert (tmp_path / "semantic_memory.txt").read_bytes() == expected_memory.encode()
    assert episodic_prompts == ["unchanged episodic learner history"]
    assert embodied.vlm_agent.semantic_log.since(1) == "@ leg 2 step 2: new fact\n"


def test_a5_removes_only_the_actor_episodic_line(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline"
    a5_dir = tmp_path / "a5"
    baseline_dir.mkdir()
    a5_dir.mkdir()
    _base, base_semantic, base_ep, base_actor, _state = _execute_step_two(
        baseline_dir, "baseline"
    )
    _a5, a5_semantic, a5_ep, a5_actor, _state = _execute_step_two(a5_dir, "a5")
    line = "## EXISTING EPISODIC MEMORY: prior reflection\n"
    assert base_semantic == a5_semantic
    assert base_ep == a5_ep
    assert line in base_actor and line not in a5_actor
    assert base_actor.replace(line, "") == a5_actor


def test_a4_findings_prompt_caps_the_retained_handoff(monkeypatch) -> None:
    from orchestrator import orchestrator_llm

    captured: dict[str, str] = {}

    def fake_call(_client, system, user, _role):
        captured.update(system=system, user=user)
        return "x" * 900

    monkeypatch.setattr(orchestrator_llm, "_llm_call", fake_call)
    result = orchestrator_llm.generate_findings_summary(
        object(), "pick up crackers", {"position": "aisle 2"}, "new fact",
        context_policy=resolve_context_policy("a4"),
    )
    assert len(result) == 600
    assert result == "x" * 600
    assert "concise factual handoff" in captured["system"]


def test_a3_does_not_make_a_findings_call(monkeypatch) -> None:
    from orchestrator import orchestrator_llm

    monkeypatch.setattr(
        orchestrator_llm,
        "generate_findings_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("A3 made a findings call")
        ),
    )
    assert orchestrator_llm._generate_findings_if_enabled(
        resolve_context_policy("a3"),
        object(),
        "completed leg",
        {"position": "checkout"},
        "learned fact",
    ) is None
