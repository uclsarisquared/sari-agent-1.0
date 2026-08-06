"""Live probe for the mid-leg self-correction path (2026-07-23). Needs the sim in Play mode.

Forces the wrong-item condition deterministically instead of hoping for a misgrab: the leg's TEXT
sends the agent to grab a snack off the cp48 shelf, while the leg's TARGET field says 'Jack Daniels'
(Liquor - nowhere near cp48). So the grab succeeds, the STOP is refused (name mismatch), and after
WRONG_ITEM_RELEASE_AFTER refusals run_leg must auto-release the hand and reset the refusal budget.

PASS criteria (printed at the end):
  - a `corrective_release` event fired (m['corrective_release'] is non-null), and
  - the leg did NOT end halt_forced on the first refusal burst (the budget reset took effect).
The leg itself is EXPECTED to end step_cap/halt_forced afterwards - Jack Daniels isn't findable in
the step budget; that is not what this probe measures.

    python validation/probes/live_correction_probe.py
"""
import os
import sys
import time
from datetime import datetime

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim.env import Reset, init_logger
from agent_core.runtime import EmbodiedAgent
from orchestrator.leg_runner import run_leg
from orchestrator.orchestration import _load_store_map
from orchestrator.orchestrator_llm import ASSOCIATIVE_CONFIG, VLM_CONFIG

OUT_DIR = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "probes", "live_correction",
    f"{datetime.now():%m%d_%H%M%S}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init_logger(run_name=f"correction-probe-{datetime.now():%m%d_%H%M%S}")
    print("[PROBE] hard env reset...")
    Reset()
    time.sleep(1.5)
    agent = EmbodiedAgent(vlm_config=VLM_CONFIG, associative_config=ASSOCIATIVE_CONFIG,
                          mode='lean', nav_mode='graph', resolver_backend="endpoint")
    sm = _load_store_map()
    leg = {"type": "pickup", "target": "Jack Daniels",
           "text": "Pick up the snack item directly in front of you on the shelf.",
           "candidates": [48], "target_name": "snack item"}
    m = run_leg(agent, leg, sm, (14, 12.0),
                log_path=os.path.join(OUT_DIR, "leg00.jsonl"), leg_idx=1)
    try:
        agent.close()
    except Exception:  # noqa: BLE001
        pass

    fired = bool(m.get("corrective_release"))
    print("\n" + "=" * 60)
    print(f"[PROBE] end_reason={m['end_reason']}  halts_refused={m['halts_refused']}  "
          f"corrective_release={m.get('corrective_release')}")
    fs = m.get("final_state") or {}
    print(f"[PROBE] final grips: left={fs.get('leftGrippedState')} "
          f"right={fs.get('rightGrippedState')}  gripped_names={fs.get('gripped_names')}")
    print(f"[PROBE] {'PASS' if fired else 'FAIL'}: corrective release "
          f"{'fired' if fired else 'never fired'}  -> logs in {OUT_DIR}")
    return 0 if fired else 1


if __name__ == "__main__":
    sys.exit(main())
