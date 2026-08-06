"""Renders BASE_SEMANTIC_MEMORY from the mapping artifacts - the Phase 4.2 shared substrate.

Replaces the hand-written store layouts that lived in memory.py (deleted 2026-07-19; git
history is the archive). Those described a store that no longer exists; this renders the SAME
KIND of document - per-shelf prose, adjacency, and a Fast Tracking coordinate table - from the
annotated checkpoint graph, so it is correct for whatever store the frozen map describes.

Fairness note (phase4.2 plan): BOTH navigation arms consume this. The VLM-nav control arm gets
correct target coordinates in its native prose+table format; the graph arm reads the same
artifacts through StoreMap. Neither arm gets knowledge the other lacks - only the EXECUTOR of
navigation differs.

Format notes, deliberate:
  - Fast Tracking entries mirror the old hand-written shape exactly
    ("-> translation: (x, 1.36, z), rotation: (0.0, yaw, 0.0)") because SYS_INST rule 12's
    worked example teaches the VLM to parse that shape. y=1.36 is the camera height the old
    tables used; yaw is the shelf perpendicular Phase 2 computed (what capture/locate face).
  - Entries are keyed "Checkpoint N", not "Shelf N" - checkpoint ids are the map's real keys,
    and inventing a renumbering would recreate the id-drift trap walk_map documents.
  - Adjacency prose comes from graph neighbors/route hints (deterministic), never the VLM -
    the graph owns spatial truth (phase3.1 decision, unchanged).
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.dirname(_THIS_DIR)  # agent/ — packages + mapping live under here
_MAPPING_DIR = os.path.join(_AGENT_DIR, "mapping")
for _p in (_AGENT_DIR, _MAPPING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402 - mapping category dirs onto sys.path (flat-import contract)

CAMERA_Y = 1.36  # the camera height the old Fast Tracking tables carried


def render_base_semantic_memory(store_map=None):
    from nav.store_map import StoreMap
    from capture_walk import perpendicular_yaw

    sm = store_map or StoreMap()
    lines = [
        "This semantic memory outlines the store's environment, rendered from the annotated "
        "checkpoint map (not hand-written). Positions are world coordinates; checkpoints are "
        "reachable standing positions.",
        "",
        "**Store contents by checkpoint** (what is visible standing at each):",
    ]

    shelf_ids = sorted(sm.shelf_checkpoints())
    for cp_id in shelf_ids:
        cp = sm.checkpoint(cp_id)
        holds = ", ".join(cp["holds"]) if cp["holds"] else "unclassified"
        summary = (cp["summary"] or "").strip()
        lines.append(f"{cp_id}. Checkpoint {cp_id} ({holds}): {summary}")

    counter = sm.counter_checkpoint()
    if counter is not None:
        c = sm.checkpoint(counter)
        lines.append(f"{counter}. Checkpoint {counter} (COUNTER / drop-off): "
                     f"{(c['summary'] or 'the checkout counter').strip()}")

    lines += ["", "**Connectivity** (which checkpoints link to which - use for rough routing):"]
    for cp_id in shelf_ids + ([counter] if counter is not None else []):
        cp = sm.checkpoint(cp_id)
        lines.append(f"- Checkpoint {cp_id} connects to: "
                     + ", ".join(str(n) for n in cp["neighbors"]))

    lines += ["", "**Fast Tracking** (use this to guide you quickly to the target object):"]
    for cp_id in shelf_ids + ([counter] if counter is not None else []):
        cp = sm.checkpoint(cp_id)
        x, z = cp["world_xz"]
        raw = sm.by_id[cp_id]
        yaw = perpendicular_yaw(raw) % 360 if raw.get("shelf_cell") else 0.0
        holds = ", ".join(cp["holds"]) if cp["holds"] else (
            "counter" if cp_id == counter else "?")
        lines.append(f"- Checkpoint {cp_id} ({holds}) has the target position -> "
                     f"translation: ({x:.2f}, {CAMERA_Y}, {z:.2f}), "
                     f"rotation: (0.0, {yaw:.1f}, 0.0)")

    return "\n".join(lines) + "\n\n"


if __name__ == "__main__":
    text = render_base_semantic_memory()
    out = os.path.join(_MAPPING_DIR, "output", "base_semantic_memory_rendered.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{len(text)} chars -> {out}")
    print(text[:1200])
