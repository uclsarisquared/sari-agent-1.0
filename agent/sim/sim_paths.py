"""Resolves paths into the Unity sim repo — a SEPARATE repo from this one (see agent/CLAUDE.md).
No single filesystem path works across machines: every prior hardcode here named one specific
person's disk layout (``C:\\Sari\\SariSandboxMY\\SariSandboxV2``, ``C:\\Users\\Tristan
Baclor\\AppData\\...``) and broke on everyone else's. Set these in config.env instead of editing
code — see config.env.example for the full field docs.
"""
import os


def sandbox_dir(required=True):
    """The Unity sim repo root (contains Assets/Resources/Data/*.json), from $SARI_SANDBOX_DIR.

    `required=False` returns None instead of raising, for callers with a documented graceful
    degradation path (e.g. subtask_completion's catalog grounding falls back to substring
    matching when the sim repo isn't available)."""
    root = os.getenv("SARI_SANDBOX_DIR")
    if not root and required:
        raise RuntimeError(
            "SARI_SANDBOX_DIR is not set. Point it at your Unity sim repo root (the folder "
            "containing Assets/Resources/Data/Categories.json) in config.env."
        )
    return root


def data_dir(required=True):
    root = sandbox_dir(required=required)
    return os.path.join(root, "Assets", "Resources", "Data") if root else None


def categories_json(required=True):
    d = data_dir(required=required)
    return os.path.join(d, "Categories.json") if d else None


def price_data_json(required=True):
    d = data_dir(required=required)
    return os.path.join(d, "PriceData.json") if d else None


def store_save_json():
    """The Unity save file the sim writes at runtime (shelf placements ground truth). No graceful
    fallback — score_index.py's per-slot scoring has no meaning without it."""
    path = os.getenv("SARI_STORE_SAVE_JSON")
    if not path:
        raise RuntimeError(
            "SARI_STORE_SAVE_JSON is not set. Point it at the Unity save file (under Unity's "
            "persistentDataPath, e.g. '.../AppData/LocalLow/Sari Sandbox/Sari Sandbox V2/"
            "Store 2 v2.json') in config.env."
        )
    return path
