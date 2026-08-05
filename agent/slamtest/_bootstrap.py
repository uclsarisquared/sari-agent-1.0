"""sys.path bootstrap for the slamtest tree.

slamtest keeps FLAT imports by contract (see agent/CLAUDE.md): modules import
each other as ``from capture_walk import ...`` even though the files live in
category subfolders (core/, graph/, drivers/, capture/, annotate/, scoring/,
app/). Importing this module inserts onto sys.path (if absent): the agent
root (parent of slamtest), the slamtest dir itself, and each category subdir —
making every slamtest module importable flat and the agent packages
(sim/, nav/, ...) importable qualified.

Scripts use it as: add slamtest/ to sys.path, then ``import _bootstrap``.
"""

import os
import sys

_SLAM_DIR = os.path.dirname(os.path.abspath(__file__))        # agent/slamtest
_OVERHAUL_DIR = os.path.dirname(_SLAM_DIR)                      # agent

_CATEGORY_DIRS = ("core", "graph", "drivers", "capture", "annotate", "scoring", "app")

for _p in [_OVERHAUL_DIR, _SLAM_DIR] + [os.path.join(_SLAM_DIR, _d) for _d in _CATEGORY_DIRS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
