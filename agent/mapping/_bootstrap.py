"""sys.path bootstrap for the mapping tree.

mapping keeps FLAT imports by contract (see agent/CLAUDE.md): modules import
each other as ``from capture_walk import ...`` even though the files live in
category subfolders (core/, graph/, drivers/, capture/, annotate/, scoring/,
app/). Importing this module inserts onto sys.path (if absent): the agent
root (parent of mapping), the mapping dir itself, and each category subdir —
making every mapping module importable flat and the agent packages
(sim/, nav/, ...) importable qualified.

Scripts use it as: add mapping/ to sys.path, then ``import _bootstrap``.
"""

import os
import sys

_MAPPING_DIR = os.path.dirname(os.path.abspath(__file__))        # agent/mapping
_AGENT_DIR = os.path.dirname(_MAPPING_DIR)                      # agent

_CATEGORY_DIRS = ("core", "graph", "drivers", "capture", "annotate", "scoring", "app")

for _p in [_AGENT_DIR, _MAPPING_DIR] + [os.path.join(_MAPPING_DIR, _d) for _d in _CATEGORY_DIRS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
