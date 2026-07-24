"""BASE_SEMANTIC_MEMORY - rendered from the annotated checkpoint map at import time.

The three hand-written store layouts that used to live here were DELETED 2026-07-19 (git
history: `07927a5` and earlier). They described stores that no longer exist - measured against
the live `Store 2 v2.json`, every Fast Tracking coordinate pointed at a wrong or out-of-store
position - and Phase 3.1 had already called their replacement: "this IS the checkpoint graph...
Phases 1-2 generate a denser, observation-derived version of that hand-written table."

Both navigation arms of the Phase 4.2 A/B consume this same document (fairness requirement -
see slamtest/plans/phase4.2_dual_stack_integration.md). Rendering happens at import: it reads
three JSONs, no sim, no model, <1s. `python memory_gen.py` writes an inspectable copy to
slamtest/output/base_semantic_memory_rendered.txt.
"""
from agent_core.memory_gen import render_base_semantic_memory

BASE_SEMANTIC_MEMORY = render_base_semantic_memory()
