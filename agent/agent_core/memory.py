"""base_semantic_memory() - rendered from the annotated checkpoint map, on first use.

The three hand-written store layouts that used to live here were DELETED 2026-07-19 (git
history: `07927a5` and earlier). They described stores that no longer exist - measured against
the live `Store 2 v2.json`, every Fast Tracking coordinate pointed at a wrong or out-of-store
position - and Phase 3.1 had already called their replacement: "this IS the checkpoint graph...
Phases 1-2 generate a denser, observation-derived version of that hand-written table."

Both navigation arms of the Phase 4.2 A/B consume this same document (fairness requirement -
see mapping/plans/phase4.2_dual_stack_integration.md). Rendering reads three JSONs, no sim, no
model, <1s. `python memory_gen.py` writes an inspectable copy to
mapping/output/base_semantic_memory_rendered.txt.

Rendering used to happen AT IMPORT, as a module constant. It doesn't any more, for two reasons:
a missing/alternate map dir turned into an ImportError traceback from a file that was only being
imported transitively (the whole toolset chain pulls this in) - and worse, the render always read
the DEFAULT map dir, so an entry point run with `--output-dir <other map>` got graph navigation
over one map and prose describing a different one. The dir is now a call argument, so the
document always describes the map the caller actually loaded, and nothing is read until an agent
asks for it. Results are cached per dir - repeated agents in one process pay once.
"""
from agent_core.memory_gen import render_base_semantic_memory

_CACHE: dict[str, str] = {}


def base_semantic_memory(output_dir=None):
    """The rendered store document for `output_dir` (None -> StoreMap's default, i.e. $SARI_MAP_DIR
    else mapping/output). Cached per dir."""
    key = output_dir or ""
    if key not in _CACHE:
        from nav.store_map import StoreMap
        _CACHE[key] = render_base_semantic_memory(StoreMap(output_dir=output_dir))
    return _CACHE[key]
