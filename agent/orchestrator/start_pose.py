"""Helpers for returning a live simulator agent to a repeatable start pose."""


def return_to_start(agent, output_dir=None):
    """Drive the agent to the fixed start checkpoint without resetting the store.

    ``output_dir`` selects the mapping artifacts loaded by ``StoreMap``. The
    map and navigation session are cached so repeated evaluation tasks use the
    same substrate and start pose.
    """
    from nav.store_map import StoreMap, NavSession

    if not hasattr(return_to_start, "_nav"):
        store_map = StoreMap(output_dir=output_dir) if output_dir else StoreMap()
        return_to_start._nav = (
            store_map,
            NavSession(store_map, stow_hands=False),
            store_map.nearest_checkpoint((-3.0, -5.0)),
        )

    store_map, nav, start_cp = return_to_start._nav
    from explore import step_agent
    from capture_walk import face

    nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
    # Use the runtime tracker rather than toggling the simulator directly; a
    # raw call would desynchronize the agent's hand state.
    agent._set_hands(False)
    nav.goto(start_cp, face_shelf=False)
    nav.pos, nav.rot = face(nav.args, nav.pos, nav.rot, 0.0)
