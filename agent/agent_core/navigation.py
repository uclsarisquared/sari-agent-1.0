"""Graph navigation, advisor routing, and navigation-backed task macros."""

from __future__ import annotations

import os
from typing import Literal, Optional

from loguru import logger

from agent_core import token_meter
from agent_core.contracts import NavigationResult
from agent_core.hands import HandController
from agent_core.prompt_loader import load_prompt


ADVISOR_SYS = load_prompt("navigation/advisor")
ADVISOR_SCHEMA = {
    "type": "object",
    "properties": {
        "next_checkpoint": {
            "type": "integer",
            "description": "checkpoint id to move to; MUST be an adjacent id",
        },
        "stop_here": {
            "type": "boolean",
            "description": "true = the target goods are visible from HERE; do not move",
        },
        "reason": {"type": "string"},
    },
    "required": ["next_checkpoint", "stop_here", "reason"],
}


class GraphNavigator:
    """Own all graph-session, candidate, and per-hop advisor state."""

    def __init__(
        self,
        hands: HandController,
        *,
        nav_mode: Literal["vlm", "graph", "graph-advised"] = "vlm",
        resolver_backend: Literal["endpoint", "claude-cli"] = "endpoint",
        advisor_backend: Literal["endpoint", "claude-cli"] = "endpoint",
        map_output_dir: Optional[str] = None,
        run_dir: Optional[str] = None,
    ) -> None:
        self.hands = hands
        self.nav_mode = nav_mode
        self.resolver_backend = resolver_backend
        self.advisor_backend = advisor_backend
        self.map_output_dir = map_output_dir
        self.run_dir = run_dir
        self.advised_llm_calls = 0
        self.advised_stats: list[dict] = []
        self.advised_shot_idx = 0
        self.graph_nav = None
        self.candidates: list[int] = []
        self.visited: set[int] = set()
        self.task: Optional[str] = None
        self.seeded: Optional[list[int]] = None
        self.seeded_name: Optional[str] = None
        self.resolution: Optional[dict] = None

    def session(self):
        if self.graph_nav is None:
            from nav.store_map import NavSession, StoreMap
            from sim.env import default_uri

            store_map = (
                StoreMap(output_dir=self.map_output_dir) if self.map_output_dir else StoreMap()
            )
            nav = NavSession(store_map, uri=default_uri(), stow_hands=False)
            self.graph_nav = (store_map, nav)
        return self.graph_nav

    def seed_candidates(self, candidates, target_name=None) -> None:
        self.seeded = list(candidates) if candidates else None
        self.seeded_name = target_name
        self.task = None

    def navigate(self, main_task: str, nav_goal: Optional[str] = None) -> NavigationResult:
        from explore import step_agent
        from nav import locate_task
        from sim.env import RequestScreenshot

        store_map, nav = self.session()
        if self.task != main_task:
            self._resolve_candidates(store_map, locate_task, main_task)
            self.task = main_task
            self.visited = set()
            if not self.candidates:
                return NavigationResult(
                    "## NAVIGATOR: could not resolve the target to any known location. "
                    "Proceed by exploring visually.\n"
                )

        remaining = [candidate for candidate in self.candidates if candidate not in self.visited]
        if not remaining:
            logger.warning("[graph-nav] all candidates visited; restarting candidate list")
            self.visited = set()
            remaining = list(self.candidates)

        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
        x, z = nav.pos[0], nav.pos[2]
        target = min(
            remaining,
            key=lambda candidate: store_map.hops(
                store_map.nearest_checkpoint((x, z)), candidate
            )
            or 99,
        )
        self.visited.add(target)
        self.hands.set_pose("rest")
        if self.nav_mode == "graph-advised":
            ok, end_checkpoint = self.advised_goto(
                store_map, nav, target, nav_goal or main_task
            )
        else:
            ok, end_checkpoint = nav.goto(target), target

        info = store_map.checkpoint(end_checkpoint)
        fresh = RequestScreenshot(save_image=False, uri=nav.args.uri)["image"]
        if not ok:
            note = (
                f"## NAVIGATOR: could not reach checkpoint {target} (path blocked). "
                f"You are at ({nav.pos[0]:.2f}, {nav.pos[2]:.2f}). Assess visually.\n"
            )
        else:
            holds = ", ".join(info["holds"]) if info["holds"] else "unknown goods"
            stopped = (
                ""
                if end_checkpoint == target
                else f" (the navigator stopped short of checkpoint {target} because the goods "
                "looked visible from here)"
            )
            note = (
                f"## ARRIVED VIA NAVIGATOR: checkpoint {end_checkpoint}{stopped}, facing a shelf "
                f"holding {holds}. {info['summary'] or ''} If the target is not visible here, "
                "choose *navigation* mode again and you will be taken to the next candidate "
                "location.\n"
            )
        return NavigationResult(note, fresh)

    def _resolve_candidates(self, store_map, locate_task, main_task: str) -> None:
        if self.seeded:
            self.candidates = [candidate for candidate in self.seeded if candidate in store_map.by_id]
            self.resolution = {
                "candidates": self.candidates,
                "target_name": self.seeded_name,
                "seeded": True,
            }
            logger.info(
                f"[graph-nav] using {len(self.candidates)} PLAN-SEEDED candidate(s): "
                f"{self.candidates}"
            )
            return

        resolve_call = locate_task.backend_callable(self.resolver_backend)
        with token_meter.role(token_meter.ROLE_RESOLVER):
            resolution, _ = locate_task.resolve(resolve_call, store_map, main_task)
        self.resolution = resolution
        candidates = resolution.get("candidates") or []
        self.candidates = [candidate for candidate in candidates if candidate in store_map.by_id]
        logger.info(
            f"[graph-nav] resolved {resolution.get('target_name')!r} "
            f"tier={resolution.get('tier')} candidates={self.candidates}"
        )

    def advised_goto(self, store_map, nav, target, nav_goal):
        from explore import step_agent
        from nav import locate_task

        ask = locate_task.backend_callable(self.advisor_backend)
        shots_dir = (
            os.path.join(self.run_dir, "advised_nav")
            if self.run_dir
            else os.path.join(store_map.output_dir, "advised_nav")
        )
        os.makedirs(shots_dir, exist_ok=True)
        target_info = store_map.checkpoint(target)
        target_holds = ", ".join(target_info["holds"]) or "unannotated"

        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
        current = store_map.nearest_checkpoint((nav.pos[0], nav.pos[2]))
        budget = 2 * (store_map.hops(current, target) or 1) + 2
        for hop in range(1, budget + 1):
            if current == target:
                return True, current
            path = store_map.hop_path(current, target)
            advice = path[1] if path and len(path) > 1 else None
            neighbors = store_map.checkpoint(current)["neighbors"]
            if not neighbors:
                break
            lines = [self._neighbor_line(store_map, neighbor, target) for neighbor in neighbors]
            shot = nav.screenshot(
                os.path.join(shots_dir, f"hop_{self.advised_shot_idx:05d}.png")
            )
            self.advised_shot_idx += 1
            prompt = self._advisor_prompt(
                store_map, current, target, target_holds, nav_goal, advice, lines
            )
            try:
                with token_meter.role(token_meter.ROLE_ADVISOR):
                    result, _ = ask(
                        ADVISOR_SYS, prompt, ADVISOR_SCHEMA, (("current view", shot),)
                    )
            except Exception as error:
                logger.warning(
                    f"[advised-nav] advisor call failed ({type(error).__name__}: {error}); "
                    "taking the advice hop"
                )
                result = {}
            self.advised_llm_calls += 1
            pick = result.get("next_checkpoint") if isinstance(result, dict) else None
            stop = bool(result.get("stop_here")) if isinstance(result, dict) else False
            reason = (result.get("reason") or "")[:200] if isinstance(result, dict) else ""
            invalid = pick not in neighbors and not stop
            if invalid:
                pick = advice if advice is not None else neighbors[0]
            record = {
                "hop": hop,
                "cur": current,
                "target": target,
                "pick": pick,
                "advice": advice,
                "agreed": not stop and pick == advice,
                "invalid": invalid,
                "stop_here": stop,
                "reason": reason,
            }
            self.advised_stats.append(record)
            logger.info(
                f"[advised-nav] hop {hop}/{budget} at cp{current} -> "
                f"{'STOP' if stop else f'cp{pick}'} (advice cp{advice}, "
                f"{'agreed' if record['agreed'] else 'INVALID' if invalid else 'deviated'})"
                f"{': ' + reason if reason else ''}"
            )
            if stop:
                nav.goto(current)
                return True, current
            if not nav.goto(pick, face_shelf=(pick == target)):
                logger.warning(
                    f"[advised-nav] executor refused cp{current} -> cp{pick}; "
                    "falling back to deterministic drive"
                )
                break
            nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
            current = store_map.nearest_checkpoint((nav.pos[0], nav.pos[2]))
        if current == target:
            return True, current
        logger.warning(
            f"[advised-nav] budget/refusal at cp{current} (target cp{target}); "
            "degrading to the graph arm's deterministic goto"
        )
        return nav.goto(target), target

    @staticmethod
    def _neighbor_line(store_map, neighbor, target) -> str:
        info = store_map.checkpoint(neighbor)
        distance = store_map.hops(neighbor, target)
        return (
            f"cp{neighbor}: {'?' if distance is None else distance} hop(s) from destination"
            + (f" | holds {', '.join(info['holds'])}" if info["holds"] else "")
            + (f" | {info['summary']}" if info["summary"] else "")
        )

    @staticmethod
    def _advisor_prompt(store_map, current, target, target_holds, nav_goal, advice, lines) -> str:
        current_summary = store_map.checkpoint(current)["summary"]
        return (
            f"## GOAL\nTask: {nav_goal}\n"
            f"Destination: checkpoint {target} (holds {target_holds})\n\n"
            f"## WHERE YOU ARE\nCheckpoint {current}"
            + (f": {current_summary}" if current_summary else "")
            + "\n\n## ADJACENT CHECKPOINTS (next_checkpoint MUST be one of these)\n"
            + "\n".join(lines)
            + (
                f"\n\n## PLANNER ADVICE\nShortest route next hop: cp{advice}"
                if advice is not None
                else ""
            )
        )

    def navigate_to_counter(self) -> NavigationResult:
        from nav.store_map import go_to_counter
        from sim.env import RequestScreenshot

        _, nav = self.session()
        self.hands.set_pose("rest")
        result = go_to_counter(nav)
        fresh = RequestScreenshot(save_image=False, uri=nav.args.uri)["image"]
        if not result.get("arrived"):
            note = (
                "## NAVIGATOR: could not reach the checkout counter "
                f"(checkpoint {result.get('checkpoint')}; "
                f"{result.get('reason', 'path blocked')}). "
                f"You are at ({result.get('x', 0.0):.2f}, "
                f"{result.get('z', 0.0):.2f}). Assess visually.\n"
            )
        else:
            note = (
                "## ARRIVED VIA NAVIGATOR: the checkout counter "
                f"(checkpoint {result['checkpoint']}). Centre the counter surface "
                "(center_to_counter), then place the held item.\n"
            )
        return NavigationResult(note, fresh)

    def checkout_held_item(self, hand: str = "auto") -> dict:
        from nav.store_map import checkout_held_item, checkout_held_items

        _, nav = self.session()
        self.hands.set_pose("rest")
        result = checkout_held_items(nav) if hand == "auto" else checkout_held_item(nav, hand=hand)
        logger.info(
            f"[checkout] success={result.get('success')} scanned={result.get('scanned')} "
            f"placed={result.get('placed')} aligned={result.get('aligned')} - "
            f"{result.get('reason')}"
        )
        return result

    def metric_approach(self, move_steps: int) -> NavigationResult:
        from sim.env import RequestScreenshot, move_forward

        self.hands.set_pose("rest")
        move_forward(move_steps)
        fresh = RequestScreenshot(save_image=False)["image"]
        note = (
            f"## MOVED {move_steps} STEP(S) (~{move_steps * 0.1:.1f} m) FORWARD to close the "
            "measured reach gap - you are still facing the same shelf. RE-CENTER on the target "
            "with center_object_on_screen (the move shifted it off-centre), then retry the grab "
            "in *manipulation*.\n"
        )
        return NavigationResult(note, fresh)
