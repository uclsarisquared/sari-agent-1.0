"""Watcher tests: scanning, collapse detection, discovery, and the HTTP surface.

Entirely offline - no sim, no coordinator, no model stack. The fixtures write the same artefacts a
real battery does (attempt.json, legNN.jsonl, legNN/stepNN.png, summary.json), so the scan path is
exercised against the real shapes rather than a mock. What is being pinned down:

  1. a healthy attempt scores clean while a looping one scores as a collapse, and tiles sort
     worst-first - the ranking IS the feature;
  2. discovery picks the newest battery, and --run-dir pins one;
  3. an attempt whose runner died shows as orphaned rather than as a live tile frozen forever, can
     be closed out into a real recorded attempt, and is never confused with whoever inherited its
     pid - while a runner that is only slow to finalize still gets the last word;
  4. the HTTP API serves state/frames/logs, and path traversal in an attempt key is refused;
  5. a rotated-aside requeue dir does not merge with the attempt that replaced it;
  6. every halt is announced exactly once whatever its outcome, and carries a replay clip inside the
     upload budget - written beside, never over, the dashboard/CLI replay.mp4;
  7. a watcher restart does not replay finishes that predate it into the channel;
  8. a human verdict is offered only where the AGENT chose to halt, is stored beside `success` and
     never over it, and reaches attempts.csv blank - not False - where nobody has looked;
  9. the dashboard renders and serves a seekable replay clip with Discord switched off entirely.

The Discord tests stub `_post` rather than the socket, except for one that stands up a throwaway HTTP
sink to pin the multipart body: that encoding is the part a wrong guess would break silently.

    python sari_bench/tests/test_watch.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sari_bench.watch import health, scan
from sari_bench.watch.server import WatchState, _safe_run_dir
from sari_bench.watch.notify import Discord

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _complete_mp4(payload: bytes = b"") -> bytes:
    def atom(name: bytes, body: bytes = b"") -> bytes:
        return struct.pack(">I4s", len(body) + 8, name) + body

    return atom(b"ftyp", b"isom\x00\x00\x02\x00") + atom(b"mdat", payload) + atom(b"moov")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _real_png(path: Path, size: tuple[int, int], index: int) -> None:
    """A frame ffmpeg can actually encode, with enough variation to cost real bits."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (18 + index * 3 % 200, 40, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(index * 4 % size[0], 10), (index * 4 % size[0] + 40, size[1] - 10)],
                   fill=(220, 200, 40))
    image.save(path)


def make_attempt(battery: Path, prompt_id: str, attempt: int, *, steps: list[dict],
                 state: str = "running", pid: int | None = None, started_ago: float = 60.0,
                 max_steps: int = 150, outcome: str = "", frames: bool = True,
                 success: bool = False, end_reason: str = "halt_granted",
                 frame_size: tuple[int, int] = (1, 1)) -> Path:
    run_dir = battery / prompt_id / f"try{attempt:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    manifest = {
        "prompt_id": prompt_id, "prompt": f"task for {prompt_id}", "family": "pickup",
        "attempt": attempt, "arm": "graph", "sandbox_id": f"sb-{prompt_id}",
        "commands_uri": "ws://127.0.0.1:51001/commands", "run_dir": str(run_dir),
        "state": state, "pid": pid, "started_epoch": now - started_ago,
        "deadline_epoch": now - started_ago + 7200, "time_limit_minutes": 120.0,
        "max_steps": max_steps, "started_at": "2026-07-25T15:00:00",
    }
    if outcome:
        manifest["outcome"] = outcome
        manifest["wall_seconds"] = started_ago
        manifest["success"] = success
        manifest["end_reason"] = end_reason
    _write(run_dir / "attempt.json", json.dumps(manifest))

    lines = [json.dumps({"event": "leg_start", "leg": 0, "type": "pickup", "text": "go get it"})]
    lines += [json.dumps({"event": "step", **step}) for step in steps]
    _write(run_dir / "leg00.jsonl", "\n".join(lines) + "\n")

    if frames:
        for index in range(1, len(steps) + 1):
            frame = run_dir / "leg00" / f"step{index:02d}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            if frame_size == (1, 1):
                frame.write_bytes(_PNG)
            else:
                _real_png(frame, frame_size, index)

    # Newline-terminated, the way logging.FileHandler writes: the log endpoint treats an unterminated
    # trailing line as one still being written.
    (run_dir / "agent.log").write_text("".join(f"log line {i}\n" for i in range(50)), encoding="utf-8")
    return run_dir


def _stamp(run_dir: Path, fields: dict[str, Any]) -> None:
    """Patches an attempt manifest the way the runner and the watcher both do."""
    manifest = json.loads((run_dir / "attempt.json").read_text(encoding="utf-8"))
    manifest.update(fields)
    _write(run_dir / "attempt.json", json.dumps(manifest))


def healthy_steps(count: int = 10) -> list[dict]:
    return [{"step": i, "mode": "navigation", "actions": [f"move_forward_{i}"],
             "pos": [float(i), 0.0, float(i)], "near_cp": f"cp{i}", "blocked": False}
            for i in range(1, count + 1)]


def looping_steps(count: int = 10) -> list[dict]:
    return [{"step": i, "mode": "manipulation" if i % 2 else "navigation",
             "actions": ["center_object"], "pos": [3.0, 0.0, 4.1], "near_cp": "cp7",
             "blocked": True}
            for i in range(1, count + 1)]


def test_health_separates_healthy_from_collapsed() -> None:
    good = health.score(healthy_steps(), seconds_since_last_step=10.0, max_steps=150)
    bad = health.score(looping_steps(), seconds_since_last_step=10.0, max_steps=150)

    assert good.level == health.LEVEL_OK, f"healthy run scored {good.score}: {good.summary}"
    assert bad.level == health.LEVEL_ALERT, f"looping run only scored {bad.score}: {bad.summary}"
    names = {signal.name for signal in bad.signals}
    assert {"spatial_loop", "action_loop", "blocked"} <= names, names

    stalled = health.score(healthy_steps(), seconds_since_last_step=1200.0)
    assert "stalled" in {s.name for s in stalled.signals}
    assert stalled.level == health.LEVEL_ALERT
    print("ok  collapse signals separate a healthy run from a looping one")


def test_scan_ranks_worst_first_and_reads_step_state() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "20260725_150000"
        _write(battery / "battery.json", json.dumps({"planned_attempts": 3, "arm": "graph"}))
        make_attempt(battery, "healthy", 1, steps=healthy_steps(), pid=os.getpid())
        make_attempt(battery, "looping", 1, steps=looping_steps(), pid=os.getpid())
        make_attempt(battery, "done", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed")

        view = scan.scan_battery(battery, time.time()).as_dict()
        keys = [a["prompt_id"] for a in view["attempts"]]
        assert keys[0] == "looping", f"worst attempt is not first: {keys}"
        assert keys[-1] == "done", f"finished attempt did not sink: {keys}"

        looping = view["attempts"][0]
        assert looping["step"] == 10, looping["step"]
        assert looping["near_cp"] == "cp7"
        assert looping["blocked"] is True
        assert looping["leg_type"] == "pickup"
        assert looping["frame"].endswith("step10.png"), looping["frame"]
        assert looping["health"]["level"] == health.LEVEL_ALERT
        assert view["counts"]["completed"] == 1
        print("ok  scan reads live step state and ranks collapsing attempts first")


def test_scan_carries_the_agents_final_response() -> None:
    """response.txt reaches the tile, is absent (not empty) mid-run, and cannot bloat the payload."""
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "20260725_150000"
        _write(battery / "battery.json", json.dumps({"planned_attempts": 3}))
        answered = make_attempt(battery, "answered", 1, steps=healthy_steps(3), state="finished",
                                outcome="completed", success=True)
        _write(answered / scan.RESPONSE_FILE, "I found the milk on the back wall and bought it.\n")
        make_attempt(battery, "midrun", 1, steps=healthy_steps(), pid=os.getpid())
        chatty = make_attempt(battery, "chatty", 1, steps=healthy_steps(3), state="finished",
                              outcome="completed")
        _write(chatty / scan.RESPONSE_FILE, "x" * (scan.RESPONSE_MAX_CHARS + 500))

        rows = {a["prompt_id"]: a for a in scan.scan_battery(battery, time.time()).as_dict()["attempts"]}
        assert rows["answered"]["response"] == "I found the milk on the back wall and bought it.", \
            rows["answered"]["response"]
        # Absent, not empty-with-a-block: an attempt still running has not answered yet.
        assert rows["midrun"]["response"] == "", rows["midrun"]["response"]
        assert len(rows["chatty"]["response"]) == scan.RESPONSE_MAX_CHARS + 1, \
            len(rows["chatty"]["response"])
        assert rows["chatty"]["response"].endswith("…")
        print("ok  the agent's final response reaches the dashboard, bounded")


def test_orphaned_attempt_is_not_shown_as_live() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        # A pid that cannot exist: the manifest says running, the process is gone.
        make_attempt(battery, "dead", 1, steps=healthy_steps(), pid=999_999_998)
        view = scan.scan_battery(battery, time.time()).as_dict()
        assert view["attempts"][0]["state"] == "orphaned", view["attempts"][0]["state"]
        assert view["attempts"][0]["alive"] is False
        print("ok  an attempt whose runner died reads as orphaned, not live")


def test_orphan_is_closed_out_rather_than_stranded() -> None:
    """The dead end this exists to remove: an attempt whose runner never came back.

    Nothing but the runner writes an attempt's closing record, so when the runner dies its attempt
    keeps `state: running` and a dead pid for good. Kill had no process left to signal and a verdict
    is refused on anything unfinished, which left the tile stuck on `orphaned` with rerunning the try
    - throwing the run away - as its only exit.
    """
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(6), pid=999_999_998)
        _stamp(run_dir, {"killed_by": "watcher", "lease_id": "lease-1", "tokens_in": 0})
        _write(run_dir / "tokens.json", json.dumps({"tokens_in": 900, "tokens_out": 40}))
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        assert scan.scan_battery(battery, time.time()).as_dict()["attempts"][0]["state"] == "orphaned"

        result = state.kill("p/try01")
        assert result["ok"] and result["closed_out"] is True, result
        # A dashboard kill, recorded exactly as the runner would have recorded it, so an orphan that
        # was killed does not read differently from a kill whose runner survived to write it down.
        assert result["outcome"] == "operator_kill", result

        manifest = json.loads((run_dir / "attempt.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "finished" and manifest["success"] is False
        assert manifest["end_reason"] == "runner_gone" and manifest["exit_code"] is None
        assert manifest["closed_out_by"] == "watcher", manifest
        assert manifest["tokens_in"] == 900, manifest["tokens_in"]
        # Measured to the run dir's last sign of life, not to the click: an orphan closed out a week
        # after it was abandoned did not run for a week.
        assert 55.0 <= manifest["wall_seconds"] <= 65.0, manifest["wall_seconds"]

        # It is on the spine now, so the report and the CSV account for it.
        rows = [json.loads(line) for line in
                (battery / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
        assert [(r["prompt_id"], r["attempt"], r["outcome"]) for r in rows] == [("p", 1, "operator_kill")]
        assert rows[0]["tokens_in"] == 900

        # And it is now an ordinary finished attempt: reviewable, and no longer killable.
        view = scan.scan_battery(battery, time.time()).as_dict()["attempts"][0]
        assert view["state"] == "finished" and view["verifiable"] is True
        assert state.kill("p/try01") == {"ok": False, "error": "attempt already finished"}
        print("ok  an orphaned attempt can be closed out, recorded, and then reviewed")


def test_close_out_leaves_the_last_word_to_a_live_runner() -> None:
    """A runner that is merely slow to finalize must not be overwritten by the watcher."""
    import threading

    from sari_bench.watch import server as server_mod

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(6), pid=999_999_998)
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        # The agent has exited; its runner is still between process.wait() and its closing write.
        finalize = threading.Timer(
            0.4, lambda: _finish(run_dir, outcome="completed", success=True))
        finalize.start()
        try:
            result = state.kill("p/try01")
        finally:
            finalize.cancel()

        assert result["ok"] and result["closed_out"] is False, result
        manifest = json.loads((run_dir / "attempt.json").read_text(encoding="utf-8"))
        assert manifest["outcome"] == "completed" and manifest["success"] is True, manifest
        assert "closed_out_by" not in manifest
        assert not (battery / "attempts.jsonl").exists(), "the runner owns the row, not the watcher"
        assert server_mod.FINALIZE_GRACE_SECONDS >= 0.4
        print("ok  close-out waits out a runner that is still finalizing, and defers to its record")


def test_a_recycled_pid_is_not_mistaken_for_the_agent() -> None:
    """A pid outlives its process. Signalling one blind is how a watcher kills a stranger."""
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        # This very process, wearing start ticks that cannot be its own: the number is live, but it
        # is not the agent. If the identity check regresses, this test SIGTERMs the test runner.
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(4), pid=os.getpid())
        _stamp(run_dir, {"process_start_ticks": "1"})
        manifest = json.loads((run_dir / "attempt.json").read_text(encoding="utf-8"))
        assert scan.agent_is_alive(manifest, os.getpid()) is False
        assert scan.scan_battery(battery, time.time()).as_dict()["attempts"][0]["state"] == "orphaned"

        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        assert state.kill("p/try01")["closed_out"] is True
        print("ok  a recycled pid reads as orphaned and is closed out, never signalled")


def test_discovery_prefers_newest_and_honours_pin() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        older = root / "20260725_100000"
        newer = root / "20260725_160000"
        make_attempt(older, "p", 1, steps=healthy_steps(2), state="finished", outcome="completed")
        time.sleep(0.02)
        make_attempt(newer, "p", 1, steps=healthy_steps(2), pid=os.getpid())
        os.utime(older, (time.time() - 500, time.time() - 500))

        found = scan.find_batteries(root)
        assert [p.name for p in found] == [newer.name, older.name], [p.name for p in found]

        auto = WatchState(bench_root=root, fixed_battery=None, discord=Discord(enabled=False))
        assert auto.resolve_battery() == newer

        pinned = WatchState(bench_root=root, fixed_battery=older, discord=Discord(enabled=False))
        assert pinned.resolve_battery() == older
        # The header's picker is how a reviewer reaches the other run, so the list of what exists
        # may not depend on how the watcher was started.
        listed = {entry["id"] for entry in pinned.snapshot()["discovered"]}
        assert listed == {newer.name, older.name}, listed
        print("ok  auto-discovery takes the newest battery; --run-dir pins one")


def test_rename_moves_the_dir_and_refuses_while_anything_is_writing() -> None:
    """The picker's pencil. A battery's id IS its directory name, so a rename is a `mv`."""
    from sari_bench.storage import RUNNER_LOCK, file_lock

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        battery = root / "20260725_100000"
        make_attempt(battery, "p", 1, steps=healthy_steps(2), state="finished", outcome="completed")
        (battery / scan.BATTERY_MANIFEST).write_text("{}", encoding="utf-8")
        stale = time.time() - scan.LIVE_GRACE_SECONDS - 60
        for path in (battery / scan.BATTERY_MANIFEST, battery):
            os.utime(path, (stale, stale))

        state = WatchState(bench_root=root, fixed_battery=None, discord=Discord(enabled=False),
                           min_interval=0.0)
        assert state.snapshot()["battery_id"] == battery.name

        # An unknown id must be an error, never a fallback onto the watched battery: falling back
        # here would rename the wrong directory and answer as though it had worked.
        assert state.rename_battery("does_not_exist", "whatever")["ok"] is False
        assert battery.is_dir(), "a rename of an unknown id touched the watched battery"

        assert state.rename_battery(battery.name, "  ../;  ")["ok"] is False, "a traversal was taken"
        assert not (root.parent / "..").is_symlink()

        renamed = state.rename_battery(battery.name, "budget meals/v2")
        assert renamed["ok"] and renamed["battery_id"] == "budget-meals-v2", renamed
        assert not battery.exists() and (root / "budget-meals-v2").is_dir()
        # The rename moved the run the watcher follows. It has to still be following it, and by the
        # new name, or the next poll reads this as a battery that has just appeared.
        assert state.battery == root / "budget-meals-v2"
        assert state.snapshot()["battery_id"] == "budget-meals-v2"

        moved = root / "budget-meals-v2"
        (root / "taken").mkdir()
        assert state.rename_battery(moved.name, "taken")["ok"] is False, "clobbered a sibling dir"
        assert moved.is_dir()

        with file_lock(moved / RUNNER_LOCK):
            refused = state.rename_battery(moved.name, "while-running")
        assert refused["ok"] is False and "runner" in refused["error"], refused
        assert moved.is_dir(), "a live battery was renamed out from under its runner"

        print("ok  a bench run renames on disk, and not while a runner or retry holds it")


def test_a_browser_can_read_a_battery_the_watcher_is_not_watching() -> None:
    """Picking another bench run in the header is a read: it must not move what the watcher follows."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        older = root / "20260725_100000"
        newer = root / "20260725_160000"
        make_attempt(older, "old", 1, steps=healthy_steps(2), state="finished", outcome="completed")
        make_attempt(newer, "new", 1, steps=healthy_steps(3), pid=os.getpid())
        os.utime(older, (time.time() - 500, time.time() - 500))

        state = WatchState(bench_root=root, fixed_battery=None, discord=Discord(enabled=False),
                           min_interval=0.0)
        assert state.view()["battery_id"] == newer.name
        assert state.view("newest")["battery_id"] == newer.name

        other = state.view(older.name)
        assert other["battery_id"] == older.name
        assert [a["prompt_id"] for a in other["attempts"]] == ["old"]
        assert other["watching_id"] == newer.name
        # The watched battery is untouched by the detour, and its notifier keeps being driven.
        assert state.battery == newer

        # An id naming nothing on disk resolves to the watched battery rather than to a path.
        for bogus in ("../..", "/etc", "nope"):
            assert state.view(bogus)["battery_id"] == newer.name, bogus
        assert state.battery_for("../..") == newer

        # Attempt routes follow the same scope, or the reviewer would read one run's log under
        # another run's tile.
        assert state.log_tail("old/try01", battery_id=older.name)["lines"][-1] == "log line 49"
        assert state.log_tail("old/try01")["lines"] == []
        assert state.frame_path("old/try01", battery_id=older.name) is not None
        assert state.frame_path("old/try01") is None
        print("ok  a browser reads any battery on disk without moving the watched one")


def test_a_running_battery_is_marked_live() -> None:
    """The picker's green dot. `.runner.lock` is the exact signal; recency covers a resumed runner."""
    from sari_bench.storage import RUNNER_LOCK, file_lock

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        make_attempt(battery, "p", 1, steps=healthy_steps(2), pid=os.getpid())
        (battery / scan.BATTERY_MANIFEST).write_text("{}", encoding="utf-8")

        with file_lock(battery / RUNNER_LOCK):
            assert scan.battery_live(battery) is True, "a held runner lock did not read as live"
        # Released, but everything in it was written a second ago, so the fallback still says live.
        assert scan.battery_live(battery) is True

        stale = time.time() - scan.LIVE_GRACE_SECONDS - 60
        for path in (battery, battery / scan.BATTERY_MANIFEST, battery / RUNNER_LOCK):
            os.utime(path, (stale, stale))
        assert scan.battery_live(battery) is False, "a finished battery still read as live"

        # A resumed runner holds no lock, and appending its first finish is what shows it is back.
        (battery / scan.ATTEMPTS_INDEX).write_text("{}\n", encoding="utf-8")
        assert scan.battery_live(battery) is True

        described = scan.describe_batteries([battery])
        assert described[0]["id"] == "b" and described[0]["live"] is True, described
        print("ok  a battery with a live runner is marked live; a finished one is not")


def test_rotated_requeue_dir_stays_separate() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=looping_steps(6), pid=os.getpid())
        aside = run_dir.with_name("try01.requeue00")
        run_dir.rename(aside)
        make_attempt(battery, "p", 1, steps=healthy_steps(4), pid=os.getpid())

        view = scan.scan_battery(battery, time.time()).as_dict()
        by_key = {a["key"]: a for a in view["attempts"]}
        assert set(by_key) == {"p/try01", "p/try01.requeue00"}, set(by_key)
        # The fresh attempt must show ITS four steps, not ten merged ones.
        assert by_key["p/try01"]["step"] == 4, by_key["p/try01"]["step"]
        assert by_key["p/try01.requeue00"]["step"] == 6
        print("ok  a rotated-aside requeue dir does not merge into its replacement")


def test_http_surface_and_traversal_refusal() -> None:
    import urllib.request
    from http.server import ThreadingHTTPServer
    from threading import Thread

    from sari_bench.watch.server import Handler

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        make_attempt(battery, "p", 1, steps=healthy_steps(5), pid=os.getpid())
        other = Path(temp) / "b2"
        make_attempt(other, "q", 1, steps=healthy_steps(2), pid=os.getpid())
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        assert _safe_run_dir(battery, "../../etc") is None, "traversal was not refused"
        assert _safe_run_dir(battery, "p/try01") is not None

        Handler.state = state
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            payload = json.loads(urllib.request.urlopen(f"{base}/api/state", timeout=5).read())
            assert payload["battery_id"] == "b"
            assert payload["mode"] == "pinned"
            assert len(payload["attempts"]) == 1

            frame = urllib.request.urlopen(f"{base}/api/attempt/p/try01/frame.png", timeout=5)
            assert frame.headers["Content-Type"] == "image/png"
            etag = frame.headers["ETag"]
            assert frame.read()[:8] == _PNG[:8]
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"{base}/api/attempt/p/try01/frame.png",
                    headers={"If-None-Match": etag},
                ), timeout=5)
            except urllib.error.HTTPError as unchanged:
                assert unchanged.code == 304
                assert unchanged.headers["ETag"] == etag
                assert unchanged.read() == b""
            else:
                raise AssertionError("an unchanged live frame was transferred again")

            log = json.loads(urllib.request.urlopen(f"{base}/api/attempt/p/try01/log", timeout=5).read())
            assert log["lines"][-1] == "log line 49", log["lines"][-1]
            full_log = json.loads(urllib.request.urlopen(
                f"{base}/api/attempt/p/try01/log?full=1", timeout=5
            ).read())
            assert full_log["lines"] == [f"log line {i}" for i in range(50)]

            # `?battery=` is what the header's picker sends. Every route has to honour it, or the
            # page would show one run's tiles over another run's frames and logs.
            switched = json.loads(urllib.request.urlopen(
                f"{base}/api/state?battery=b2", timeout=5
            ).read())
            assert switched["battery_id"] == "b2", switched["battery_id"]
            assert switched["watching_id"] == "b"
            scoped = json.loads(urllib.request.urlopen(
                f"{base}/api/attempt/q/try01/log?lines=1&battery=b2", timeout=5
            ).read())
            assert scoped["lines"] == ["log line 49"], scoped["lines"]
            unscoped = json.loads(urllib.request.urlopen(
                f"{base}/api/attempt/q/try01/log", timeout=5
            ).read())
            assert unscoped["lines"] == [], "an unscoped key read another battery's log"

            page = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
            assert "const FRAME_POLL_MS = 250" in page
            assert "Sari Bench" in page
        finally:
            server.shutdown()
            server.server_close()
        print("ok  HTTP serves state, frames and logs; traversal keys are refused")


def test_log_cursor_appends_only_what_is_new() -> None:
    """The dashboard terminal appends, so the endpoint must hand back deltas, not the tail again."""
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(2), pid=os.getpid())
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        log = run_dir / "agent.log"

        first = state.log_tail("p/try01", lines=5)
        assert first["lines"] == [f"log line {i}" for i in range(45, 50)], first["lines"]
        assert first["offset"] == log.stat().st_size
        assert first["size"] == log.stat().st_size
        assert first["partial"] == "" and first["reset"] is False

        full = state.log_tail("p/try01", full=True)
        assert full["lines"] == [f"log line {i}" for i in range(50)], full["lines"]
        assert full["offset"] == log.stat().st_size

        # Nothing written since: an empty delta, and the cursor stays put.
        quiet = state.log_tail("p/try01", since=first["offset"])
        assert quiet["lines"] == [] and quiet["offset"] == first["offset"]
        assert quiet["size"] == first["size"]

        with log.open("a", encoding="utf-8") as handle:
            handle.write("fresh line\n")
        delta = state.log_tail("p/try01", since=first["offset"])
        assert delta["lines"] == ["fresh line"], delta["lines"]
        assert delta["size"] > first["size"]

        # A half-written line is previewed but does not advance the cursor, so it is delivered once
        # more - whole - when its newline lands, rather than arriving in two pieces.
        with log.open("a", encoding="utf-8") as handle:
            handle.write("half a li")
        mid = state.log_tail("p/try01", since=delta["offset"])
        assert mid["lines"] == [] and mid["partial"] == "half a li", mid
        assert mid["offset"] == delta["offset"]
        with log.open("a", encoding="utf-8") as handle:
            handle.write("ne\n")
        whole = state.log_tail("p/try01", since=mid["offset"])
        assert whole["lines"] == ["half a line"] and whole["partial"] == "", whole

        # Truncation (or rotation) puts the cursor past the end; the client is told to start over.
        log.write_text("restarted\n", encoding="utf-8")
        after = state.log_tail("p/try01", since=whole["offset"])
        assert after["reset"] is True and after["lines"] == ["restarted"], after
        print("ok  the log endpoint serves byte-cursor deltas, partial lines and truncation resets")


def test_full_log_is_not_limited_by_tail_window() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(2), pid=os.getpid())
        log = run_dir / "agent.log"
        expected = [f"{i:04d} " + ("x" * 80) for i in range(3000)]
        log.write_text("\n".join(expected) + "\n", encoding="utf-8")
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        full = state.log_tail("p/try01", full=True)
        assert full["lines"] == expected
        assert full["offset"] == log.stat().st_size
        print("ok  full-log reads are not limited by the tail line or byte windows")


def test_snapshot_exposes_log_size_for_change_driven_terminals() -> None:
    """A tile must notice late output even after the attempt becomes orphaned or finished."""
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(2), pid=999_999_999)

        first = scan.scan_battery(battery, time.time()).as_dict()["attempts"][0]
        assert first["state"] == "orphaned"
        assert first["log_bytes"] == (run_dir / "agent.log").stat().st_size

        with (run_dir / "agent.log").open("a", encoding="utf-8") as handle:
            handle.write("late runner cleanup\n")
        second = scan.scan_battery(battery, time.time()).as_dict()["attempts"][0]
        assert second["log_bytes"] > first["log_bytes"]
        print("ok  snapshots expose late log growth after an orphan transition")


def test_log_query_params_are_clamped() -> None:
    from sari_bench.watch.server import LOG_MAX_LINES, _int_param

    query = {"lines": ["9999"], "since": ["-4"], "junk": ["nope"]}
    assert _int_param(query, "lines", 25, 1, LOG_MAX_LINES) == LOG_MAX_LINES
    assert _int_param(query, "since", None, 0, None) == 0
    assert _int_param(query, "junk", 7, 0, None) == 7, "unparseable values must fall back"
    assert _int_param({}, "since", None, 0, None) is None, "absent means bootstrap, not zero"
    print("ok  log query params are clamped and fall back safely")


def test_report_and_kill_stamp() -> None:
    from sari_bench.report import collect

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(4), state="finished",
                               outcome="completed")
        _write(run_dir / "summary.json", json.dumps({
            "task": "t", "arm": "graph", "success": True, "legs_planned": 1, "legs_completed": 1,
            "legs": [{"type": "pickup", "text": "go", "success": True, "end_reason": "halt_granted",
                      "timesteps": 4, "llm_calls": 9, "errors": 0, "halts_refused": 1}],
        }))
        _write(battery / "attempts.jsonl", json.dumps({
            "prompt_id": "p", "attempt": 1, "outcome": "completed", "success": True,
            "wall_seconds": 61.0, "run_dir": str(run_dir), "requeues": 0, "sandbox_id": "sb-p",
        }) + "\n")

        attempts, legs = collect(battery)
        assert len(attempts) == 1 and len(legs) == 1
        assert attempts[0]["success"] is True
        assert attempts[0]["wall_minutes"] == 1.02, attempts[0]["wall_minutes"]
        assert legs[0]["timesteps"] == 4 and legs[0]["halts_refused"] == 1

        # A finished attempt cannot be killed.
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        result = state.kill("p/try01")
        assert result["ok"] is False and "finished" in result["error"], result
        assert state.kill("../../etc")["ok"] is False
        print("ok  report flattens attempts/legs; kill refuses finished and traversal keys")


def _finish(run_dir: Path, *, outcome: str, success: bool = False,
            end_reason: str = "halt_granted") -> None:
    """Stamps an attempt closed the way the runner does when its agent exits."""
    manifest = json.loads((run_dir / "attempt.json").read_text(encoding="utf-8"))
    manifest.update({"state": "finished", "outcome": outcome, "success": success,
                     "end_reason": end_reason, "wall_seconds": 61.0, "pid": None})
    _write(run_dir / "attempt.json", json.dumps(manifest))


def _recording_discord() -> tuple[Discord, list[tuple[dict, Path | None]]]:
    """A Discord that is 'enabled' but records instead of sending."""
    posts: list[tuple[dict, Path | None]] = []
    discord = Discord(webhook_url="http://127.0.0.1:1/never-called")
    discord._post = lambda payload, attachment=None: posts.append((payload, attachment))  # type: ignore[method-assign]
    return discord, posts


def _finish_titles(posts: list[tuple[dict, Path | None]]) -> list[str]:
    titles = [p[0]["embeds"][0].get("title", "") for p in posts]
    return [t for t in titles if t.startswith("Attempt")]


def test_target_bitrate_math() -> None:
    from sari_bench import video

    # 150 frames at 6 fps is 25s; 8 MB of payload over 25s with 5% held back.
    assert video.target_bitrate(150, 6.0, 8_000_000) == 2_432_000, video.target_bitrate(150, 6.0)
    assert video.target_bitrate(150, 4.0, 8_000_000) == 1_621_333

    # A very long attempt hits the floor rather than asking for an unwatchable bitrate.
    assert video.target_bitrate(10_000, 6.0, 8_000_000) == video.MIN_VIDEO_BITRATE

    # A one-frame clip must not divide by a sub-second duration and blow up.
    assert video.target_bitrate(1, 6.0, 8_000_000) == int(8_000_000 * 8 * 0.95)
    print("ok  upload bitrate is derived from clip duration and clamped at both ends")


def test_every_finish_notifies_once() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        _write(battery / "battery.json", json.dumps({"planned_attempts": 9, "arm": "graph"}))
        make_attempt(battery, "won", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True)
        make_attempt(battery, "missed", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=False)
        make_attempt(battery, "broke", 1, steps=healthy_steps(3), state="finished",
                     outcome="agent_error")
        make_attempt(battery, "killed", 1, steps=healthy_steps(3), state="finished",
                     outcome="operator_kill")

        discord, posts = _recording_discord()
        state = WatchState(bench_root=Path(temp), fixed_battery=battery, discord=discord,
                           replay=None, min_interval=0.0)
        state.snapshot(force=True)

        titles = _finish_titles(posts)
        assert len(titles) == 4, titles
        assert any("succeeded" in t and "won" in t for t in titles), titles
        assert any("goal not met" in t and "missed" in t for t in titles), titles
        assert any("failed" in t and "broke" in t for t in titles), titles
        assert any("killed" in t and "killed" in t for t in titles), titles

        colors = {p[0]["embeds"][0]["title"].split(":")[0]: p[0]["embeds"][0]["color"]
                  for p in posts if p[0]["embeds"][0].get("title", "").startswith("Attempt")}
        assert colors["Attempt succeeded"] == 0x4FA96B, colors
        assert colors["Attempt failed"] == 0xE0553F, colors

        state.snapshot(force=True)
        assert len(_finish_titles(posts)) == 4, "a second pass re-announced finishes"
        print("ok  every halt is announced exactly once, successes included")


def test_agent_error_is_failure_and_automatic_invalid() -> None:
    """Old contradictory manifests are repaired in the view and excluded exactly like an E vote."""
    from sari_bench.report import collect

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        make_attempt(
            battery, "crashed", 1, steps=healthy_steps(2), state="finished",
            outcome="agent_error", success=True,
        )

        view = scan.scan_battery(battery, time.time()).as_dict()
        attempt = view["attempts"][0]
        assert attempt["outcome"] == "agent_error"
        assert attempt["success"] is False
        assert attempt["verified"] is True
        assert attempt["verified_verdict"] == "invalid"
        assert attempt["verified_success"] is None
        assert view["counts"].get("success", 0) == 0
        assert view["counts"]["verified_invalid"] == 1
        assert view["counts"].get("awaiting_verdict", 0) == 0

        rows, _ = collect(battery)
        assert rows[0]["success"] is False
        assert rows[0]["verified_verdict"] == "invalid"
        assert rows[0]["success_final"] == ""
    print("ok  agent_error is a failure and automatically excluded as invalid")


def test_finish_attaches_replay_mp4() -> None:
    import shutil as _shutil

    from sari_bench import video
    from sari_bench.watch.replay import ReplayNotifier

    if _shutil.which("ffmpeg") is None:
        print("--  skipped replay render test: ffmpeg not on PATH")
        return

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        # Start it live, so the watcher sees the halt happen rather than finding it already done -
        # a finish that predates the watcher is deliberately seeded silently.
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(12), pid=os.getpid(),
                               frame_size=(320, 240))

        discord, posts = _recording_discord()
        worker = ReplayNotifier(discord, max_bytes=8_000_000, width=320, fps=6.0)
        assert worker.announce_enabled
        worker.start()
        try:
            state = WatchState(bench_root=Path(temp), fixed_battery=battery, discord=discord,
                               replay=worker, min_interval=0.0)
            state.snapshot(force=True)
            assert _finish_titles(posts) == []

            _finish(run_dir, outcome="completed", success=True)
            state.snapshot(force=True)
            worker._queue.join()

            attached = [p[1] for p in posts if p[0]["embeds"][0].get("title", "").startswith("Attempt")]
            assert len(attached) == 1, attached
            clip = attached[0]
            assert clip == run_dir / video.UPLOAD_NAME, clip
            size = clip.stat().st_size
            assert 0 < size <= 8_000_000, size
            # The full dashboard replay is queued by the same finish transition, without waiting
            # for a reviewer to open the modal.
            replay_clip = run_dir / video.REPLAY_NAME
            assert replay_clip.is_file() and replay_clip.stat().st_size > 0

            stamp = clip.stat().st_mtime_ns
            assert video.render_for_upload(run_dir, max_bytes=8_000_000, width=320) == clip
            assert clip.stat().st_mtime_ns == stamp, "an existing in-budget clip was re-rendered"

            state.snapshot(force=True)
            worker._queue.join()
            assert len(_finish_titles(posts)) == 1, "the halt was announced twice"
        finally:
            worker.stop()
        print(f"ok  a halt posts a {size / 1e6:.2f} MB replay clip, rendered once and reused")


def test_replay_seed_suppresses_backfill() -> None:
    from sari_bench.watch.replay import ReplayNotifier

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        for name in ("a", "b", "c"):
            make_attempt(battery, name, 1, steps=healthy_steps(2), state="finished",
                         outcome="completed", success=True)

        discord, posts = _recording_discord()
        worker = ReplayNotifier(discord)
        worker.start()
        try:
            state = WatchState(bench_root=Path(temp), fixed_battery=battery, discord=discord,
                               replay=worker, min_interval=0.0)
            state.snapshot(force=True)
            worker._queue.join()
            assert _finish_titles(posts) == [], "a restart replayed finishes that predate it"

            make_attempt(battery, "d", 1, steps=healthy_steps(2), state="finished",
                         outcome="agent_error")
            state.snapshot(force=True)
            worker._queue.join()
            titles = _finish_titles(posts)
            assert len(titles) == 1 and "d" in titles[0], titles
        finally:
            worker.stop()
        print("ok  a watcher restart seeds old finishes silently and still catches new ones")


def test_replay_worker_prioritizes_announcements_and_rejects_corrupt_clips() -> None:
    from sari_bench import video
    from sari_bench.watch.replay import RENDERING, ReplayNotifier

    events: list[str] = []

    class RecordingDiscord:
        enabled = True

        def attempt_finished(self, _attempt: dict[str, Any], *, video: Path | None = None) -> None:
            events.append("posted")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        archive_dir = root / "archive"
        announce_dir = root / "announce"
        archive_dir.mkdir()
        announce_dir.mkdir()
        # Nonempty is insufficient: a killed ffmpeg has no moov atom and must be replaced.
        (archive_dir / video.REPLAY_NAME).write_bytes(b"\x00\x00\x00\x18ftypmp42partial")

        worker = ReplayNotifier(RecordingDiscord())  # type: ignore[arg-type]
        assert worker.enqueue("archive/try01", archive_dir) == RENDERING
        assert worker.submit({
            "key": "announce/try01",
            "run_dir": str(announce_dir),
            "end_reason": "halt_granted",
        })

        original_render = video.render
        original_upload = video.render_for_upload

        def fake_render(*_args: Any, **_kwargs: Any) -> None:
            events.append("archive")
            return None

        def fake_upload(*_args: Any, **_kwargs: Any) -> None:
            events.append("announcement")
            return None

        video.render = fake_render  # type: ignore[assignment]
        video.render_for_upload = fake_upload  # type: ignore[assignment]
        worker.start()
        try:
            worker._queue.join()
        finally:
            worker.stop()
            video.render = original_render
            video.render_for_upload = original_upload

        assert events == ["announcement", "posted", "archive"], events
    print("ok  notification clips outrank archive renders and corrupt mp4s are re-rendered")


def test_full_replay_queue_falls_back_to_text_notification() -> None:
    from sari_bench.watch.replay import QUEUE_MAXSIZE, ReplayNotifier

    class EnabledDiscord:
        enabled = True

    with tempfile.TemporaryDirectory() as temp:
        worker = ReplayNotifier(EnabledDiscord())  # type: ignore[arg-type]
        root = Path(temp)
        for index in range(QUEUE_MAXSIZE):
            run_dir = root / str(index)
            run_dir.mkdir()
            worker.enqueue(f"p/try{index:02d}", run_dir)
        key = "urgent/try01"
        assert not worker.submit({
            "key": key,
            "run_dir": str(root),
            "end_reason": "halt_granted",
        })
        assert key not in worker._claimed
    print("ok  a full replay queue triggers immediate text-notification fallback")


def test_finished_run_queues_dashboard_replay_without_discord() -> None:
    """Finishing is the render trigger; opening the modal is only a fallback for older runs."""
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(2), state="finished",
                               outcome="completed", success=True)
        discord, posts = _recording_discord()
        discord.enabled = False
        queued: list[tuple[str, Path]] = []

        class RecordingReplay:
            def seed(self, _attempts: list[dict[str, Any]]) -> None:
                pass

            def enqueue(self, key: str, path: Path) -> str:
                queued.append((key, path))
                return "rendering"

        state = WatchState(
            bench_root=Path(temp),
            fixed_battery=battery,
            discord=discord,
            replay=RecordingReplay(),  # type: ignore[arg-type]
            min_interval=0.0,
        )
        state.snapshot(force=True)

        assert queued == [("p/try01", run_dir)], queued
        assert posts == [], "auto-rendering unexpectedly enabled Discord notifications"
    print("ok  a finished run queues its dashboard replay even with Discord off")


def test_every_finished_attempt_is_reviewable() -> None:
    """The gate on the verdict buttons.

    How an attempt ended does not affect whether a human can judge it. Only unfinished attempts are
    withheld because their outcome can still change.
    """
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        make_attempt(battery, "granted", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")
        make_attempt(battery, "backstop", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="completed_no_stop")
        make_attempt(battery, "capped", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=False, end_reason="step_cap")
        make_attempt(battery, "timedout", 1, steps=healthy_steps(3), state="finished",
                     outcome="harness_timeout", success=False, end_reason="time_cap")
        make_attempt(battery, "forced", 1, steps=healthy_steps(3), state="finished",
                     outcome="operator_kill", success=False, end_reason="halt_forced")
        make_attempt(battery, "live", 1, steps=healthy_steps(3), pid=os.getpid())

        view = scan.scan_battery(battery, time.time()).as_dict()
        by_id = {a["prompt_id"]: a for a in view["attempts"]}
        assert by_id["granted"]["verifiable"] is True
        assert by_id["backstop"]["verifiable"] is True
        assert by_id["capped"]["verifiable"] is True
        assert by_id["timedout"]["verifiable"] is True
        assert by_id["forced"]["verifiable"] is True
        assert by_id["live"]["verifiable"] is False, "a running attempt offered a verdict"
        assert all(a["verified"] is False for a in view["attempts"])
        assert all(a["verified_success"] is None for a in view["attempts"])
        assert view["counts"]["awaiting_verdict"] == 5, view["counts"]
        print("ok  every finished attempt is offered for human review")


def test_verdict_round_trip_and_refusals() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        make_attempt(battery, "p", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")
        make_attempt(battery, "capped", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="step_cap")
        make_attempt(battery, "live", 1, steps=healthy_steps(3), pid=os.getpid())
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        result = state.verdict("p/try01", "fail", note="never actually grabbed it", by="tester")
        assert result["ok"] is True, result
        manifest = json.loads((battery / "p" / "try01" / "attempt.json").read_text())
        assert manifest["verified_success"] is False
        assert manifest["verified_by"] == "tester"
        assert manifest["verified_note"] == "never actually grabbed it"
        # The predicate's own verdict is untouched: the disagreement is the point.
        assert manifest["success"] is True, "the human verdict overwrote the measured one"

        view = scan.scan_battery(battery, time.time()).as_dict()
        reviewed = {a["prompt_id"]: a for a in view["attempts"]}["p"]
        assert reviewed["verified"] is True and reviewed["verified_success"] is False
        assert reviewed["success"] is True
        assert view["counts"]["verified"] == 1 and view["counts"]["verified_fail"] == 1
        assert view["counts"]["disagree"] == 1, view["counts"]

        # Correctable: the other button overwrites.
        assert state.verdict("p/try01", "pass")["ok"] is True
        manifest = json.loads((battery / "p" / "try01" / "attempt.json").read_text())
        assert manifest["verified_success"] is True

        # Cleared, an attempt reads as never looked at - not as a verdict of False.
        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": True}
        manifest = json.loads((battery / "p" / "try01" / "attempt.json").read_text())
        assert "verified_success" not in manifest and "verified_by" not in manifest
        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": False}

        # The server re-checks eligibility: capped is finished and allowed, live is not.
        assert state.verdict("capped/try01", "fail")["ok"] is True
        refused = state.verdict("live/try01", "pass")
        assert refused["ok"] is False and "only finished attempts" in refused["error"], refused
        assert state.verdict("../../etc", "pass")["ok"] is False
        assert state.clear_verdict("../../etc")["ok"] is False
        print("ok  verdicts round-trip beside `success`, overwrite, clear, and refuse unfinished attempts")


def test_invalid_verdict_is_excluded_rather_than_failed() -> None:
    """The third verdict: a run the reviewer throws out, counted in neither column.

    The load-bearing property is that `verified_success` is ABSENT, not False. Every existing
    reader - the report's `success_final`, the runner's winner check, any pivot table someone
    already built - treats that key as the whole answer, so a broken run left behind a False there
    would be totalled as a human-confirmed failure by all of them at once.
    """
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        _write(battery / "battery.json", json.dumps({"tries": 2, "prompts": [{"id": "p"}]}))
        make_attempt(battery, "p", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")
        # No pid: an unstarted sibling is still cancellable and gets its stop stamped, but nothing
        # here can signal a real process - least of all this test's own.
        make_attempt(battery, "p", 2, steps=healthy_steps(3))
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        manifest_path = battery / "p" / "try01" / "attempt.json"
        assert state.verdict("p/try01", "invalid", by="tester")["ok"] is True
        manifest = json.loads(manifest_path.read_text())
        assert manifest["verified_verdict"] == "invalid"
        assert "verified_success" not in manifest, "an excluded run reads as a verified failure"
        assert manifest["success"] is True, "the human verdict overwrote the measured one"

        view = scan.scan_battery(battery, time.time()).as_dict()
        judged = {a["prompt_id"]: a for a in view["attempts"] if a["attempt"] == 1}["p"]
        assert judged["verified"] is True and judged["verified_verdict"] == "invalid"
        assert judged["verified_success"] is None
        assert view["counts"]["verified_invalid"] == 1, view["counts"]
        assert "verified_fail" not in view["counts"], view["counts"]
        # The predicate said success and the reviewer did not say fail - there is no disagreement,
        # only an attempt nobody is willing to draw a conclusion from.
        assert "disagree" not in view["counts"], view["counts"]

        # An excluded try decides nothing, so the prompt's remaining tries must be left alone.
        plan = json.loads((battery / "battery.json").read_text())
        assert not plan.get("human_verified_winners"), plan
        sibling = json.loads((battery / "p" / "try02" / "attempt.json").read_text())
        assert "stop_reason" not in sibling, sibling

        # Downgrading a pass to invalid has to take the old boolean with it.
        assert state.verdict("p/try01", "pass", by="tester")["ok"] is True
        assert json.loads(manifest_path.read_text())["verified_success"] is True
        assert state.verdict("p/try01", "invalid", by="tester")["ok"] is True
        manifest = json.loads(manifest_path.read_text())
        assert "verified_success" not in manifest, manifest
        assert manifest["verified_verdict"] == "invalid"

        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": True}
        assert "verified_verdict" not in json.loads(manifest_path.read_text())

        assert state.verdict("p/try01", "bogus")["ok"] is False
        print("ok  an invalid verdict excludes a run instead of failing it, and cancels no siblings")


def test_already_successful_verdict_is_excluded_and_applied_automatically() -> None:
    """The fourth verdict: a try halted because a sibling had already won the prompt.

    It is excluded on exactly the terms `invalid` is - no `verified_success` at all, nothing added
    to either column - but it is not a finding about the harness, and the runner's own
    cancellations carry it without anyone having to press A.
    """
    from sari_bench.report import collect

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        _write(battery / "battery.json", json.dumps({"tries": 2, "prompts": [{"id": "p"}]}))
        make_attempt(battery, "p", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")
        # What the runner writes for a try it never dispatched because the prompt was already won.
        make_attempt(battery, "p", 2, steps=healthy_steps(1), state="finished",
                     outcome="skipped", success=False, end_reason="already_successful")
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)

        view = scan.scan_battery(battery, time.time()).as_dict()
        cancelled = {a["key"]: a for a in view["attempts"]}["p/try02"]
        assert cancelled["verified"] is True
        assert cancelled["verified_verdict"] == "already_successful"
        assert cancelled["verified_success"] is None
        assert view["counts"]["verified_already_successful"] == 1, view["counts"]
        # The whole point of the automatic mark: it leaves the review queue instead of sitting in it.
        assert view["counts"].get("awaiting_verdict", 0) == 1, view["counts"]
        assert "disagree" not in view["counts"], view["counts"]

        rows = {row["attempt"]: row for row in collect(battery)[0]}
        assert rows[2]["verified_verdict"] == "already_successful"
        assert rows[2]["verified_success"] == ""
        assert rows[2]["success_final"] == "", "an excluded try answered the prompt"
        assert rows[2]["verdict_agrees"] == ""

        # By hand, on a try that ran: same exclusion, and no boolean left behind by an earlier pass.
        manifest_path = battery / "p" / "try01" / "attempt.json"
        assert state.verdict("p/try01", "pass", by="tester")["ok"] is True
        assert json.loads(manifest_path.read_text())["verified_success"] is True
        result = state.verdict("p/try01", "already_successful", by="tester")
        assert result["ok"] is True and result["verified_success"] is None
        manifest = json.loads(manifest_path.read_text())
        assert manifest["verified_verdict"] == "already_successful"
        assert "verified_success" not in manifest, manifest
        assert manifest["success"] is True, "the verdict overwrote the measured answer"

        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": True}
        assert "verified_verdict" not in json.loads(manifest_path.read_text())
    print("ok  already_successful excludes a halted try and is applied to cancellations by itself")


def test_retry_config_preserves_completion_guard() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        _write(
            battery / "battery.json",
            json.dumps({
                "coordinator": "ws://127.0.0.1:9000",
                "completion_guard": "vlm",
            }),
        )
        state = WatchState(
            bench_root=Path(temp),
            fixed_battery=battery,
            discord=Discord(enabled=False),
            min_interval=0.0,
        )
        config = state._retry_config(battery, {"command": []})
        assert config["completion_guard"] == "vlm", config
    print("ok  watcher retry preserves the battery completion guard")


def test_invalid_verdict_in_the_report() -> None:
    from sari_bench.report import ATTEMPT_COLUMNS, collect

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        make_attempt(battery, "broken", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")
        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        assert state.verdict("broken/try01", "invalid", by="tester")["ok"] is True

        rows, _ = collect(battery)
        row = rows[0]
        assert set(ATTEMPT_COLUMNS) >= {"verified_verdict"}
        assert row["verified_verdict"] == "invalid"
        assert row["verified"] is True
        # Blank, never False: these three columns are what anyone groups by, and an excluded run
        # must fall out of every one of those groupings rather than land in the failure bucket.
        assert row["verified_success"] == "", repr(row["verified_success"])
        assert row["verdict_agrees"] == "", repr(row["verdict_agrees"])
        assert row["success_final"] == "", repr(row["success_final"])
        print("ok  the report reports an invalid run as excluded, not as a failure")


def test_success_verdict_cancels_only_same_prompt_siblings() -> None:
    """A JSON family label is grouping metadata, not a cancellation boundary."""
    from sari_bench.watch import server as server_mod

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        _write(
            battery / "battery.json",
            json.dumps({
                "tries": 4,
                "planned_attempts": 8,
                "prompts": [
                    {"id": "p", "family": "pickup", "prompt": "winner"},
                    {"id": "other", "family": "pickup", "prompt": "same family"},
                ],
            }),
        )
        make_attempt(battery, "p", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")
        starting = make_attempt(battery, "p", 2, steps=[], state="starting", pid=None)
        running = make_attempt(battery, "p", 3, steps=healthy_steps(1), pid=424242)
        unrelated = make_attempt(
            battery, "other", 1, steps=healthy_steps(1), pid=434343
        )
        state = WatchState(
            bench_root=Path(temp),
            fixed_battery=battery,
            discord=Discord(enabled=False),
            min_interval=0.0,
        )

        signalled: list[tuple[int, int]] = []
        old_killpg, old_getpgid = server_mod.os.killpg, server_mod.os.getpgid
        server_mod.os.getpgid = lambda pid: pid
        server_mod.os.killpg = lambda pgid, sig: signalled.append((pgid, sig))
        try:
            result = state.verdict("p/try01", "pass", by="reviewer")
        finally:
            server_mod.os.killpg, server_mod.os.getpgid = old_killpg, old_getpgid

        assert result["siblings_stopped"] == 1, result
        assert result["siblings_skipped"] == 2, result  # starting try02 + queued try04
        assert [pid for pid, _sig in signalled] == [424242], signalled
        for path in (starting, running):
            manifest = json.loads((path / "attempt.json").read_text())
            assert manifest["stop_reason"] == "already_successful", manifest
            assert manifest["stop_requested_by"] == "reviewer", manifest
            assert manifest["winning_attempt_key"] == "p/try01", manifest
        assert "stop_reason" not in json.loads((unrelated / "attempt.json").read_text())

        durable = json.loads((battery / "battery.json").read_text())
        assert durable["human_verified_winners"]["p"]["winning_attempt_key"] == "p/try01"
        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": True}
        durable = json.loads((battery / "battery.json").read_text())
        assert "p" in durable["human_verified_winners"], "clearing restored cancelled siblings"

        # Failing another review is ordinary metadata and never grows the cancellation set.
        make_attempt(battery, "other", 2, steps=healthy_steps(2), state="finished",
                     outcome="completed", success=False, end_reason="halt_granted")
        failed = state.verdict("other/try02", "fail", by="reviewer")
        assert failed["siblings_stopped"] == 0 and failed["siblings_skipped"] == 0
        dashboard = (
            Path(server_mod.STATIC_DIR) / "dashboard.html"
        ).read_text(encoding="utf-8")
        assert "irreversibly stops or skips every other try of this prompt" in dashboard
        assert "will not restart them" in dashboard

        administrative = make_attempt(
            battery, "p", 4, steps=[], state="finished", outcome="skipped",
            success=False, end_reason="already_successful", frames=False,
        )
        admin_manifest = json.loads((administrative / "attempt.json").read_text())
        admin_manifest["winning_attempt_key"] = "p/try01"
        _write(administrative / "attempt.json", json.dumps(admin_manifest))

        class NoReplay:
            def request(self, *_args: Any) -> str:
                raise AssertionError("administrative skip requested a replay render")

        state.replay = NoReplay()  # type: ignore[assignment]
        assert state.replay_status("p/try04") == ("unavailable", None)
        discord, posts = _recording_discord()
        discord.attempt_finished({
            "key": "p/try04", "end_reason": "already_successful", "outcome": "skipped",
        })
        assert posts == [], "administrative skip emitted a Discord finish notification"
        print("ok  successful review durably cancels only sibling tries, and fail/clear do not undo it")


def test_response_body_ignores_client_disconnects_only() -> None:
    """A browser abandoning a video seek is routine; other write failures remain visible."""
    from sari_bench.watch.server import Handler

    class Disconnected:
        def __init__(self, error: OSError) -> None:
            self.error = error

        def write(self, _body: bytes) -> None:
            raise self.error

    handler = object.__new__(Handler)
    for error in (BrokenPipeError(), ConnectionResetError(), ConnectionAbortedError()):
        handler.wfile = Disconnected(error)  # type: ignore[assignment]
        handler._write_body(b"response")

    handler.wfile = Disconnected(OSError("disk-like write failure"))  # type: ignore[assignment]
    try:
        handler._write_body(b"response")
    except OSError as error:
        assert str(error) == "disk-like write failure"
    else:
        raise AssertionError("an unrelated response write error was swallowed")
    print("ok  client disconnects do not emit handler errors; unrelated write failures still do")


def test_verdict_and_replay_over_http() -> None:
    """The review surface end to end, without needing ffmpeg for the parts that are not the encode.

    The clip for the ready path is pre-placed, exactly as an already-announced halt would have left
    it: `request()` serves an in-budget replay.discord.mp4 straight off disk. That keeps the routing,
    the range handling and the verdict round trip covered on a host with no ffmpeg, and leaves only
    the encode itself to `test_finish_attaches_replay_mp4`.
    """
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from threading import Thread

    from sari_bench import video
    from sari_bench.watch.replay import ReplayNotifier
    from sari_bench.watch.server import Handler

    def request(path: str, *, data: bytes | None = None,
                headers: dict[str, str] | None = None) -> tuple[int, bytes, Any]:
        req = urllib.request.Request(f"{base}{path}", data=data, headers=headers or {},
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        run_dir = make_attempt(battery, "p", 1, steps=healthy_steps(6), state="finished",
                               outcome="completed", success=True, end_reason="halt_granted")
        clip_bytes = _complete_mp4(bytes(range(256)) * 8)
        (run_dir / video.REPLAY_NAME).write_bytes(clip_bytes)
        make_attempt(battery, "bare", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted", frames=False)

        # Discord OFF: the dashboard's review flow must work with no webhook configured at all.
        discord, posts = _recording_discord()
        discord.enabled = False
        worker = ReplayNotifier(discord, max_bytes=8_000_000, width=320, fps=6.0)
        assert worker.render_enabled is True, "rendering was disabled along with Discord"
        assert worker.announce_enabled is False
        worker.start()

        state = WatchState(bench_root=Path(temp), fixed_battery=battery, discord=discord,
                           replay=worker, min_interval=0.0)
        Handler.state = state
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            code, body, headers = request("/api/attempt/p/try01/replay.mp4")
            assert code == 200, (code, body[:200])
            assert headers["Content-Type"] == "video/mp4"
            assert headers["Accept-Ranges"] == "bytes", "the clip advertises as unseekable"
            assert body == clip_bytes
            total = len(clip_bytes)

            # Seekable, or the reviewer can only watch the clip straight through.
            code, chunk, headers = request("/api/attempt/p/try01/replay.mp4",
                                           headers={"Range": "bytes=0-99"})
            assert code == 206, code
            assert chunk == clip_bytes[:100], len(chunk)
            assert headers["Content-Range"] == f"bytes 0-99/{total}", headers["Content-Range"]

            # An open-ended range, which is what a browser sends to start streaming.
            code, chunk, headers = request("/api/attempt/p/try01/replay.mp4",
                                           headers={"Range": "bytes=64-"})
            assert code == 206 and chunk == clip_bytes[64:], len(chunk)
            assert headers["Content-Range"] == f"bytes 64-{total - 1}/{total}"

            # A suffix range, and one that runs off the end.
            code, chunk, _ = request("/api/attempt/p/try01/replay.mp4",
                                     headers={"Range": "bytes=-32"})
            assert code == 206 and chunk == clip_bytes[-32:], len(chunk)
            code, _, _ = request("/api/attempt/p/try01/replay.mp4",
                                 headers={"Range": f"bytes={total + 10}-"})
            assert code == 416, code

            # An attempt with no frames is terminal, not re-queued on every 2s poll.
            code, body_bare, _ = request("/api/attempt/bare/try01/replay.mp4")
            assert code == 202, (code, body_bare[:200])
            worker._queue.join()
            code, body_bare, _ = request("/api/attempt/bare/try01/replay.mp4")
            assert code == 409, (code, body_bare[:200])
            assert json.loads(body_bare)["status"] == "unavailable"

            code, body, _ = request("/api/attempt/p/try01/verdict",
                                    data=json.dumps({"success": False, "note": "wrong shelf"}).encode())
            assert code == 200 and json.loads(body)["ok"] is True, (code, body[:200])

            payload = json.loads(request("/api/state")[1])
            card = {a["prompt_id"]: a for a in payload["attempts"]}["p"]
            assert card["verified"] is True and card["verified_success"] is False
            assert card["success"] is True and card["verifiable"] is True
            assert payload["counts"]["disagree"] == 1, payload["counts"]

            code, body, _ = request("/api/attempt/p/try01/verdict/clear", data=b"")
            assert code == 200 and json.loads(body)["cleared"] is True

            # A malformed body is rejected rather than stored as a falsey verdict.
            code, _, _ = request("/api/attempt/p/try01/verdict", data=b'{"success": "yes"}')
            assert code == 400, code
            code, _, _ = request("/api/attempt/p/try01/verdict", data=b"not json")
            assert code == 400, code
            code, body_success, _ = request(
                "/api/attempt/p/try01/verdict",
                data=json.dumps({"success": True}).encode(),
            )
            success_payload = json.loads(body_success)
            assert code == 200, code
            assert success_payload["sibling_cancellations"] == {"stopped": 0, "skipped": 0}

            # The three-value field, and a body that names no verdict the server knows.
            code, body, _ = request("/api/attempt/p/try01/verdict",
                                    data=json.dumps({"verdict": "invalid"}).encode())
            assert code == 200 and json.loads(body)["ok"] is True, (code, body[:200])
            card = {a["prompt_id"]: a for a in json.loads(request("/api/state")[1])["attempts"]}["p"]
            assert card["verified_verdict"] == "invalid" and card["verified_success"] is None
            code, _, _ = request("/api/attempt/p/try01/verdict", data=b'{"verdict": "maybe"}')
            assert code == 400, code

            # Kill still routes correctly now that do_POST dispatches on three suffixes.
            code, body, _ = request("/api/attempt/p/try01/kill", data=b"")
            assert code == 400 and "finished" in json.loads(body)["error"], body

            retried: list[str] = []
            state.retry = lambda key, **_: (retried.append(key) or {
                "ok": True, "key": key, "retry_state": "stopping"
            })  # type: ignore[method-assign]
            code, body, _ = request("/api/attempt/p/try01/retry", data=b"")
            assert code == 202 and json.loads(body)["ok"] is True, body
            assert retried == ["p/try01"]

            # Renaming over HTTP, last because it moves the directory every assert above named.
            # A live battery is refused, so the fixture has to look finished first.
            stale = time.time() - scan.LIVE_GRACE_SECONDS - 60
            for path in (battery / scan.BATTERY_MANIFEST, battery / scan.ATTEMPTS_INDEX, battery):
                if path.exists():
                    os.utime(path, (stale, stale))
            code, body, _ = request("/api/battery/rename",
                                    data=json.dumps({"battery": "b", "name": "ablation run"}).encode())
            assert code == 200, (code, body[:200])
            assert json.loads(body)["battery_id"] == "ablation-run", body
            assert (Path(temp) / "ablation-run").is_dir() and not battery.exists()
            # --run-dir held the old path. If it did not move too, the watcher is now pinned to a
            # directory that no longer exists and every later poll answers with nothing.
            assert state.fixed_battery == Path(temp) / "ablation-run"
            assert json.loads(request("/api/state")[1])["battery_id"] == "ablation-run"
            code, body, _ = request("/api/battery/rename",
                                    data=json.dumps({"battery": "gone", "name": "x"}).encode())
            assert code == 400 and json.loads(body)["ok"] is False, body

            assert posts == [], "the dashboard review flow posted to Discord"
        finally:
            server.shutdown()
            server.server_close()
            worker.stop()
        print("ok  HTTP records verdicts and serves a seekable replay with Discord off")


def test_report_carries_the_human_verdict() -> None:
    from sari_bench.report import ATTEMPT_COLUMNS, collect

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        # "judged" gets the agent's summary.json; "unseen" deliberately does not, so the row has to
        # fall back to the manifest for `success` the way the dashboard does. The two surfaces
        # disagreeing on `success` would silently invert `verdict_agrees`.
        judged = make_attempt(battery, "judged", 1, steps=healthy_steps(3), state="finished",
                              outcome="completed", success=True, end_reason="halt_granted")
        _write(judged / "summary.json", json.dumps({"success": True, "legs": []}))
        make_attempt(battery, "unseen", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")

        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        assert state.verdict("judged/try01", "fail", by="tester")["ok"] is True

        rows = {row["prompt_id"]: row for row in collect(battery)[0]}

        disputed = rows["judged"]
        assert disputed["success"] is True, "the predicate's verdict was overwritten"
        assert disputed["verified"] is True and disputed["verified_success"] is False
        assert disputed["verdict_agrees"] is False, disputed["verdict_agrees"]
        assert disputed["success_final"] is False, "success_final ignored the human"
        assert disputed["verified_by"] == "tester"

        # Unreviewed must be blank, never False: a pivot must not read "not looked at" as "failed".
        unseen = rows["unseen"]
        assert unseen["verified"] is False
        assert unseen["verified_success"] == "", repr(unseen["verified_success"])
        assert unseen["verdict_agrees"] == "", repr(unseen["verdict_agrees"])
        assert unseen["success_final"] is True
        assert unseen["verifiable"] is True

        assert {"verified", "verified_success", "verdict_agrees",
                "success_final"} <= set(ATTEMPT_COLUMNS)

        # The CSV and the dashboard must not disagree about `success`, or the card says the human
        # contradicted the predicate while `verdict_agrees` says they matched.
        on_screen = {a["prompt_id"]: a for a in scan.scan_battery(battery, time.time()).as_dict()["attempts"]}
        for name, row in rows.items():
            assert row["success"] == on_screen[name]["success"], (name, row["success"])
        assert on_screen["judged"]["verified_success"] != on_screen["judged"]["success"]
        print("ok  attempts.csv reports the human verdict beside the predicate's, blank when unreviewed")


def test_report_csv_route_matches_the_cli() -> None:
    """The overview tab's export button and `sari_bench report` must be the same bytes.

    Two ways to get the same table is one way to get two answers, so the route is a thin wrapper
    around report.collect rather than its own flattening. What is pinned here is that it stays thin:
    same columns, same rows, same blank-not-False treatment of an unreviewed attempt.
    """
    import csv
    import io
    import urllib.request
    from http.server import ThreadingHTTPServer
    from threading import Thread

    from sari_bench.report import ATTEMPT_COLUMNS, collect
    from sari_bench.watch.server import Handler

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        judged = make_attempt(battery, "judged", 1, steps=healthy_steps(3), state="finished",
                              outcome="completed", success=True, end_reason="halt_granted")
        _write(judged / "summary.json", json.dumps({"success": True, "legs": []}))
        make_attempt(battery, "unseen", 1, steps=healthy_steps(3), state="finished",
                     outcome="completed", success=True, end_reason="halt_granted")

        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        assert state.verdict("judged/try01", "fail", by="tester")["ok"] is True

        Handler.state = state
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/api/report.csv", timeout=10
            ) as response:
                assert response.status == 200
                assert response.headers["Content-Type"].startswith("text/csv")
                # An attachment, so the browser saves it rather than rendering a wall of text.
                assert response.headers["Content-Disposition"] == 'attachment; filename="b-attempts.csv"'
                body = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        # Byte-for-byte what `python -m sari_bench report` would have written for this battery.
        expected = io.StringIO(newline="")
        writer = csv.DictWriter(expected, fieldnames=ATTEMPT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(collect(battery)[0])
        assert body == expected.getvalue(), "the route and the CLI produced different CSVs"

    served = list(csv.DictReader(io.StringIO(body)))
    assert list(served[0]) == ATTEMPT_COLUMNS, "the route drifted from the CLI's columns"
    assert [row["prompt_id"] for row in served] == ["judged", "unseen"], "rows lost or reordered"
    assert served[0]["verified"] == "True" and served[0]["verified_success"] == "False"
    assert served[0]["success"] == "True", "the predicate's verdict was overwritten"
    # Blank, never False: an unreviewed attempt must not read as "a human said it failed".
    assert served[1]["verified_success"] == "", repr(served[1]["verified_success"])
    assert served[1]["verdict_agrees"] == "", repr(served[1]["verdict_agrees"])
    print("ok  /api/report.csv serves the CLI's attempts.csv as a named download")


def test_roles_grain_reports_per_reasoner_token_spend() -> None:
    """The ablation export: one row per attempt PER ROLE, from tokens.json's `by_role` block.

    What is pinned is the thing the whole per-role change exists for - that a battery can answer
    "what was the guard costing us" - plus the two honest-reporting rules around it: an attempt with
    no role block contributes no rows at all (not a row of zeroes, which would read as "that
    component was free"), and the shares total to 1.0 over the rows that do exist.
    """
    import csv
    import io
    import urllib.request
    from http.server import ThreadingHTTPServer
    from threading import Thread

    from sari_bench.report import ROLE_COLUMNS, collect, role_rows
    from sari_bench.watch.server import Handler

    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        metered = make_attempt(battery, "metered", 1, steps=healthy_steps(3), state="finished",
                               outcome="completed", success=True, end_reason="halt_granted")
        _write(metered / "summary.json", json.dumps({"success": True, "legs": []}))
        _write(metered / "tokens.json", json.dumps({
            "tokens_in": 900, "tokens_out": 100, "calls": 6, "untracked_calls": 0,
            "by_role": {
                "guard": {"tokens_in": 300, "tokens_out": 20, "calls": 2},
                "actor": {"tokens_in": 600, "tokens_out": 80, "calls": 4},
            },
            "tokens_total": 1000,
        }))
        # An attempt from before per-role accounting: totals only, no `by_role` at all.
        legacy = make_attempt(battery, "legacy", 1, steps=healthy_steps(3), state="finished",
                              outcome="completed", success=True, end_reason="halt_granted")
        _write(legacy / "tokens.json", json.dumps({"tokens_in": 500, "tokens_out": 50, "calls": 3}))

        attempts, _legs = collect(battery)
        rows = role_rows(attempts)
        assert [row["prompt_id"] for row in rows] == ["metered", "metered"], rows
        # Pipeline order, not insertion or alphabetical order: actor comes before guard.
        assert [row["role"] for row in rows] == ["actor", "guard"], rows
        assert [row["tokens_total"] for row in rows] == [680, 320], rows
        assert [row["calls"] for row in rows] == [4, 2], rows
        assert round(sum(row["share_in"] for row in rows), 6) == 1.0, rows
        # The join key back to attempts.csv, and the arm an ablation groups by.
        assert rows[0]["run_dir"] == str(metered) and rows[0]["arm"] == "graph"

        state = WatchState(bench_root=Path(temp), fixed_battery=battery,
                           discord=Discord(enabled=False), min_interval=0.0)
        Handler.state = state
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/api/report.csv?grain=roles", timeout=10
            ) as response:
                assert response.status == 200
                assert response.headers["Content-Disposition"] == 'attachment; filename="b-roles.csv"'
                body = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        expected = io.StringIO(newline="")
        writer = csv.DictWriter(expected, fieldnames=ROLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        assert body == expected.getvalue(), "the roles route and the CLI produced different CSVs"

    served = list(csv.DictReader(io.StringIO(body)))
    assert list(served[0]) == ROLE_COLUMNS, "the roles route drifted from the CLI's columns"
    # The private carrier key must never surface in an exported CSV.
    assert "_tokens_by_role" not in body
    print("ok  /api/report.csv?grain=roles serves the per-reasoner token grain")


def test_responder_role_is_ordered_after_findings() -> None:
    roles = {"unattributed": {}, "responder": {}, "findings": {}, "actor": {}}
    assert scan.sorted_roles(roles) == ["actor", "findings", "responder", "unattributed"]


def test_multipart_upload_round_trip() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from sari_bench.watch import notify as notify_mod

    bodies: list[bytes] = []
    types: list[str] = []

    class Sink(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:
            types.append(self.headers["Content-Type"])
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as temp:
            clip = Path(temp) / "replay.discord.mp4"
            clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"payload" * 100)
            discord = Discord(webhook_url=f"http://127.0.0.1:{server.server_port}/hook")
            discord._post({"embeds": [{"title": "Attempt succeeded: p try 1"}]}, attachment=clip)

            assert types[0].startswith("multipart/form-data; boundary="), types
            body = bodies[0]
            assert b'name="payload_json"' in body
            assert b'name="files[0]"; filename="replay.discord.mp4"' in body
            assert b"Content-Type: video/mp4" in body, body[:400]

            # Over the cap, the message still goes out - just as text.
            fat = Path(temp) / "fat.mp4"
            fat.write_bytes(b"\x00" * (notify_mod.MAX_ATTACHMENT_BYTES + 1))
            discord._post({"embeds": [{"title": "Attempt failed: p try 2"}]}, attachment=fat)
            assert types[1] == "application/json", types
    finally:
        server.shutdown()
        server.server_close()
    print("ok  an mp4 uploads as video/mp4; an oversize clip degrades to a text post")


def main() -> int:
    test_health_separates_healthy_from_collapsed()
    test_scan_ranks_worst_first_and_reads_step_state()
    test_scan_carries_the_agents_final_response()
    test_orphaned_attempt_is_not_shown_as_live()
    test_orphan_is_closed_out_rather_than_stranded()
    test_close_out_leaves_the_last_word_to_a_live_runner()
    test_a_recycled_pid_is_not_mistaken_for_the_agent()
    test_discovery_prefers_newest_and_honours_pin()
    test_rename_moves_the_dir_and_refuses_while_anything_is_writing()
    test_a_browser_can_read_a_battery_the_watcher_is_not_watching()
    test_a_running_battery_is_marked_live()
    test_rotated_requeue_dir_stays_separate()
    test_http_surface_and_traversal_refusal()
    test_log_cursor_appends_only_what_is_new()
    test_full_log_is_not_limited_by_tail_window()
    test_snapshot_exposes_log_size_for_change_driven_terminals()
    test_log_query_params_are_clamped()
    test_report_and_kill_stamp()
    test_target_bitrate_math()
    test_every_finish_notifies_once()
    test_agent_error_is_failure_and_automatic_invalid()
    test_finish_attaches_replay_mp4()
    test_replay_seed_suppresses_backfill()
    test_replay_worker_prioritizes_announcements_and_rejects_corrupt_clips()
    test_full_replay_queue_falls_back_to_text_notification()
    test_finished_run_queues_dashboard_replay_without_discord()
    test_every_finished_attempt_is_reviewable()
    test_verdict_round_trip_and_refusals()
    test_invalid_verdict_is_excluded_rather_than_failed()
    test_retry_config_preserves_completion_guard()
    test_invalid_verdict_in_the_report()
    test_success_verdict_cancels_only_same_prompt_siblings()
    test_response_body_ignores_client_disconnects_only()
    test_verdict_and_replay_over_http()
    test_report_carries_the_human_verdict()
    test_report_csv_route_matches_the_cli()
    test_roles_grain_reports_per_reasoner_token_spend()
    test_multipart_upload_round_trip()
    print("\nAll watch tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
