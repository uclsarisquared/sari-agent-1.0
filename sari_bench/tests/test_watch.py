"""Watcher tests: scanning, collapse detection, discovery, and the HTTP surface.

Entirely offline - no sim, no coordinator, no model stack. The fixtures write the same artefacts a
real battery does (attempt.json, legNN.jsonl, legNN/stepNN.png, summary.json), so the scan path is
exercised against the real shapes rather than a mock. What is being pinned down:

  1. a healthy attempt scores clean while a looping one scores as a collapse, and tiles sort
     worst-first - the ranking IS the feature;
  2. discovery picks the newest battery, and --run-dir pins one;
  3. an attempt whose runner died shows as orphaned rather than as a live tile frozen forever;
  4. the HTTP API serves state/frames/logs, and path traversal in an attempt key is refused;
  5. a rotated-aside requeue dir does not merge with the attempt that replaced it;
  6. every halt is announced exactly once whatever its outcome, and carries a replay clip inside the
     upload budget - written beside, never over, the CLI's uncapped replay.mp4;
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

    (run_dir / "agent.log").write_text("\n".join(f"log line {i}" for i in range(50)), encoding="utf-8")
    return run_dir


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


def test_orphaned_attempt_is_not_shown_as_live() -> None:
    with tempfile.TemporaryDirectory() as temp:
        battery = Path(temp) / "b"
        # A pid that cannot exist: the manifest says running, the process is gone.
        make_attempt(battery, "dead", 1, steps=healthy_steps(), pid=999_999_998)
        view = scan.scan_battery(battery, time.time()).as_dict()
        assert view["attempts"][0]["state"] == "orphaned", view["attempts"][0]["state"]
        assert view["attempts"][0]["alive"] is False
        print("ok  an attempt whose runner died reads as orphaned, not live")


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
        print("ok  auto-discovery takes the newest battery; --run-dir pins one")


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
            assert frame.read()[:8] == _PNG[:8]

            log = json.loads(urllib.request.urlopen(f"{base}/api/attempt/p/try01/log", timeout=5).read())
            assert log["lines"][-1] == "log line 49", log["lines"][-1]

            page = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
            assert "Sari Bench" in page
        finally:
            server.shutdown()
            server.server_close()
        print("ok  HTTP serves state, frames and logs; traversal keys are refused")


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
            # The CLI's uncapped artefact must be left for the CLI to write.
            assert not (run_dir / "replay.mp4").exists(), "auto-render clobbered replay.mp4"

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


def test_only_voluntary_halts_are_reviewable() -> None:
    """The gate on the verdict buttons.

    A run the harness cut off at its step or time cap never claimed to be done, so there is nothing
    for a human to confirm or deny. Only `halt_granted` (the agent said STOP and the predicate granted
    it) and `completed_no_stop` (the backstop) put a claim on the record worth checking.
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
        make_attempt(battery, "live", 1, steps=healthy_steps(3), pid=os.getpid())

        view = scan.scan_battery(battery, time.time()).as_dict()
        by_id = {a["prompt_id"]: a for a in view["attempts"]}
        assert by_id["granted"]["verifiable"] is True
        assert by_id["backstop"]["verifiable"] is True
        assert by_id["capped"]["verifiable"] is False, "a step-capped run offered a verdict"
        assert by_id["timedout"]["verifiable"] is False
        assert by_id["live"]["verifiable"] is False, "a running attempt offered a verdict"
        assert all(a["verified"] is False for a in view["attempts"])
        assert all(a["verified_success"] is None for a in view["attempts"])
        assert view["counts"]["awaiting_verdict"] == 2, view["counts"]
        print("ok  only attempts the agent halted itself are offered for human review")


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

        result = state.verdict("p/try01", False, note="never actually grabbed it", by="tester")
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
        assert state.verdict("p/try01", True)["ok"] is True
        manifest = json.loads((battery / "p" / "try01" / "attempt.json").read_text())
        assert manifest["verified_success"] is True

        # Cleared, an attempt reads as never looked at - not as a verdict of False.
        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": True}
        manifest = json.loads((battery / "p" / "try01" / "attempt.json").read_text())
        assert "verified_success" not in manifest and "verified_by" not in manifest
        assert state.clear_verdict("p/try01") == {"ok": True, "cleared": False}

        # The server re-checks eligibility: the button is not the only gate.
        for key in ("capped/try01", "live/try01"):
            refused = state.verdict(key, True)
            assert refused["ok"] is False and "not reviewable" in refused["error"], refused
        assert state.verdict("../../etc", True)["ok"] is False
        assert state.clear_verdict("../../etc")["ok"] is False
        print("ok  verdicts round-trip beside `success`, overwrite, clear, and refuse the ineligible")


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
        clip_bytes = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 8
        (run_dir / video.UPLOAD_NAME).write_bytes(clip_bytes)
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
            code, _, _ = request("/api/attempt/p/try01/verdict",
                                 data=json.dumps({"success": True}).encode())
            assert code == 200, code

            # Kill still routes correctly now that do_POST dispatches on three suffixes.
            code, body, _ = request("/api/attempt/p/try01/kill", data=b"")
            assert code == 400 and "finished" in json.loads(body)["error"], body

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
        assert state.verdict("judged/try01", False, by="tester")["ok"] is True

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
    test_orphaned_attempt_is_not_shown_as_live()
    test_discovery_prefers_newest_and_honours_pin()
    test_rotated_requeue_dir_stays_separate()
    test_http_surface_and_traversal_refusal()
    test_report_and_kill_stamp()
    test_target_bitrate_math()
    test_every_finish_notifies_once()
    test_finish_attaches_replay_mp4()
    test_replay_seed_suppresses_backfill()
    test_only_voluntary_halts_are_reviewable()
    test_verdict_round_trip_and_refusals()
    test_verdict_and_replay_over_http()
    test_report_carries_the_human_verdict()
    test_multipart_upload_round_trip()
    print("\nAll watch tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
