"""Offline unit tests for agent_core/token_meter.py - the per-run token accounting the distributed
benchmark reports as tokens in/out.

The whole point of the meter is that it counts calls it never sees the source of, so these drive the
REAL OpenAI SDK against a throwaway localhost HTTP server that speaks the chat-completions schema.
No sim, no model stack, no network beyond loopback.

    python tests/test_token_meter.py   # or: pytest ...
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI

from agent_core import token_meter


class _Handler(BaseHTTPRequestHandler):
    """Answers any chat-completions POST with a fixed usage block."""

    prompt_tokens = 100
    completion_tokens = 20

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's contract
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model", "test-model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": self.prompt_tokens,
                      "completion_tokens": self.completion_tokens,
                      "total_tokens": self.prompt_tokens + self.completion_tokens},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):  # keep the test output clean
        pass


def _server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def _call(client, model="Qwen/Qwen3.6-27B"):
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hi"}])


def test_counts_every_sdk_call_and_splits_by_model():
    token_meter.install()
    token_meter.reset()
    server, base_url = _server()
    try:
        client = OpenAI(base_url=base_url, api_key="test")
        _call(client)
        _call(client)
        _call(client, model="other-model")
    finally:
        server.shutdown()

    totals = token_meter.totals()
    assert totals["calls"] == 3, totals
    assert totals["tokens_in"] == 300, totals
    assert totals["tokens_out"] == 60, totals
    assert totals["tokens_total"] == 360, totals
    assert totals["untracked_calls"] == 0, totals
    assert totals["by_model"]["Qwen/Qwen3.6-27B"]["calls"] == 2, totals["by_model"]
    assert totals["by_model"]["other-model"]["tokens_in"] == 100, totals["by_model"]


def test_delta_measures_one_leg():
    token_meter.install()
    token_meter.reset()
    server, base_url = _server()
    try:
        client = OpenAI(base_url=base_url, api_key="test")
        _call(client)
        before = token_meter.snapshot()
        _call(client)
        _call(client)
    finally:
        server.shutdown()

    leg = token_meter.delta(before)
    assert leg == {"tokens_in": 200, "tokens_out": 40, "calls": 2,
                   "by_role": {token_meter.UNATTRIBUTED:
                               {"tokens_in": 200, "tokens_out": 40, "calls": 2}}}, leg


def test_roles_attribute_calls_and_total_to_the_whole():
    """The point of roles: an ablation must be able to read off what one component was costing."""
    token_meter.install()
    token_meter.reset()
    server, base_url = _server()
    try:
        client = OpenAI(base_url=base_url, api_key="test")
        with token_meter.role(token_meter.ROLE_ACTOR):
            _call(client)
            _call(client)
        with token_meter.role(token_meter.ROLE_GUARD):
            _call(client)
        _call(client)   # outside any block: unattributed, never dropped and never guessed at
    finally:
        server.shutdown()

    by_role = token_meter.totals()["by_role"]
    assert by_role[token_meter.ROLE_ACTOR] == {"tokens_in": 200, "tokens_out": 40, "calls": 2}
    assert by_role[token_meter.ROLE_GUARD] == {"tokens_in": 100, "tokens_out": 20, "calls": 1}
    assert by_role[token_meter.UNATTRIBUTED] == {"tokens_in": 100, "tokens_out": 20, "calls": 1}
    # The rows must re-total to the whole, or a share column computed from them is a lie.
    assert sum(row["calls"] for row in by_role.values()) == token_meter.totals()["calls"]
    assert sum(row["tokens_in"] for row in by_role.values()) == token_meter.totals()["tokens_in"]


def test_innermost_role_wins_and_resets_after_a_failure():
    """Perception called from inside an actor step is perception's cost, and a call that raised must
    not leave its role behind for the next one."""
    token_meter.install()
    token_meter.reset()
    server, base_url = _server()
    try:
        client = OpenAI(base_url=base_url, api_key="test")
        with token_meter.role(token_meter.ROLE_ACTOR):
            with token_meter.role(token_meter.ROLE_PERCEPTION):
                _call(client)
            _call(client)
        try:
            with token_meter.role(token_meter.ROLE_GUARD):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert token_meter.current_role() == token_meter.UNATTRIBUTED
        _call(client)
    finally:
        server.shutdown()

    by_role = token_meter.totals()["by_role"]
    assert by_role[token_meter.ROLE_PERCEPTION]["calls"] == 1, by_role
    assert by_role[token_meter.ROLE_ACTOR]["calls"] == 1, by_role
    assert token_meter.ROLE_GUARD not in by_role, by_role
    assert by_role[token_meter.UNATTRIBUTED]["calls"] == 1, by_role


def test_record_external_counts_the_raw_http_backends():
    """nav.locate_task's qwen backend posts with `requests` and never touches the patched SDK, so
    without this path the advisor and the map resolver are missing from the totals entirely."""
    token_meter.install()
    token_meter.reset()
    token_meter.record_external("Qwen/Qwen3.6-27B",
                                {"prompt_tokens": 700, "completion_tokens": 30},
                                token_meter.ROLE_ADVISOR)
    totals = token_meter.totals()
    assert totals["calls"] == 1, totals
    assert totals["tokens_total"] == 730, totals
    assert totals["by_role"][token_meter.ROLE_ADVISOR]["tokens_in"] == 700, totals["by_role"]
    # A body with no usage block is counted as untracked, exactly like a streamed SDK response.
    token_meter.record_external("Qwen/Qwen3.6-27B", None, token_meter.ROLE_ADVISOR)
    assert token_meter.totals()["untracked_calls"] == 1, token_meter.totals()


def test_responder_has_a_dedicated_token_role():
    token_meter.install()
    token_meter.reset()
    token_meter.record_external(
        "Qwen/Qwen3.6-27B",
        {"prompt_tokens": 250, "completion_tokens": 30},
        token_meter.ROLE_RESPONDER,
    )
    assert token_meter.ROLE_RESPONDER in token_meter.ROLES
    assert token_meter.totals()["by_role"][token_meter.ROLE_RESPONDER] == {
        "tokens_in": 250,
        "tokens_out": 30,
        "calls": 1,
    }


def test_dump_writes_tokens_json():
    token_meter.install()
    token_meter.reset()
    server, base_url = _server()
    try:
        client = OpenAI(base_url=base_url, api_key="test")
        with tempfile.TemporaryDirectory() as run_dir:
            _call(client)
            token_meter.dump(run_dir)
            payload = json.loads(
                open(os.path.join(run_dir, "tokens.json"), encoding="utf-8").read())
    finally:
        server.shutdown()

    # The benchmark runner reads exactly these keys off a killed attempt.
    assert payload["tokens_in"] == 100, payload
    assert payload["tokens_out"] == 20, payload
    assert payload["tokens_total"] == 120, payload


def test_install_is_idempotent():
    """A second install must not wrap the wrapper - that would double-count every call."""
    token_meter.install()
    token_meter.install()
    token_meter.reset()
    server, base_url = _server()
    try:
        _call(OpenAI(base_url=base_url, api_key="test"))
    finally:
        server.shutdown()
    assert token_meter.totals()["calls"] == 1, token_meter.totals()


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
