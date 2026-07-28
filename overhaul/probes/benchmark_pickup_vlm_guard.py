"""Benchmark the isolated pickup VLM guard on one saved image."""

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

_OVERHAUL = Path(__file__).resolve().parent.parent
_REPO = _OVERHAUL.parent
if str(_OVERHAUL) not in sys.path:
    sys.path.insert(0, str(_OVERHAUL))

from dotenv import load_dotenv
from openai import OpenAI

from orchestrator.pickup_vlm_guard import classify_pickup

DEFAULT_FRAME = (
    _REPO / "bench_runs" / "20260727_020820" / "easy_02" / "try04" / "capture"
    / "frame000348-1785090368627510158.jpg"
)
DEFAULT_SKU = "LESLIE_S_CLOVER_CHIPS_CHEESE_24G"
DEFAULT_TARGET = "Clover Chips"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


def _client(base_url=None, api_key=None):
    load_dotenv(_REPO / "config.env")
    url = base_url or os.getenv("OPENAI_API_URL")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not (url and key):
        raise RuntimeError(
            "Set OPENAI_API_URL and OPENAI_API_KEY (or pass --base-url and --api-key).")
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlsplit(url)
    netloc = parsed.netloc if parsed.port else f"{parsed.hostname}:8000"
    resolved = f"{parsed.scheme}://{netloc}/v1"
    return OpenAI(base_url=resolved, api_key=key, max_retries=0), resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--sku", default=DEFAULT_SKU)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON report path")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    frame = args.frame.resolve()
    image_b64 = base64.b64encode(frame.read_bytes()).decode("utf-8")
    media_type = "image/jpeg" if frame.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    client, resolved_url = _client(args.base_url, args.api_key)
    config = SimpleNamespace(
        temperature=0.5,
        max_tokens=1536,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    results = []
    for index in range(1, args.runs + 1):
        verdict = classify_pickup(
            client, args.model, config, image_b64, args.sku, args.target, media_type)
        results.append(verdict)
        print(
            f"run {index}/{args.runs}: match={verdict['match']} "
            f"conclusive={verdict['conclusive']} latency={verdict['latency_ms']} ms "
            f"reason={verdict['reason']}"
        )

    latencies = [row["latency_ms"] for row in results]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame": str(frame),
        "sku": args.sku,
        "target": args.target,
        "model": args.model,
        "base_url": resolved_url,
        "runs": results,
        "summary": {
            "count": len(results),
            "positive": sum(row["match"] is True for row in results),
            "conclusive": sum(row["conclusive"] is True for row in results),
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        },
    }
    print(json.dumps(report["summary"], indent=2))

    if args.out:
        output = args.out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"report: {output}")


if __name__ == "__main__":
    main()
