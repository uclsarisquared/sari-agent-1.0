"""Replay deterministic and VLM pickup guards over identical saved captures (no simulator).

Manifest JSON may be a list or ``{"rows": [...]}``. Each row requires:
``{"id", "image", "sku", "target", "expected"}``, where expected is a boolean and image paths are
resolved relative to the manifest. The report preserves every decision/reason/latency and confusion
counts for both backends.

Do not treat a report as trusted-run validation unless its manifest includes the saved CDO corned
beef/breakfast positive, the held Clover Chips/specific Chipsy negative, and another genuine
attribute mismatch. This repository currently does not contain identifiable held-item captures for
the first two controls, so the probe intentionally does not manufacture a baseline manifest.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2] / "agent"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from orchestrator.pickup_vlm_guard import classify_pickup
from orchestrator.subtask_completion import completion_predicate
from agent_core.models import agent_model
from agent_core.llm import normalize_endpoint_root

load_dotenv(_ROOT.parent / "config.env")


def _load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a non-empty JSON list or object with a non-empty rows list")
    required = {"image", "sku", "target", "expected"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"row {index} must contain {sorted(required)}")
        if type(row["expected"]) is not bool:
            raise ValueError(f"row {index} expected must be a JSON boolean")
    return rows


def _confusion(rows, key):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unavailable": 0}
    for row in rows:
        decision = row[key]["match"]
        expected = row["expected"]
        counts[("t" if decision == expected else "f") + ("p" if decision else "n")] += 1
        if key == "vlm" and not row[key]["conclusive"]:
            counts["unavailable"] += 1
    return counts


def _runtime_client(base_url=None, api_key=None):
    url = base_url or os.getenv("OPENAI_API_URL")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not (url and key):
        raise RuntimeError("set OPENAI_API_URL and OPENAI_API_KEY (or pass --base-url/--api-key)")
    url = f"{normalize_endpoint_root(url)}/v1"
    return OpenAI(base_url=url, api_key=key, max_retries=0), url


def run(manifest_path, output_path, base_url=None, api_key=None,
        model=None):
    model = model or agent_model()
    manifest_path = Path(manifest_path).resolve()
    rows = _load_manifest(manifest_path)
    client, resolved_url = _runtime_client(base_url, api_key)
    config = SimpleNamespace(
        temperature=0.5,
        max_tokens=1536,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    results = []
    for index, source in enumerate(rows):
        image_path = Path(source["image"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        media_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        state = {
            "leftGrippedState": True,
            "rightGrippedState": False,
            "leftHoveredObject": "None",
            "rightHoveredObject": "None",
            "gripped_name": source["sku"],
            "gripped_names": {"left": source["sku"], "right": None},
        }
        det_started = time.monotonic()
        deterministic, det_reason = completion_predicate(
            {"type": "pickup", "target": source["target"]}, state)
        det_latency = round((time.monotonic() - det_started) * 1000, 1)
        vlm = classify_pickup(client, model, config, image_b64,
                              source["sku"], source["target"], media_type)
        results.append({
            "id": source.get("id", index),
            "image": str(image_path),
            "sku": source["sku"],
            "target": source["target"],
            "expected": source["expected"],
            "deterministic": {
                "match": deterministic, "reason": det_reason, "latency_ms": det_latency},
            "vlm": vlm,
        })
        print(f"{source.get('id', index)}: expected={source['expected']} "
              f"deterministic={deterministic} vlm={vlm['match']} "
              f"conclusive={vlm['conclusive']} {vlm['latency_ms']}ms")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "model": model,
        "base_url": resolved_url,
        "rows": results,
        "confusion": {
            "deterministic": _confusion(results, "deterministic"),
            "vlm": _confusion(results, "vlm"),
        },
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"report: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="saved replay manifest JSON")
    parser.add_argument("--out", required=True, help="result report JSON to write")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None, help="defaults to $SARI_MODEL")
    args = parser.parse_args()
    run(args.manifest, args.out, args.base_url, args.api_key, args.model)


if __name__ == "__main__":
    main()
