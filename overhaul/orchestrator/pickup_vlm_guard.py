"""Import-cheap, stateless VLM classifiers for pickup and inspection completion.

The runtime injects its existing OpenAI-compatible client and model configuration. Calls go directly
to that client with a fresh two-message context, never through the actor's history-bearing
``send_message`` method.
"""

import json
import time


GUARD_SYSTEM = (
    "You are a strict pickup-completion classifier for a grocery-store robot. Decide whether the "
    "HELD SKU is a valid instance of the requested TARGET. Use ordinary product knowledge and the "
    "current image only as supporting visual context. Attribute/category descriptions may match a "
    "specific product (for example, a canned breakfast food), but a shared broad category word does "
    "not make two specifically named products equivalent. Return only the required JSON object."
)

INSPECT_GUARD_SYSTEM = (
    "You are a strict visual inspection verifier for a grocery-store robot. Given the exact image "
    "the robot used, an observation QUERY, and the robot's REPORTED ANSWER, decide whether the image "
    "conclusively supports that answer. This is read-only reporting: do not infer manipulation or "
    "physical-state completion. If the relevant objects, count, label, or date are not sufficiently "
    "visible, return match=false. Return only the required JSON object."
)

_SCHEMA = {
    "type": "object",
    "properties": {"match": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["match", "reason"],
    "additionalProperties": False,
}


def _refusal(reason, latency_ms, sku):
    return {"match": False, "reason": reason, "conclusive": False,
            "latency_ms": round(latency_ms, 1), "sku": sku}


def classify_pickup(client, model, config, image_b64, held_sku, target,
                    image_media_type="image/png"):
    """Make exactly one 30-second attempt and return a normalized plain-dict verdict."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": GUARD_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text",
             "text": f"HELD SKU: {held_sku}\nTARGET: {target}\nDoes the held SKU match the target?"},
        ]},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": getattr(config, "temperature", 0),
        "max_tokens": min(getattr(config, "max_tokens", 256), 256),
        "timeout": 30,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "pickup_guard", "schema": _SCHEMA, "strict": True},
        },
    }
    extra_body = getattr(config, "extra_body", None)
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        # Disable the OpenAI SDK's own automatic retries without mutating the shared actor client.
        # Fake/minimal clients used by offline tests need not implement with_options.
        request_client = (client.with_options(max_retries=0)
                          if callable(getattr(client, "with_options", None)) else client)
        response = request_client.chat.completions.create(**kwargs)
        parsed = json.loads(response.choices[0].message.content)
        if (not isinstance(parsed, dict) or set(parsed) != {"match", "reason"}
                or type(parsed.get("match")) is not bool
                or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()):
            raise ValueError("expected a strict boolean match and non-empty string reason")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1), "sku": held_sku}
    except Exception as exc:  # timeout, API, response shape, and JSON all fail closed; no retry
        latency = (time.monotonic() - started) * 1000
        return _refusal(f"VLM guard unavailable ({type(exc).__name__}: {exc})", latency, held_sku)


def classify_inspection(client, model, config, image_b64, query, answer, auxiliary_context,
                        image_media_type="image/png"):
    """Verify a free-form inspection answer against one actor-visible frame, failing closed."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": INSPECT_GUARD_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"QUERY: {query}\nREPORTED ANSWER: {answer}\n"
                f"AUXILIARY CONTEXT: {json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
                "Does the image conclusively support the reported answer?"
            )},
        ]},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": getattr(config, "temperature", 0),
        "max_tokens": min(getattr(config, "max_tokens", 256), 256),
        "timeout": 30,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "inspect_guard", "schema": _SCHEMA, "strict": True},
        },
    }
    extra_body = getattr(config, "extra_body", None)
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        request_client = (client.with_options(max_retries=0)
                          if callable(getattr(client, "with_options", None)) else client)
        response = request_client.chat.completions.create(**kwargs)
        parsed = json.loads(response.choices[0].message.content)
        if (not isinstance(parsed, dict) or set(parsed) != {"match", "reason"}
                or type(parsed.get("match")) is not bool
                or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()):
            raise ValueError("expected a strict boolean match and non-empty string reason")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1)}
    except Exception as exc:  # timeout, API, response shape, and JSON all fail closed
        latency = (time.monotonic() - started) * 1000
        return {"match": False,
                "reason": f"VLM inspection guard unavailable ({type(exc).__name__}: {exc})",
                "conclusive": False, "latency_ms": round(latency, 1)}


def make_inspect_guard(client, model, config, image_b64, on_verdict=None):
    """Return an image-bound, per-step cached callback matching ``predicate_inspect``'s contract."""
    cache = {}

    def guard(query, answer, auxiliary_context):
        try:
            aux_key = json.dumps(auxiliary_context, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:  # defensive; classify_inspection still receives the original context
            aux_key = repr(auxiliary_context)
        key = (str(query), str(answer), aux_key)
        reused = key in cache
        if reused:
            cached = cache[key]
            verdict = dict(cached) if isinstance(cached, dict) else cached
        else:
            verdict = classify_inspection(
                client, model, config, image_b64, query, answer, auxiliary_context)
            cache[key] = dict(verdict) if isinstance(verdict, dict) else verdict
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(query, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def evaluate_hands(client, model, config, image_b64, target, held_skus):
    """Classify each unique SKU once; return ``(per_hand_verdicts, call_count)``."""
    per_hand, by_sku, calls = {}, {}, 0
    for side in ("left", "right"):
        sku = (held_skus or {}).get(side)
        if not sku:
            continue
        if sku in by_sku:
            verdict = dict(by_sku[sku])
            verdict["reused"] = True
        else:
            verdict = classify_pickup(client, model, config, image_b64, sku, target)
            verdict["reused"] = False
            by_sku[sku] = dict(verdict)
            calls += 1
        per_hand[side] = verdict
    return per_hand, calls
