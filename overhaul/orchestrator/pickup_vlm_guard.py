"""Import-cheap, stateless VLM classifiers for completion guards.

The runtime injects its existing OpenAI-compatible client and model configuration. Calls go directly
to that client with a fresh two-message context, never through the actor's history-bearing
``send_message`` method.
"""

import json
import time

from agent_core import token_meter


def _guard_call(client, kwargs):
    """One classifier request, billed to the `guard` role.

    Disables the OpenAI SDK's own automatic retries without mutating the shared actor client;
    fake/minimal clients used by offline tests need not implement ``with_options``. Every classifier
    in this module goes through here, which is also what makes the guards a single line item in the
    per-role token accounting - an ablation that drops the completion guard reads its saving off
    that one row.
    """
    request_client = (client.with_options(max_retries=0)
                      if callable(getattr(client, "with_options", None)) else client)
    with token_meter.role(token_meter.ROLE_GUARD):
        return request_client.chat.completions.create(**kwargs)


GUARD_SYSTEM = (
    "You are a strict pickup-completion classifier for a grocery-store robot. Decide whether the "
    "HELD SKU is a valid instance of the requested TARGET. Use ordinary product knowledge and the "
    "current image only as supporting visual context. Attribute/category descriptions may match a "
    "specific product (for example, a canned breakfast food), but a shared broad category word does "
    "not make two specifically named products equivalent. Return only the required JSON object."
)

INSPECT_GUARD_SYSTEM = (
    "You are a visual inspection verifier for a grocery-store robot. You receive one or more images "
    "the robot actually used, in the order it observed them, an observation QUERY, and the robot's "
    "REPORTED ANSWER. Decide whether those images, CONSIDERED TOGETHER, plausibly support that "
    "answer. A robot holds one item per hand and can only turn ONE item's printed label toward the "
    "camera at a time, so when the query spans several items each item's supporting evidence will "
    "normally come from a DIFFERENT image: judge each claimed reading against whichever supplied "
    "image shows that item, and never reject an answer merely because no single image shows every "
    "item at once. Every item, value, or reading the answer asserts must still be supported by some "
    "supplied image; if any item the query needs has no image showing its relevant label, return "
    "match=false. This is read-only reporting: do not infer manipulation or "
    "physical-state completion. If the query concerns nutritional facts, ingredients, an expiration "
    "date, or other printed item text, require the relevant printed surface to directly face the "
    "camera and be visible in the supplied image that supports it; the sandbox's rendered textures "
    "are frequently low-"
    "resolution, so exact character-by-character legibility is NOT required — a label that is "
    "visibly present and facing the camera, even if small, slightly blurred, or only partly sharp, "
    "is sufficient support as long as the reported answer is plausible given what can be made out. "
    "An edge-on, oblique, upside-down, largely hidden, or merely glimpsed label must still return "
    "match=false. If the relevant objects, count, label, or date are not sufficiently visible at "
    "all, return match=false. The reported answer must directly and concretely answer the query "
    "with the requested facts, values, count, date, or identification. A statement about visibility "
    "or a needed next action—such as saying the label is unreadable, needs rotation, or needs a "
    "better view—is not an answer to the query and must return match=false even when the image "
    "supports that statement. Return only the required JSON object."
)

INSPECTION_VISIBILITY_SYSTEM = (
    "You are a fresh-frame visibility gate inside a deterministic held-item inspection tool. "
    "Use only the supplied image and INSPECTION REQUEST. Return match=true only when the specific "
    "printed information requested—such as a Nutrition Facts panel or an expiration date—is directly "
    "facing the camera AND LEGIBLE enough for the main agent to read and report its actual values. "
    "LEGIBILITY IS MANDATORY, not optional: the individual words, numbers, and date characters needed "
    "to answer the request must be clearly readable and unambiguous at the image's current resolution. "
    "A label that is merely visible, recognizable, or known to be present must return match=false when "
    "its requested text is too small, blurred, pixelated, obscured by glare, cropped, angled, or partly "
    "hidden. For Nutrition Facts, seeing the panel outline or the words 'Nutrition Facts' is not enough; "
    "the relevant value rows must be readable. For an expiration date, every character of the complete "
    "date must be readable without guessing. If uncertain whether the requested text can actually be "
    "read, return match=false. Seeing only the package, another side panel, or a barcode is also "
    "insufficient. Do not extract or report the values yourself and do not reason about actions or "
    "rotation; only decide whether the requested text is both visible and legible. PaddleOCR text "
    "may be supplied as UNTRUSTED AUXILIARY EVIDENCE. Use it to help resolve tiny characters only "
    "when it is consistent with the requested label visibly facing the camera; unrelated package or "
    "background OCR must not cause match=true, and OCR never relaxes the front-facing requirement. "
    "Return only the required JSON object."
)

INSPECTION_LABEL_PRESENCE_SYSTEM = (
    "You are a fresh-frame label-location gate inside a deterministic held-item inspection tool. "
    "Use only the supplied image and INSPECTION REQUEST. This check is deliberately less strict than "
    "the separate legibility gate. Return match=true when the specific requested label surface—such "
    "as the Nutrition Facts panel or the printed expiration-date area—is recognizably present AND "
    "DIRECTLY FACING THE CAMERA, even when its words or numbers are still too small or blurry to "
    "read. This is a side-locking decision: match=true tells the robot to STOP ROTATING permanently, "
    "so front-facing orientation is mandatory. The label plane should look substantially flat-on, "
    "with little perspective foreshortening; readable lines seen obliquely, as a narrow trapezoid, or "
    "around a package edge must return match=false so rotation continues until the panel faces the "
    "camera squarely. Do not match merely because the package, a barcode, ingredients panel, unrelated "
    "text, or a thin edge of an unknown label is visible. If uncertain whether this is the requested "
    "label or whether it is directly front-facing, return match=false. Do not extract values. Return "
    "only the required JSON object."
)

COMPARE_GUARD_SYSTEM = (
    "You are a strict visual comparison verifier for a grocery-store robot. You receive labeled "
    "candidate images, a comparison CRITERION, and the robot's REPORTED CHOICE. Decide whether the "
    "images, considered together, conclusively support that choice. Compare only evidence directly "
    "visible in the supplied images; do not use a product catalog, hidden store data, or unsupported "
    "product knowledge. If any candidate, relevant label, value, or distinguishing feature needed "
    "for the comparison is not sufficiently visible, return match=false. Return only the required "
    "JSON object."
)

UNKNOWN_GUARD_SYSTEM = (
    "You are a strict visual task-completion verifier for a grocery-store robot. Given the robot's "
    "current image, an unstructured TASK, and its COMPLETION CLAIM, decide whether the image "
    "conclusively supports that the task is complete. You may verify observation or manipulation "
    "state only when the relevant object and state are directly visible. Do not use a product "
    "catalog, hidden store data, or infer off-camera actions or outcomes. If the visible evidence is "
    "insufficient, return match=false. Return only the required JSON object."
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
        response = _guard_call(client, kwargs)
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
                        image_media_type="image/png", evidence_frames=None):
    """Verify a free-form inspection answer against the actor-visible frame plus earlier evidence.

    A multi-item inspect leg is UNVERIFIABLE from one frame (2026-07-29, measured): a robot can only
    turn one held item's printed label toward the camera at a time, so a two-item sugar comparison
    reads item A at step N and item B at step M and no single frame ever shows both. The guard used
    to see only the current frame and refused every STOP until the refusal cap - correct about the
    image, wrong about the unit of evidence. `evidence_frames` is the leg's ordered ledger of frames
    that already satisfied the inspection macro's visibility gate, each `{"label", "image_b64"}`;
    they are sent BEFORE the current frame, mirroring classify_compare's labeled-candidate layout.
    Empty/absent evidence reproduces the original single-image request exactly.
    """
    started = time.monotonic()
    content = []
    for index, frame in enumerate(evidence_frames or [], 1):
        label = str((frame or {}).get("label") or "").strip()
        frame_b64 = str((frame or {}).get("image_b64") or "").strip()
        if not frame_b64:
            continue   # a ledger entry with no frame is context-only; never send an empty image
        content.extend([
            {"type": "text", "text": f"EVIDENCE {index}: {label or 'earlier observation'}"},
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{frame_b64}"}},
        ])
    if content:
        content.append({"type": "text", "text": "CURRENT FRAME:"})
    content.extend([
        {"type": "image_url",
         "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
        {"type": "text", "text": (
            f"QUERY: {query}\nREPORTED ANSWER: {answer}\n"
            f"AUXILIARY CONTEXT: {json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
            "Do the supplied images conclusively support the reported answer?"
        )},
    ])
    messages = [
        {"role": "system", "content": INSPECT_GUARD_SYSTEM},
        {"role": "user", "content": content},
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
        response = _guard_call(client, kwargs)
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


def classify_inspection_visibility(client, model, config, image_b64, query,
                                   image_media_type="image/png", ocr_lines=None):
    """Fresh-context check for whether the requested printed information is currently legible."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": INSPECTION_VISIBILITY_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"INSPECTION REQUEST: {str(query or '').strip()}\n"
                f"PADDLEOCR AUXILIARY TEXT: "
                f"{json.dumps(list(ocr_lines or []), ensure_ascii=False)}\n"
                "Is the requested printed information directly visible and legible now?"
            )},
        ]},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": min(getattr(config, "max_tokens", 256), 256),
        "timeout": 30,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "inspection_visibility",
                "schema": _SCHEMA,
                "strict": True,
            },
        },
    }
    extra_body = getattr(config, "extra_body", None)
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        response = _guard_call(client, kwargs)
        parsed = json.loads(response.choices[0].message.content)
        if (not isinstance(parsed, dict) or set(parsed) != {"match", "reason"}
                or type(parsed.get("match")) is not bool
                or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()):
            raise ValueError("expected a strict boolean match and non-empty string reason")
        latency = (time.monotonic() - started) * 1000
        return {
            "match": parsed["match"],
            "reason": parsed["reason"].strip(),
            "conclusive": True,
            "latency_ms": round(latency, 1),
        }
    except Exception as exc:
        latency = (time.monotonic() - started) * 1000
        return {
            "match": False,
            "reason": (
                f"VLM inspection visibility check unavailable "
                f"({type(exc).__name__}: {exc})"
            ),
            "conclusive": False,
            "latency_ms": round(latency, 1),
        }


def classify_inspection_label_presence(client, model, config, image_b64, query,
                                       image_media_type="image/png"):
    """Fresh-context check for a recognizable requested label, without requiring legibility."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": INSPECTION_LABEL_PRESENCE_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"INSPECTION REQUEST: {str(query or '').strip()}\n"
                "Is the specific requested label recognizably present on this facing side, even if "
                "its values are not yet legible?"
            )},
        ]},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": min(getattr(config, "max_tokens", 256), 256),
        "timeout": 30,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "inspection_label_presence",
                "schema": _SCHEMA,
                "strict": True,
            },
        },
    }
    extra_body = getattr(config, "extra_body", None)
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        response = _guard_call(client, kwargs)
        parsed = json.loads(response.choices[0].message.content)
        if (not isinstance(parsed, dict) or set(parsed) != {"match", "reason"}
                or type(parsed.get("match")) is not bool
                or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()):
            raise ValueError("expected a strict boolean match and non-empty string reason")
        latency = (time.monotonic() - started) * 1000
        return {
            "match": parsed["match"],
            "reason": parsed["reason"].strip(),
            "conclusive": True,
            "latency_ms": round(latency, 1),
        }
    except Exception as exc:
        latency = (time.monotonic() - started) * 1000
        return {
            "match": False,
            "reason": (
                f"VLM inspection label-presence check unavailable "
                f"({type(exc).__name__}: {exc})"
            ),
            "conclusive": False,
            "latency_ms": round(latency, 1),
        }


def classify_compare(client, model, config, candidate_frames, criterion, answer,
                     auxiliary_context, image_media_type="image/png"):
    """Verify one reported choice against an ordered, labeled set of candidate frames."""
    started = time.monotonic()
    try:
        frames = list(candidate_frames or [])
        if len(frames) < 2:
            raise ValueError("at least two labeled candidate frames are required")
        content = []
        for index, frame in enumerate(frames, 1):
            if isinstance(frame, dict):
                target, image_b64 = frame.get("target"), frame.get("image_b64")
            else:
                target, image_b64 = frame
            target = str(target or "").strip()
            image_b64 = str(image_b64 or "").strip()
            if not target or not image_b64:
                raise ValueError(f"candidate {index} is missing a label or image")
            content.extend([
                {"type": "text", "text": f"CANDIDATE {index}: {target}"},
                {"type": "image_url",
                 "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            ])
        content.append({"type": "text", "text": (
            f"CRITERION: {criterion}\nREPORTED CHOICE: {answer}\n"
            f"AUXILIARY CONTEXT: "
            f"{json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
            "Do the labeled candidate images conclusively support the reported choice?"
        )})
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": COMPARE_GUARD_SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": getattr(config, "temperature", 0),
            "max_tokens": min(getattr(config, "max_tokens", 256), 256),
            "timeout": 30,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "compare_guard", "schema": _SCHEMA, "strict": True},
            },
        }
        extra_body = getattr(config, "extra_body", None)
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = _guard_call(client, kwargs)
        parsed = json.loads(response.choices[0].message.content)
        if (not isinstance(parsed, dict) or set(parsed) != {"match", "reason"}
                or type(parsed.get("match")) is not bool
                or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip()):
            raise ValueError("expected a strict boolean match and non-empty string reason")
        latency = (time.monotonic() - started) * 1000
        return {"match": parsed["match"], "reason": parsed["reason"].strip(),
                "conclusive": True, "latency_ms": round(latency, 1)}
    except Exception as exc:  # timeout, API, input, response shape, and JSON all fail closed
        latency = (time.monotonic() - started) * 1000
        return {"match": False,
                "reason": f"VLM compare guard unavailable ({type(exc).__name__}: {exc})",
                "conclusive": False, "latency_ms": round(latency, 1)}


def classify_unknown(client, model, config, image_b64, task, claim, auxiliary_context,
                     image_media_type="image/png"):
    """Verify an unstructured task-completion claim against the actor-visible current frame."""
    started = time.monotonic()
    messages = [
        {"role": "system", "content": UNKNOWN_GUARD_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"}},
            {"type": "text", "text": (
                f"TASK: {task}\nCOMPLETION CLAIM: {claim}\n"
                f"AUXILIARY CONTEXT: "
                f"{json.dumps(auxiliary_context, ensure_ascii=False, default=str)}\n"
                "Does the current image conclusively support completion of the task?"
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
            "json_schema": {"name": "unknown_guard", "schema": _SCHEMA, "strict": True},
        },
    }
    extra_body = getattr(config, "extra_body", None)
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        response = _guard_call(client, kwargs)
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
                "reason": f"VLM unknown guard unavailable ({type(exc).__name__}: {exc})",
                "conclusive": False, "latency_ms": round(latency, 1)}


def make_inspect_guard(client, model, config, image_b64, on_verdict=None, evidence_frames=None):
    """Return an image-bound, per-step cached callback matching ``predicate_inspect``'s contract.

    ``evidence_frames`` is the leg's accumulated inspection ledger (see ``classify_inspection``); it
    is fixed for the lifetime of this per-step closure, so the cache key need not cover it.
    """
    frames = [
        {"label": str((frame or {}).get("label") or ""),
         "image_b64": str((frame or {}).get("image_b64") or "")}
        for frame in (evidence_frames or [])
    ]
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
                client, model, config, image_b64, query, answer, auxiliary_context,
                evidence_frames=frames)
            cache[key] = dict(verdict) if isinstance(verdict, dict) else verdict
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(query, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def make_compare_guard(client, model, config, candidate_frames, on_verdict=None):
    """Return a candidate-frame-bound cached callback for ``predicate_compare``."""
    frames = tuple(
        (str(frame.get("target") or ""), str(frame.get("image_b64") or ""))
        if isinstance(frame, dict) else (str(frame[0]), str(frame[1]))
        for frame in (candidate_frames or [])
    )
    cache = {}

    def guard(criterion, answer, auxiliary_context):
        try:
            aux_key = json.dumps(auxiliary_context, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            aux_key = repr(auxiliary_context)
        key = (str(criterion), str(answer), aux_key)
        reused = key in cache
        if reused:
            verdict = dict(cache[key])
        else:
            verdict = classify_compare(
                client, model, config, frames, criterion, answer, auxiliary_context)
            cache[key] = dict(verdict)
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(criterion, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def make_unknown_guard(client, model, config, image_b64, on_verdict=None):
    """Return an image-bound, per-step cached callback for ``predicate_unknown``."""
    cache = {}

    def guard(task, claim, auxiliary_context):
        try:
            aux_key = json.dumps(auxiliary_context, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            aux_key = repr(auxiliary_context)
        key = (str(task), str(claim), aux_key)
        reused = key in cache
        if reused:
            verdict = dict(cache[key])
        else:
            verdict = classify_unknown(
                client, model, config, image_b64, task, claim, auxiliary_context)
            cache[key] = dict(verdict)
            guard.call_count += 1
        if callable(on_verdict):
            on_verdict(task, auxiliary_context, verdict, reused)
        return verdict

    guard.call_count = 0
    return guard


def cache_compare_candidate_frames(cache, targets, candidate_sets, nearest_checkpoint,
                                   image_b64, step):
    """Capture the first frame seen at each candidate's resolved checkpoint.

    Candidate identity is positional: two targets may resolve to the same checkpoint and still
    receive separately labeled evidence entries. Malformed/unresolved sets simply capture nothing,
    leaving the VLM path to fail closed when no complete guard can be built.
    """
    if not isinstance(cache, dict) or not isinstance(candidate_sets, list):
        return 0
    targets = list(targets or [])
    if len(targets) != len(candidate_sets):
        return 0
    captured = 0
    for index, checkpoints in enumerate(candidate_sets):
        if (index in cache or not checkpoints
                or nearest_checkpoint not in set(checkpoints)):
            continue
        cache[index] = {
            "target": targets[index],
            "image_b64": image_b64,
            "checkpoint": nearest_checkpoint,
            "step": step,
        }
        captured += 1
    return captured


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
