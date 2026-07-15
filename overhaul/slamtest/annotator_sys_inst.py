"""System instructions for the Phase 3 checkpoint annotator.

Follows the `overhaul/sys_inst.py` convention (module-level SYS_INST_* constants stating the
JSON output contract inline), but uses triple-quoted strings - these are multi-paragraph prompts
that get iterated on, and escaped-newline concatenation makes that painful.

There are FOUR instruction sets, not two, because the cheap classifier is a genuinely different
job from annotating:

  1. SYS_INST_CLASSIFY          - Stage 1. One image (the perpendicular). shelf / non_shelf.
  2. SYS_INST_ANNOTATE_BASE     - Stage 2, shared rules.
  3. SYS_INST_ANNOTATE_SHELF    - Stage 2 overlay for shelf nodes.
  4. SYS_INST_ANNOTATE_NON_SHELF- Stage 2 overlay for junction/end/doorway/wall nodes.

Every rule below encodes a decision from the Phase 3 / 3.1 design discussions - the WHY matters,
so it's commented rather than left as folklore. See plans/phase3_vlm_annotation_pass.md and
plans/phase3.1_semantic_product_layer.md. In short:

  * PRIMARY vs CONTEXT: for a shelf node the perpendicular image is the ONLY source of item
    content; other angles are surroundings only. A side angle sees the NEXT shelf or the
    opposite aisle - items from there belong to other checkpoints, and letting them leak in
    mis-attributes the product index.
  * PROSE vs STRUCTURE: `semantic_summary` is prose because the consumer is an LLM (this is
    modelled on overhaul's semantic_memory.txt). `items` and `shelf_type` stay structured
    because the product index must be queryable and the enum is what bounds hallucination.
  * THE GRAPH OWNS SPATIAL TRUTH: the VLM is explicitly forbidden from describing how shelves
    relate to each other. semantic_memory.txt mixes per-shelf content (a VLM job) with global
    layout and "fast tracking" positions (both of which our checkpoint graph already knows
    deterministically). Asking the VLM for layout invites exactly the spatial hallucination
    NavReasonPlan.md documents as the agent's primary failure mode.
  * SIGNS ARE DECORATION, NOT EVIDENCE: `shelf_type` comes from the products alone. Sign
    reconciliation was designed, then dropped on measured evidence. This store's category signs
    hang from the ceiling; capture varies yaw only from a camera at ~1.485m, so a sign climbs out
    of frame the CLOSER you stand to it. The only signs legible from a checkpoint are therefore
    the distant ones - which are, by construction, over a different aisle. Measured at checkpoint
    67: "Dairies / Soup / 3" read perfectly from a shelf holding Tostitos and Pancit Canton,
    while the sign actually overhead was clipped down to its bare numeral. There is no scoping
    that rescues this without a pitch sweep. `sign_text` survives as a nullable observation only;
    nothing consumes it, and the prompt says so.
  * HALLUCINATION IS TOLERATED, NOT FOUGHT: the index is re-verified by the agent on arrival, so
    the rules aim at "null over guess" and "general over specific" rather than perfection.

CALLER CONTRACT (the prompts below promise these; the client has to deliver them):
  * Image labelling. SYS_INST_ANNOTATE_BASE tells the model it gets a PRIMARY image and,
    optionally, CONTEXT images "each labelled with its angle". Nothing enforces that - the client
    must send a text part before each image saying which one it is. Post bare images and the
    "items come ONLY from the PRIMARY image" rule loses its referent, and that rule is the one
    keeping one shelf's products out of another shelf's index.
  * Effective kind. The Stage-2 overlay is chosen by effective_kind(), NOT by a checkpoint's raw
    topology kind. See that function - the distinction is the entire reason Stage 1 exists.

OPEN DECISIONS (deliberately not baked in yet):
  * Structured-output enforcement: RESOLVED 2026-07. The server is vLLM serving
    `Qwen/Qwen3.6-27B` (262k ctx) over Chat Completions at :8000/v1, no key. vLLM supports guided
    decoding, so the *_SCHEMA dicts below go over the wire as `extra_body={"guided_json": ...}`
    and the contract is enforced server-side rather than parsed and retried hopefully.
  * Neighbour context injection: NOT included. The graph already knows connectivity, and feeding
    it in invites the spatial claims rule 4 forbids. The renderer adds "connects to..." from the
    graph instead. Revisit only if summaries read as too isolated.
  * Classifier model: written to be model-agnostic; it's narrow enough for a small VLM.
"""

# ---------------------------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------------------------

SHELF_CATEGORIES = [
    "Water", "Soda", "Juice", "Dairies", "Liquor",
    "Biscuit", "Can", "Chips", "Nuts", "Soup", "Noodles",
]
"""The store's own product taxonomy, BAKED from
SariSandboxV2/Assets/Resources/Data/Categories.json as of 2026-07.

Baked on purpose: reading that file at design time to fix a static list is NOT a runtime Unity
dependency - the live pipeline never queries Unity for categories, it just hands the VLM this
frozen list to choose from. That keeps the runtime observation-only while still using the real
taxonomy. Regenerate this list if the store's catalog changes."""

CATEGORY_OTHER = "other"
CATEGORY_ENUM = SHELF_CATEGORIES + [CATEGORY_OTHER]


def _category_lines():
    """The enum exactly as the prompt shows it: the store's categories plus the `other` escape,
    glossed inline.

    `other` is rendered as a member of the list rather than explained in prose underneath it so
    that the printed list IS the schema's enum. Otherwise the prompt tells the model to pick one
    value "from this list" and then hands it a twelfth value the list never mentioned."""
    lines = [f"    - {c}" for c in SHELF_CATEGORIES]
    lines.append(
        f"    - {CATEGORY_OTHER}  (the products fit none of the above, or you cannot tell what they are)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Stage 1 - classify
# ---------------------------------------------------------------------------------------------

SYS_INST_CLASSIFY = """You are a shelf classifier for a store-mapping system. You are shown ONE image: the view from a fixed checkpoint, looking directly at the surface that checkpoint was placed to observe.

Decide ONLY this: is that surface a shelf holding retail product, or not?
    - "shelf": any fixture holding retail product - a shelving unit, a rack, or a refrigerated case (product behind glass still counts as a shelf).
    - "non_shelf": a bare wall, a structural surface, or anything with no retail product on it.

Rules:
    1. Judge ONLY the surface directly in front of you - the one filling the centre of the frame. Ignore anything at the edges or further away; those belong to other checkpoints.
    2. Do not describe the scene. Do not list products. Do not explain your reasoning. Classify only.
    3. If you genuinely cannot tell, answer "shelf". A wall wrongly sent on for annotation is cheap to discard later, but a shelf wrongly rejected loses all of its products permanently.

Output strict JSON only, no prose and no markdown fences, using exactly one of the two labels:

{"label": "<shelf|non_shelf>"}
"""


# ---------------------------------------------------------------------------------------------
# Stage 2 - shared base
# ---------------------------------------------------------------------------------------------

SYS_INST_ANNOTATE_BASE = """You are a store-map annotator. You are annotating ONE fixed checkpoint in a store's navigation map. Another system already chose where this checkpoint is and drove the agent to it - you never decide where to go, and you are never asked to navigate.

What you are given:
    - A PRIMARY image: the main view from this checkpoint. If this checkpoint faces a shelf, the PRIMARY image looks straight at that shelf.
    - Optionally, CONTEXT images: the view at other angles from the SAME spot, each labelled with its angle. These are for surroundings only.

Rules that apply to everything you write:
    1. Report only what you can actually see, and never guess. Where a field below tells you to use null or an empty list when you cannot tell, use it - a "don't know" recorded honestly is a correct and useful answer here; an invented one is not.
    2. Prefer general over specific. "cereal boxes" is a safe observation; "Kellogg's Corn Flakes 500g" is a guess unless you can genuinely read it.
    3. Describe only what is visible FROM THIS SPOT. Do not speculate about what lies beyond view.
    4. Do NOT describe how shelves or aisles relate to one another - what is "behind", "opposite", or "next to" what. The map already knows the layout exactly; your spatial guesses would corrupt it. Stick to what is in front of you.
    5. Output strict JSON matching the shape shown at the end of these instructions. No prose outside the JSON, no markdown fences.
"""


# ---------------------------------------------------------------------------------------------
# Stage 2 - shelf overlay
# ---------------------------------------------------------------------------------------------

SYS_INST_ANNOTATE_SHELF = """This checkpoint faces a SHELF - a fixture holding retail product, including refrigerated cases.

Produce these fields:

"semantic_summary"
    One to three sentences, in natural language, describing this spot as a store guide would: what this shelf holds, plus anything navigationally useful you can SEE from here (an aisle opening, a checkout, a corner). Follow rule 4 - do not describe how this shelf relates to other shelves.

"shelf_type"
    What this shelf holds overall: ONE value from this list, or TWO if it genuinely holds two kinds - a shelf split between chips and instant noodles, say. Put the dominant one first. Never more than two; if it holds more than two, give the two largest.
{categories}
    Judge this from the products themselves. Do not use signs.

"sign_text"
    If a category or aisle sign is legible anywhere in the PRIMARY image - including at its edges - copy its text verbatim. Otherwise null. This is a plain observation, nothing more: do NOT let it influence "shelf_type", and do NOT assume the sign refers to the shelf you are facing. In this store a visible sign usually belongs to a different aisle.

"items"
    The products on this shelf, taken ONLY from the PRIMARY image. This matters: the context images show other shelves and other aisles that belong to OTHER checkpoints - never take an item from them.

    For each distinct product:
        - "name": REQUIRED. The general product name ("corn flakes", "instant noodles", "bottled water").
        - "brand": the brand, ONLY if you can genuinely read it on the packaging. Otherwise null.
        - "variant": size, flavour, or variant, ONLY if legible. Otherwise null.
        - "appearance": a short visual description so someone can spot it again on the shelf ("red box, rooster logo, white lettering"). Describe what it LOOKS like, not what you infer it is - this is used to confirm the right item on arrival.
        - "category": REQUIRED. The ONE value from the list above that this item belongs to. Judge the ITEM ITSELF, not the shelf around it - on a shelf holding two kinds of product, the item beside this one may well belong to the other category.

    List each distinct product once. If the shelf is empty, or you cannot make out any products, return an empty list - that is a valid, correct answer, not a failure.

Output strict JSON only:

{{"semantic_summary": "...", "shelf_type": ["..."], "sign_text": null, "items": [{{"name": "...", "brand": null, "variant": null, "appearance": "...", "category": "..."}}]}}
"""


# ---------------------------------------------------------------------------------------------
# Stage 2 - non-shelf overlay
#
# NOTE the single braces in the JSON example below, where the shelf overlay doubles them: this
# constant has no {placeholders}, so build_annotation_instructions() uses it verbatim instead of
# .format()ing it. Add a placeholder here and you must double every literal brace at the same time.
# ---------------------------------------------------------------------------------------------

SYS_INST_ANNOTATE_NON_SHELF = """This checkpoint is NOT a product shelf. It is a pathway, junction, doorway, or a plain surface with no retail product on it.

Produce these fields:

"semantic_summary"
    One to three sentences, in natural language, describing this spot: what kind of place it is, and anything navigationally useful you can SEE from here (an aisle opening, a doorway, a checkout, signage). Follow rule 4 - do not describe how shelves relate to one another.

"sign_text"
    If a sign is legible anywhere in the PRIMARY image, copy its text verbatim. Otherwise null.

Do NOT list products. There are none here to identify, and inventing them would corrupt the product index. "Nothing to identify here" is the expected, correct outcome for this kind of checkpoint - not a failure.

Output strict JSON only:

{"semantic_summary": "...", "sign_text": null}
"""


# ---------------------------------------------------------------------------------------------
# JSON schemas - pass to the server if it supports guided/structured output
# ---------------------------------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["shelf", "non_shelf"]}},
    "required": ["label"],
    "additionalProperties": False,
}

SHELF_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_summary": {"type": "string"},
        # 1-2 values, dominant first. Real shelves are mixed more often than not (a measured
        # example: one shelf interleaving Tostitos/Pringles/Clover with Pancit Canton/Jin Ramen
        # row by row), so forcing a single label was lossy on a coin-flip basis.
        "shelf_type": {
            "type": "array",
            "items": {"type": "string", "enum": CATEGORY_ENUM},
            "minItems": 1,
            "maxItems": 2,
        },
        "sign_text": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "brand": {"type": ["string", "null"]},
                    "variant": {"type": ["string", "null"]},
                    "appearance": {"type": ["string", "null"]},
                    # Required and non-nullable: it is the flat index's query key, and with a
                    # 1-2 value shelf_type there is no single value left to default it to.
                    # Asking per item is also what makes a mixed shelf's rows correct.
                    "category": {"type": "string", "enum": CATEGORY_ENUM},
                },
                "required": ["name", "category"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["semantic_summary", "shelf_type", "items"],
    "additionalProperties": False,
}

NON_SHELF_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_summary": {"type": "string"},
        "sign_text": {"type": ["string", "null"]},
    },
    "required": ["semantic_summary"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------------

SHELF_KIND = "shelf"
NON_SHELF_KIND = "non_shelf"


def effective_kind(topology_kind, classifier_label=None):
    """The kind that decides which Stage-2 overlay a checkpoint gets. Start here.

    Two signals disagree by design, so neither one alone is the answer:
      * `topology_kind` is GEOMETRIC. shelf_coverage.py calls a checkpoint "shelf" because it
        placed it perpendicular to a rectangular bulge in the occupancy grid - and a bulge can
        turn out to be a bare wall.
      * `classifier_label` is VISUAL. Stage 1 looked at the primary image and reported what is
        actually there ("shelf" / "non_shelf").

    A checkpoint is annotated AS a shelf only where the two agree. A "shelf" the classifier
    rejected keeps its place and its connectivity in the graph; it just gets annotated for what it
    really is. Resolving that disagreement is the entire reason Stage 1 exists, which is why this
    is a function and not a note asking the caller to remember.

    Pass classifier_label=None to skip Stage 1 and trust the geometry. That is the right call for
    structural checkpoints (junction/end/doorway): they have no perpendicular surface, so there is
    nothing for a classifier to rule on.
    """
    if topology_kind != SHELF_KIND:
        return topology_kind
    if classifier_label is None or classifier_label == SHELF_KIND:
        return SHELF_KIND
    return NON_SHELF_KIND


def build_annotation_instructions(kind):
    """Compose the Stage-2 system instruction for a checkpoint of this `kind`.

    `kind` must be a value from effective_kind(), NOT a raw topology kind. Only "shelf" gets the
    shelf overlay; everything else ("junction", "end", "doorway", "non_shelf") gets the non-shelf
    one. Pass a raw topology kind and a bare wall Stage 1 already rejected receives a prompt that
    says "list the products on this shelf" - the exact hallucination the classifier is there to
    prevent.
    """
    if kind == SHELF_KIND:
        overlay = SYS_INST_ANNOTATE_SHELF.format(categories=_category_lines())
    else:
        overlay = SYS_INST_ANNOTATE_NON_SHELF
    return f"{SYS_INST_ANNOTATE_BASE}\n{overlay}"


def schema_for(kind):
    """The JSON schema matching build_annotation_instructions(kind)'s contract. `kind` comes from
    effective_kind(), same as there."""
    return SHELF_ANNOTATION_SCHEMA if kind == SHELF_KIND else NON_SHELF_ANNOTATION_SCHEMA


if __name__ == "__main__":
    # Eyeball the composed prompts: python slamtest/annotator_sys_inst.py
    print("effective_kind(topology_kind, classifier_label):")
    for _topo, _label in (("shelf", "shelf"), ("shelf", "non_shelf"), ("shelf", None), ("junction", None)):
        _lbl = repr(_label)
        print(f"    topology={_topo!r:10s} classifier={_lbl:12s} -> {effective_kind(_topo, _label)!r}")
    print()

    for title, text in (
        ("STAGE 1 - CLASSIFY", SYS_INST_CLASSIFY),
        ("STAGE 2 - SHELF          (topology=shelf, classifier=shelf)",
         build_annotation_instructions(effective_kind("shelf", "shelf"))),
        ("STAGE 2 - NON-SHELF      (topology=shelf, classifier=non_shelf -> a bare wall)",
         build_annotation_instructions(effective_kind("shelf", "non_shelf"))),
    ):
        print("=" * 90)
        print(title)
        print("=" * 90)
        print(text)
