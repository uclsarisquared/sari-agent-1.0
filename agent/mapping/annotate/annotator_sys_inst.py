"""Prompt composition and JSON schemas for the Phase 3 checkpoint annotator.

The instruction text lives under ``agent/prompts/annotator``; this module retains the public
``SYS_INST_*`` exports, executable schemas, and composition rules used by all annotator backends.

There are FOUR instruction sets, not two, because the cheap classifier is a genuinely different
job from annotating:

  1. SYS_INST_CLASSIFY          - Stage 1. One image (the perpendicular). shelf / non_shelf.
  2. SYS_INST_ANNOTATE_BASE     - Stage 2, shared rules.
  3. SYS_INST_ANNOTATE_SHELF    - Stage 2 overlay for shelf nodes.
  4. SYS_INST_ANNOTATE_NON_SHELF- Stage 2 overlay for junction/end/doorway/wall nodes.

Every rule below encodes a decision from the Phase 3 / 3.1 design discussions - the WHY matters,
so it's commented rather than left as folklore. See plans/phase3_vlm_annotation_pass.md and
plans/phase3.1_semantic_product_layer.md. In short:

  * MULTI-VIEW, ALL ITEM-BEARING: a shelf node is annotated from the STANDING view plus (when
    captured) the CROUCHED view - the SAME shelf from the SAME spot at a lower camera height, which
    reaches the bottom rows the standing frame clips or renders oblique. The `items` rule reads
    from ALL views and dedups the overlap. Cross-shelf leakage (a neighbouring shelf intruding at a
    frame edge) is barred by the "only THIS shelf" rule, NOT by restricting to a single view.
  * PROSE vs STRUCTURE: `semantic_summary` is prose because the consumer is an LLM (this is
    modelled on agent's semantic_memory.txt). `items` and `shelf_type` stay structured
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

VIEW VOCABULARY (2026-07-20): the prompts name the views STANDING / CROUCHED, matching what
capture_walk actually sends (the primary standing frame + the crouch shot) and what the backends
label them. This replaced the earlier STRAIGHT / DOWN / UP wording, which advertised pitch views
that never arrived. A/B'd on sonnet (the annotator baseline) old-vocab vs new-vocab, K=6 each on
cp015 + cp017: cp015 dead-even (~9 items both), cp017 pure noise (both swing 0-2 on that sparse
shelf) - i.e. zero recall regression, a consistency fix not a quality change. A first n=1 qwen probe
looked like a regression (11->6) but did not replicate; qwen is not the annotator and misbehaved.

MEASURED - DO NOT RE-ATTEMPT (2026-07, all on cp067 captures):
  * THERE IS NO `brand` FIELD, on purpose. It used to be its own field; the name/brand SPLIT turned
    out to be ambiguous, and the model resolved it differently run to run. Three runs on one image:
    2x "mode A" (name = a generic category - "potato chips" x11 - with the real identity in brand),
    1x "mode B" (name = "Clover Chips", brand = a redundant "Clover"). Mode A was arguably the model
    OBEYING the old wording, whose examples ("corn flakes", "instant noodles", "bottled water") were
    all generic - so `name`, the flat index's ONLY query key, came back unsearchable in most runs
    while everything looked fine. Folding brand into `name` dissolves the ambiguity: 3/3 runs then
    returned specific names, zero generic.
  * Do NOT try to prompt the brand hallucination away. An explicit "null unless you can literally
    read it; do not infer; repeating the product's name here is wrong" bullet made every axis worse
    on the identical image: items 9 -> 19 (one entry per FACING) and names collapsed to "chips" x11.
    It hallucinated regardless ("Mama"). Reverted.
  * NAME POLICY (adopted 2026-07 after a user review of a real pass): a name must be TEXT READ OFF
    THE LABEL; an item whose label can't be read is OMITTED, never named by description or grouped.
    This SUPERSEDES the earlier "fall back to a generic product name" clause, which was producing
    exactly the junk the review flagged - grouping rows ("assorted soda bottles", "bottled water
    (assorted brands)") and appearance rows ("green chip bag", "canned goods (gold lid)"). Crucially
    it does NOT re-trigger the mode-flip in the bullet above: that flip happened because a generic
    name ("potato chips") was an available escape hatch; forbidding generic/description names removes
    the hatch, so the model's only moves are "read the label" or "omit the item" - no generic bucket
    to flee into. Validated on the two reviewed captures: cp015 10->8 items (both "assorted" rows
    gone, 8 real brands kept), cp017 9->2 (seven description rows gone, Tostitos + Mackerel kept),
    zero generic names in either. The cost is RECALL: an unreadable label is now a missing row, not a
    described one - which is the point (a described row was never findable anyway); the fix for a
    genuinely-wanted-but-unreadable item is resolution (tiling / closer), not a looser name rule. The
    focused single-question pass still reads a brand the omnibus form can't ("read the brand, or
    answer UNREADABLE" -> "LuckyMe!"); that remains the only measured way to raise recall on hard
    labels without inventing.
  * Reading distance IS load-bearing for LARGE printed text, and does nothing for small logos.
    Stepping the agent in at cp067: at 1.54m variants came back mostly null; at 1.20m they came back
    correct ("Mild", "Cheesier", "Original", "Sour Cream & Onion") and a previously-missed product
    ("Platos") appeared. The brand logo stayed unreadable throughout. shelf_coverage's
    MAX_READING_DISTANCE_M was 1.5m - parked just outside the legible range.
  * Thinking must be OFF (chat_template_kwargs={"enable_thinking": False}). With it on, Qwen3.6
    spent all 2048 tokens reasoning, fell into a repetition loop, and returned content=None.
    Thinking OFF is necessary but not sufficient: on a dense shelf of near-identical facings, greedy
    decoding (temperature=0) can still loop INSIDE the items array until it hits max_tokens (cp020,
    2026-07-24 - a stack of Pancit Canton cups). Sampling penalties break the loop but gut recall;
    see the measured note at the finish_reason=="length" branch in annotate_qwen.py. This is a qwen
    decoding weakness, not a prompt bug - the claude-cli baseline does not loop on cp020.

CALLER CONTRACT (the prompts below promise these; the client has to deliver them):
  * Image labelling. SYS_INST_ANNOTATE_BASE tells the model each image is labelled STANDING or
    CROUCHED. Nothing enforces that - the client must send a text part before each image saying
    which one it is (annotate_pass ships the primary as STANDING and the crouch shot as CROUCHED;
    annotate_qwen / annotate_claude_cli label the same way). Post bare images and the multi-view
    "enumerate across every view and dedup the overlap" rule loses its referent.
  * Effective kind. The Stage-2 overlay is chosen by effective_kind(), NOT by a checkpoint's raw
    topology kind. See that function - the distinction is the entire reason Stage 1 exists.

OPEN DECISIONS (deliberately not baked in yet):
  * Structured-output enforcement: RESOLVED 2026-07, measured against a vLLM server serving the
    then-default model (262k ctx) over Chat Completions at :8000/v1. Send the *_SCHEMA
    dicts as `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ...,
    "strict": True}}`. NOT as `guided_json` - this build SILENTLY IGNORES that spelling, which is
    invisible in normal use because the prompt alone already yields conforming JSON. It only shows
    up against a negative control: given a schema whose sole legal label was "banana", guided_json
    returned the model's own "non_shelf" while response_format returned "banana". vLLM's strict
    mode is also more permissive than OpenAI's - it accepts `sign_text` being absent from
    `required` alongside `additionalProperties: False`, which OpenAI strict would reject.
  * Neighbour context injection: NOT included. The graph already knows connectivity, and feeding
    it in invites the spatial claims rule 4 forbids. The renderer adds "connects to..." from the
    graph instead. Revisit only if summaries read as too isolated.
  * Classifier model: written to be model-agnostic; it's narrow enough for a small VLM.
"""

from agent_core.prompt_loader import load_prompt


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

# MEASURED 2026-07-21 (rule 1, sonnet): the original rule 1 judged ONLY the CENTRE of the frame,
# which rejected every checkpoint placed at an aisle END or in a GAP between units - there the
# shelf fills one HALF of the frame (off-centre) while open floor/wall fills the centre, so the
# model judged the floor and returned non_shelf, discarding a densely-stocked, readable shelf.
# Found on cp35/49/53 (+cp31, a fridge/shelf gap that was already flip-flopping). Rewriting rule 1
# to count a NEAR shelf anywhere in the frame flipped all four to shelf 12/12 while the genuine
# walls cp22/28/36/40 stayed non_shelf 8/8 (negative control) - a recall fix with no wall
# regression. The anti-leakage intent survives as "ignore only a SMALL, DISTANT edge shelf".
SYS_INST_CLASSIFY = load_prompt("annotator/classify")


# ---------------------------------------------------------------------------------------------
# Stage 2 - shared base
# ---------------------------------------------------------------------------------------------

SYS_INST_ANNOTATE_BASE = load_prompt("annotator/base")


# ---------------------------------------------------------------------------------------------
# Stage 2 - shelf overlay
# ---------------------------------------------------------------------------------------------

SYS_INST_ANNOTATE_SHELF = load_prompt("annotator/shelf")


# ---------------------------------------------------------------------------------------------
# Stage 2 - non-shelf overlay
#
# NOTE the non-shelf asset has no ``{placeholders}``, so build_annotation_instructions() uses it
# verbatim. The shelf asset is formatted with ``categories`` and must double its literal JSON braces.
# ---------------------------------------------------------------------------------------------

SYS_INST_ANNOTATE_NON_SHELF = load_prompt("annotator/non_shelf")


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
                    # Brand is folded INTO name, deliberately - see the measured note above.
                    "name": {"type": "string"},
                    "variant": {"type": ["string", "null"]},
                    # String, not number: the tag may read "29.95", "29", or partially. Kept
                    # nullable and NOT required - a mis-paired price is worse than a missing one,
                    # and the catalog's pricePHP gives a ground truth to score these against.
                    "price": {"type": ["string", "null"]},
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
    # Eyeball the composed prompts: python mapping/annotate/annotator_sys_inst.py
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
