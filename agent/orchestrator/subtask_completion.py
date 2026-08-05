"""subtask_completion.py - typed-subtask contract + pluggable pickup completion predicates.

Lightweight and sim-free BY DESIGN: no torch/agent/env/openai imports, so the offline predicate
tests and the decomposer A/B harness load in milliseconds (importing subtask_agents pulls the whole
model stack, ~25 s). subtask_agents.py imports the loaded prompt export, parser, and predicates FROM
HERE; the prompt text itself lives in ``agent/prompts/orchestrator/decomposer.md``.

The default pickup, compare, and unknown guards remain deterministic. An optional VLM backend may
inject per-hand pickup verdicts, a labeled multi-frame compare verdict, or a current-frame unknown
task verdict, while every inspect leg requires its own image-bound VLM verdict. Deterministic
physical and structural prerequisites remain additive. A VLM `STOP` is still only a *request* these
predicates grant or refuse.

Type vocabulary is CLOSED: pickup | checkout | compare | goto | inspect. An untypeable decomposition degrades
to `unknown`, which falls back to the pre-6.3 keyword guards (behaviour preserved, logged as
`untyped`). `checkout` REPLACES the master plan's `place`: 6.2 collapsed scan-then-bag into the one
deterministic macro `store_map.checkout_held_item`, whose measured `{scanned, placed}` verdict the
checkout predicate reads rather than re-deriving.
"""

import json
import logging
import os
import re

from agent_core.prompt_loader import load_prompt
from sim.sim_paths import categories_json  # lightweight (stdlib os only) - safe to import here

logger = logging.getLogger(__name__)

# Closed type vocabulary. `checkout` (not `place`) - see the module docstring.
SUBTASK_TYPES = ("pickup", "checkout", "compare", "goto", "inspect")

# A leg whose halt is refused this many times force-continues the LOOP (never a fake grant) with the
# last refusal reason left in state, so a confused agent can't spin forever on STOP. The leg is
# marked `halt_forced` so 6.4 can count it. (phase6.3 plan, "Refusal cap".)
HALT_REFUSAL_CAP = 3

# The symmetric twin of the refusal cap: if the completion predicate stays satisfied for this many
# CONSECUTIVE steps and the agent still never proposes STOP (it keeps pursuing future goals from
# persistent memory), the leg ends success=True with end_reason='completed_no_stop'. One cap stops
# infinite STOP-spam; this stops infinite not-stopping. 6.4 reports halt_granted vs completed_no_stop
# as the completion-detection health signal (did the agent know it was done, or need the backstop?).
COMPLETION_BACKSTOP = 3

# Mid-leg self-correction (2026-07-23): after this many refused STOPs on a pickup leg while a hand
# verifiably holds the WRONG item (mismatched_hands below), run_leg force-releases that hand (drops
# the item) and resets the refusal budget ONCE per leg, so the agent gets a real second attempt at
# the grab instead of spinning its remaining refusals into halt_forced. Set below HALT_REFUSAL_CAP
# on purpose: one refusal is guidance the agent may act on itself; two identical refusals mean it
# is not correcting, so code corrects.
WRONG_ITEM_RELEASE_AFTER = 2


# ---------------------------------------------------------------------------
# Typed decomposer contract
# ---------------------------------------------------------------------------

TYPED_DECOMPOSER_SYSTEM = load_prompt("orchestrator/decomposer")


def parse_decomposition(raw: str, original_task: str) -> list:
    """Turn the decomposer LLM's raw output into a list of typed subtask dicts.

    Every returned item is a dict with at least {"type", "text"}, "type" in SUBTASK_TYPES or
    "unknown". Robust by construction - ANY failure degrades to a single `unknown` subtask carrying
    the original task text (which run_leg then handles with the pre-6.3 keyword guards), never an
    exception. Old-style bare strings (a legacy or half-migrated decomposer) are wrapped as `unknown`
    too, so a mixed response never crashes.
    """
    array_match = re.search(r'\[[\s\S]*\]', raw or "")
    if not array_match:
        return [{"type": "unknown", "text": original_task}]
    try:
        items = json.loads(array_match.group(0))
    except (json.JSONDecodeError, ValueError):
        return [{"type": "unknown", "text": original_task}]
    if not isinstance(items, list) or not items:
        return [{"type": "unknown", "text": original_task}]

    out = []
    for item in items:
        if isinstance(item, str):
            out.append({"type": "unknown", "text": item})
            continue
        if not isinstance(item, dict):
            continue  # skip a stray non-object rather than poison the whole plan
        text = str(item.get("text") or "").strip()
        t = str(item.get("type") or "").strip().lower()
        if t not in SUBTASK_TYPES:
            # Untypeable element: preserve its instruction, fall back to keyword guards.
            out.append({"type": "unknown", "text": text or original_task})
            continue
        if t == "inspect" and not str(item.get("query") or "").strip():
            # An inspection without its verification question cannot be guarded. Degrade safely
            # instead of emitting a malformed member of the closed typed contract.
            out.append({"type": "unknown", "text": text or original_task})
            continue
        norm = {"type": t, "text": text or original_task}
        # Carry the type-specific structured fields through untouched when present.
        for field in ("target", "location", "targets", "criterion"):
            if item.get(field) is not None:
                norm[field] = item[field]
        if t == "inspect":
            norm["query"] = str(item["query"]).strip()
        # `count` is normalized here (int, >=1) so every downstream reader gets a clean value; an
        # unparseable count is dropped (default-1 behaviour) rather than poisoning the leg.
        if item.get("count") is not None:
            try:
                norm["count"] = max(1, int(item["count"]))
            except (TypeError, ValueError) as error:
                logger.warning(
                    "[subtask_completion] invalid pickup count %r (%s); defaulting to 1",
                    item["count"], error,
                )
        out.append(norm)
    return out or [{"type": "unknown", "text": original_task}]


# ---------------------------------------------------------------------------
# Name matching (the pickup / compare grounding primitive)
# ---------------------------------------------------------------------------

# Tokens that carry no product identity - dropped before overlap so a target like "the milk" doesn't
# match on "the". Kept deliberately small; colour/qualifier words (green, large, orange) are content.
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "with", "from", "for",
    "some", "any", "one", "it", "its", "that", "this", "these", "those", "item", "items",
    "product", "pick", "up", "grab", "get", "take", "lift", "bring", "carry", "please",
}


def _tokens(text: str) -> list:
    """Content tokens of a product name/phrase, lowercased, stopwords + <=2-char noise dropped."""
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS]


# --- catalog grounding (2026-07-23) -----------------------------------------------------------
# Two catalog sources, both LAZY + FAILURE-SILENT by design (a machine without them degrades to the
# plain substring check, keeping this module sim-free and import-cheap):
#   1. Categories.json - the full category->SKU index (all ~250 SKUs). Path comes from
#      $SARI_SANDBOX_DIR (config.env; see sim.sim_paths) - there is no path to the separate sim repo
#      that works across machines, so it is read as a setting rather than hardcoded per-developer
#      (the previous absolute broke on every machine but the one it was written on, silently, into
#      this same failure-silent fallback - see agent/CLAUDE.md open thread on catalog grounding).
#      Gives BOTH category membership (a 'Biscuits' target -> any Biscuit SKU) AND the set of
#      category-name words, which are GENERIC (see below).
#   2. products_final_shelf_reconciled.json - the annotated per-SKU records (~56 SKUs) carrying a
#      clean NAME + VARIANT + APPEARANCE. Used to ENRICH the held item's identity text: the raw sim
#      SKU string is munged ('JACK_AND_JILL_PIATTOS_...'), the reconciled name is clean ('Piattos').
#      APPEARANCE (colour/packaging) rides along as a SOFT, ADDITIVE tie-breaker - it can only HELP a
#      target token match (e.g. 'green Piattos' vs a 'green bag' appearance), never block one, so the
#      78% of SKUs with no record never cause a false refusal.
_RECONCILED_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "mapping", "output", "products_final_shelf_reconciled.json")
_CATEGORY_LEXICON = None    # {singularized category word: [sku_lower, ...]}
_GENERIC_NAMES = None       # {category-name tokens} - matching on these alone is not product identity
_RECONCILED_INDEX = None    # {sku_lower: "clean name variant appearance ..."} (category EXCLUDED)
_CATALOG_LOADED = False     # True iff Categories.json was actually read, vs the alias-only fallback.
                            # _category_lexicon() alone can't tell a real catalog from an absent one:
                            # _CATEGORY_ALIASES always merges in 5 pseudo-categories, so the dict is
                            # truthy either way. Anything that needs to distinguish "no sim repo" from
                            # "sim repo present" (e.g. a test's skip guard) should check this flag.

# A size/pack token ('70g', '500ml', '5x18g') is a spec, not identity - matching on it alone is as
# weak as matching on a category word, so it is treated generic too. Pure-digit brands ('555') are
# NOT size tokens (no unit), so they stay distinctive.
_SIZE_TOKEN = re.compile(r"^\d+(g|kg|mg|ml|l|pcs)$|^\d+x\d+")

# Colloquial words the store catalog has no clean category/token for -> the specific SKUs they name,
# matched as lowercase substrings (same containment the category lexicon uses; brand+product suffices).
# Each key is merged into the lexicon as a pseudo-category AND marked generic, so a BARE alias target
# ('cereal') resolves to its SKU list while a NAMED product still wins on its own distinctive token
# ('Coco Pops' -> coco_pops). Lists are kept TIGHT (only true members) so an alias can never grant a
# neighbour that merely shares a broad category - it does NOT alias to a whole loose category.
# Every list below was checked against the frozen catalog (measure, don't assume): the coverage sweep
# is in tests + the 2026-07-24 session notes. WHY each is needed rather than left to substring match:
#   cereal    - MEASURED refusal (run 2026-07-24: target 'cereal', held KELLOGG'S_COCO_POPS_30G,
#               refused): no cereal SKU contains "cereal"; all six are filed under Biscuit by brand.
#   candy /   - both alias the eat-as-is chocolate BARS (Schogetten, Lindt, M&M, Kinder Tronky - a
#   chocolate   choco wafer bar, counted here per user call 2026-07-24). 'candy' matched 0 SKUs;
#               'chocolate' matched 21 across 4 categories as a FLAVOR word - the alias TIGHTENS it to
#               the bars, while a choco-flavored biscuit still matches on its brand ('Pocky chocolate').
#   cheese    - dairy cheese only (Eden, Magnolia Cheezee/DailyQuezo). 'cheese'/'cheesy' is a flavor
#               across Chips/Can/Nuts (19 raw matches); a flavored snack still matches on its brand.
#   yogurt    - Cimory, Dutchmill Proyo, Pascual Greek, Bauer FruFru. Dutchmill *Delight* (cultured
#               milk, not yogurt) is deliberately OUT.
_CATEGORY_ALIASES = {
    "cereal": ["coco_pops", "froot_loops", "frosties",
               "corn_flakes", "honeystars", "kokokrunch"],
    "candy": ["schogetten", "lindt", "m&m", "kinder"],
    "chocolate": ["schogetten", "lindt", "m&m", "kinder"],
    "cheese": ["eden", "cheezee", "dailyquezo"],
    "yogurt": ["cimory", "proyo", "pascual_greek", "frufru"],
}

# Multi-word colloquialisms the single-token path can't resolve: 'energy drink' tokenizes to
# energy+drink, and 'drink' alone matches Sprite/Pocari/Delmonte while MISSING Sting (no 'energy' in
# its SKU). Keyed on the whole PHRASE, tested as a normalized substring of the target. A matched
# phrase contributes its words to the generic set FOR THAT TARGET ONLY (see blob_matches_target), so
# 'Cobra energy drink' still prefers the distinctive 'cobra' (won't grant a Red Bull) while a bare
# 'energy drink' resolves to the tight list. This keeps bare 'milk' UNaffected - it's only neutralized
# when the actual phrase 'soy milk' / 'oat milk' is present, so 'pick up milk' still matches real milk.
_PHRASE_ALIASES = {
    "energy drink": ["cobra", "redbull", "sting"],
    "soy milk": ["vitamilk"],
    "oat milk": ["oatside"],
}


def _singular(word: str) -> str:
    """Cheap plural folding so a target token 'biscuits' finds category 'Biscuit' ('ies'->'y',
    trailing 's' dropped). Not linguistics - just enough for the catalog's category names."""
    w = str(word or "").lower()
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _category_lexicon() -> dict:
    """Load normalized catalog categories and fallback aliases for target matching."""
    global _CATEGORY_LEXICON, _GENERIC_NAMES, _CATALOG_LOADED
    if _CATEGORY_LEXICON is None:
        lex, generic = {}, set()
        _CATALOG_LOADED = False
        categories_path = categories_json(required=False)
        if categories_path:
            try:
                with open(categories_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                for cat in data.get("Categories", []):
                    name = _singular(cat.get("Category"))
                    if name:
                        lex[name] = [str(i).lower() for i in (cat.get("Items") or [])]
                    # Every token of a category name is a generic type-word (both plural and
                    # singular forms, so 'chips'/'chip', 'biscuits'/'biscuit' all count).
                    for tok in _tokens(cat.get("Category")):
                        generic.add(tok)
                        generic.add(_singular(tok))
                _CATALOG_LOADED = True
            except (OSError, ValueError, TypeError, AttributeError) as error:
                lex, generic = {}, set()
                logger.error(
                    "[subtask_completion] failed to load Categories.json from %s (%s: %s); "
                    "using alias-only category matching",
                    categories_path, type(error).__name__, error,
                )
        if not _CATALOG_LOADED and not categories_path:
            logger.warning(
                "[subtask_completion] Categories.json was not found; using alias-only "
                "category matching"
            )
        # Colloquial aliases (see _CATEGORY_ALIASES): merged AFTER the file load so they apply even
        # when the sim repo is absent (they carry their own SKU substrings), and via setdefault so a
        # real catalog category of the same name always wins. Each alias key is also generic, so a
        # bare-alias target ('cereal') falls to category membership instead of demanding a substring.
        for alias, sku_substrs in _CATEGORY_ALIASES.items():
            key = _singular(alias)
            lex.setdefault(key, list(sku_substrs))
            generic.add(alias)
            generic.add(key)
        _CATEGORY_LEXICON = lex
        _GENERIC_NAMES = generic
    return _CATEGORY_LEXICON


def _generic_names() -> set:
    """Return catalog terms that identify a category rather than a specific SKU."""
    _category_lexicon()   # populates _GENERIC_NAMES as a side effect
    return _GENERIC_NAMES or set()


def _is_generic(tok: str) -> bool:
    """A token that carries no product IDENTITY on its own: a catalog category-name word (chips,
    biscuit, can, tuna-less 'canned'... whatever the category names are) or a size/pack spec."""
    return (tok in _generic_names() or _singular(tok) in _generic_names()
            or bool(_SIZE_TOKEN.match(tok)))


def _reconciled_index() -> dict:
    """{sku_lower -> clean identity text} from the annotated reconciled products. Merges EVERY record
    for a SKU (it can appear at several checkpoints), joining name + variant + appearance - but NOT
    category (a category word is generic; folding it in would re-introduce exactly the noise the
    distinctive-token rule strips). Failure-silent: no file -> empty -> matching just uses the raw SKU
    string, as before."""
    global _RECONCILED_INDEX
    if _RECONCILED_INDEX is None:
        idx = {}
        try:
            with open(_RECONCILED_JSON, encoding="utf-8") as fh:
                recs = json.load(fh)
            for r in recs if isinstance(recs, list) else []:
                sku = r.get("sku")
                if not sku:
                    continue
                parts = [str(r.get(f) or "") for f in ("name", "variant", "appearance")]
                txt = " ".join(p for p in parts if p).lower()
                if txt:
                    idx[str(sku).lower()] = (idx.get(str(sku).lower(), "") + " " + txt).strip()
        except (OSError, ValueError):
            idx = {}
        _RECONCILED_INDEX = idx
    return _RECONCILED_INDEX


def _norm_phrase(text) -> str:
    """Lowercase + collapse every non-alphanumeric run to a single space, so a multi-word phrase alias
    ('energy drink') can be tested as a plain substring of a target ('an Energy-Drink!')."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def blob_matches_target(blob, target) -> bool:
    """True iff `blob` (a held/hovered SKU string, any case) is the `target` product.

    Identity rests on DISTINCTIVE tokens, not generic ones (2026-07-23). A target token is generic if
    it names a catalog CATEGORY ('chips', 'biscuit') or is a size spec ('70g') - matching on one of
    those alone is not identity. This fixes a MEASURED false GRANT (run 0723_094628-era: target
    'Chipsy Corn Chips Nacho Crispies BBQ', the agent held LESLIE_S_CLOVER_CHIPS_CHEESE - a DIFFERENT
    product - yet the lone shared category word 'chips' granted it). The correct Chipsy SKU shares
    chipsy/nacho/crispies/bbq; Clover shares only the generic 'chips', so it is now refused.

    Rules, in order:
      - no content tokens in the target -> True (can't ground it, don't block).
      - the held SKU string is ENRICHED with its reconciled clean name/variant/appearance when known
        (soft, additive: appearance colour can only HELP a match like 'green Piattos', never block).
      - a multi-word PHRASE alias present in the target ('energy drink', 'soy milk') neutralizes its
        own words but not a co-occurring brand, so 'Cobra energy drink' still needs 'cobra'.
      - if the target has any DISTINCTIVE token -> match iff one of them is in the enriched text.
      - else a matched phrase resolves to its SKU list; else a PURELY generic target ('Biscuits',
        'cereal', 'candy') falls to category / single-word-alias membership.
    """
    kws = _tokens(target)
    if not kws:
        return True
    blob = str(blob or "").lower()
    # Enrich the held identity text with the clean reconciled name/variant/appearance of any SKU
    # named in the blob (soft; absent for un-annotated SKUs, which then match on the raw string only).
    full = blob
    for sku, txt in _reconciled_index().items():
        if sku in blob:
            full += " " + txt
    # Phrase aliases (see _PHRASE_ALIASES): a multi-word colloquialism present in the target resolves
    # to a tight SKU list. Its component words go generic FOR THIS TARGET, so a more specific brand
    # token still wins ('Cobra energy drink' -> cobra, never a stray Red Bull); a bare phrase falls to
    # its SKU list below. Component words are neutralized ONLY when the phrase is actually present, so
    # bare 'milk' keeps matching real milk.
    ntarget = _norm_phrase(target)
    phrase_words, phrase_skus = set(), []
    for phrase, skus in _PHRASE_ALIASES.items():
        if phrase in ntarget:
            phrase_words.update(_tokens(phrase))
            phrase_skus.extend(skus)
    distinctive = [k for k in kws if not _is_generic(k) and k not in phrase_words]
    if distinctive:
        return any(k in full for k in distinctive)
    if phrase_skus:
        return any(s in blob for s in phrase_skus)
    # Purely-generic target: category (or single-word alias) membership - any member SKU counts.
    lex = _category_lexicon()
    for k in kws:
        skus = lex.get(_singular(k))
        if skus and any(s in blob for s in skus):
            return True
    return False


def name_matches(state: dict, keywords) -> bool:
    """True iff any keyword appears in the agent's hovered/gripped object blob. Re-homed here from
    pickup_navigation (its single caller now imports it) so the pickup predicate and the 6.4 eval share ONE
    implementation. Same substring-containment semantics as before - deliberately loose; the phase6.3
    plan flags tightening as a 6.4-audit follow-up, not a guess to make now.

    The blob includes `gripped_name` - the name the grab tool reported AT THE GRIP - because the live
    `<side>HoveredObject` fields clear to 'null' once the hand retracts from the shelf (a measured
    false-negative: the item is still held, but hovered no longer names it). `gripped_name` is the
    durable record run_leg keeps while a hand is gripping; it is '' for callers (pickup_navigation) that
    don't set it, so this stays backward-safe. Dual-hand: run_leg space-joins BOTH hands' recorded
    names into it (per-hand detail in `gripped_names`), so a match on EITHER held item counts."""
    fields = [state.get("leftHoveredObject") or "", state.get("rightHoveredObject") or "",
              state.get("gripped_name") or ""]
    blob = " ".join(str(f) for f in fields).lower()
    return any(str(k).lower() in blob for k in keywords)


def name_overlap(state: dict, target: str) -> bool:
    """True iff the gripped/hovered object name overlaps the decomposer's free-text `target` -
    token overlap OR category membership (blob_matches_target). An empty/degenerate target (no
    content tokens) returns True - we can't ground it, so we don't block on it (the grip itself is
    still required by the pickup predicate)."""
    fields = [state.get("leftHoveredObject") or "", state.get("rightHoveredObject") or "",
              state.get("gripped_name") or ""]
    return blob_matches_target(" ".join(str(f) for f in fields), target)


def pickup_has_target(sub: dict) -> bool:
    """Whether a pickup has enough target content to invoke a grounding backend."""
    return bool(_tokens((sub or {}).get("target") or ""))


def _validate_guard_backend(guard_backend):
    """Reject completion-guard backends outside the supported deterministic/VLM set."""
    if guard_backend not in ("deterministic", "vlm"):
        raise ValueError(f"unknown completion guard backend: {guard_backend!r}")


def _guard_verdict(guard_verdicts, side):
    """Return one normalized injected verdict, or None (strict bools only)."""
    if not isinstance(guard_verdicts, dict):
        return None
    verdict = guard_verdicts.get(side)
    if not isinstance(verdict, dict) or type(verdict.get("match")) is not bool:
        return None
    return verdict


def mismatched_hands(sub: dict, state: dict, start_grips=(),
                     guard_backend="deterministic", guard_verdicts=None) -> list:
    """The sides currently gripping an item that measurably ISN'T this pickup leg's target - the
    hands run_leg's mid-leg self-correction may auto-release (2026-07-23). Conservative by
    construction; a hand is listed only when ALL of:
      (a) it is gripping NOW,
      (b) it was EMPTY at leg start (`start_grips`) - never drop an item carried in from a previous
          leg; that item belongs to an earlier subtask's outcome,
      (c) its grip-time recorded name (state['gripped_names']) exists AND fails the target match -
          a hand with NO recorded name is never released (we don't drop what we can't identify).
    An untargeted pickup (no content tokens) mismatches nothing. Under the VLM backend, only an
    injected conclusive negative is actionable: missing/failed verdicts refuse completion but never
    cause a corrective release, and deterministic matching is not used as a silent fallback."""
    _validate_guard_backend(guard_backend)
    target = (sub or {}).get("target") or ""
    if not _tokens(target):
        return []
    names = state.get("gripped_names")
    names = names if isinstance(names, dict) else {}
    start = set(start_grips or ())
    out = []
    for side in ("left", "right"):
        name = names.get(side)
        if not (state.get(f"{side}GrippedState") and side not in start and name):
            continue
        if guard_backend == "vlm":
            verdict = _guard_verdict(guard_verdicts, side)
            if verdict and verdict["match"] is False and verdict.get("conclusive") is True:
                out.append(side)
        elif not blob_matches_target(name, target):
            out.append(side)
    return out


# ---------------------------------------------------------------------------
# Completion predicates - each: (subtask_dict, state[, final_text]) -> (granted: bool, reason: str)
# ---------------------------------------------------------------------------

def _gripping(state: dict) -> bool:
    """Return whether either simulator hand is currently gripping an item."""
    return bool(state.get("leftGrippedState") or state.get("rightGrippedState"))


# Pre-6.3 keyword sets, preserved verbatim for the `unknown` fallback so an untypeable subtask keeps
# exactly the old behaviour (no better, no worse - just no longer the ONLY guard).
_PICKUP_KW = ("pick up", "grab", "get", "take", "lift")
_DROP_KW = ("drop", "place", "put down", "set down", "release", "leave")


def predicate_pickup(sub: dict, state: dict, guard_backend="deterministic",
                     guard_verdicts=None) -> tuple:
    """Grant iff a hand grips AND (when a target names a product) the gripped/hovered name overlaps
    it. Refuses the exact failure the old drop-agnostic guard could not see on a paraphrase, and the
    wrong-item grab a bare grip-check would pass.

    `count` (default 1, from the decomposer contract): how many of the target must be held AT ONCE -
    'pick up 2 X' is one leg with count=2, not two legs (a second same-SKU pickup leg would grant
    instantly on the first leg's carry). Counting needs run_leg's per-hand `gripped_names`; a runner
    without it (pickup_navigation's flat loop) degrades to the single-item check, flagged [unverified
    count] - honest, never a silent block the wiring can't feed.

    `guard_backend="vlm"` replaces targeted name matching with injected per-hand verdicts. Missing,
    malformed, and failed verdicts fail closed; deterministic matching is never a VLM fallback."""
    _validate_guard_backend(guard_backend)
    try:
        count = max(1, int(sub.get("count") or 1))
    except (TypeError, ValueError) as error:
        logger.warning(
            "[subtask_completion] invalid pickup count %r (%s); defaulting to 1",
            sub.get("count"), error,
        )
        count = 1
    if not _gripping(state):
        return False, "pickup not complete: no hand is gripping anything yet"
    target = sub.get("target") or ""
    if guard_backend == "vlm" and _tokens(target):
        names = state.get("gripped_names")
        names = names if isinstance(names, dict) else {}
        held = [side for side in ("left", "right") if state.get(f"{side}GrippedState")]
        matched, match_reasons, failures = [], [], []
        for side in held:
            verdict = _guard_verdict(guard_verdicts, side)
            if verdict is None:
                failures.append(f"{side}: no valid VLM guard verdict")
            elif verdict["match"]:
                matched.append(side)
                match_reasons.append(f"{side}: {verdict.get('reason') or 'matched'}")
            else:
                failures.append(f"{side}: {verdict.get('reason') or 'VLM guard refused the match'}")
        if len(matched) >= count:
            return True, (f"pickup complete: VLM guard matched {len(matched)} held item(s) "
                          f"to {target!r} ({count} required); {'; '.join(match_reasons)}")
        held_desc = {side: names.get(side) for side in held}
        return False, (f"pickup not complete: VLM guard matched {len(matched)} of {count} "
                       f"{target!r} (hands: {held_desc}); "
                       f"{'; '.join(failures) or 'not enough matching held items'}")
    if count > 1:
        names = state.get("gripped_names")
        if not isinstance(names, dict):
            if target and not name_overlap(state, target):
                held = (state.get("gripped_name") or state.get("leftHoveredObject")
                        or state.get("rightHoveredObject"))
                return False, (f"gripping, but the held item ({held!r}) does not match target "
                               f"{target!r} - grab the right item")
            return True, (f"pickup granted [unverified count: runner tracks no per-hand names, "
                          f"cannot verify {count} held]")
        matched = [side for side in ("left", "right")
                   if state.get(f"{side}GrippedState")
                   and blob_matches_target(names.get(side) or "", target)]
        if len(matched) >= count:
            return True, f"pickup complete: holding {len(matched)} matching item(s) ({count} required)"
        held_desc = {s: names.get(s) for s in ("left", "right") if state.get(f"{s}GrippedState")}
        return False, (f"pickup not complete: holding {len(matched)} of {count} "
                       f"{target or 'item'!s} (hands: {held_desc}) - the agent has two hands, "
                       "grab another with the free hand")
    if target:
        if not name_overlap(state, target):
            held = (state.get("gripped_name") or state.get("leftHoveredObject")
                    or state.get("rightHoveredObject"))
            return False, (f"gripping, but the held item ({held!r}) does not match target {target!r} - "
                           "grab the right item")
        return True, "pickup complete: gripping the target item"
    # Dual-hand (2026-07-23), UNTARGETED pickups only: a grip carried IN from a previous leg must not
    # satisfy THIS leg's pickup. run_leg sets `new_grip_this_leg` when a hand that was empty at leg
    # start grips; when the runner provides it (pickup_navigation's flat loop does not - key absent skips
    # the check, old behaviour), an untargeted pickup needs a NEW grip, not just any grip. A TARGETED
    # pickup grants on the name match above regardless - the right item in hand is the right item,
    # whichever leg grabbed it.
    if state.get("new_grip_this_leg") is False:
        return False, ("pickup not complete: still holding only the item carried in from a previous "
                       "subtask - grab THIS subtask's item (a hand is free)")
    return True, "pickup complete: gripping an item"


def predicate_checkout(sub: dict, state: dict) -> tuple:
    """Grant iff the checkout macro reported scanned AND placed (both MEASURED - OCR receipt delta /
    released-under-a-placeable-verdict) for the item it just bagged AND no hand is left gripping.
    Reads `last_checkout` (the macro's own result dict surfaced into state); it does NOT re-derive
    the verdict.

    ONE checkout leg = 'scan the items I'm holding' (2026-07-23 fix). The decomposer emits a SINGLE
    checkout leg for everything carried at once ('scan THEM and bag THEM' - the two-hand carry is one
    checkout moment, not two legs), so this is complete only when NOTHING is left in hand. The prior
    version granted as soon as the ONE hand the macro bagged from was empty, which ended the leg after
    the first item and left the second unscanned (measured: run 0723_094628_graph - the leg halted
    with a Pepero still gripped in the right hand). Now a still-holding OTHER hand refuses and sends
    the agent to check that item out too; the checkout macro (hand='auto') targets the remaining hand
    on the next call, so the existing self-correction loop drives the second scan with no new tool."""
    lc = state.get("last_checkout")
    if not lc:
        return False, ("checkout not attempted yet: dispatch the checkout tool "
                       "(it drives to the counter, scans, and bags in one call)")
    if not lc.get("scanned"):
        return False, f"checkout incomplete: item not scanned - {lc.get('reason', 'retry / re-align')}"
    if not lc.get("placed"):
        return False, f"checkout incomplete: scanned but not bagged - {lc.get('reason', 'retry the bag')}"
    # The hand the macro bagged from must actually be empty now - a placed verdict with that same
    # hand still gripping is an inconsistent macro result, retry it.
    hand = lc.get("hand")
    if hand in ("left", "right") and state.get(f"{hand}GrippedState"):
        return False, (f"checkout reported placed but the {hand} hand (the one it bagged from) "
                       f"is still gripping - inconsistent, retry")
    # Any OTHER hand still carrying an item means the leg isn't done - a second item was picked up and
    # only the first is scanned. Send the agent to check the remaining one out too before it STOPs.
    still = [s for s in ("left", "right") if state.get(f"{s}GrippedState")]
    if still:
        return False, (f"one item scanned and bagged, but the {'/'.join(still)} hand is still holding "
                       f"an item - dispatch the checkout tool again (it targets the hand still "
                       f"holding) and scan that one too before you STOP")
    return True, "checkout complete: all held items scanned and bagged (measured)"


def predicate_goto(sub: dict, state: dict) -> tuple:
    """Grant iff the agent's nearest checkpoint is (one of) the leg's resolved target checkpoint(s).
    Both are injected at leg start (Phase 6.3 #1): `sub['target_checkpoint']` - resolved ONCE at plan
    time by the map resolver, an int OR a list (a product/area can live on several shelves) - and
    `state['nearest_checkpoint']` (localised each step from the live pose + the map). If either is
    unavailable the predicate cannot verify, so it grants on the VLM's judgment and says so - honest
    [unverified], never a silent strictness the wiring can't feed."""
    target_cp = sub.get("target_checkpoint")
    near_cp = state.get("nearest_checkpoint")
    if target_cp is None or near_cp is None:
        return True, (f"goto granted [unverified]: checkpoint info unavailable "
                      f"(target={target_cp}, nearest={near_cp})")
    targets = set(target_cp) if isinstance(target_cp, (list, tuple, set)) else {target_cp}
    if not targets:
        return True, "goto granted [unverified]: target resolved to no checkpoint"
    if near_cp in targets:
        return True, f"goto complete: at checkpoint {near_cp} (target {sorted(targets)})"
    return False, f"goto not complete: nearest checkpoint is {near_cp}, target is {sorted(targets)}"


def predicate_compare(sub: dict, state: dict, final_text: str = "",
                      guard_backend="deterministic", compare_guard=None) -> tuple:
    """Grant iff (a) the agent's final output names a choice from `targets`, AND (b) it actually went
    and LOOKED at both candidates - it visited each candidate's checkpoint. With the opt-in VLM
    backend, those deterministic prerequisites are followed by a conclusive positive verdict over
    the labeled frames cached at the candidate checkpoints.

    The visited-both check (Phase 6.3 #4) turns "decide from the camera, not the product index" from a
    prompt instruction into a deterministic gate: `sub['candidate_sets']` (one resolved checkpoint list
    per target, from plan_legs) must each intersect `state['visited_checkpoints']` (the cumulative
    task-level visit trace). It is DEFENSIVE - if the candidates didn't resolve (no `candidate_sets`),
    the visit half is skipped and flagged [unverified] rather than wrongly blocking; a physical-
    inspection eval should ensure they resolve. With no declared targets we can't check the choice
    either, so we grant [unverified]."""
    targets = sub.get("targets") or []
    if not targets:
        if guard_backend == "vlm":
            return False, "compare not complete: no candidate targets are available for the VLM guard"
        return True, "compare granted [unverified]: no candidate targets to check the choice against"
    blob = str(final_text or "").lower()
    named = next((cand for cand in targets if any(tok in blob for tok in _tokens(cand))), None)
    if named is None:
        return False, ("compare not complete: final output must name your choice from "
                       f"{targets} with what you observed")
    # (b) visited-both, when the candidates resolved to checkpoints.
    cand_sets = sub.get("candidate_sets")
    if cand_sets:
        visited = state.get("visited_checkpoints") or set()
        visited = set(visited)
        unseen = [s for s in cand_sets if s and not (set(s) & visited)]
        if unseen:
            return False, (f"compare not complete: chose {named!r} but never stood at "
                           f"candidate checkpoint(s) {unseen} - go LOOK at both before deciding")
        deterministic_reason = f"compare complete: chose {named!r} after visiting both candidates"
    else:
        deterministic_reason = (
            f"compare complete: chose {named!r} "
            "[unverified: candidates had no resolved checkpoints]")
    if guard_backend == "deterministic":
        return True, deterministic_reason
    if not callable(compare_guard):
        return False, "compare not complete: VLM compare guard is unavailable"
    criterion = str(sub.get("criterion") or sub.get("text") or "").strip()
    if not criterion:
        return False, "compare not complete: no comparison criterion is available"
    aux = {
        "targets": list(targets),
        "named_choice": named,
        "candidate_sets": sub.get("candidate_sets"),
        "visited_checkpoints": sorted(state.get("visited_checkpoints") or []),
        "nearest_checkpoint": state.get("nearest_checkpoint"),
    }
    try:
        verdict = compare_guard(criterion, str(final_text or "").strip(), aux)
    except Exception as exc:  # noqa: BLE001 - guards must fail closed
        return False, (f"compare not complete: VLM compare guard failed "
                       f"({type(exc).__name__}: {exc})")
    if (not isinstance(verdict, dict)
            or type(verdict.get("match")) is not bool
            or verdict.get("conclusive") is not True
            or not isinstance(verdict.get("reason"), str)
            or not verdict["reason"].strip()):
        return False, "compare not complete: VLM compare guard returned no conclusive verdict"
    if not verdict["match"]:
        return False, f"compare not complete: VLM rejected the choice ({verdict['reason'].strip()})"
    return True, f"{deterministic_reason}; VLM verified the choice ({verdict['reason'].strip()})"


def predicate_unknown(sub: dict, state: dict, final_text: str = "",
                      guard_backend="deterministic", unknown_guard=None) -> tuple:
    """Apply the legacy keyword/grip checks to an untypeable subtask.

    The deterministic backend preserves the pre-6.3 behavior. The opt-in VLM backend keeps those
    checks as prerequisites, then verifies the free-text completion claim against the current frame.
    """
    text = str(sub.get("text") or "").lower()
    grip = _gripping(state)
    is_pickup = any(kw in text for kw in _PICKUP_KW)
    is_drop = any(kw in text for kw in _DROP_KW)
    if is_pickup and not grip:
        return False, "STOP blocked (untyped): pick-up task but nothing is gripped"
    if is_drop and grip:
        # Dual-hand (2026-07-23): a drop leg that RELEASED a leg-start grip is done even if the other
        # hand still carries its own item. run_leg sets `released_grip_this_leg`; runners that don't
        # (pickup_navigation's flat loop - key absent, falsy) keep the old any-grip block.
        if state.get("released_grip_this_leg"):
            deterministic_reason = (
                "halt granted (untyped): drop task released its item "
                "(the other hand's carried item is its own subtask's business)")
        else:
            return False, "STOP blocked (untyped): drop task but a hand is still gripping"
    else:
        deterministic_reason = "halt granted (untyped): no keyword guard blocks it"
    if guard_backend == "deterministic":
        return True, deterministic_reason
    if not callable(unknown_guard):
        return False, "STOP blocked (untyped): VLM unknown-task guard is unavailable"
    task = str(sub.get("text") or "").strip()
    if not task:
        return False, "STOP blocked (untyped): no task text is available for the VLM guard"
    claim = (str(final_text or "").strip()
             or "The agent requested STOP and claims the task is complete.")
    names = state.get("gripped_names")
    aux = {
        "gripped_name": state.get("gripped_name"),
        "gripped_names": dict(names) if isinstance(names, dict) else names,
        "left_gripped": bool(state.get("leftGrippedState")),
        "right_gripped": bool(state.get("rightGrippedState")),
        "released_grip_this_leg": bool(state.get("released_grip_this_leg")),
        "nearest_checkpoint": state.get("nearest_checkpoint"),
    }
    try:
        verdict = unknown_guard(task, claim, aux)
    except Exception as exc:  # noqa: BLE001 - guards must fail closed
        return False, (f"STOP blocked (untyped): VLM unknown-task guard failed "
                       f"({type(exc).__name__}: {exc})")
    if (not isinstance(verdict, dict)
            or type(verdict.get("match")) is not bool
            or verdict.get("conclusive") is not True
            or not isinstance(verdict.get("reason"), str)
            or not verdict["reason"].strip()):
        return False, "STOP blocked (untyped): VLM unknown-task guard returned no conclusive verdict"
    if not verdict["match"]:
        return False, (f"STOP blocked (untyped): VLM rejected task completion "
                       f"({verdict['reason'].strip()})")
    return True, f"{deterministic_reason}; VLM verified completion ({verdict['reason'].strip()})"


def inspection_evidence_gap(state: dict) -> list:
    """Held hands a two-item inspect leg has not yet read a label from - ``[]`` when none.

    Deterministic pre-check for the multi-item inspection failure measured 2026-07-29: with an item
    in each hand, the agent read one label, reported the comparison, and the single-frame VLM guard
    refused ("the other item's label is not visible in the image") - advice the agent CANNOT act on,
    because only one held item can face the camera at a time. The gap check answers the actionable
    question instead: which held item has no evidence frame yet.

    SCOPED to two gripped hands ON PURPOSE. A single-held-item (or hands-free) inspect leg keeps its
    pre-2026-07-29 behaviour exactly - it may legitimately be answered from the shelf view without
    the held-item macro ever running, and blocking that would be a new false refusal. `state
    ['inspection_evidence']` is run_leg's per-leg ledger of frames that passed the inspection macro's
    visibility gate; a hand whose ledger SKU no longer matches what it holds counts as uncovered (the
    item was swapped), and an absent ledger leaves every gripped hand uncovered.
    """
    state = state if isinstance(state, dict) else {}
    gripped = [side for side in ("left", "right") if state.get(f"{side}GrippedState")]
    if len(gripped) < 2:
        return []
    names = state.get("gripped_names")
    names = names if isinstance(names, dict) else {}
    evidence = state.get("inspection_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    gaps = []
    for side in gripped:
        held = str(names.get(side) or "").strip()
        covered = any(
            isinstance(row, dict) and row.get("hand") == side
            and (not held or not str(row.get("sku") or "").strip()
                 or str(row["sku"]).strip() == held)
            for row in evidence
        )
        if not covered:
            gaps.append({"hand": side, "sku": held or None})
    return gaps


def predicate_inspect(sub: dict, state: dict, final_text: str = "",
                      inspect_guard=None) -> tuple:
    """Grant an observation/reporting leg only on a conclusive positive injected VLM verdict.

    The callback is deliberately image-agnostic here: the runtime binds it to the exact screenshot
    used by the actor, plus the leg's accumulated evidence frames. Missing inputs, callback failures,
    and malformed/inconclusive results all refuse completion. A blank answer is rejected before the
    callback is touched, and so is a two-item leg with an unread held item (see
    ``inspection_evidence_gap``) - a free refusal that tells the agent which item to inspect next
    instead of spending a VLM call to say the frame is insufficient.
    """
    answer = str(final_text or "").strip()
    if not answer:
        return False, "inspection not complete: no answer was reported"
    query = str((sub or {}).get("query") or "").strip()
    if not query:
        return False, "inspection not complete: no inspection query is available"
    gaps = inspection_evidence_gap(state)
    if gaps:
        unread = ", ".join(
            f"{gap['hand']} hand ({gap['sku'] or 'unknown item'})" for gap in gaps)
        return False, (
            f"inspection not complete: you are holding two items but have not read a label from "
            f"the {unread} yet - run inspect_held_item_{gaps[0]['hand']} and read that item before "
            f"you STOP (each item must be inspected in its own turn; they cannot both face the "
            f"camera at once)")
    if not callable(inspect_guard):
        return False, "inspection not complete: VLM inspection guard is unavailable"
    names = state.get("gripped_names")
    evidence = state.get("inspection_evidence")
    aux = {
        "gripped_name": state.get("gripped_name"),
        "gripped_names": dict(names) if isinstance(names, dict) else names,
        "nearest_checkpoint": state.get("nearest_checkpoint"),
        # Which earlier frame shows which item, so the verdict can attribute each reading. Frame
        # bytes never travel here - the guard closure holds them.
        "inspection_evidence": list(evidence) if isinstance(evidence, list) else None,
    }
    try:
        verdict = inspect_guard(query, answer, aux)
    except Exception as exc:  # noqa: BLE001 - guards must fail closed
        return False, (f"inspection not complete: VLM inspection guard failed "
                       f"({type(exc).__name__}: {exc})")
    if (not isinstance(verdict, dict)
            or type(verdict.get("match")) is not bool
            or verdict.get("conclusive") is not True
            or not isinstance(verdict.get("reason"), str)
            or not verdict["reason"].strip()):
        return False, "inspection not complete: VLM inspection guard returned no conclusive verdict"
    if verdict["match"]:
        return True, f"inspection complete: VLM verified the reported answer ({verdict['reason'].strip()})"
    return False, f"inspection not complete: VLM rejected the reported answer ({verdict['reason'].strip()})"


def reported_completion_answer(response: dict) -> str:
    """Extract the structured STOP answer without falling back to its termination placeholder."""
    if not isinstance(response, dict):
        return ""
    answer = response.get("reported_answer")
    return answer.strip() if isinstance(answer, str) else ""


def reported_inspection_answer(response: dict) -> str:
    """Backward-compatible inspection-specific name for ``reported_completion_answer``."""
    return reported_completion_answer(response)


def held_item_inspection_active(sub: dict, state: dict) -> bool:
    """Whether this timestep may expose the constrained held-item inspection controls."""
    return bool(
        isinstance(sub, dict)
        and sub.get("type") == "inspect"
        and isinstance(state, dict)
        and (state.get("leftGrippedState") or state.get("rightGrippedState"))
    )


def planned_subtask_metrics(legs) -> dict:
    """Stable planned-leg vocabulary counts plus the unknown-leg adoption rate."""
    counts = {t: 0 for t in (*SUBTASK_TYPES, "unknown")}
    rows = list(legs or [])
    for leg in rows:
        t = leg.get("type") if isinstance(leg, dict) else "unknown"
        counts[t if t in counts else "unknown"] += 1
    return {
        "planned_leg_type_counts": counts,
        "unknown_subtask_rate": (counts["unknown"] / len(rows)) if rows else 0.0,
    }


def inspect_scope_violation(action: str, step: int, pre_action_state: dict,
                            dispatch_result: dict) -> dict:
    """Build an audit event for a hard-blocked inspect-scope violation, or return ``None``."""
    action = str(action or "")
    result = dispatch_result or {}
    if not result.get("inspect_scope_violation"):
        return None
    grabs = {"extend_arm_until_grabbed", "extend_arm_until_grabbed_left",
             "extend_arm_until_grabbed_right"}
    macros = {"checkout_held_item", "checkout_held_item_left", "checkout_held_item_right"}
    if action in grabs:
        kind = "grip_or_grab"
    elif action in ("grip_left", "grip_right"):
        side = action.split("_", 1)[1]
        kind = ("grip_toggle_release"
                if pre_action_state.get(f"{side}GrippedState") else "grip_or_grab")
    elif action in macros:
        kind = "checkout_macro"
    elif action in {"move_forward", "move_backward", "move_left", "move_right"}:
        kind = "body_translation"
    else:
        kind = "disallowed_action"
    return {
        "event": "inspect_scope_violation",
        "step": step,
        "action": action,
        "kind": kind,
        "blocked": True,
        "executed": False,
        "reason": result.get("reason"),
        "pre_action_grip_state": {
            "left": bool(pre_action_state.get("leftGrippedState")),
            "right": bool(pre_action_state.get("rightGrippedState")),
        },
    }


def completion_predicate(sub: dict, state: dict, final_text: str = "",
                         guard_backend="deterministic", guard_verdicts=None,
                         inspect_guard=None, compare_guard=None,
                         unknown_guard=None) -> tuple:
    """Dispatch a subtask's halt request to its typed predicate. Returns (granted, reason). Any
    unrecognized type routes to the `unknown` keyword fallback, so this never raises on a malformed
    subtask."""
    _validate_guard_backend(guard_backend)
    t = (sub or {}).get("type")
    if t == "pickup":
        return predicate_pickup(sub, state, guard_backend=guard_backend,
                                guard_verdicts=guard_verdicts)
    if t == "checkout":
        return predicate_checkout(sub, state)
    if t == "goto":
        return predicate_goto(sub, state)
    if t == "compare":
        return predicate_compare(
            sub, state, final_text, guard_backend=guard_backend,
            compare_guard=compare_guard)
    if t == "inspect":
        return predicate_inspect(sub, state, final_text, inspect_guard=inspect_guard)
    return predicate_unknown(
        sub, state, final_text, guard_backend=guard_backend,
        unknown_guard=unknown_guard)
