"""Phase 4 step 0.2b - resolver correctness, WITH the model in the loop. No sim needed.

Phase 4 is evaluated on LOCATING tasks only (no manipulation, no centring - see the phase doc's
Evaluation contract). This scores the resolver across the five task phrasings a store agent
actually receives, which are five DIFFERENT resolution problems, not one:

  precise_name  full formal product name, as a catalog/packaging would write it
                ("Coca-Cola Light 500ml", "Lucky Me! Pancit Canton")
                -> tests: does formal phrasing still land on the index's looser name?
  general_name  the product TYPE, no brand ("corned beef", "instant noodles", "butter cookies")
                -> tests: can it gather the brand-named rows that ARE that type? The index is
                   brand-heavy ('555 Corned Beef', 'CDO Corned Beef'), so this is a
                   one-to-MANY expansion, the opposite shape from precise_name.
  ingredient    "an item containing peanuts" / "something with milk"
                -> tests knowledge the index DOES NOT CARRY. products_*.json has name/variant/
                   price/appearance/category and NO ingredients; the annotator never read an
                   ingredients panel. So this stratum measures the resolver's WORLD KNOWLEDGE
                   against the catalog's allergen ground truth (159/250 SKUs carry an
                   `allergens` string). Expect this to be the weakest stratum; that is the
                   point of measuring it. See the [decision] note in the phase doc about
                   whether to enrich the index offline from PriceData.json.
  shelf_with    "a shelf with chips", "a shelf with canned goods" -> category-level. Every
                checkpoint holding that shelf_type is equally correct, so recall matters more
                than precision here and the rubric weights accordingly.
  slang         "coke", "chichirya", "de lata", "softdrinks" - including Filipino store slang,
                which is what this store's users would actually say
                -> tests the semantic bridge no deterministic matcher can make.

Scoring is about RESOLUTION, not stock: "Coke Zero" resolving to the cola checkpoints is
CORRECT even though the store doesn't stock it. Found/not-found is the driving+verification
layer's job, measured separately by the live trial set.

    python mapping/scoring/eval_resolver.py --backend qwen        # needs $OPENAI_API_URL + $OPENAI_API_KEY
    python mapping/scoring/eval_resolver.py --backend claude-cli
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/scoring
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from nav.store_map import StoreMap  # noqa: E402
from nav import locate_task  # noqa: E402
from sim import sim_paths  # noqa: E402

CATALOG = sim_paths.price_data_json()   # $SARI_SANDBOX_DIR/Assets/Resources/Data/PriceData.json


def squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("'s", ""))


def sku_squash(sku):
    return squash("".join(t for t in sku.split("_")
                          if not re.fullmatch(r"[0-9.]+(ML|L|G|KG|OZ|X[0-9.]+G)?", t)))


def cps_matching(sm, *substrings):
    """Checkpoints whose index rows' names contain any of these substrings (squashed)."""
    subs = [squash(s) for s in substrings]
    return {r["checkpoint_id"] for r in sm.products
            if any(s in squash(r["name"]) for s in subs)}


def cps_with_allergen(sm, allergen):
    """Checkpoints holding an index row whose name matches a catalog SKU carrying `allergen`.
    This is the ONLY defensible expectation set for the ingredient stratum: it joins the
    catalog's ground-truth allergen string to the index by name, so it credits the resolver for
    any genuinely correct answer and never for a guess the store cannot support."""
    with open(CATALOG, encoding="utf-8") as f:
        catalog = json.load(f)
    skus = [k for k, v in catalog.items()
            if v.get("allergens") and allergen.lower() in v["allergens"].lower()]
    sq = [sku_squash(s) for s in skus]
    out = set()
    for r in sm.products:
        n = squash(r["name"])
        if n and any(n in s or s in n for s in sq):
            out.add(r["checkpoint_id"])
    return out


def build_tasks(sm):
    """Expectations derive from the artifacts + catalog at runtime, so this cannot drift stale."""
    cat = lambda c: set(sm.category_checkpoints(c))
    T = []

    # 1. PRECISE NAME - formal/full product name
    T += [
        ("Find the Coca-Cola Light 500ml.", "precise_name", cps_matching(sm, "coca-cola"), "name"),
        ("Find the Lucky Me! Pancit Canton.", "precise_name", cps_matching(sm, "pancit canton"), "name"),
        ("Find the Century Tuna.", "precise_name", cps_matching(sm, "century tuna"), "name"),
        ("Find the Leslie's Clover Chips.", "precise_name", cps_matching(sm, "clover chips"), "name"),
        ("Find the Danisa Butter Cookies.", "precise_name", cps_matching(sm, "danisa"), "name"),
    ]

    # 2. GENERAL NAME - product type, no brand. One-to-MANY.
    T += [
        ("Find corned beef.", "general_name", cps_matching(sm, "corned beef", "carne norte", "karne norte"), "name"),
        ("Find instant noodles.", "general_name", cps_matching(sm, "pancit canton", "ramen", "mi goreng", "noodle"), "name"),
        ("Find butter cookies.", "general_name", cps_matching(sm, "butter cookies", "danisa"), "name"),
        ("Find bottled water.", "general_name", cat("Water"), "name"),
        ("Find sardines.", "general_name", cps_matching(sm, "sardines"), "name"),
    ]

    # 3. INGREDIENT - knowledge the index does not carry; scored vs catalog allergens.
    # ONLY DISCRIMINATING allergens. MEASURED 2026-07-19: Milk covers 20/24 shelf checkpoints,
    # Wheat 19/24, Soy 17/24 - "find something with milk" is satisfied by nearly every shelf in
    # the store, so a resolver that answered "everywhere" would score ~1.0 and the task would
    # measure nothing. Fish (3), Tree Nuts (3), Crustacean (2) and Egg (2) actually
    # discriminate. Do not re-add the common allergens as scored tasks.
    T += [
        ("Find an item containing fish.", "ingredient", cps_with_allergen(sm, "Fish"), None),
        ("Find something containing tree nuts.", "ingredient", cps_with_allergen(sm, "Tree Nuts"), None),
        ("Find a product with shellfish in it.", "ingredient", cps_with_allergen(sm, "Crustacean"), None),
        ("Find an item that contains egg.", "ingredient", cps_with_allergen(sm, "Egg"), None),
        ("Find an item containing peanuts.", "ingredient", cps_with_allergen(sm, "Peanuts"), None),
    ]

    # 4. SHELF WITH X - category level; recall-weighted
    T += [
        ("Find a shelf with chips.", "shelf_with", cat("Chips"), "category"),
        ("Find a shelf with canned goods.", "shelf_with", cat("Can"), "category"),
        ("Find a shelf with instant noodles.", "shelf_with", cat("Noodles"), "category"),
        ("Find a shelf with biscuits.", "shelf_with", cat("Biscuit"), "category"),
        ("Find a shelf with soda.", "shelf_with", cat("Soda"), "category"),
    ]

    # 5. SLANG / colloquial, incl. Filipino store vocabulary
    T += [
        ("Find the coke.", "slang", cps_matching(sm, "coca-cola"), "name"),
        ("Where are the chichirya?", "slang", cat("Chips"), None),
        ("Find the de lata.", "slang", cat("Can"), None),
        ("Find some softdrinks.", "slang", cat("Soda"), None),
        ("Find the mineral water.", "slang", cat("Water"), None),
    ]
    return T


def score_one(stratum, expect_cps, expect_tier, result):
    got = set(result.get("candidates") or [])
    tier = result.get("tier")
    if not expect_cps:
        # No defensible expectation -> honest refusal is the only scoreable pass.
        return (1.0, "refused (nothing to find)") if not got else (0.0, f"invented {sorted(got)}")
    if not got:
        return 0.0, f"no candidates (tier={tier})"
    inter = got & expect_cps
    precision = len(inter) / len(got)
    recall = len(inter) / len(expect_cps)
    if stratum in ("shelf_with", "general_name", "ingredient"):
        # One-to-many strata: missing valid shelves is the real failure; a spare candidate
        # costs one wasted visit, which the driving layer self-corrects. Recall-weighted.
        base = 0.35 * precision + 0.65 * recall
    else:
        base = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tier_ok = (expect_tier is None) or (tier == expect_tier)
    detail = (f"tier={tier}{'' if tier_ok else f'(want {expect_tier})'} "
              f"cand={sorted(got)} P={precision:.2f} R={recall:.2f}")
    return (0.85 * base + 0.15 * (1.0 if tier_ok else 0.0)), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["claude-cli", "qwen"], default="qwen")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--only", default=None, help="run one stratum only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sm = StoreMap()
    call = locate_task.make_backend(args)
    tasks = [t for t in build_tasks(sm) if not args.only or t[1] == args.only]

    rows, by_stratum = [], {}
    t0 = time.time()
    print(f"== 0.2b resolver eval: backend={args.backend}, {len(tasks)} tasks ==")
    cur = None
    for task, stratum, expect_cps, expect_tier in tasks:
        if stratum != cur:
            print(f"\n-- {stratum} --")
            cur = stratum
        t1 = time.time()
        try:
            result, _ = locate_task.resolve(call, sm, task)
            err = None
        except Exception as e:
            result, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t1
        score, detail = (0.0, err[:110]) if err else score_one(stratum, expect_cps, expect_tier, result)
        rows.append({"task": task, "stratum": stratum, "score": round(score, 3),
                     "expected": sorted(expect_cps), "expected_tier": expect_tier,
                     "result": result, "detail": detail, "seconds": round(dt, 1)})
        by_stratum.setdefault(stratum, []).append(score)
        print(f"  [{score:4.2f}] {task:<42} {detail}  ({dt:.0f}s)")

    print(f"\n== summary ({time.time()-t0:.0f}s) ==")
    for s, sc in by_stratum.items():
        print(f"   {s:<13} {sum(sc)/len(sc):.2f}  (n={len(sc)})")
    overall = sum(r["score"] for r in rows) / len(rows)
    print(f"   {'OVERALL':<13} {overall:.2f}")

    out = args.out or os.path.join(_MAPPING_DIR, "output",
                                   f"resolver_eval_{args.backend}_{datetime.now():%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "rows": rows,
                   "by_stratum": {s: sum(v)/len(v) for s, v in by_stratum.items()},
                   "overall": overall}, f, indent=2, ensure_ascii=False)
    print(f"   -> {out}")


if __name__ == "__main__":
    main()
