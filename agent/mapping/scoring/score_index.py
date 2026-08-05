"""Phase 4 step 0.2a - OFFLINE index coverage scoring. No model, no sim.

Scores the annotated product index (products_<tag>.json) against two ground truths:

  1. The Unity catalog (PriceData.json, 232 SKUs) - "how much of the store's vocabulary does
     the index know a location for, by NAME?" This is deliberately a DETERMINISTIC match and
     therefore a LOWER BOUND: the LLM resolver bridges name gaps this scorer cannot ("Coke
     Zero" -> "Coca-Cola", measured). Use it for the reconciler decision, never as the
     resolution denominator - that is 0.2b's job, with the model in the loop.
  2. The live store's placements (Store 2 v2.json `shelfItems`) - slot-level truth, which
     exists ONLY for shelves 6/7/8 (the back fridges; 128 placements, 50 distinct SKUs as of
     2026-07-19). For those we can also ask the harder question: is the SKU indexed at a
     checkpoint that actually FACES its shelf?

Matching: squashed-alphanumeric substring, both directions ("cocacola" vs
"cocacolaregular500ml"). Crude on purpose - a fancier matcher here would smuggle resolver
intelligence into a number that is supposed to measure the INDEX, not the matcher.

    python mapping/scoring/score_index.py
    python mapping/scoring/score_index.py --products-tag final_shelf --verbose
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/scoring
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)
from sim import sim_paths  # noqa: E402

CATALOG_DIR = sim_paths.data_dir()          # $SARI_SANDBOX_DIR/Assets/Resources/Data (config.env)
STORE_JSON = sim_paths.store_save_json()    # $SARI_STORE_SAVE_JSON (config.env)

# Fridge shelves span (x_min, x_max) at z=6.0, face at z ~= 5.5; a checkpoint "faces" one if
# its x lies within the span +- slack and it sits on the approach line z ~= 4.55.
FRIDGE_SPANS = {6: (-5.125, -3.875), 7: (-3.625, -2.375), 8: (-2.125, -0.875)}
APPROACH_Z = (4.0, 5.2)
X_SLACK = 0.45  # a checkpoint at a shelf boundary sees both sides; measured on cp19/cp20


def squash(s):
    # "'s" dropped BEFORE squashing: "Nature's Spring" must meet NATURE_SPRING_500ML, and the
    # possessive otherwise leaves a stray 's' that defeats the substring test (measured: it
    # produced a false MISSING and a false ghost for the same product in one run).
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("'s", ""))


def sku_squash(sku):
    """Drop size/packaging tokens (500ML, 87.9G, 1L...) - they never appear in VLM names."""
    toks = [t for t in sku.split("_")
            if not re.fullmatch(r"[0-9.]+(ML|L|G|KG|OZ)?", t)]
    return squash("".join(toks))


def name_matches_sku(name_sq, sku_sq):
    return bool(name_sq) and (name_sq in sku_sq or sku_sq in name_sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-tag", default="final_shelf")
    ap.add_argument("--output-dir", default=os.path.join(_MAPPING_DIR, "output"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.output_dir, f"products_{args.products_tag}.json"),
              encoding="utf-8") as f:
        products = json.load(f)
    with open(os.path.join(CATALOG_DIR, "PriceData.json"), encoding="utf-8") as f:
        catalog = list(json.load(f).keys())
    with open(os.path.join(CATALOG_DIR, "Categories.json"), encoding="utf-8") as f:
        cat_of = {sku: c["Category"] for c in json.load(f)["Categories"] for sku in c["Items"]}
    with open(STORE_JSON, encoding="utf-8") as f:
        store = json.load(f)

    index_names = sorted({r["name"] for r in products})
    index_sq = [(n, squash(n)) for n in index_names]
    rows_by_name = defaultdict(list)
    for r in products:
        rows_by_name[r["name"]].append(r["checkpoint_id"])

    # ---- 1. catalog-wide name coverage (lower bound) ----
    per_cat = defaultdict(lambda: [0, 0])
    matched = {}
    for sku in catalog:
        sq = sku_squash(sku)
        hit = next((n for n, nsq in index_sq if name_matches_sku(nsq, sq)), None)
        matched[sku] = hit
        cat = cat_of.get(sku, "?")
        per_cat[cat][1] += 1
        if hit:
            per_cat[cat][0] += 1

    total_hit = sum(1 for v in matched.values() if v)
    print(f"== 0.2a catalog-wide name coverage (DETERMINISTIC LOWER BOUND) ==")
    print(f"   {total_hit}/{len(catalog)} catalog SKUs have a name-matched index row "
          f"({100*total_hit/len(catalog):.0f}%)")
    for cat in sorted(per_cat, key=lambda c: per_cat[c][0]/max(per_cat[c][1],1)):
        h, t = per_cat[cat]
        print(f"   {cat:<10} {h:>3}/{t:<3} ({100*h/max(t,1):3.0f}%)")

    # ---- 2. fridge region: placement truth ----
    placed = defaultdict(set)          # sku -> {shelf_id}
    for key, row in store["shelfItems"].items():
        shelf_id = int(re.match(r"ID(\d+)_", key).group(1))
        for it in row["items"]:
            placed[it["name"]].add(shelf_id)

    # which checkpoints face each fridge shelf
    from nav.store_map import StoreMap
    sm = StoreMap(output_dir=args.output_dir)
    facing = defaultdict(list)
    for cp_id in sm.shelf_checkpoints():
        x, z = sm.checkpoint(cp_id)["world_xz"]
        if APPROACH_Z[0] <= z <= APPROACH_Z[1]:
            for sid, (x0, x1) in FRIDGE_SPANS.items():
                if x0 - X_SLACK <= x <= x1 + X_SLACK:
                    facing[sid].append(cp_id)

    print(f"\n== 0.2a fridge-region truth (shelves 6/7/8 only - the ONLY slot-level truth) ==")
    print(f"   facing checkpoints: " +
          ", ".join(f"shelf{sid}->{sorted(cps)}" for sid, cps in sorted(facing.items())))
    n_any = n_right = 0
    misses = []
    for sku, shelf_ids in sorted(placed.items()):
        sq = sku_squash(sku)
        hits = [n for n, nsq in index_sq if name_matches_sku(nsq, sq)]
        ok_any = bool(hits)
        ok_right = any(cp in facing[sid]
                       for n in hits for cp in rows_by_name[n] for sid in shelf_ids)
        n_any += ok_any
        n_right += ok_right
        if not ok_any:
            misses.append(sku)
        if args.verbose:
            mark = "AT-SHELF" if ok_right else ("indexed " if ok_any else "MISSING ")
            print(f"   [{mark}] {sku:<45} shelves={sorted(shelf_ids)} via={hits[:2]}")
    n = len(placed)
    print(f"   indexed anywhere      : {n_any}/{n} ({100*n_any/n:.0f}%)")
    print(f"   indexed at a checkpoint facing its shelf: {n_right}/{n} ({100*n_right/n:.0f}%)")
    print(f"   missing entirely      : {n - n_any}")
    for sku in misses:
        print(f"     MISSING: {sku}  ({cat_of.get(sku, '?')})")

    # ---- 3. inverse: index rows with no catalog referent (hallucination floor) ----
    ghosts = [n for n, nsq in index_sq
              if not any(name_matches_sku(nsq, sku_squash(s)) for s in catalog)]
    print(f"\n== index names with NO catalog match (misread or invented): "
          f"{len(ghosts)}/{len(index_names)} ==")
    for g in ghosts:
        print(f"   {g}  @cp{sorted(set(rows_by_name[g]))}")


if __name__ == "__main__":
    main()
