"""Snap-to-catalog reconciler (CLAUDE.md open thread 2; Phase 4 step 0.3 said build it).

Fuzzy-matches every annotated product name to a canonical SKU from the Unity catalog
(PriceData.json + Categories.json), producing products_<tag>_reconciled.json - the same rows,
ENRICHED, never merged and never renamed:

    + sku                canonical catalog ID, or null when no defensible match
    + sku_candidates     when several SKUs tie (brand-only reads: "Coca-Cola" vs
                         REGULAR/LIGHT/ZERO/ORIGINAL) - the variant list, sku stays null
    + variant_uncertain  true for those brand-only reads (CLAUDE.md asked for exactly this flag)
    + match              {method, score, runner_up} - the audit trail for every snap

Design stances, and why:

  * NULL OVER GUESS. A wrong snap poisons the index worse than a ghost name does - a ghost
    fails a lookup loudly, a wrong snap routes the agent somewhere confidently wrong. The
    thresholds are calibrated on measured danger cases: "Hunt's Corned Beef" and "555 Corned
    Beef" have NO catalog referent and sit one brand-token from real SKUs
    (ARGENTINA_CORNED_BEEF, 555_TUNA_*); both must stay null. Runner-up margin + category veto
    are what hold that line, not the raw ratio alone.
  * ROWS ARE NEVER MERGED and `name` is never overwritten - Phase 3.1's no-merge rule stands.
    Fragmentation ("Knock Knacks"/"Knock Krack"/"Knock Kracks") collapses at QUERY time because
    all three carry the same sku, not because the rows became one.
  * OFFLINE, DETERMINISTIC, FREE. Pure string algebra over two JSONs; no model, no sim. The
    catalog path is PINNED - three stale worktree copies of the Unity project exist under
    Assets/Scripts/.claude/worktrees/, and a glob would find four PriceData.jsons.

    python mapping/scoring/reconcile_products.py                    # writes *_reconciled.json + report
    python mapping/scoring/reconcile_products.py --self-test        # calibration cases only, no write
"""
import argparse
import difflib
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
CATALOG_DIR = sim_paths.data_dir()   # $SARI_SANDBOX_DIR/Assets/Resources/Data (config.env)

# Tokens that carry no identity: grammar, packaging, and the filler words annotators add.
STOPWORDS = {"the", "a", "an", "in", "with", "of", "and", "flavored", "flavor", "pack",
             "packs", "twin", "original", "premium", "traditional", "classic", "regular",
             "style", "home"}
# 'original' etc. ARE identity for variant disambiguation, but at the NAME->SKU stage they
# cause more false splits than they resolve ("Ritz Crackers Original" vs RITZ_ORIGINAL_3PACK
# works either way; "Danisa Traditional Butter Cookies" vs DSFAPS_DANISA_BUTTERCOOKIES fails
# WITHOUT dropping them). Variant selection happens on sku_candidates later, not here.

# A size token REQUIRES a unit. Bare numbers are NOT sizes - "555" is a brand, and dropping it
# turned "555 Corned Beef" into "corned beef", which containment-matched every corned beef in
# the catalog (measured self-test failure). '87' from "87.9G" survives as SKU-side noise, which
# the directional scorer ignores.
SIZE_RE = re.compile(r"^[0-9.,]+(ML|L|G|KG|OZ|PCS|PACK|X[0-9.,x]*G?)$|^[0-9]+X[0-9.]+G?$", re.I)


def tokens_of(s):
    s = s.lower().replace("'s", "").replace("&", " and ")
    toks = re.split(r"[^a-z0-9]+", s)
    # len<=2 tokens are noise, not identity: the 's' in LESLIE_S "contains into" every string
    # (measured: it welded Piattos to Leslie's at 1.0), and 'a'/'go' add nothing a longer
    # token doesn't. Dropping them is what makes containment scoring safe.
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS and not SIZE_RE.match(t)]


def squash(s):
    return "".join(tokens_of(s))


def tok_sim(a, b):
    if a == b:
        return 1.0
    # Containment counts as a full match: catalog SKUs concatenate words (BLACKCURRANTWAFERS,
    # BBQOVERLOAD, AJINOMOTO), so 'wafers'/'bbq'/'aji' must meet them at full strength - the
    # plain ratio gives 'wafers'~'blackcurrantwafers' 0.50 and the token gate kills a correct
    # snap (measured: Bissin wafers, Aji Soup & Go, Pic-A all false-nulled). The floor is on
    # the SHORTER string (>=3, with <=2-char tokens already dropped in tokens_of - a first
    # attempt gated only the NAME side and LESLIE_S's 's' welded Piattos to Leslie's at 1.0).
    # Deliberate consequences, all covered in self_test: 'king'-in-'kangkongking' passes here
    # but "Bangkok King" still nulls on its other token (bangkok~kangkongking 0.63 < 0.65);
    # 'hunt'-in-'hunts' passes but "Hunt's Corned Beef" nulls on corned/beef vs PORK&BEANS.
    # ...but partial containment is NOT identity: 'pica' is a prefix of 'picattos' by
    # coincidence, and at a flat 1.0 it outscored the true referent PIATTOS (0.933), snapping
    # the misread to the wrong REAL product - the worst outcome the reconciler can produce.
    # Scaling by length ratio keeps 'wafers'-in-'blackcurrantwafers' strong (0.90) while
    # 'pica'-in-'picattos' (0.925) drops just below a near-perfect whole-token match, which
    # puts PIATTOS back on top and demotes PICA to family-candidate at worst.
    if min(len(a), len(b)) >= 3 and (a in b or b in a):
        return 0.85 + 0.15 * (min(len(a), len(b)) / max(len(a), len(b)))
    return difflib.SequenceMatcher(None, a, b).ratio()


def name_score(name_toks, sku_toks):
    """Directional token alignment: every NAME token must find a home among the SKU's tokens
    (the SKU may carry extra tokens - brand prefixes, variants - without penalty; extra NAME
    tokens ARE penalised via the averaging). Length-weighted so 'podioron'~'polvoron' counts
    more than 'hop'~'hop'."""
    if not name_toks or not sku_toks:
        return 0.0
    num = den = 0.0
    for t in name_toks:
        best = max(tok_sim(t, s) for s in sku_toks)
        w = len(t)
        num += w * best
        den += w
    return num / den


class Reconciler:
    def __init__(self):
        with open(os.path.join(CATALOG_DIR, "PriceData.json"), encoding="utf-8") as f:
            self.catalog = list(json.load(f).keys())
        with open(os.path.join(CATALOG_DIR, "Categories.json"), encoding="utf-8") as f:
            self.cat_of = {sku: c["Category"]
                           for c in json.load(f)["Categories"] for sku in c["Items"]}
        self.sku_toks = {sku: tokens_of(sku.replace("_", " ")) for sku in self.catalog}
        self.sku_sq = {sku: "".join(self.sku_toks[sku]) for sku in self.catalog}
        # UNfiltered squash, for the all-short-token fallback: RC_COLA's own 'rc' token is
        # dropped by tokens_of too, so the filtered squash is 'cola' and 'rc' can never prefix
        # it (measured self-test failure).
        self.sku_raw = {sku: re.sub(r"[^a-z0-9]", "", sku.lower()) for sku in self.catalog}

    def match(self, name, category=None):
        """-> dict(sku, sku_candidates, variant_uncertain, match). Thresholds calibrated on the
        measured cases in self_test() - change them there first."""
        n_toks = tokens_of(name)
        n_sq = "".join(n_toks)
        if not n_toks:
            # Names that are ALL short tokens ("RC") vanish under the <=2-char filter and
            # scored 0.00 against their own product (RC -> RC_COLA, measured). Fallback:
            # raw-squash PREFIX match only - prefix, not containment, because 'rc' is a
            # substring of 'supeRCrunch' and containment would family them together.
            raw = re.sub(r"[^a-z0-9]", "", name.lower())
            hits = sorted(sku for sku, sq in self.sku_raw.items() if sq.startswith(raw))
            if len(hits) == 1:
                return {"sku": hits[0], "sku_candidates": None, "variant_uncertain": False,
                        "match": {"method": "prefix", "score": 1.0, "runner_up": None}}
            if hits:
                return {"sku": None, "sku_candidates": hits, "variant_uncertain": True,
                        "match": {"method": "prefix_family", "score": 1.0, "runner_up": None}}
            return {"sku": None, "sku_candidates": None, "variant_uncertain": False,
                    "match": {"method": "none", "score": 0.0, "runner_up": 0.0}}
        scored = []
        for sku in self.catalog:
            s_sq = self.sku_sq[sku]
            if n_sq and (n_sq in s_sq or s_sq in n_sq):
                scored.append((1.0, sku, "exact"))
            else:
                scored.append((name_score(n_toks, self.sku_toks[sku]), sku, "fuzzy"))
        scored.sort(reverse=True)
        best_score, best_sku, method = scored[0]

        # Ties/near-ties: same-brand variant family (brand-only read) vs genuine ambiguity.
        near = [(sc, sk) for sc, sk, _ in scored if sc >= best_score - 0.04]
        if method == "exact":
            exact_near = [(sc, sk) for sc, sk in near if sc == 1.0]
            if len(exact_near) > 1:
                # "Coca-Cola" containment-matches every COCACOLA_* variant: brand-only read.
                return {"sku": None,
                        "sku_candidates": sorted(sk for _, sk in exact_near),
                        "variant_uncertain": True,
                        "match": {"method": "brand_only", "score": 1.0, "runner_up": None}}
            return {"sku": best_sku, "sku_candidates": None, "variant_uncertain": False,
                    "match": {"method": "exact", "score": 1.0,
                              "runner_up": round(scored[1][0], 3)}}

        # Fuzzy tier. THREE gates hold the null-over-guess line (all calibrated in self_test):
        #  - overall score >= 0.76 ("Knock Krack" -> KNICK_KNACKS scores 0.764; the 0.76-0.86
        #    band is why the audit list in the report exists)
        #  - EVERY name token >= 0.65 against its best SKU token. This is the brand gate:
        #    "Hunt's Corned Beef" scores 0.886 overall against STAR_NUTRIMEATS_CORNED_BEEF_
        #    CHUNKY_CHEESE because corned+beef align perfectly and 'hunt'~'chunky' happens to
        #    hit 0.600 exactly (common block "hun") - averages hide brands; minima don't, but
        #    only if the gate clears coincidental blocks. MEASURED landscape: impostor pairs
        #    top out at 0.600 (hunt~chunky); true misreads bottom at 0.727 (krack~knacks).
        #    0.65 sits in the middle of that gap. Re-measure before moving it.
        #  - category consistency when both sides know their category.
        runner = next((sc for sc, sk, _ in scored[1:] if sk != best_sku), 0.0)
        cat_sku = self.cat_of.get(best_sku)
        # Chips<->Biscuit is exempt from the category veto: the store's own taxonomy blurs the
        # snack aisle (CLAUDE.md: cereals are filed under Biscuit; measured here: the annotator
        # filed "Knock Krack" under Chips while the catalog says Knick Knacks is Biscuit, and
        # the veto turned a calibrated-correct snap into a ghost IN PRODUCTION ONLY - the
        # self-test passed because it doesn't pass categories. Divergence between those two is
        # always a category-veto smell.)
        blurry = {frozenset(("Chips", "Biscuit"))}
        cat_veto = (category and cat_sku and category != "other"
                    and category != cat_sku
                    and frozenset((category, cat_sku)) not in blurry)
        min_tok = (min(max(tok_sim(t, s) for s in self.sku_toks[best_sku]) for t in n_toks)
                   if n_toks else 0.0)
        if best_score >= 0.76 and min_tok >= 0.65 and not cat_veto:
            fam = [(sc, sk) for sc, sk in near if sc >= 0.76]
            if len(fam) > 1:
                return {"sku": None, "sku_candidates": sorted(sk for _, sk in fam),
                        "variant_uncertain": True,
                        "match": {"method": "fuzzy_family", "score": round(best_score, 3),
                                  "runner_up": round(runner, 3)}}
            return {"sku": best_sku, "sku_candidates": None, "variant_uncertain": False,
                    "match": {"method": "fuzzy", "score": round(best_score, 3),
                              "runner_up": round(runner, 3)}}
        return {"sku": None, "sku_candidates": None, "variant_uncertain": False,
                "match": {"method": "none", "score": round(best_score, 3),
                          "runner_up": round(runner, 3),
                          "nearest": best_sku, "cat_veto": bool(cat_veto)}}


# Calibration truth - measured 2026-07-19 against the real ghost list. POSITIVE cases must
# snap to (a SKU containing) the fragment; NEGATIVE cases must return sku=None because the
# catalog holds no such product and the nearest SKU is a DIFFERENT product one brand-token
# away. These are the reconciler's regression suite; tune thresholds HERE, never in prod runs.
POSITIVE = [
    ("Ritz Riginal", "RITZ_ORIGINAL"),
    ("Ritz Crackers Riginal", "RITZ_ORIGINAL"),
    ("Ritz Crackers Original", "RITZ_ORIGINAL"),
    ("Knock Knacks", "KNICK_KNACKS"),
    ("Knock Krack", "KNICK_KNACKS"),
    ("Knock Kracks", "KNICK_KNACKS"),
    ("Knick Knacks", "KNICK_KNACKS"),
    ("Cheezy Corn Crunchy", "CHEEZY_CORN_CRUNCH"),
    ("Danisa Traditional Butter Cookies", "DANISA"),
    ("Leslie's Clover Chips Cheesier", "CLOVER"),
    ("Piattos", "PIATTOS"),
    ("Picattos", "PIATTOS"),
    ("Lava Cake", "LAVACAKE"),
    ("Mamon Twin Packs", "MAMON"),
    # Recovered by containment-aware tok_sim (concatenated SKU words) - keep these pinned:
    ("Bisin Blackcurrant Wafers", "BISSIN"),
    ("Aji Soup & Go", "SOUP&GO"),
    ("Pic-A BBQ Overload", "PICA_BBQOVERLOAD"),
    # Production-only regressions - these MUST run with the category the index actually
    # carries, because they exercise gates the bare name never hits:
    ("Knock Krack", "KNICK_KNACKS", "Chips"),   # Chips/Biscuit blurry-pair veto exemption
    ("RC", "RC_COLA", "Soda"),                  # all-short-token name -> raw prefix fallback
]
NEGATIVE = [
    "Hunt's Corned Beef",      # catalog has only HUNTS_PORK&BEANS; nearest is ARGENTINA_CORNED_BEEF
    "555 Corned Beef",         # 555_* is tuna/sardines only
    "Bangkok King",            # not in catalog at all
    "Century Vienna Sausage",  # vienna sausage exists only as LIBBYS/WAGI
    "Hunts Beef Loaf",         # beef loaf exists only as CDO/EL_RANCHO/STAR
    # Right BRAND, but probably a DIFFERENT product than the catalog's one HoP entry
    # (SOFTCOOKIES) - "crisp rice" vs "soft cookies". Was first listed as a positive; demoted
    # deliberately: snapping right-brand-wrong-product is the exact failure null-over-guess
    # exists to prevent, and 'podioron'~'houseofpolvoron' can't clear the token gate anyway.
    "House of Podioron HOP Crisp Rice",
]
BRAND_ONLY = ["Coca-Cola", "Pepsi"]  # must yield variant_uncertain + candidates, sku=None


def self_test(rec, verbose=True):
    fails = []
    for case in POSITIVE:
        name, want = case[0], case[1]
        r = rec.match(name, case[2] if len(case) > 2 else None)
        ok = r["sku"] and want in r["sku"]
        # a variant family containing the right product also passes (e.g. Clover Chips sizes)
        if not ok and r["sku_candidates"]:
            ok = any(want in c for c in r["sku_candidates"])
        if not ok:
            fails.append(f"POS  {name!r}: got {r['sku'] or r['sku_candidates']} "
                         f"({r['match']['method']} {r['match']['score']})")
        elif verbose:
            print(f"  ok POS  {name:<36} -> {r['sku'] or r['sku_candidates']}"
                  f"  [{r['match']['method']} {r['match']['score']}]")
    for name in NEGATIVE:
        r = rec.match(name)
        if r["sku"] is not None or r["sku_candidates"]:
            fails.append(f"NEG  {name!r}: wrongly snapped to "
                         f"{r['sku'] or r['sku_candidates']} ({r['match']})")
        elif verbose:
            print(f"  ok NEG  {name:<36} -> null  [nearest {r['match'].get('nearest')} "
                  f"{r['match']['score']}]")
    for name in BRAND_ONLY:
        r = rec.match(name)
        if not r["variant_uncertain"] or not r["sku_candidates"]:
            fails.append(f"BRAND {name!r}: expected variant_uncertain+candidates, got {r}")
        elif verbose:
            print(f"  ok BRAND {name:<35} -> {len(r['sku_candidates'])} variants, "
                  f"variant_uncertain")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-tag", default="final_shelf")
    ap.add_argument("--output-dir", default=os.path.join(_MAPPING_DIR, "output"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rec = Reconciler()
    print(f"== self-test ({len(POSITIVE)} positive / {len(NEGATIVE)} negative / "
          f"{len(BRAND_ONLY)} brand-only) ==")
    fails = self_test(rec)
    if fails:
        print("\nSELF-TEST FAILURES:")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("  ALL PASS")
    if args.self_test:
        return

    src = os.path.join(args.output_dir, f"products_{args.products_tag}.json")
    with open(src, encoding="utf-8") as f:
        products = json.load(f)

    stats = defaultdict(int)
    cache = {}
    for row in products:
        key = (row["name"], row.get("category"))
        if key not in cache:
            cache[key] = rec.match(row["name"], row.get("category"))
        row.update(cache[key])
        stats[row["match"]["method"]] += 1

    out = os.path.join(args.output_dir, f"products_{args.products_tag}_reconciled.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=1, ensure_ascii=False)

    names = {}
    for row in products:
        names.setdefault(row["name"], row)
    n_names = len(names)
    snapped = {n: r for n, r in names.items() if r["sku"]}
    family = {n: r for n, r in names.items() if r["variant_uncertain"]}
    ghosts = {n: r for n, r in names.items() if not r["sku"] and not r["sku_candidates"]}

    print(f"\n== reconciliation of {len(products)} rows / {n_names} distinct names ==")
    print(f"   rows by method: {dict(stats)}")
    print(f"   names -> single SKU        : {len(snapped)}")
    print(f"   names -> variant family    : {len(family)}  (variant_uncertain)")
    print(f"   names -> null (ghosts)     : {len(ghosts)}")
    print(f"\n-- fuzzy snaps (the reconciler's actual work - AUDIT THESE) --")
    for n, r in sorted(names.items()):
        m = r["match"]
        if m["method"] in ("fuzzy", "fuzzy_family"):
            tgt = r["sku"] or f"family{r['sku_candidates']}"
            print(f"   {m['score']:.2f}  {n:<42} -> {tgt}")
    print(f"\n-- remaining ghosts (no defensible catalog referent) --")
    for n, r in sorted(ghosts.items()):
        print(f"   {r['match']['score']:.2f}  {n:<42} (nearest {r['match'].get('nearest')})")
    print(f"\n   -> {out}")


if __name__ == "__main__":
    main()
