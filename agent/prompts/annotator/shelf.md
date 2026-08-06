This checkpoint faces a SHELF - a fixture holding retail product, including refrigerated cases.

Produce these fields:

"semantic_summary"
    One to three sentences, in natural language, describing this spot as a store guide would: what this shelf holds, plus anything navigationally useful you can SEE from here (an aisle opening, a checkout, a corner). Follow rule 4 - do not describe how this shelf relates to other shelves.

"shelf_type"
    What this shelf holds overall: ONE value from this list, or TWO if it genuinely holds two kinds - a shelf split between chips and instant noodles, say. Put the dominant one first. Never more than two; if it holds more than two, give the two largest.
{categories}
    Judge this from the products themselves. Do not use signs.

"sign_text"
    If a category or aisle sign is legible anywhere in ANY view - including at the edges of a frame - copy its text verbatim. Otherwise null. This is a plain observation, nothing more: do NOT let it influence "shelf_type", and do NOT assume the sign refers to the shelf you are facing. In this store a visible sign usually belongs to a different aisle.

"items"
    The products on this shelf, taken from ALL the views you were given. They are one shelf at different camera heights, and each reaches rows the others cut off - the CROUCHED view in particular shows face-on the bottom rows that the STANDING view clips or renders oblique. Enumerate across every view; a product is no less real for appearing in only one of them.

    The views OVERLAP, so list each distinct product EXACTLY ONCE however many views show it. The same box seen in two views is one entry, not two.

    Only THIS shelf counts. If a neighbouring shelf or aisle intrudes at the edge of a frame, its products belong to a different checkpoint - leave them out.

    For each distinct product:
        - "name": REQUIRED, and it must be the text PRINTED ON THE ITEM'S LABEL - the brand and product name as actually written on the packaging ("Lucky Me! Pancit Canton", "Coca-Cola", "Tostitos"). Read it off the label. Do NOT name an item from its shape, its colour, or your sense of what it probably is - this is the only field anyone searches on later, so it has to be the real printed name.
          ONLY include an item whose label you can read. If a label is turned away, too small, or too blurred to make out, LEAVE THAT ITEM OUT. Do not add it under a description like "green chip bag", "canned goods", or "bottled water" - an item you can only describe, not read, is not a usable entry, and omitting it is the correct choice, not a failure.
          ONE entry is ONE product. Never merge several items into a single row and never generalise: "assorted soda bottles", "various cans", "bottled water (assorted brands)", "mixed chips" are all forbidden. If five distinct bottles are legible, that is five entries; the bottles you cannot read are simply left out.
        - "variant": size, flavour, or variant, ONLY if legible. Otherwise null.
        - "price": the number printed on THIS product's shelf price tag, copied exactly as shown ("29.95"). The tags sit in a row along the shelf edge, each under the product it belongs to. Use null if you cannot read the digits, and ALSO use null if you cannot tell which tag is this product's - a price paired with the wrong product is worse than no price.
        - "appearance": a short visual description so someone can spot it again on the shelf ("red box, rooster logo, white lettering"). Describe what it LOOKS like, not what you infer it is - this is used to confirm the right item on arrival.
        - "category": REQUIRED. The ONE value from the list above that this item belongs to. Judge the ITEM ITSELF, not the shelf around it - on a shelf holding two kinds of product, the item beside this one may well belong to the other category.

    A short list of products you could genuinely read beats a long list padded with descriptions. If you cannot read any label, or the shelf is empty, return an empty list - that is a valid, correct answer, not a failure.

Output strict JSON only:

{{"semantic_summary": "...", "shelf_type": ["..."], "sign_text": null, "items": [{{"name": "...", "variant": null, "price": null, "appearance": "...", "category": "..."}}]}}
