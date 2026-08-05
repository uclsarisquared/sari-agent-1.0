This checkpoint is NOT a product shelf. It is a pathway, junction, doorway, or a plain surface with no retail product on it.

Produce these fields:

"semantic_summary"
    One to three sentences, in natural language, describing this spot: what kind of place it is, and anything navigationally useful you can SEE from here (an aisle opening, a doorway, a checkout, signage). Follow rule 4 - do not describe how shelves relate to one another.

"sign_text"
    If a sign is legible anywhere in ANY view, copy its text verbatim. Otherwise null.

Do NOT list products. There are none here to identify, and inventing them would corrupt the product index. "Nothing to identify here" is the expected, correct outcome for this kind of checkpoint - not a failure.

Output strict JSON only:

{"semantic_summary": "...", "sign_text": null}
