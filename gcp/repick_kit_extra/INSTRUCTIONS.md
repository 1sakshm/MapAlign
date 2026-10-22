# Supplementary evaluation points

57 OSM candidates in the 15 protocol cells that have no evaluation point (cells: r0c0, r0c4, r1c0, r1c1, r1c2, r2c3, r2c4, r3c1, r3c2, r3c3, r4c0, r4c1, r4c2, r4c3, r4c4). Each is at least 1 km from every existing point. Many will not exist on the 1926 sheet. Skip unless you are sure: accept a point only if the same road, railway or stream exists on the 1926 map and you can see where they meet. A nearby look-alike is worse than a skip. Pick with the timed tool:

`python -m mapalign.picktool gcp/repick_kit_extra/repick_extra_template.csv --picker "Your Name"`

Or pick in the browser (all three kits in one page, served on localhost only):
`python -m mapalign.pickweb --picker "Your Name"`

Then score the frozen methods on them: `python -m mapalign.robustness extra_points --points gcp/repick_kit_extra/repick_extra_template.csv`
