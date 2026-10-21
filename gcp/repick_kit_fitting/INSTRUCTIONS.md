# Independent re-pick

For each point, open `<point_id>.jpg`. The left panel is the 1926 scan with full-resolution pixel coordinates on its grid; the right panel is today's OSM with the target feature marked by a cross. Find the same feature on the 1926 map and write its pixel x/y (read from the grid labels, or measure in any image viewer on `data/raw/57_D11_Mysore_1926.jpg`) into the template. Record how many seconds each point took. Do not look at `gcp/*_points.csv` or any results before finishing.

Then run: `python -m mapalign.robustness picker --repick <filled template>`
