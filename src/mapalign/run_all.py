"""Reproduce the whole experiment in order.

Usage:
    python -m mapalign.run_all            # fetch (if missing) -> render -> SIFT -> LoFTR -> Method C -> rasters -> evaluate

SIFT is expected to "fail" (exit code 2 = ran, no valid transform); that outcome is a result and does not
stop the run. Any other non-zero exit stops it.
"""

from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger("mapalign.run_all")

STEPS = [
    (["mapalign.fetch", "all"], (0,)),
    (["mapalign.reference"], (0,)),
    (["mapalign.pipeline", "sift"], (0, 2)),
    (["mapalign.pipeline", "loftr"], (0, 2)),
    (["mapalign.georeference", "datum"], (0,)),
    (["mapalign.georeference", "manual"], (0,)),
    (["mapalign.georeference", "raster", "loftr"], (0, 2)),
    (["mapalign.georeference", "raster", "manual"], (0, 2)),
    (["mapalign.georeference", "verify", "loftr"], (0, 2)),
    (["mapalign.georeference", "verify", "manual"], (0, 2)),
    (["mapalign.evaluate"], (0,)),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for args, ok_codes in STEPS:
        log.info("==> python -m %s", " ".join(args))
        code = subprocess.call([sys.executable, "-m", *args])
        if code not in ok_codes:
            log.error("step failed with exit code %d: %s", code, " ".join(args))
            return code
        if code == 2:
            log.warning("%s: ran without a valid transform (recorded as a result)", " ".join(args))
    log.info("All steps finished. See results/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
