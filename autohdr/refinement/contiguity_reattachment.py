"""LINK — rejoin a clipped singleton that capture-continues a cluster AND matches it.

Root cause this pass targets
----------------------------
A heavily clipped frame at the END of a bracket's exposure ladder (a saturated
near-white frame, mean > 225, or a crushed near-black frame, mean < 30) can carry
too little anchor signal for the earlier reattach passes to clear their floor, so
it strands as its own singleton even though it is obviously the bracket's last
frame. Example: GT 11992's QUC09117 (B254) stranding away from QUC09113-09116.

Signal (training-free, deterministic, browser-portable)
-------------------------------------------------------
Two independent cues must AGREE before we merge (merging is the dangerous
direction — a wrong merge fuses two scenes):

  1. CAPTURE CONTIGUITY — the singleton's capture sequence number directly extends
     exactly ONE cluster's run (same camera prefix, number == that cluster's
     min-1 or max+1). A bracket is one contiguous run, so its trailing clipped
     frame continues the run; a genuinely different bracket has its own run.
  2. STRUCTURAL AGREEMENT — the singleton's surviving extreme structure (bright
     light spots for near-black, dark objects for near-white) covers the cluster's
     well-exposed template at or above ``_COV_FLOOR``. This rejects a different
     scene that merely happens to be numbered consecutively (e.g. two DSC0596x
     brackets from one camera): its structure does not match, so coverage is low.

Contiguity alone would over-merge consecutive-but-different brackets; coverage
alone is too weak at the clipped tail. Requiring BOTH, with a unique contiguous
neighbour, is safe — it holds the data/large 1302/1302 gate.

Reads only pixels + filenames (capture order, a trusted signal here). No labels,
no ML, no randomness.
"""
from __future__ import annotations

import re

import numpy as np

from ..features import extreme_anchor as _ea
from .context import RefinementContext

_DARK_MAX = 30.0     # near-black: bright-light-source regime
_WHITE_MIN = 225.0   # near-white: dark-object regime
_COV_FLOOR = 0.5     # structural coverage the singleton must reach to rejoin
_COV_MARGIN = 0.2    # ...and must beat the next-best contiguous neighbour by this
#                      (a continuous camera roll makes the trailing frame contiguous
#                      with BOTH its own bracket and the NEXT one; coverage breaks
#                      the tie — it matches its own scene far better)


def _seq_key(filename: str):
    stem = re.sub(r"^g\d+_", "", filename.rsplit(".", 1)[0])
    m = re.search(r"(\d+)$", stem)
    return (stem[: m.start()], int(m.group(1))) if m else (filename, -1)


class ContiguityReattachment:
    """Merge a clipped singleton into the one cluster it capture-continues and
    structurally matches. ``apply(labels, ctx) -> labels``."""

    def apply(self, labels: np.ndarray, ctx: RefinementContext) -> np.ndarray:
        gray = getattr(ctx, "gray", None)
        if gray is None or ctx.filenames is None:
            return labels
        labels = labels.copy()
        B = ctx.brightness

        clusters: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)

        # precompute each cluster's (prefix -> set of sequence numbers) capture runs
        runs: dict[int, dict[str, set[int]]] = {}
        for c, members in clusters.items():
            pf_nums: dict[str, set[int]] = {}
            for k in members:
                pf, nf = _seq_key(ctx.filenames[k])
                if nf >= 0:
                    pf_nums.setdefault(pf, set()).add(nf)
            runs[c] = pf_nums

        for c, members in list(clusters.items()):
            if len(members) != 1:
                continue  # only true singletons are candidates
            k = members[0]
            polarity = _ea.clip_polarity(gray[k])
            if polarity is None:
                continue  # not clipped — leave to the edge-based passes
            pf, nf = _seq_key(ctx.filenames[k])
            if nf < 0:
                continue  # unparseable sequence — cannot assert contiguity

            # candidate clusters whose run this frame directly extends. A continuous
            # camera roll makes the trailing clipped frame contiguous with BOTH its
            # own bracket and the next one, so there can be several — score each by
            # structural coverage and let the best win by a margin (its own scene's
            # lights/objects match; a different scene's do not).
            scored: list[tuple[float, int]] = []
            for oc, pf_nums in runs.items():
                if oc == c:
                    continue
                nums = pf_nums.get(pf)
                if not nums or not (nf - 1 in nums or nf + 1 in nums):
                    continue
                well_tiles = [gray[j] for j in clusters[oc] if _ea.is_well_exposed(gray[j])]
                tmpl = _ea.build_template(well_tiles) if well_tiles else None
                if tmpl is None:
                    continue
                scored.append((_ea.coverage_score(gray[k], tmpl, polarity=polarity), oc))
            if not scored:
                continue
            scored.sort(reverse=True)
            best, host = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if best >= _COV_FLOOR and best - runner >= _COV_MARGIN:
                labels[k] = host
                clusters[host].append(k)
                runs[host].setdefault(pf, set()).add(nf)
                clusters[c] = []
        return labels
