# Ideation — Simplifying HDR-Bracket Grouping into a Principled Formulation

_Date: 2026-06-22 · Mode: repo-grounded · Focus: replace the 6-pass refinement stack with a simpler, mathematically defensible model without regressing fixable-only 1302/1302._

## Grounding Context

**Current architecture.** Coarse wavelet+gradient descriptor clusterer (`FusionClusterer`) + six hand-tuned refinement passes (FIX2 orphan reattach, FIX2c clipped reattach, FIX4 cluster merge, FIX5 hi-res split, FIX6 capture-run split, FIX7 scene-change split), each with magic thresholds. `autohdr/refinement/`.

**Decisive prior learning — "THE WALL."** A single all-pairs affinity + connected-components + global threshold formulation was already tested and plateaus at ~0.94–0.95. Reason: HDR brackets are sparse dark→mid→bright *chains* whose ends don't correlate directly, while weak between-group bridges must be rejected — chain-honoring and bridge-rejecting are in direct tension. Leiden/spectral/HDBSCAN all underperformed connected-components; bridge-rejecting heuristics broke legitimate chains. **A naive unified affinity formula regresses.**

**Decisive external finding.** The canonical literature answer (Ward MTB 2003; Google temporal-event-clustering patent US7640218; stereo illumination-invariance work) is **capture-order single-linkage**: graph = the 1-D capture-order chain (each frame linked only to temporal neighbors), edge weight = exposure-invariant dissimilarity, cut at change-points. This avoids THE WALL because adjacency carries the chain (no dark↔bright direct link needed) and non-adjacent frames are never compared (no spurious bridges). EXIF (`ExposureMode=2`, `SequenceImageNumber`, timestamp gap) is the authoritative prior *when present*; pixels are the universal fallback. Clipped frames → Ward exclusion bitmap / Debevec hat-weight (drop untrustworthy pixels), then capture-order fallback when too few valid pixels remain.

**Hard constraints (from memory + user).** Fixable-only must hold ≥0.99 (currently 1302/1302). Threshold must stay per-run adaptive (plateau selector) — a global constant breaks cross-size (0.44@500 → 0.62@5041). Determinism required (sorted filenames, full SVD). Known-UNSAFE dead-ends: 2-frame capture-adjacency merge, camera-prefix split, exposure-period split of contiguous multi-brackets, triangle de-bridge, Leiden, tiled-min ZNCC, census/log-gradient descriptors.

## Topic Axes

1. **Similarity primitive** — what exposure-invariant pixel signal scores a pair.
2. **Graph structure** — which pairs get compared (all-pairs vs capture-order chain).
3. **Clipped / information-floor handling** — frames with too little signal to score.
4. **Decision/threshold rule** — global vs adaptive vs change-point.
5. **Metadata exploitation** — EXIF/capture-order as prior vs ignored.
6. **Refactor strategy** — how to de-hardcode without regressing.

---

## Survivors (ranked)

### 1. Capture-order-structured affinity (the principled core) — axis: graph structure
Reframe the whole pipeline as **one** declarative model: graph = capture-order chain, edge weight = masked edge-ZNCC with a Ward/Debevec exclusion mask, components cut where adjacent dissimilarity exceeds an adaptive cut. This is the literature's canonical formulation and the only "single affinity" framing that provably *avoids* THE WALL (it never needs a dark↔bright direct link, never forms all-pairs bridges).
- **Basis** — `external:` Ward 2003 MTB + US7640218 temporal clustering + stereo illumination-invariance; `direct:` matches the repo's validated gradient-ZNCC signal.
- **Why it matters** — collapses six imperative passes into graph-construction + weighting + adaptive cut, each with a physical meaning. Directly answers the "too hardcoded" worry.
- **Risk / open question** — must reproduce the work each pass does today (FIX5 lookalike-room split, FIX6 fused-scene split) as edge decisions; capture-order must only *structure* candidate edges, never decide merges alone (that's the KNOWN-UNSAFE dead-end). Prove byte-identical 1302/1302 before deleting any pass.

### 2. Probe for EXIF / capture metadata — axis: metadata exploitation
The pipeline is pixel-only and ignores metadata entirely. The literature says EXIF `ExposureMode=2` + `SequenceImageNumber` is *authoritative* when present, and timestamp gaps strongly delimit brackets. **Cheap to verify, potentially decisive.**
- **Basis** — `external:` HDRSoft/Photomatix, Lightroom AEB practice; `reasoned:` if the source JPEGs carry AEB EXIF, grouping becomes near-trivial for the metadata-present subset.
- **Why it matters** — could replace heuristics outright for a large fraction of images; turns the hard pixel problem into a fallback for the metadata-absent tail.
- **Next step** — `exiftool` probe a sample of `data/large/images` for `ExposureMode`, `ExposureBracketValue`, `SequenceImageNumber`, `DateTimeOriginal`/`SubSecTime`. **Do this first — it may reshape everything.**

### 3. Safe de-hardcoding refactor: declarative constraints, prove byte-identical, then simplify — axis: refactor strategy
The lowest-risk path the learnings explicitly recommend: refactor the six imperative passes into one declarative affinity-with-constraints model that produces the **same edge decisions**, prove identical 1302/1302 + 30k output, *then* simplify thresholds. Cleans the architecture without betting the benchmark.
- **Basis** — `direct:` memory note "unify the bookkeeping… prove byte-identical before changing any threshold."
- **Why it matters** — gets the user the maintainability win they're asking for with zero accuracy risk; every past "cleaner formula that changed decisions" regressed full_subset.

### 4. Adaptive valley threshold from the bimodal score distribution — axis: decision rule
Replace remaining magic thresholds (0.35, 0.45, etc.) with the natural valley of the adjacent-pair ZNCC histogram (within-bracket ≈1.0, cross-bracket ≈0.5–0.8). Generalizes the existing plateau selector to the edge-cut decision.
- **Basis** — `external:` literature notes threshold has no universal value; valley-finding on bimodal distributions is standard; `direct:` repo already uses a per-run plateau selector.
- **Why it matters** — removes the single largest source of heuristic fragility and the cross-size brittleness; principled and per-run.

### 5. One principled clipped-frame rule (Ward exclusion + capture-order fallback) — axis: clipped handling
Unify FIX2/FIX2c/reattachment into a single rule: compute similarity only on valid (non-clipped) pixels; if valid-pixel count is below a floor, decide membership by capture-order neighbor, not by a noisy score. Formalizes what masked-ZNCC already gestures at.
- **Basis** — `external:` Ward exclusion bitmap, Debevec hat weight, Mertens well-exposedness; `direct:` memory note on pure-clip frames at the information floor matching wrong groups.
- **Why it matters** — turns two heuristic passes into one defensible rule and fixes the unresolved clipped strands (10125/10129/10463).

### 6. Durable repo learnings doc (capture THE WALL + dead-ends) — axis: refactor strategy
The hard-won dead-ends and THE WALL live only in personal auto-memory + an iteration log. Write `docs/DESIGN.md` (or `docs/solutions/`) recording them so the next refactor doesn't re-derive them.
- **Basis** — `direct:` learnings agent's meta-note (no `docs/solutions/` exists).
- **Why it matters** — compounding; protects the benchmark from well-meaning future simplifications.

---

## Rejected (with reasons)

- **Naive all-pairs affinity + CC + single global threshold** — REJECT: proven to hit THE WALL (~0.94). This was my original instinct; the research killed it.
- **Learned pair-classifier / embedding boundary** — REJECT: abandoned; bakes a density-tied boundary, collapses cross-size (train 500 → test 5k → all singletons 0.0865).
- **Leiden / spectral / HDBSCAN clustering** — REJECT: measured worse than connected-components (0.9187 vs 0.9526); overkill; discards the temporal-order prior that makes the chain formulation work.
- **Census-on-gradients / log-gradient descriptor** — REJECT: census/log-gradient descriptors over-merged in past trials; tiled-min ZNCC fires on exposure clipping.
- **Pure exposure-period (EV-cycle) splitting** — REJECT as a *decision* rule: splitting contiguous multi-brackets by EV cycle is a KNOWN-UNSAFE dead-end; usable only as a weak prior, not a cut.

## EXIF probe results (2026-06-22, ran #2)

Probed `data/large/images` with Pillow (no `exiftool` available).

- **Coverage is all-or-nothing per group:** of 300 sampled groups, **31% have EXIF on every frame, 69% on none, 0% mixed.** Frame-level presence 29%. EXIF presence is a clean per-group property.
- **Where present, it's gold:** clean exposure ramps (`ExposureTime` 0.0015→0.016, etc.), AEB `ExposureMode=2`, `ExposureBiasValue` textbook (-1.7/0.3/2.3), `DateTimeOriginal` tightly clustered (same/adjacent second). **No `SubSecTimeOriginal`** (0/46) — timestamps collide within a second, so timestamp alone can't order frames inside a bracket.
- **It reaches cases pixels physically can't.** Of the 7 information-limited unfixable groups (pixel-dead black/white frames), **3 carry full EXIF — 56453, 56460, 22901** — so timestamp+exposure grouping could *solve* them despite dead pixels. The other 4 (35098, 10370, 82871, 70877) have no EXIF and stay unrecoverable. Several multi-scene mislabels also carry EXIF (40667, 40599, 40615, 38084, 73234), where timestamps give independent evidence of distinct captures.

**Verdict.** EXIF cannot be the backbone (69% stripped) — pixel + capture-order must remain the foundation (reinforces Survivor #1). EXIF's real value is as an **opportunistic fast-path + information-floor tie-breaker**: it can group the 31% EXIF-present groups deterministically and rescue ~3 of the 7 pixel-dead groups.

> ⚠️ **Generalization caveat — verify before investing.** 69% of this dataset is *already* EXIF-stripped, which strongly suggests the grading/submission images may be stripped too. If so, an EXIF fast-path is a local-benchmark crutch that won't transfer to the held-out test (and the real deliverable is a client-side browser grouper). **Confirm the evaluation images retain EXIF before building on it.** Treat EXIF as a gracefully-degrading accelerator, not a dependency.

## Pixel-only residual research (2026-06-22) — resolving the 69% without EXIF

**Audit result: the live grouper is 100% grayscale.** Gradient descriptor, wavelet embedding, masked correlation, brightness — all grayscale. The color cache `img128c.npz` is decoded but used *only* for report thumbnails (`build_disagreement_report.py:45`); the grouper never consumes it. (The previously-removed color signal was a *learned* CNN `embed_color.onnx` — not hand-crafted color.)

**New-signal-class ranking (training-free, for the residual):**

1. **Normalized rg-chromaticity histogram — exposure-invariant COLOR. Highest return.**
   - `r=R/(R+G+B), g=G/(R+G+B)` cancels the exposure scalar (uniform brightness change divides out) → invariant to exposure, shading, approx white-balance. Textbook illumination-invariant color (Drew ICCV'98; rg-chromaticity).
   - Targets the **dominant residual: wrong-merges of look-alike-different scenes** (12226 house/field, 10464/10886 similar rooms) — different rooms differ in paint/wood/upholstery color even when edge structure matches.
   - ~0 compute, scales to 100k, **browser-portable (pure math, no model)** — fits the real deliverable.
   - Guards: compute on well-exposed pixels only (chromaticity breaks under clipping); use as a *separation/veto* signal (block merges of color-divergent clusters), never as a primary linker (don't break HDR chains); must not reintroduce KNOWN-UNSAFE over-splits.

2. **Local-feature geometric verification** (RootSIFT / SuperPoint+LightGlue inlier count) — decisive same-scene vs different-scene test for *textured* pairs (inliers: same-scene 30–200+, different 0–5). But needs a model/GPU, not browser-clean; degenerates on textureless scenes. Second-stage verifier on ambiguous candidate pairs only.

3. **DINOv2 zero-shot embedding** — semantic tiebreaker for low-texture scenes (empty rooms, fields) where both color and keypoints are weak. Heavy (GPU index ~33 min/100k); optional.

**THE WALL (chain-vs-bridge), principled resolution:** compare only **same-exposure-level** frames directly (median brightness is the exposure proxy; only build affinity edges between similar-brightness frames), and chain across exposures via capture order. Prevents the dark-A↔dark-B cross-bracket false merge that flat ZNCC + global threshold cannot avoid. Closest published analog: US7640218 temporal event clustering with a variance cost — HDR version swaps timestamp-variance for exposure-level-variance + monotonicity. Genuinely novel as a graph formulation.

**Clipped/dead frames:** PRNU/sensor fingerprint is NOT viable (arxiv 2407.00543: degrades to noise floor under clipping; exclusion fails to even activate). Only recoverable signals are the few non-clipped pixels (masked features/chromaticity) + capture order — confirming the current approach is already correct.

### Added survivors
- **S7. Exposure-invariant color veto (rg-chromaticity)** — top pick for the residual; training-free, browser-portable, data already decoded.
- **S8. Same-exposure-level edge graph** — the principled WALL-dodge; pairs with Survivor #1.

## Recommended sequence
1. **Probe EXIF (#2)** — one `exiftool` sample; may reshape the whole plan.
2. **Capture-order-structured affinity prototype (#1)** — built behind the existing harness, measured against fixable-only 1302/1302 + 30k disagreements, *additive* to current pipeline until it matches/beats.
3. If it holds → **safe refactor (#3)** + **adaptive threshold (#4)** + **clipped rule (#5)**, deleting passes only after byte-identical proof.
4. **Record learnings (#6).**
