# Ideation — Robust blob-constellation matching for clipped frames

_Date: 2026-06-22 · Mode: repo-grounded · Focus: tell same-scene from different-scene near-black/near-white frames by matching their light-blob / dark-blob constellations — "see how smarter people solve this."_

## Grounding

**The problem.** A near-black frame is a few BRIGHT blobs (lights/windows) on black; a near-white frame is a few DARK blobs (objects) on white. Scene identity = where the blobs are. We must decide same-scene (blobs coincide) vs different-scene (blobs displaced) for clipped frames where masked edge-ZNCC is blind (no co-valid overlap). This is the **Cluster-B information-floor** that blocked 5 cases (1038/10280, 10464/10613, 10886/10593, 14288/14037, 14279/14983).

**What we tried and why it failed (measured):**
- Coarse-grid bright-pixel IoU (24×24, top-6%) → 0.4 for everything (whole-frame dilution).
- Greedy nearest-neighbour "chamfer" between blob centroids → **distributions overlap** (foreign 0.018–0.097 vs same-scene 0.008–0.283). Root cause: chamfer **averages in the misses**, and blob COUNT drifts with exposure.

**Hard constraints.** Training-free, deterministic, cheap, browser-portable (pure pixel/geometry ops, no ML). Must not regress the large-set fixable 1302/1302. Acts only on heavily-clipped frames.

## Topic axes
1. **Blob detection** — getting an exposure-stable set of blob positions.
2. **The matcher** — aligning two small point sets robustly to missing/spurious points.
3. **The discriminator** — the scalar that cleanly separates same vs different scene.
4. **Integration** — where this plugs into AnchorSplitter/AnchorReattachment.

---

## Survivors (ranked)

### 1. RANSAC inlier-count discriminator (replace chamfer) — axis: discriminator. **Highest leverage, build first.**
Hypothesize a transform (translation, or similarity for tiny scale/shift) from a minimal blob correspondence, apply it, and **count inlier blobs** (those landing within tolerance of a counterpart). The score is the inlier count, not an average distance.
- **Basis** — `external:` astrometry.net verification; SupeRANSAC; fingerprint LSA-R; particle-tracking bipartite matching all use inlier count.
- **Why it matters** — immune to missing/spurious blobs *by construction* (they're non-inliers, no penalty). A wrong scene can't accumulate many consistent inliers by chance at small N, so same/different separate **sharply** — the exact property chamfer lacked. This alone may crack the 5 cases.
- **Cheap:** for our case (same camera ⇒ ~no rotation, near-pure translation) a **2-point RANSAC** (or even all-pairs translation voting) is ~50 lines.

### 2. MSER blob detection (replace fixed-threshold top-K) — axis: detection.
Maximally Stable Extremal Regions: regions defined by pixel *ordering* → invariant to any monotonic intensity (exposure) change; the stability criterion keeps only blobs whose area persists across a threshold band → **stabilizes blob count** across exposures. Run MSER+ for dark blobs (white frames) and MSER- (inverted) for bright blobs (black frames).
- **Basis** — `external:` Matas et al. 2002; OpenCV ships `cv2.MSER`.
- **Why it matters** — fixes the *upstream* cause of chamfer's failure (count drift). Feeds clean, repeatable blobs to the matcher. (Persistence/topological thresholding is the same idea if MSER is overkill.)

### 3. Quad-hash / triangle-hash constellation matching (astrometry) — axis: matcher. **The gold standard if 2-point RANSAC isn't enough.**
Build a transform-invariant hash from point *quads* (astrometry.net) or sorted-side-length *triangles* (STD, 2022): place the two farthest points of a quad at (0,0)/(1,1), hash the other two → invariant to translation/rotation/scale; matching hashes vote for a transform, verified by inlier count.
- **Basis** — `external:` Lang et al. 2010 (astrometry.net); STD arxiv 2209.12435.
- **Why it matters** — the canonical solved instance of "match bright points on black, robust to missing/spurious." ~100 lines, no catalog needed (build a tiny in-memory index per cluster). Overkill if pure translation holds, but the principled fallback.

### 4. Shape-Context descriptors + Hungarian assignment — axis: matcher.
Describe each blob by a log-polar histogram of vectors to all other blobs (translation-invariant; scale-invariant via mean-distance normalization). Match via optimal bipartite assignment (Hungarian) with dummy nodes for outliers; assignment cost = score.
- **Basis** — `external:` Belongie & Malik 2000.
- **Why it matters** — a strong alternative discriminator that handles unequal cardinality cleanly; more robust than greedy NN. Slightly heavier than RANSAC; keep as a backup.

### 5. Modified Hausdorff (mean, not max) as a sparse-N tiebreaker — axis: discriminator.
After best alignment, score by **mean** directed nearest-neighbour distance (Dubuisson & Jain 1994 — best Hausdorff variant), used only when N < 4 (quad/RANSAC degenerates).
- **Basis** — `external:` Dubuisson & Jain 1994.
- **Why it matters** — cheap continuous fallback for 1–3 blob frames where geometric verification has too few points.

---

## Rejected (with reasons)
- **Greedy nearest-neighbour chamfer** — what we built; averages in the misses, fooled by count drift → measured overlap.
- **Coarse-grid bright-pixel IoU** — dilutes the signal over the whole frame → ~0.4 for everything.
- **Plain ICP (no outlier model)** — residual-after-fit distributions overlap for outlier-heavy small sets; CPD/trimmed-ICP better but heavier than RANSAC inlier-count for our N.
- **Raw (max) Hausdorff** — one outlier blob blows up the score.
- **Any learned/semantic detector (SuperPoint, DINOv2)** — violates training-free + browser-portable.

## Recommended pipeline (a strong CV engineer's default)
**MSER (both polarities) → 2-point/quad RANSAC → inlier-count threshold**, with modified-Hausdorff tiebreaker for N<4. Training-free, deterministic, browser-portable (~200 lines; OpenCV has MSER + RANSAC). Plug the inlier-count score into AnchorSplitter (split when a clipped frame's blobs don't inlier-match the scene) and AnchorReattachment (link when they do).

## Recommended first step
**Spike the RANSAC inlier-count discriminator** on the 5 failing cases vs the same-scene safety set — even with our existing (imperfect) blob detector. It's the cheapest, highest-leverage idea and directly targets *why* chamfer overlapped. If inlier-count separates cleanly, swap in MSER detection and integrate. If it's marginal, escalate to quad-hash.
