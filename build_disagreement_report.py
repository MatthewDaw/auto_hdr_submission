"""
Self-contained HTML report of every place our model DISAGREES with the ground
truth, across one or more datasets. For each GT group we don't reproduce exactly,
we show its frames with TWO color bars per image:

    row 1 = the GT cluster   (one color per ground-truth group)
    row 2 = OUR cluster      (one color per predicted cluster)

So a split reads as "top row one color, bottom row many colors"; a merge reads as
"our color repeating across two GT blocks". Within each GT block, frames are
sorted by our cluster, then by exposure (darkest -> lightest). Blocks are sorted
by GT group id.

Run dump_autohdr_labels.py <data_dir> first for each dataset (writes pred_labels.json).
Usage: build_disagreement_report.py <out.html> <data_dir>[:Label] [<data_dir>[:Label] ...]
"""
import sys, csv, json, base64
import sys as _sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, cv2

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from autohdr.features.masked_correlation import MaskedCorrelation
from anomaly_section import build_section as build_anomaly_section

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("disagreement_report.html")
SPECS = sys.argv[2:] or ["data/large:Large set (5041 images)", "sample:Sample set (500 images)"]
THUMB = 140

PAL = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22",
       "#e84393", "#00b894", "#0984e3", "#fdcb6e", "#6c5ce7", "#d63031", "#00cec9",
       "#a29bfe", "#fab1a0", "#55efc4", "#ffeaa7", "#fd79a8", "#74b9ff"]


def color(idx):  # stable color per integer id
    return PAL[idx % len(PAL)]


def load_dataset(data_dir: Path):
    gt = {}
    for row in csv.DictReader(open(data_dir / "public_manifest.csv", encoding="utf-8")):
        gt[row["filename"]] = row["group_id"]
    pred = json.load(open(data_dir / "pred_labels.json"))
    col = np.load(data_dir / "img128c.npz", allow_pickle=True)
    imgs = col["imgs"]; idxof = {f: i for i, f in enumerate(col["files"])}
    return gt, pred, imgs, idxof, data_dir


def thumb(imgs, idxof, f):
    im = cv2.resize(imgs[idxof[f]], (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def section(label, data_dir: Path):
    gt, pred, imgs, idxof, ddir = load_dataset(data_dir)
    gt_groups = defaultdict(list)
    for f, g in gt.items():
        gt_groups[g].append(f)
    our_sets = defaultdict(set)
    for f, c in pred.items():
        our_sets[c].add(f)
    our_framesets = {frozenset(s) for s in our_sets.values()}  # precompute once

    # stable integer ids for coloring
    gt_cid = {g: i for i, g in enumerate(sorted(gt_groups))}
    our_cids = sorted({pred[f] for f in gt})
    our_cid = {c: i for i, c in enumerate(our_cids)}

    # a GT group disagrees when its frame set is not reproduced exactly by us
    disagreements = [(g, frames) for g, frames in gt_groups.items()
                     if frozenset(frames) not in our_framesets]

    # For each disagreeing GT group, the block shows every frame in any of our
    # clusters that touch it — so a merge surfaces the FOREIGN frames we pulled in
    # (rendered with their own GT color on top), not just the group's own frames.
    block_frames = {}
    for g, frames in disagreements:
        clusters_here = {pred[f] for f in frames}
        block_frames[g] = sorted(
            {o for c in clusters_here for o in our_sets[c]} | set(frames))

    # brightness only for the frames we will actually render
    needed = {f for fs in block_frames.values() for f in fs}
    raw = np.load(ddir / "raw256.npz", allow_pickle=True)
    ridx = {f: i for i, f in enumerate(raw["files"])}
    rimgs = raw["imgs"]
    bright = {f: float(rimgs[ridx[f]].mean()) for f in needed if f in ridx}

    # owner of each of our clusters = the GT group holding the most of its frames
    owner = {}
    for c, s in our_sets.items():
        cnt = Counter(gt[o] for o in s if o in gt)
        owner[c] = cnt.most_common(1)[0][0] if cnt else None

    def rep_zncc(reps):
        """Pairwise masked-ZNCC (+ co-valid overlap) between representative frames."""
        labels = sorted(reps)
        local = [reps[l] for l in labels]
        mc = MaskedCorrelation(np.stack([rimgs[ridx[f]] for f in local]))
        out = []
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                z, o = mc.score(a, b)
                out.append((labels[a], labels[b], z, o))
        return out

    def well_rep(frames):
        return min((f for f in frames if f in ridx),
                   key=lambda f: abs(bright[f] - 120), default=None)

    blocks = []
    tally = {"gterr": 0, "review": 0}
    for g, frames in sorted(disagreements, key=lambda kv: kv[0]):
        ours_here = {pred[f] for f in frames}
        own_clusters = sorted(c for c in ours_here if owner.get(c) == g)
        leak_clusters = sorted(c for c in ours_here if owner.get(c) != g)

        verdict, dissim_html = "frame-set mismatch", ""
        # assessment = (css_class, headline) — our read on GT-error vs model-failure
        assess = ("review", "MODEL-FAILURE CANDIDATE — review")
        if len(own_clusters) >= 2 and not leak_clusters:
            # genuine angle-split: this group's frames cleanly occupy >=2 of OUR
            # clusters, each owned by this group -> the GT lumped distinct angles.
            verdict = (f"we SPLIT it into {len(own_clusters)} clusters "
                       f"(scene not perfectly still — camera moved or in-scene motion)")
            reps = {c: well_rep([f for f in frames if pred[f] == c]) for c in own_clusters}
            reps = {c: f for c, f in reps.items() if f}
            if len(reps) >= 2:
                dis = rep_zncc(reps)
                worst = max(z for _, _, z, _ in dis)
                min_overlap = min(o for _, _, _, o in dis)
                if min_overlap < 8000:
                    tag = ("a clipped/low-signal frame — too few valid pixels to "
                           "confirm; likely a reattachment miss, not a real split")
                    assess = ("review", "MODEL-FAILURE CANDIDATE — clipped frame split off, review")
                elif worst < 0.45:
                    tag = ("low ⇒ scene genuinely differs (camera angle or in-scene "
                           "motion) — not HDR-mergeable, so GT wrongly lumps them")
                    assess = ("gterr", f"LIKELY GT ERROR — different scenes/angles/motion (max ZNCC {worst:.2f})")
                else:
                    tag = "borderline similarity"
                    assess = ("review", f"POSSIBLE OVER-SPLIT — clusters look similar (ZNCC {worst:.2f}), review")
                pairs = " · ".join(f"{a}↔{b} <b>{z:.2f}</b>" for a, b, z, _ in dis)
                dissim_html = (f'<div class=dis>well-exposed edge-ZNCC between our '
                               f'clusters: {pairs} &nbsp;⇒&nbsp; {tag}</div>')
        elif leak_clusters:
            # our error: some of this group's frames leaked into a cluster owned by
            # a DIFFERENT GT group (a wrong merge across genuinely different scenes).
            victims = sorted({owner[c] for c in leak_clusters if owner.get(c)})
            n_leaked = sum(1 for f in frames if pred[f] in leak_clusters)
            verdict = (f"OUR ERROR — {n_leaked} frame(s) wrongly merged into GT group "
                       f"{', '.join(victims)}'s cluster")
            assess = ("review", f"MODEL-FAILURE CANDIDATE — {n_leaked} frame(s) merged "
                      f"into GT {', '.join(victims)} (near-duplicate / wrong merge)")
            home = max(own_clusters, key=lambda c: sum(pred[f] == c for f in frames),
                       default=None)
            reps = {}
            if home is not None:
                reps[f"GT{g}·{home}"] = well_rep([f for f in frames if pred[f] == home])
            for c in leak_clusters:
                reps[f"GT{owner[c]}·{c}"] = well_rep(our_sets[c])
            reps = {k: f for k, f in reps.items() if f}
            if len(reps) >= 2:
                dis = rep_zncc(reps)
                pairs = " · ".join(f"{a} vs {b} <b>{z:.2f}</b>" for a, b, z, _ in dis)
                dissim_html = (f'<div class=dis>edge-ZNCC across the merged scenes: '
                               f'{pairs} &nbsp;⇒&nbsp; genuinely different scenes — '
                               f'GT was right to separate them, our merge is the error</div>')
        elif len(own_clusters) == 1:
            verdict = "frame-set mismatch (extra/missing frame in our cluster)"
            assess = ("review", "MODEL-FAILURE CANDIDATE — extra/missing frame (shared-frame artifact?)")

        # render the full block (group's frames + any foreign frames we merged in),
        # sorted by our cluster, then exposure
        render = sorted(block_frames[g], key=lambda f: (our_cid[pred[f]], bright.get(f, 0)))
        cells = []
        for f in render:
            fg = gt.get(f, g)               # this frame's TRUE gt group
            gc = color(gt_cid[fg]); oc = color(our_cid[pred[f]])
            foreign = " foreign" if fg != g else ""
            cells.append(
                f'<figure class="t{foreign}"><img src="data:image/jpeg;base64,{thumb(imgs, idxof, f)}">'
                f'<div class=bar style="background:{gc}">GT {fg}</div>'
                f'<div class=bar style="background:{oc}">ours {pred[f]}</div>'
                f'<figcaption>B{bright.get(f, 0):.0f}</figcaption></figure>')
        tally[assess[0]] = tally.get(assess[0], 0) + 1
        blocks.append(
            f'<div class=grp><div class="assess {assess[0]}">{assess[1]}</div>'
            f'<div class=h>GT group <b>{g}</b> · {len(frames)} frames '
            f'· <span class=v>{verdict}</span></div>{dissim_html}'
            f'<div class=r>{"".join(cells)}</div></div>')

    matched = len(gt_groups) - len(disagreements)
    return (f'<section><h2>{label} '
            f'<span class=count>{len(disagreements)} disagreements · '
            f'{matched}/{len(gt_groups)} matched · '
            f'<span style="color:#7fe0a0">{tally["gterr"]} likely GT-error</span> · '
            f'<span style="color:#ffb86b">{tally["review"]} to review</span></span></h2>'
            f'{"".join(blocks)}</section>')


datasets = []
for spec in SPECS:
    if ":" in spec and not spec[1:3] == ":\\":
        path, label = spec.split(":", 1)
    else:
        path, label = spec, spec
    datasets.append((path, label))

# the anomaly analysis needs an unfixable.json (the GT-error labels) to score against
anomaly_datasets = [(p, l) for p, l in datasets if (Path(p) / "unfixable.json").exists()]
anomaly_html = build_anomaly_section(anomaly_datasets) if anomaly_datasets else ""
sections = [section(label, Path(path)) for path, label in datasets]

html = f"""<!doctype html><html><head><meta charset=utf-8><title>Where we disagree with GT</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:32px;max-width:1300px;margin:auto}}
 h1{{font-size:24px;margin:0 0 4px}}
 .lead{{color:#9aa3af;max-width:880px;margin:0 0 26px}}
 .legend{{display:flex;gap:18px;margin:10px 0 22px;font-size:13px;color:#b9c0cc}}
 .legend div{{display:flex;align-items:center;gap:6px}}
 .sw{{width:34px;height:11px;border-radius:2px;display:inline-block}}
 section{{margin:34px 0}}
 h2{{font-size:19px;border-bottom:2px solid #2a2f3a;padding-bottom:6px}}
 .count{{font-size:13px;color:#8b93a1;font-weight:400}}
 .grp{{margin:14px 0;border:1px solid #262b35;border-radius:10px;padding:12px;background:#161a22}}
 .h{{color:#cfd6e0;margin-bottom:8px}}
 .v{{color:#f0a35c;font-style:italic}}
 .assess{{display:inline-block;font-size:12px;font-weight:700;padding:3px 10px;border-radius:5px;margin-bottom:8px}}
 .assess.gterr{{background:#13351f;color:#7fe0a0;border:1px solid #2ecc71}}
 .assess.review{{background:#3a2410;color:#ffb86b;border:1px solid #e67e22}}
 .dis{{font-size:12px;color:#9fd3ff;background:#10202e;border-left:3px solid #3a7bd5;
   padding:6px 10px;border-radius:4px;margin:0 0 9px}}
 p.cat{{color:#b9c0cc;background:#161a22;border-left:3px solid #3498db;padding:10px 14px;border-radius:4px;max-width:900px}}
 .anom{{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px}}
 .r{{display:flex;flex-wrap:wrap;gap:7px}}
 figure.t{{margin:0;width:{THUMB}px}}
 figure.t img{{width:{THUMB}px;height:{THUMB}px;object-fit:cover;border-radius:6px 6px 0 0;display:block}}
 /* foreign frames are distinguished by their GT color bar only (no box/tag),
    so images stay clean for side-by-side comparison */
 .bar{{font-size:10px;color:#fff;font-weight:600;text-align:center;padding:1px 0;
       text-shadow:0 0 2px rgba(0,0,0,.6)}}
 figcaption{{font-size:10.5px;color:#cfd6e0;background:#0f1115;text-align:center;
             padding:2px;border-radius:0 0 6px 6px}}
</style></head><body>
<h1>Where our model disagrees with the ground truth</h1>
<p class=lead>Every ground-truth group we do not reproduce exactly. Each thumbnail
carries two color bars: the <b>top bar is the GT cluster</b> (one color per
ground-truth group) and the <b>bottom bar is our predicted cluster</b>. A top row
of one color over a bottom row of several colors is a place where GT lumps
distinct camera angles together and we split them — a GT mismatch by design.
Within each block, frames are ordered by our cluster, then by exposure (dark →
light); blocks are ordered by GT group id.</p>
<div class=legend>
  <div><span class=sw style="background:#3498db"></span> top bar = GT cluster</div>
  <div><span class=sw style="background:#2ecc71"></span> bottom bar = our cluster</div>
</div>
{anomaly_html}
{''.join(sections)}
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html)/1024:.0f} KB)")
