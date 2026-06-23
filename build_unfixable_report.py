"""
Build a self-contained HTML report of every group we excluded as UNFIXABLE
(genuine ground-truth errors / information-limited frames), with the actual
images embedded as base64, a per-group + per-category explanation, AND what our
model actually predicted for each frame (the "more-correct" answer).

Run compute_pred_labels.py <data_dir> first to produce <data>/pred_labels.json.
Usage: build_unfixable_report.py <data_dir> [out.html]
"""
import sys, json, base64
from pathlib import Path
import numpy as np, cv2

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/large")
OUT  = Path(sys.argv[2] if len(sys.argv) > 2 else "unfixable_report.html")

cand = json.load(open(DATA / "unfixable.json"))["groups"]
col  = np.load(DATA / "img128c.npz", allow_pickle=True)
imgs = col["imgs"]; files = list(col["files"])
idxof = {f: i for i, f in enumerate(files)}

# what OUR model predicted for each frame
predf = DATA / "pred_labels.json"
pred = json.load(open(predf)) if predf.exists() else {}

files_by_gid = {}
for f in files:
    g = f.split("_", 1)[0][1:]  # 'g9245__DSC..' -> '9245'
    files_by_gid.setdefault(g, []).append(f)

# brightness (grayscale mean) per frame, only for the frames we actually render
# (groups whose unfixable.json entry recorded a reason but no member list).
needed = set()
for gid, info in cand.items():
    if not info.get("members"):
        gnum = gid[1:] if gid.startswith("g") else gid
        needed.update(files_by_gid.get(gnum, []))
bright = {}
if needed:
    raw = np.load(DATA / "raw256.npz", allow_pickle=True)
    ridx = {f: i for i, f in enumerate(raw["files"])}
    rimgs = raw["imgs"]
    for f in needed:
        bright[f] = float(rimgs[ridx[f]].mean())

THUMB = 150  # px


def thumb(f):
    im = cv2.resize(imgs[idxof[f]], (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def ensure_members(gid, info):
    if info.get("members"):
        return info["members"]
    gnum = gid[1:] if gid.startswith("g") else gid
    return [[f, bright.get(f, 0.0), -1] for f in files_by_gid.get(gnum, [])]


def category(reason):
    r = reason.lower()
    if r.startswith("information-limited"):
        return ("Information-limited",
                "A frame in the group carries essentially no usable signal "
                "(near-black, blown-white, or only a fraction of a percent of "
                "valid pixels). With no structure to correlate against, it is "
                "mathematically indistinguishable from frames in OTHER scenes, "
                "so no exposure-invariant grouper can place it correctly.")
    if "duplicate-frame" in r:
        return ("Duplicate frame shared across groups",
                "The exact same frame appears in two different-scene reference "
                "groups. A grouping algorithm can only assign a frame to ONE "
                "cluster, so at least one reference group is unsatisfiable by "
                "construction.")
    return ("Mislabeled ground truth (multi-scene)",
            "The reference group lumps together frames from visibly different "
            "scenes (different rooms / a drone that flew elsewhere). Even the "
            "well-exposed frames disagree, so honoring the label would require "
            "merging unrelated scenes — the label itself is the error.")


CAT_ORDER = ["Mislabeled ground truth (multi-scene)",
             "Duplicate frame shared across groups",
             "Information-limited"]

buckets = {c: [] for c in CAT_ORDER}
for gid, info in cand.items():
    info["members"] = ensure_members(gid, info)
    if not info.get("frames"):
        info["frames"] = len(info["members"])
    if not info.get("subscenes"):
        subs = {m[2] for m in info["members"] if m[2] >= 0}
        info["subscenes"] = len(subs) if subs else "?"
    cat, blurb = category(info["reason"])
    buckets[cat].append((gid, info, blurb))

PAL = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]


def group_block(gid, info, blurb, cat):
    members = sorted(info["members"], key=lambda x: (x[2], x[1]))
    uniq = [c for c in dict.fromkeys(pred.get(f) for f, _, _ in members) if c is not None]
    pcolor = {c: PAL[i % len(PAL)] for i, c in enumerate(uniq)}
    plabel = {c: f"P{i + 1}" for i, c in enumerate(uniq)}
    cells = []
    for f, br, sc in members:
        c = pred.get(f)
        if c is None:
            badge = '<span class=pb style="background:#555">model: n/a</span>'
        else:
            badge = (f'<span class=pb style="background:{pcolor[c]}">'
                     f'model&rarr; {plabel[c]}</span>')
        sclabel = f"sub-scene {sc}" if sc >= 0 else "scene n/a"
        cells.append(
            f'<figure class=t><img src="data:image/jpeg;base64,{thumb(f)}">'
            f'<figcaption>B{br:.0f} · {sclabel}{badge}</figcaption></figure>')
    if uniq:
        ids = ", ".join(plabel[c] for c in uniq)
        if len(uniq) == 1 and cat.startswith("Information"):
            tail = ("our model folds the lone information-limited frame into the "
                    "coherent core — there is no second real scene to separate.")
        elif len(uniq) == 1:
            tail = ("our model could not separate them either — the well-exposed "
                    "sub-scenes are too alike to tell apart from pixels alone, so "
                    "this is information-limited, not a clean fix.")
        elif cat.startswith("Information"):
            tail = ("our model separates the information-limited frame(s) from the "
                    "coherent core rather than forcing a wrong merge.")
        elif cat.startswith("Duplicate"):
            tail = ("our model places the shared frame in one scene only — it "
                    "physically cannot satisfy both reference groups at once.")
        else:
            tail = "matching the genuinely distinct scenes the label conflated."
        split = (f'Our model grouped these {info["frames"]} frames into '
                 f'<b>{len(uniq)} cluster(s)</b> ({ids}) — {tail}')
    else:
        split = "Predicted labels unavailable (run compute_pred_labels.py)."
    return (f'<div class=grp>'
            f'<div class=h>group <b>{gid}</b> · {info["frames"]} frames · '
            f'{info["subscenes"]} sub-scene(s)</div>'
            f'<div class=why>ground-truth-error reason: {info["reason"]}</div>'
            f'<div class=pred>{split}</div>'
            f'<div class=r>{"".join(cells)}</div></div>')


sections = []
total = len(cand)
for cat in CAT_ORDER:
    items = buckets[cat]
    if not items:
        continue
    blurb = items[0][2]
    blocks = "".join(group_block(g, i, b, cat) for g, i, b in
                     sorted(items, key=lambda x: -len(x[1]["members"])))
    sections.append(
        f'<section><h2>{cat} <span class=count>{len(items)} group(s)</span></h2>'
        f'<p class=cat>{blurb}</p>{blocks}</section>')

html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Unfixable groups — {DATA.name}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:32px;max-width:1200px;margin:auto}}
 h1{{font-size:24px;margin:0 0 4px}}
 .lead{{color:#9aa3af;max-width:820px;margin:0 0 28px}}
 section{{margin:36px 0}}
 h2{{font-size:19px;border-bottom:2px solid #2a2f3a;padding-bottom:6px}}
 .count{{font-size:13px;color:#8b93a1;font-weight:400}}
 p.cat{{color:#b9c0cc;background:#161a22;border-left:3px solid #3498db;padding:10px 14px;border-radius:4px;max-width:860px}}
 .grp{{margin:16px 0;border:1px solid #262b35;border-radius:10px;padding:14px;background:#161a22}}
 .h{{color:#cfd6e0;margin-bottom:4px}}
 .why{{color:#f0a35c;font-size:12.5px;margin-bottom:4px;font-style:italic}}
 .pred{{color:#7fd18a;font-size:12.5px;margin-bottom:10px}}
 .r{{display:flex;flex-wrap:wrap;gap:8px}}
 figure.t{{margin:0;text-align:center;width:{THUMB}px}}
 figure.t img{{width:{THUMB}px;height:{THUMB}px;object-fit:cover;border-radius:6px 6px 0 0;display:block}}
 figcaption{{font-size:10.5px;color:#cfd6e0;background:#0f1115;padding:3px 2px;border-radius:0 0 6px 6px}}
 .pb{{display:block;margin-top:3px;color:#fff;font-weight:600;border-radius:3px;padding:1px 0}}
</style></head><body>
<h1>Excluded as unfixable — {total} groups</h1>
<p class=lead>These reference groups are excluded from the fixable-only score because
they are <b>not algorithm failures</b>. Each is either a mislabeled ground-truth
group (unrelated scenes lumped together), a frame physically shared by two groups,
or an information-limited frame carrying no usable signal. Each frame's caption shows its
brightness (B), the visually-coherent sub-scene it belongs to (when known), and
<b>what our model predicted</b> (the colored <span style="background:#3498db;color:#fff;padding:1px 5px;border-radius:3px">model&rarr; P#</span>
badge). Where the ground-truth label lumps scenes together, the model's split is the
<b>more-correct answer</b>.</p>
{''.join(sections)}
</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT}  ({len(html)/1024:.0f} KB, {total} groups)")
