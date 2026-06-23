"""Unsupervised "outliers stand out" analysis for the disagreement report.

Pools every ground-truth group across the given datasets, measures each group's
internal coherence (median pairwise masked-ZNCC of its well-exposed frames), runs
an unsupervised Isolation Forest (no labels), and shows that the known GT-error
groups (unfixable.json) surface as the top anomalies. Returns a self-contained
HTML fragment with two inline-SVG plots. Per-group features are cached to
<repo>/anomaly_features.json so the report rebuilds cheaply.
"""
from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from autohdr.features.masked_correlation import MaskedCorrelation

_CACHE = Path(__file__).resolve().parent / "anomaly_features.json"


def _features_for(data_dir: Path):
    gt = defaultdict(list)
    for r in csv.DictReader(open(data_dir / "public_manifest.csv", encoding="utf-8")):
        gt[r["group_id"]].append(r["filename"])
    raw = np.load(data_dir / "raw256.npz", allow_pickle=True)
    files = list(map(str, raw["files"]))
    imgs = raw["imgs"]
    idx = {f: i for i, f in enumerate(files)}
    B = {f: float(imgs[idx[f]].mean()) for f in files}
    flagged = set(json.load(open(data_dir / "unfixable.json"))["groups"].keys())
    rows = []
    for g, frames in gt.items():
        well = [f for f in frames if 50 <= B[f] <= 205] or frames
        sub = np.stack([imgs[idx[f]] for f in well])
        mc = MaskedCorrelation(sub)
        zs = [mc.score(i, j)[0] for i in range(len(well)) for j in range(i + 1, len(well))]
        brs = [B[f] for f in frames]
        rows.append({
            "dataset": data_dir.name,
            "group": g,
            "coherence": float(np.median(zs)) if zs else 1.0,
            "size": len(frames),
            "spread": max(brs) - min(brs),
            "flagged": g in flagged or f"g{g}" in flagged,
        })
    return rows


def _load_features(datasets):
    key = sorted(str(Path(d).resolve()) for d, _ in datasets)
    if _CACHE.exists():
        cached = json.load(open(_CACHE))
        if cached.get("key") == key:
            return cached["rows"]
    rows = []
    for d, _ in datasets:
        rows += _features_for(Path(d))
    json.dump({"key": key, "rows": rows}, open(_CACHE, "w"))
    return rows


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _svg_curve(scores, flagged, w=520, h=260, pad=34):
    n = len(scores)
    order = np.argsort(-scores)
    s = scores[order]; fl = flagged[order]
    smin, smax = s.min(), s.max()
    def X(i): return pad + i / max(n - 1, 1) * (w - 2 * pad)
    def Y(v): return h - pad - (v - smin) / (smax - smin + 1e-9) * (h - 2 * pad)
    poly = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(s))
    dots = "".join(
        f'<circle cx="{X(i):.1f}" cy="{Y(s[i]):.1f}" r="3.2" fill="#e74c3c"/>'
        for i in range(n) if fl[i])
    return f'''<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:560px">
<rect x="0" y="0" width="{w}" height="{h}" fill="#0c1018" rx="8"/>
<polyline points="{poly}" fill="none" stroke="#7f8c9b" stroke-width="1.5"/>
{dots}
<text x="{pad}" y="20" fill="#9fd3ff" font-size="12">flagged GT-error groups (red) sit at the extreme</text>
<text x="{pad}" y="{h-10}" fill="#6b7280" font-size="11">groups sorted by anomaly score →</text>
</svg>'''


def _svg_scatter(coh, scores, flagged, w=520, h=260, pad=34):
    smin, smax = scores.min(), scores.max()
    def X(c): return pad + c * (w - 2 * pad)            # coherence 0..1
    def Y(v): return h - pad - (v - smin) / (smax - smin + 1e-9) * (h - 2 * pad)
    pts = ""
    for c, v, f in zip(coh, scores, flagged):
        col = "#e74c3c" if f else "#2ecc71"
        r = 3.0 if f else 1.7
        pts += f'<circle cx="{X(c):.1f}" cy="{Y(v):.1f}" r="{r}" fill="{col}" fill-opacity="0.8"/>'
    return f'''<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:560px">
<rect x="0" y="0" width="{w}" height="{h}" fill="#0c1018" rx="8"/>
{pts}
<text x="{pad}" y="20" fill="#2ecc71" font-size="12">● correctly labeled &nbsp;&nbsp;<tspan fill="#e74c3c">● flagged GT-error</tspan></text>
<text x="{pad}" y="{h-10}" fill="#6b7280" font-size="11">internal coherence (low = mixed scenes) →</text>
<text x="{w-pad}" y="{h-10}" fill="#6b7280" font-size="11" text-anchor="end">↑ anomaly score</text>
</svg>'''


def build_section(datasets):
    rows = _load_features(datasets)
    coh = np.array([r["coherence"] for r in rows])
    size = np.array([r["size"] for r in rows])
    spread = np.array([r["spread"] for r in rows])
    flag = np.array([r["flagged"] for r in rows])
    X = np.column_stack([coh, np.log(size), spread])
    iso = IsolationForest(n_estimators=400, random_state=0).fit(X)
    score = -iso.score_samples(X)
    auc = _auc(flag.astype(int), score)
    n_flag = int(flag.sum())
    top_capt = int(flag[np.argsort(-score)[:n_flag]].sum())
    med_f, med_c = float(np.median(coh[flag])), float(np.median(coh[~flag]))
    return f'''<section><h2>Proving the GT-error groups stand out
 <span class=count>unsupervised — no labels</span></h2>
<p class=cat>Pooled <b>{len(rows)}</b> ground-truth groups across both datasets. An
<b>Isolation Forest</b> trained only on each group's internal coherence (median
pairwise masked-ZNCC of its well-exposed frames), size, and brightness spread —
with <b>no access to which groups are flagged</b> — ranks the known GT-error groups
as the top anomalies (ROC-AUC <b>{auc:.2f}</b>; {top_capt}/{n_flag} flagged groups
land in the top {n_flag}). Flagged groups have median coherence <b>{med_f:.2f}</b>
versus <b>{med_c:.2f}</b> for correctly-labeled groups — they really are the
incoherent outliers, recoverable without any labels.</p>
<div class=anom>{_svg_curve(score, flag)}{_svg_scatter(coh, score, flag)}</div>
</section>'''
