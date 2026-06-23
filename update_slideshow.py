# -*- coding: utf-8 -*-
"""Bring algorithm_slideshow.html up to date: remove all CNN/MobileNet/ONNX
content (we are training-free now) and reflect the real refinement pipeline
(11 passes, not 3). Idempotent-ish: asserts each anchor is present exactly once."""
import re
from pathlib import Path

P = Path("algorithm_slideshow.html")
html = P.read_text(encoding="utf-8")

EM = "—"  # em dash used throughout the deck

repls = [
    # --- "A learned complement" slide -> "A complementary signal" -----------
    ("A learned complement. Gradient correlation is precise",
     "A complementary signal. Gradient correlation is precise"),
    (f"So we add a small contrastive embedding {EM} a MobileNet trained so "
     "same-angle photos land close together and different rooms land apart, "
     "even across exposures.",
     f"So we add a second signal that captures each room's overall identity {EM} "
     "a compact, training-free embedding built from classic transforms, shown "
     "on the next slides."),
    ("<h2>A learned complement</h2>", "<h2>A complementary signal</h2>"),
    ("<h3>Color embedding</h3><p>Small MobileNet, contrastive-trained on "
     "color + structure.",
     "<h3>Room-identity embedding</h3><p><b>Training-free</b> transform "
     "(next slides)."),

    # --- "Our pick" slide: strip CNN comparisons ---------------------------
    (f"and it is far faster to extract {EM} about a tenth of a millisecond for "
     "eigenface, under one millisecond for wavelet, versus several for the "
     f"network, and the gap is far larger in the browser. And it is not a "
     "compromise: on the full five thousand image set it actually beat the "
     "network at the base level, and matched it through the whole pipeline "
     "within a couple of groups. Simpler, faster, and just as accurate.",
     f"and it is fast to extract {EM} about a tenth of a millisecond for "
     "eigenface, under one millisecond for wavelet. And it is no compromise: it "
     "carries the full pipeline to a perfect fixable score on the five thousand "
     "image set. Simpler, faster, and fully training-free."),
    (f"far faster {EM} yet on the 5,000-image set it <b>beats the CNN at base "
     "level</b> and matches it end-to-end.",
     f"far faster {EM} and on the 5,000-image set it <b>carries the full "
     "pipeline to a perfect fixable score</b>."),
    ('<span class="pill">large-set base fusion: <b class="good">0.987</b> '
     'vs CNN 0.977</span>',
     '<span class="pill">large-set base fusion: <b class="good">0.987</b></span>'),
    ('<span class="pill">full pipeline: <b class="good">1.000</b> fixable '
     '(0.981 raw) ' + EM + ' beats CNN&#39;s 0.992</span>',
     '<span class="pill">full pipeline: <b class="good">1.000</b> fixable '
     '(0.981 raw)</span>'),
    ('      <tr><td>With CNN extract<sup>†</sup></td><td>16.1 s</td>'
     '<td>30.0 s</td><td class="big">46.1 s &nbsp;(9.1 ms/img)</td></tr>\n',
     ''),

    # --- refinement overview: "Three" -> the real set ----------------------
    ("Now we refine the groups with three targeted passes.",
     "Now we refine the groups with a sequence of targeted passes."),
    ("<h2>Three refinement passes</h2>", "<h2>Refinement passes</h2>"),
    ('<p class="mut">FIX 2, 4, 5 ' + EM + ' each targets one failure, each '
     'validated for zero collateral.</p>',
     '<p class="mut">Each pass targets one failure ' + EM + ' re-attaching '
     'stranded clipped frames or splitting over-merged scenes ' + EM + ' and is '
     'validated for zero collateral. Three worked examples follow; the full set '
     'is on the summary slide.</p>'),

    # --- browser slide: no model to ship -----------------------------------
    (f"which ports directly to JavaScript or WebGL. The one learned piece, the "
     "embedding, is a four-megabyte O N N X model that runs client-side with "
     "onnx-runtime web.",
     "which ports directly to JavaScript or WebGL. There is no model to "
     "download and nothing to train: the embedding is fit per shoot in a "
     "fraction of a second."),
    ('      <span class="step">embedding</span><span class="arr">→ 4 MB ONNX '
     '· onnxruntime-web</span>',
     '      <span class="step">wavelet + eigenface embedding</span>'
     '<span class="arr">→ typed-array JS · fit per-run, no model</span>'),
    ('      <li>No server, no upload ' + EM + ' <b class="acc">fully '
     'client-side and offline</b>.</li>',
     '      <li><b class="acc">No model download, nothing to train</b> ' + EM +
     ' every stage is classic array math.</li>\n      <li>No server, no upload '
     + EM + ' fully client-side and offline.</li>'),

    # --- final one-line slide ----------------------------------------------
    ("fuse a structural correlation with a learned embedding, pick the "
     "threshold per run, form groups, then refine with three targeted passes",
     "fuse a structural correlation with a training-free embedding, pick the "
     "threshold per run, form groups, then refine with a sequence of targeted "
     "passes"),
    ('<span class="step"><b>FIX 2 · 4 · 5</b></span>',
     '<span class="step"><b>11 refinement passes</b></span>'),

    # --- results table -----------------------------------------------------
    (f'<tr><td>5,000-photo {EM} raw, all groups</td><td class="big" '
     'style="font-size:1.2em">0.981</td></tr>',
     f'<tr><td>5,000-photo {EM} raw, all groups</td><td class="big" '
     'style="font-size:1.2em">0.984</td></tr>\n'
     f'      <tr><td>30,000-photo set {EM} raw, all groups</td><td class="big" '
     'style="font-size:1.2em">0.971</td></tr>'),
    (f'<p class="mut">Generalizes across a 13× size change {EM} the '
     'per-run threshold adapts automatically.</p>',
     f'<p class="mut">Generalizes across a 60× size change (500 ' + EM +
     ' 30,000 images) ' + EM + ' the per-run threshold adapts automatically. '
     'Across all <b>9,592</b> groups tested, just <b class="good">22</b> are '
     'true model failures ' + EM + ' <b class="good">99.77%</b> once the ~2.6% '
     'genuinely mislabeled groups are set aside.</p>'),
]

for old, new in repls:
    n = html.count(old)
    assert n == 1, f"expected 1 match, got {n} for: {old[:70]!r}"
    html = html.replace(old, new)

# --- remove the CNN footnote paragraph (apostrophe-safe via regex) ----------
html, k = re.subn(
    r'<p class="mut" style="font-size:\.72em;margin:\.2em 0"><sup>†</sup>'
    r'same pipeline.*?</p>',
    '<p class="mut" style="font-size:.72em;margin:.2em 0">Extraction is a '
    'wavelet transform plus one matrix multiply per image &mdash; '
    'sub-millisecond, with no model to download and nothing to train.</p>',
    html, flags=re.S)
assert k == 1, f"CNN footnote: expected 1, got {k}"

# --- remove the entire "First attempt: a learned CNN embedding" slide -------
start = html.index('<section class="slide" data-narr="Our first version of the '
                   'room-identity signal')
line_start = html.rfind("\n", 0, start) + 1
end = html.index("</section>", start) + len("</section>")
html = html[:line_start] + html[end:]
assert "First attempt: a learned CNN embedding" not in html

# --- insert the full-pipeline summary slide after the FIX 5 slide -----------
anchor = ('<p class="mut">The tightness guard distinguishes true over-merges '
          'from legitimately varied groups.</p>\n  </section>')
assert html.count(anchor) == 1
summary = '''
  <section class="slide" data-narr="Those three are just the worked examples. In total the pipeline runs eleven refinement passes, each a small deterministic masked-correlation test, in two families. Re-attach and merge passes rescue a frame or piece that was wrongly stranded: a clipped orphan, a split-off cluster, a capture-run piece, a drift-shifted frame, or one matched by its surviving bright or dark anchor spots. Split passes pull apart scenes that were wrongly merged: by capture run, by a localized scene difference, by an anchor mismatch, or by a foreign clipped frame. Every pass holds the fixable score at a perfect one.">
    <h2>The full refinement pipeline</h2>
    <p class="mut">FIX 2 / 4 / 5 are the worked examples. In all, <b class="acc">eleven</b> deterministic masked-correlation passes run in order, in two families:</p>
    <div class="grid">
      <div class="box"><h3 class="acc">Re-attach &amp; merge</h3><p>rescue a wrongly-stranded frame or piece</p><p class="mut" style="font-size:.8em">exposure-ladder orphan (2) &middot; cluster merge (4) &middot; clipped-uniqueness (2c) &middot; bright/dark <b>anchor</b> (12) &middot; camera-<b>drift</b> re-align (16)</p></div>
      <div class="box"><h3 class="acc">Split</h3><p>separate two scenes wrongly merged</p><p class="mut" style="font-size:.8em">high-res (5) &middot; capture-run (6) &middot; scene-change (7) &middot; localized scene-difference (10) &middot; anchor mismatch (11) &middot; foreign clipped frame (17)</p></div>
    </div>
    <p class="mut">Splits fire only on confident masked-ZNCC evidence; re-attaches only when a clipped frame's surviving structure clearly matches. <span class="good">All hold fixable-only = 1.000.</span></p>
  </section>'''
html = html.replace(anchor, anchor + "\n" + summary)

# --- fix the static initial slide counter (auto-corrected by JS, but tidy) --
n_slides = html.count('class="slide"')
html = re.sub(r'<div id="cnt">1 / \d+</div>',
              f'<div id="cnt">1 / {n_slides}</div>', html)

P.write_text(html, encoding="utf-8")
print(f"updated {P}: {n_slides} slides")
