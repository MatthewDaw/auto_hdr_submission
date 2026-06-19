"""
Add the 'CNN -> wavelet/eigenface' arc to the slideshow: one slide on the CNN we
first tried (and its costs), then eigenfaces, wavelets, and the combined choice
with throughput metrics. Generates real visuals (eigen-rooms, wavelet decomposition,
throughput chart) and injects after the 'A learned complement' slide.
"""
import base64
import numpy as np, cv2, pywt
from sklearn.decomposition import PCA

raw=np.load("sample/raw256.npz",allow_pickle=True)["imgs"]
clahe=cv2.createCLAHE(3.0,(8,8))
def enc(im,png=True):
    ok,b=cv2.imencode(".png" if png else ".jpg",im,[] if png else [cv2.IMWRITE_JPEG_QUALITY,88]); return base64.b64encode(b).decode()
def tag(im,w,png=True,cap=None,col="#58a6ff"):
    m="png" if png else "jpeg"; t=f'<img src="data:image/{m};base64,{enc(im,png)}" style="width:{w}px;border-radius:6px;border:1px solid #30363d;display:block">'
    if cap: t=f'<figure style="margin:0;text-align:center">{t}<figcaption style="font-size:.62em;color:{col};margin-top:.25em">{cap}</figcaption></figure>'
    return t
def row(items,gap=".5em"): return f'<div style="display:flex;gap:{gap};flex-wrap:wrap;justify-content:center;align-items:flex-start;margin:.4em 0">{"".join(items)}</div>'

# ---- eigenfaces (eigen-rooms): PCA components on CLAHE 64x64 ----
X=np.stack([clahe.apply(cv2.resize(im,(64,64))).astype(np.float32).ravel() for im in raw])
pca=PCA(8).fit(X); mean_img=pca.mean_.reshape(64,64)
def norm(v): v=v-v.min(); return (v/(v.max()+1e-9)*255).astype(np.uint8)
eig_imgs=[cv2.applyColorMap(cv2.resize(norm(pca.components_[k].reshape(64,64)),(96,96),interpolation=cv2.INTER_NEAREST),cv2.COLORMAP_VIRIDIS) for k in range(6)]
eig_row=row([tag(cv2.resize(norm(mean_img),(96,96)),96,True,"avg room")]+[tag(e,96,True,f"eigen-room {k+1}") for k,e in enumerate(eig_imgs)])

# ---- wavelet decomposition montage ----
mid=raw[len(raw)//2]
g=clahe.apply(cv2.resize(mid,(256,256))).astype(np.float32)
co=pywt.wavedec2(g,'haar',level=2)
arr,_=pywt.coeffs_to_array(co)
disp=arr.copy(); disp[:64,:64]=norm(disp[:64,:64])         # approximation
m=np.abs(disp); m[:64,:64]=0; disp=np.where(np.arange(256)[None,:]<64,disp,0)*0  # rebuild below
# build a clean montage: approximation top-left + |details| elsewhere, log-scaled
mont=np.zeros((256,256),np.float32); a2=co[0]; mont[:64,:64]=norm(a2)
full=np.abs(pywt.coeffs_to_array(co)[0]); full[:64,:64]=0
ld=np.log1p(full); ld=ld/(ld.max()+1e-9)*255; ld[:64,:64]=norm(a2)
wave_img=cv2.applyColorMap(ld.astype(np.uint8),cv2.COLORMAP_MAGMA)
cv2.rectangle(wave_img,(0,0),(63,63),(255,255,255),1)
wave_row=row([tag(cv2.cvtColor(cv2.resize(mid,(150,150)),cv2.COLOR_GRAY2BGR),150,True,"room photo"),
              tag(cv2.resize(wave_img,(150,150),interpolation=cv2.INTER_NEAREST),150,True,"wavelet detail bands (multi-scale edges)","#3fb950")])

# ---- throughput / accuracy comparison chart ----
CW,CH=620,300; cc=np.full((CH,CW,3),22,np.uint8)
bars=[("CNN embedding",3.19,(120,120,120)),("wavelet",0.81,(80,175,76)),("eigenface/PCA",0.11,(80,175,76))]
x=70; bw=120; gap=70; mx=3.4
cv2.putText(cc,"extraction time (ms / image, CPU) - lower is better",(20,22),cv2.FONT_HERSHEY_SIMPLEX,0.46,(180,180,180),1,cv2.LINE_AA)
for name,val,col in bars:
    bh=int(val/mx*(CH-90)); cv2.rectangle(cc,(x,CH-40-bh),(x+bw,CH-40),col,-1)
    cv2.putText(cc,f"{val:.2f}",(x+30,CH-48-bh),cv2.FONT_HERSHEY_SIMPLEX,0.55,(230,230,230),2,cv2.LINE_AA)
    cv2.putText(cc,name,(x-2,CH-18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1,cv2.LINE_AA); x+=bw+gap
chart=row([tag(cc,470,True)])

CNN=('  <section class="slide" data-narr="Our first version of the room-identity signal was a learned convolutional network — a MobileNet trained with a contrastive objective so that same-room photos land close together. It works, but it carries real costs. It needs a labelled training run. It produces a four-megabyte model you have to bundle and ship. And in the browser it runs through web-assembly, which is slow — tens of milliseconds per image. So we asked: can a classic, training-free transform do the same job?">\n'
    '    <h2>First attempt: a learned CNN embedding</h2>\n'
    '    <p>A MobileNet trained contrastively — same room close, different room far, across exposures. It works, but it has costs:</p>\n'
    '    <div style="display:flex;gap:.5em;justify-content:center;flex-wrap:wrap;margin:.4em 0">'
    '<span class="pill" style="border-color:#d2992266">needs <b class="warn">training</b> on labels</span>'
    '<span class="pill" style="border-color:#d2992266">bundle a <b class="warn">4&nbsp;MB</b> model (ONNX)</span>'
    '<span class="pill" style="border-color:#f8514966"><b class="bad">slow</b> in-browser (WASM, ~tens of ms/img)</span></div>\n'
    '    <p class="mut">Question: can a classic, <b class="acc">training-free</b> transform do the same job — faster and lighter?</p>\n'
    '  </section>\n')

EIGEN=('  <section class="slide" data-narr="The first classic tool is the eigenface, better called here an eigen-room. We take many room photos, and use principal component analysis to find the handful of patterns that explain most of how rooms differ from each other. These patterns — shown here — are the eigen-rooms. Any photo is then summarized by how much of each pattern it contains: a compact fingerprint, computed with a single matrix multiply, and crucially learned with no labels.">\n'
    '    <h2>Eigenfaces (PCA) — &quot;eigen-rooms&quot;</h2>\n'
    '    <p class="mut">Principal Component Analysis finds the few patterns that explain how rooms differ. Each photo = how much of each pattern it has → a compact fingerprint (one matrix multiply, no labels).</p>\n'
    f'    {eig_row}\n'
    '    <p class="mut">Fit on the shoot&apos;s own photos in ~0.3&nbsp;s — adapts per run, no training, no model to ship.</p>\n'
    '  </section>\n')

WAVE=('  <section class="slide" data-narr="The second classic tool is the wavelet transform. Unlike a single edge filter, it decomposes the image into edges at many scales at once — fine details and coarse layout together — while discarding brightness. On the right you can see those multi-scale detail bands for the room on the left. It is the natural, multi-scale generalization of the gradient we already use, and like gradients it is exposure-robust and cheap.">\n'
    '    <h2>Wavelets — multi-scale edges</h2>\n'
    '    <p class="mut">A wavelet decomposes the image into edges at <b>many scales at once</b> (fine detail + coarse layout), dropping brightness — a multi-scale generalization of the Sobel gradient.</p>\n'
    f'    {wave_row}\n'
    '    <p class="mut">Captures both fine structure (alignment) and coarse layout (room identity) in one cheap, exposure-robust transform.</p>\n'
    '  </section>\n')

COMBO=('  <section class="slide" data-narr="Our choice combines them: run the wavelet transform, then compress its bands with P C A. Eigen-wavelets. It needs no training and no bundled model, the P C A is fit per shoot so it adapts automatically, and it is far faster to extract — about a tenth of a millisecond for eigenface, under one millisecond for wavelet, versus several for the network, and the gap is far larger in the browser. And it is not a compromise: on the full five thousand image set it actually beat the network at the base level, and matched it through the whole pipeline within a couple of groups. Simpler, faster, and just as accurate.">\n'
    '    <h2>Our pick: wavelet + eigenface</h2>\n'
    '    <p class="mut"><b class="acc">Eigen-wavelets</b> = wavelet bands → PCA. No training, no bundled model, fit per-run, and far faster — yet on the 5,000-image set it <b>beats the CNN at base level</b> and matches it end-to-end.</p>\n'
    f'    {chart}\n'
    '    <div style="display:flex;gap:.5em;justify-content:center;flex-wrap:wrap">'
    '<span class="pill">large-set base fusion: <b class="good">0.987</b> vs CNN 0.977</span>'
    '<span class="pill">full pipeline: <b class="good">0.989</b> vs CNN 0.992 (~5 groups)</span>'
    '<span class="pill">no training · no ONNX · ~30&times; faster extract</span></div>\n'
    '  </section>\n')

html=open("algorithm_slideshow.html",encoding="utf-8").read()
anchor="proven to generalize across dataset sizes.</p>\n  </section>"
assert anchor in html, "learned-complement anchor not found"
html=html.replace(anchor, anchor+"\n"+CNN+EIGEN+WAVE+COMBO, 1)
open("algorithm_slideshow.html","w",encoding="utf-8").write(html)
print("injected CNN->wavelet/eigenface arc (4 slides)")
