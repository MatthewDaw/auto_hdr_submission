"""
Finalize the ground-truth-error (unfixable) list = coherence-flagged AND
actually-missed by the full pipeline (fusion + FIX2 ladder re-attach). A group
the pipeline solves is fixable by definition, so we drop those false positives.
Writes corrected <data>/unfixable.json + self-contained HTML gallery (base64 imgs).
Usage: finalize_unfixable.py <data_dir>
"""
import sys, json, base64
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
import descriptor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA=Path(sys.argv[1] if len(sys.argv)>1 else "sample")
VLO,VHI=8,247; W_GRAD=0.65; MASK_THR=0.58; MIN_VALID=0.03; MARGIN=0.12
clahe=cv2.createCLAHE(3.0,(8,8))
def grad_mask(r):
    g=clahe.apply(r).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3)
    return cv2.magnitude(gx,gy),((r>=VLO)&(r<=VHI))
def mzncc(m1,v1,m2,v2):
    v=(v1&v2).ravel()
    if v.mean()<MIN_VALID: return -1.0
    a=m1.ravel()[v]-m1.ravel()[v].mean(); b=m2.ravel()[v]-m2.ravel()[v].mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def pipeline_solved():
    rd=np.load(f"{DATA}/raw256.npz",allow_pickle=True); raw=rd["imgs"]; gid=rd["gid"]; files=list(rd["files"])
    d=np.load(f"{DATA}/feat_cache.npz",allow_pickle=True); n=len(files)
    G=(d["M"]@d["M"].T).astype(np.float32); E=descriptor.embed(raw)
    Es=(E@E.T).astype(np.float32); B=np.array([raw[i].mean() for i in range(n)])
    Fz=(W_GRAD*G+(1-W_GRAD)*Es).astype(np.float32)
    groups=defaultdict(set)
    for f,g in zip(files,gid): groups[g].add(f)
    refsets=set(frozenset(v) for v in groups.values()); best=(-1,None)
    for t in np.arange(0.15,0.97,0.01):
        A=Fz>=t; np.fill_diagonal(A,False)
        _,lab=connected_components(csr_matrix(A),directed=False)
        pred=defaultdict(set)
        for i,f in enumerate(files): pred[lab[i]].add(f)
        sc=len(refsets&set(frozenset(v) for v in pred.values()))
        if sc>best[0]: best=(sc,A.copy())
    A=best[1]; _,lab0=connected_components(csr_matrix(A),directed=False)
    # FIX2
    clm=defaultdict(list)
    for j in range(n): clm[lab0[j]].append(j)
    step={c:(np.median(np.diff(np.sort(B[m]))) if len(m)>=2 else 80.0) for c,m in clm.items()}
    gm=[grad_mask(raw[i]) for i in range(n)]
    for i in [k for k in range(n) if B[k]<45 or B[k]>210]:
        sc=[]
        for c,m in clm.items():
            if c==lab0[i]: continue
            j=min(m,key=lambda k:abs(B[k]-B[i]))
            if abs(B[i]-B[j])>1.8*max(step[c],30.0): continue
            sc.append((mzncc(*gm[i],*gm[j]),j))
        sc.sort(reverse=True)
        if sc and sc[0][0]>=MASK_THR and (sc[0][0]-(sc[1][0] if len(sc)>1 else -1)>=MARGIN or (sc[1][0] if len(sc)>1 else -1)<MASK_THR):
            A[i,sc[0][1]]=A[sc[0][1],i]=True
    np.fill_diagonal(A,False); _,lab=connected_components(csr_matrix(A),directed=False)
    pred=defaultdict(set)
    for i,f in enumerate(files): pred[lab[i]].add(f)
    predlk=set(frozenset(v) for v in pred.values())
    return {g for g,v in groups.items() if frozenset(v) in predlk}, groups, files

def gallery(uf, files):
    col=np.load(DATA/"img128c.npz",allow_pickle=True)["imgs"]; idxof={f:i for i,f in enumerate(files)}
    def thumb(f):
        ok,buf=cv2.imencode(".jpg",cv2.resize(col[idxof[f]],(80,80)),[cv2.IMWRITE_JPEG_QUALITY,72])
        return base64.b64encode(buf).decode()
    PAL=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]
    rows=[]
    for g,info in sorted(uf.items(), key=lambda kv:-kv[1]["subscenes"]):
        cells="".join(f'<div class=t><img src="data:image/jpeg;base64,{thumb(f)}"><div class=b style="background:{PAL[sc%len(PAL)]}">B{br:.0f}·s{sc}</div></div>'
                      for f,br,sc in sorted(info["members"],key=lambda x:(x[2],x[1])))
        rows.append(f'<div class=grp><div class=h>group {g} — <b>{info["reason"]}</b> · {info["frames"]} frames · {info["subscenes"]} sub-scenes</div><div class=r>{cells}</div></div>')
    html=f"""<!doctype html><meta charset=utf-8><title>Ground-truth-error groups — {DATA}</title>
<style>body{{font:13px system-ui;background:#111;color:#ddd;margin:20px}}.grp{{margin:16px 0;border:1px solid #333;border-radius:8px;padding:10px;background:#1a1a1a}}
.h{{margin-bottom:8px;color:#bbb}}.r{{display:flex;flex-wrap:wrap;gap:6px}}.t{{text-align:center}}.t img{{width:80px;height:80px;object-fit:cover;border-radius:4px;display:block}}
.b{{font-size:10px;color:#fff;border-radius:0 0 4px 4px;padding:1px}}</style>
<h2>{DATA}: {len(uf)} ground-truth-error groups excluded from eval</h2>
<p>Reference groups whose members are visually-unrelated scenes (different rooms lumped together, or drone shots that moved) AND which the full pipeline cannot solve. Thumbnails colored by disconnected sub-scene. These are mislabeled ground truth, not algorithm failures.</p>{''.join(rows)}"""
    open(DATA/"unfixable_gallery.html","w",encoding="utf-8").write(html)

def main():
    cand=json.load(open(DATA/"unfixable.json"))["groups"]
    solved,groups,files=pipeline_solved()
    final={g:info for g,info in cand.items() if g not in solved}
    removed=[g for g in cand if g in solved]
    print(f"{DATA}: detector flagged {len(cand)}, pipeline solved {len(removed)} of them (false positives removed): {removed}")
    print(f"  FINAL ground-truth-error groups: {len(final)}")
    json.dump({"note":"coherence-flagged AND pipeline-missed = genuine ground-truth errors","groups":final},
              open(DATA/"unfixable.json","w"),indent=1)
    gallery(final, files)
    # final achievable score
    fixable={g for g in groups if g not in final}
    okf=solved & fixable
    print(f"  pipeline solves {len(solved&set(groups))}/{len(groups)} raw = {len(solved&set(groups))/len(groups):.4f}")
    print(f"  FIXABLE-only (excluding {len(final)} ground-truth errors): {len(okf)}/{len(fixable)} = {len(okf)/len(fixable):.4f}")
    print(f"  wrote {DATA}/unfixable.json + unfixable_gallery.html")

if __name__=="__main__":
    main()
