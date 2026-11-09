"""Mark corresponding points on each pair in the browser (localhost only).

Click a point in the historical image, then the same point in the modern photo. Repeat for 8-10 points per pair:
sharp building corners, window corners, spire tips or the top of a column, spread over the building. Scroll to
zoom, drag to pan, and press Z to undo the last click. Every completed point is saved straight away to
styles/marks/<pair_id>.csv, so you can close the page and come back.

Do this before any matcher is run on the pairs, and without looking at matcher output. Once every pair is
marked, `python styles/mark.py --lock` writes marks/LOCK.json and the marks are frozen.

    python styles/fetch.py            # once, downloads the images
    python styles/mark.py             # then open http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import PAIRS, image_path  # noqa: E402

MARKS = Path(__file__).resolve().parent / "marks"
LOCK = MARKS / "LOCK.json"
FIELDS = ["point_id", "hist_x", "hist_y", "modern_x", "modern_y", "description"]


def pairs() -> list[dict]:
    return list(csv.DictReader(open(PAIRS, encoding="utf-8")))


def read_marks(pid: str) -> list[dict]:
    p = MARKS / f"{pid}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def write_marks(pid: str, points: list[dict]) -> None:
    if LOCK.exists():
        raise PermissionError("marks are locked")
    with open(MARKS / f"{pid}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, p in enumerate(points, 1):
            w.writerow({"point_id": f"P{i:02d}", **{k: round(float(p[k]), 2) for k in FIELDS[1:5]},
                        "description": p.get("description", "")})


def lock() -> int:
    files = sorted(MARKS.glob("*.csv"))
    files = [f for f in files if not f.name.startswith("_")]
    missing = [r["pair_id"] for r in pairs() if not (MARKS / f"{r['pair_id']}.csv").exists()]
    short = [f.stem for f in files if len(read_marks(f.stem)) < 8]
    if missing or short:
        print("not locking. unmarked:", missing, "fewer than 8 points:", short)
        return 1
    LOCK.write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=1))
    print(f"locked {len(files)} mark files in {LOCK}")
    return 0


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Mark pairs</title><style>
body{margin:0;font:14px system-ui,sans-serif;background:#f4f1ea;color:#222}
header{display:flex;gap:12px;align-items:center;padding:8px 12px;background:#e8e2d4;flex-wrap:wrap}
main{display:flex;gap:8px;padding:8px;height:calc(100vh - 140px)}
.pane{flex:1;position:relative;background:#ddd;overflow:hidden;cursor:crosshair}
.pane canvas{position:absolute;inset:0;width:100%;height:100%}
.pane .lab{position:absolute;left:8px;top:6px;background:#fffc;padding:2px 6px;border-radius:3px;z-index:2}
.pane.next{outline:4px solid #b5482a}
footer{padding:4px 12px;font-size:13px}
button{font:inherit;padding:4px 10px}select{font:inherit}
</style></head><body>
<header><select id="sel"></select><button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
<b id="info"></b><span id="hint"></span><button id="undo">undo (Z)</button><button id="clear">clear pair</button></header>
<main><div class="pane" id="ph"><span class="lab">historical</span><canvas></canvas></div>
<div class="pane" id="pm"><span class="lab">modern</span><canvas></canvas></div></main>
<footer id="foot"></footer>
<script>
const C=['#b5482a','#2e7d32','#c88a12','#6b4c9a','#00796b','#ad1457','#5d4037','#0277bd','#827717','#d84315','#37474f','#558b2f'];
let P=[],idx=0,pts=[],pending=null,locked=false;const V={h:{s:1,x:0,y:0,img:new Image()},m:{s:1,x:0,y:0,img:new Image()}};
const panes={h:document.querySelector('#ph'),m:document.querySelector('#pm')};
async function j(u,o){const r=await fetch(u,o);if(!r.ok)throw new Error(await r.text());return r.json()}
function fit(k){const v=V[k],c=panes[k].getBoundingClientRect();v.s=Math.min(c.width/v.img.width,c.height/v.img.height);v.x=(c.width-v.img.width*v.s)/2;v.y=(c.height-v.img.height*v.s)/2}
function draw(k){const cv=panes[k].querySelector('canvas'),r=panes[k].getBoundingClientRect();cv.width=r.width;cv.height=r.height;
 const g=cv.getContext('2d'),v=V[k];g.drawImage(v.img,v.x,v.y,v.img.width*v.s,v.img.height*v.s);
 const all=pts.slice();if(pending)all.push(pending);
 all.forEach((p,i)=>{const X=k=='h'?p.hist_x:p.modern_x,Y=k=='h'?p.hist_y:p.modern_y;if(X==null)return;
  const sx=v.x+(X+.5)*v.s,sy=v.y+(Y+.5)*v.s;g.strokeStyle=C[i%C.length];g.lineWidth=2;g.beginPath();
  g.moveTo(sx-9,sy);g.lineTo(sx+9,sy);g.moveTo(sx,sy-9);g.lineTo(sx,sy+9);g.stroke();g.fillStyle=C[i%C.length];
  g.font='bold 13px system-ui';g.fillText(i+1,sx+6,sy-6)});}
function redraw(){draw('h');draw('m');panes.h.classList.toggle('next',!pending);panes.m.classList.toggle('next',!!pending);
 document.querySelector('#hint').textContent=locked?'LOCKED':pending?'now click the SAME point in the modern photo':'click a point in the historical image';
 const p=P[idx];document.querySelector('#info').textContent=`${idx+1}/${P.length}  ${p.pair_id}  [${p.style}]  ${pts.length} points`;
 document.querySelector('#foot').textContent=`${p.landmark}, ${p.city}. ${p.notes||''}  Aim for 8-10 points spread over the building.`}
async function load(i){idx=(i+P.length)%P.length;const p=P[idx];document.querySelector('#sel').value=idx;pending=null;
 pts=(await j('/api/marks/'+p.pair_id)).map(r=>({hist_x:+r.hist_x,hist_y:+r.hist_y,modern_x:+r.modern_x,modern_y:+r.modern_y,description:r.description||''}));
 await Promise.all(['h','m'].map(k=>new Promise(res=>{V[k].img.onload=()=>{fit(k);res()};V[k].img.src=`/img/${p.pair_id}/${k=='h'?'historical':'modern'}`})));redraw()}
async function save(){const p=P[idx];await j('/api/marks/'+p.pair_id,{method:'POST',body:JSON.stringify(pts)});
 const s=await j('/api/pairs');P.forEach((q,i)=>q.n=s[i].n);fillSel()}
function fillSel(){document.querySelector('#sel').innerHTML=P.map((p,i)=>`<option value=${i}>${p.n>=8?'✓':'  '} ${p.pair_id} (${p.n})</option>`).join('');document.querySelector('#sel').value=idx}
for(const k of ['h','m']){const el=panes[k];let drag=null,moved=false;
 el.addEventListener('wheel',e=>{e.preventDefault();const r=el.getBoundingClientRect(),v=V[k],f=e.deltaY<0?1.25:0.8,mx=e.clientX-r.left,my=e.clientY-r.top;
  v.x=mx-(mx-v.x)*f;v.y=my-(my-v.y)*f;v.s*=f;draw(k)},{passive:false});
 el.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,vx:V[k].x,vy:V[k].y};moved=false});
 window.addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)moved=true;
  if(moved){V[k].x=drag.vx+dx;V[k].y=drag.vy+dy;draw(k)}});
 window.addEventListener('mouseup',async e=>{if(!drag)return;const wasDrag=moved;drag=null;if(wasDrag||locked||e.target.closest('.pane')!==el)return;
  const r=el.getBoundingClientRect(),v=V[k],X=(e.clientX-r.left-v.x)/v.s-.5,Y=(e.clientY-r.top-v.y)/v.s-.5;
  if(X<-.5||Y<-.5||X>v.img.width-.5||Y>v.img.height-.5)return;
  if(k=='h'&&!pending){pending={hist_x:X,hist_y:Y,modern_x:null,modern_y:null,description:''}}
  else if(k=='m'&&pending){pending.modern_x=X;pending.modern_y=Y;pts.push(pending);pending=null;await save()}
  redraw()})}
async function undo(){if(pending){pending=null}else if(pts.length){pts.pop();await save()}redraw()}
document.addEventListener('keydown',e=>{if(e.key=='z'||e.key=='Z')undo()});
document.querySelector('#undo').onclick=undo;
document.querySelector('#clear').onclick=async()=>{if(confirm('Remove all points for this pair?')){pts=[];pending=null;await save();redraw()}};
document.querySelector('#prev').onclick=()=>load(idx-1);document.querySelector('#next').onclick=()=>load(idx+1);
document.querySelector('#sel').onchange=e=>load(+e.target.value);window.onresize=()=>{fit('h');fit('m');redraw()};
(async()=>{const s=await j('/api/pairs');P=s;locked=s.length&&s[0].locked;fillSel();const first=P.findIndex(p=>p.n<8);await load(first<0?0:first)})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parts = self.path.strip("/").split("/")
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if parts[:2] == ["api", "pairs"]:
            locked = LOCK.exists()
            out = [{**{k: r[k] for k in ("pair_id", "style", "landmark", "city", "notes")},
                    "n": len(read_marks(r["pair_id"])), "locked": locked} for r in pairs()]
            return self._send(200, json.dumps(out).encode())
        if parts[:2] == ["api", "marks"] and len(parts) == 3:
            return self._send(200, json.dumps(read_marks(parts[2])).encode())
        if parts[0] == "img" and len(parts) == 3 and parts[2] in ("historical", "modern"):
            p = image_path(parts[1], parts[2])
            if p.exists():
                return self._send(200, p.read_bytes(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if parts[:2] == ["api", "marks"] and len(parts) == 3 and parts[2] in {r["pair_id"] for r in pairs()}:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                write_marks(parts[2], body)
            except PermissionError as e:
                return self._send(403, str(e).encode(), "text/plain")
            return self._send(200, b"{}")
        self._send(404, b"not found", "text/plain")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--lock", action="store_true", help="freeze the marks (SHA-256 of every file into marks/LOCK.json)")
    args = ap.parse_args(argv)
    MARKS.mkdir(exist_ok=True)
    if args.lock:
        return lock()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Marking {len(pairs())} pairs at {url}  (Ctrl+C to stop; every point is saved as you go)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
