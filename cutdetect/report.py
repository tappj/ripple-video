# Embedded HTML/CSS/JS is kept on readable asset lines.
# ruff: noqa: E501
"""Self-contained Phase 5 HTML detection report."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


def _thumbnail_data(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def render_debug_report(
    predictions_path: str | Path,
    normalized_signals_path: str | Path,
    thumbnails_dir: str | Path,
    labels_path: str | Path | None = None,
) -> str:
    """Render a portable timeline, signal plot, and detection thumbnail strip."""
    prediction_file = Path(predictions_path)
    contract = json.loads(prediction_file.read_text(encoding="utf-8"))
    with np.load(normalized_signals_path, allow_pickle=False) as archive:
        traces = {
            name: archive[name].astype(float).tolist()
            for name in archive.files
            if name not in {"agreement", "available_weight"}
        }
    thumbs = Path(thumbnails_dir)
    labels: list[dict[str, object]] = []
    if labels_path is not None:
        label_root: object = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        if isinstance(label_root, dict):
            label_root = label_root.get("labels", [])
        if isinstance(label_root, list):
            labels = [item for item in label_root if isinstance(item, dict)]
    cuts = []
    for item in contract["cuts"]:
        frame = int(item["frame"])
        before = thumbs / f"frame_{max(0, frame - 1):06d}.jpg"
        after = thumbs / f"frame_{frame:06d}.jpg"
        cuts.append(
            {
                **item,
                "before": _thumbnail_data(before) if before.is_file() else "",
                "after": _thumbnail_data(after) if after.is_file() else "",
            }
        )
    payload = json.dumps(
        {
            "video": contract["video"],
            "cuts": cuts,
            "diagnostics": contract["diagnostics"],
            "traces": traces,
            "labels": labels,
        }
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cutdetect · debug report</title><style>
:root {{ color-scheme: dark; font-family: "Avenir Next", Avenir, sans-serif; --paper:#efe9dc; --ink:#151515; --red:#ff4b33; --blue:#37b7e8; }}
* {{ box-sizing:border-box }} body {{ margin:0;background:#111;color:var(--paper) }}
header {{ padding:48px clamp(24px,7vw,100px) 30px;border-bottom:1px solid #444 }} h1 {{ margin:0;font:800 clamp(44px,9vw,110px)/.8 Georgia,serif;letter-spacing:-.07em }}
.meta {{ display:flex;gap:30px;margin-top:28px;font-family:monospace;text-transform:uppercase }} main {{ padding:38px clamp(24px,7vw,100px) 100px }}
canvas {{ width:100%;height:430px;background:#171717;border-block:1px solid #444 }} h2 {{ font:700 32px Georgia,serif;margin:60px 0 20px }}
.cuts {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:2px;background:#444 }} article {{ background:#171717;padding:16px }}
.pair {{ display:grid;grid-template-columns:1fr 1fr;gap:2px }} img {{ width:100%;display:block }} .stamp {{ color:var(--red);font:700 24px monospace;margin:12px 0 4px }}
small {{ color:#aaa;font-family:monospace }} pre {{ white-space:pre-wrap;color:#aaa }}
</style></head><body><header><h1>CUT<br>REPORT</h1><div class="meta" id="meta"></div></header><main>
<h2>Signal timeline</h2><canvas id="plot"></canvas><h2>Detected boundaries</h2><section class="cuts" id="cuts"></section>
<h2>Diagnostics</h2><pre id="diagnostics"></pre></main><script>
const DATA={payload};const meta=document.querySelector('#meta');meta.textContent=`${{DATA.video.frame_count}} frames · ${{DATA.video.fps}} fps · ${{DATA.cuts.length}} cuts${{DATA.labels.length ? ` · ${{DATA.labels.length}} labels` : ''}}`;
const root=document.querySelector('#cuts');for(const cut of DATA.cuts){{const item=document.createElement('article');item.innerHTML=`<div class="pair"><img src="${{cut.before}}"><img src="${{cut.after}}"></div><div class="stamp">${{cut.time_sec.toFixed(3)}}s</div><small>FRAME ${{cut.frame}} · CONF ${{cut.confidence.toFixed(3)}} · ${{cut.agreement_count}} SIGNALS</small>`;root.append(item)}}
document.querySelector('#diagnostics').textContent=JSON.stringify(DATA.diagnostics,null,2);const canvas=document.querySelector('#plot');const dpr=devicePixelRatio||1;canvas.width=canvas.clientWidth*dpr;canvas.height=canvas.clientHeight*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);const w=canvas.clientWidth,h=canvas.clientHeight;
const names=Object.keys(DATA.traces);names.forEach((name,row)=>{{const vals=DATA.traces[name],y0=(row+.5)*h/names.length;c.strokeStyle=name==='fused'?'#ff4b33':'#37b7e8';c.globalAlpha=name==='fused'?1:.35;c.beginPath();vals.forEach((v,i)=>{{const x=i*w/(vals.length-1),y=y0-(Number.isFinite(v)?v:0)*h/names.length*.45;i?c.lineTo(x,y):c.moveTo(x,y)}});c.stroke()}});c.globalAlpha=1;c.strokeStyle='#efe9dc';for(const cut of DATA.cuts){{const x=(cut.frame-1)*w/(DATA.video.frame_count-2);c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}}c.strokeStyle='#ff4b33';c.lineWidth=2;for(const label of DATA.labels){{const x=(label.frame-1)*w/(DATA.video.frame_count-2);c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}}
</script></body></html>"""


def write_debug_report(
    predictions_path: str | Path,
    normalized_signals_path: str | Path,
    thumbnails_dir: str | Path,
    output_path: str | Path,
    labels_path: str | Path | None = None,
) -> Path:
    """Write a self-contained HTML report and return its absolute path."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_debug_report(predictions_path, normalized_signals_path, thumbnails_dir, labels_path),
        encoding="utf-8",
    )
    return destination
