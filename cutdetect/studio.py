# Embedded application assets intentionally preserve readable HTML/CSS/JS lines.
# ruff: noqa: E501, RUF001
"""Local drag-and-drop studio for detecting and exporting jump-cut clips."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, unquote, urlparse

from cutdetect.config import IngestConfig, LabelConfig, StudioConfig
from cutdetect.detect import run_detection
from cutdetect.export import split_from_predictions
from cutdetect.ingest import ingest_video
from cutdetect.label import prepare_thumbnails
from cutdetect.report import write_debug_report


def render_studio_html(default_sensitivity: float) -> str:
    """Render the dependency-free local Cut Room application."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cut Room · jump-cut studio</title><style>
:root{{--ink:#121210;--paper:#f0eadc;--signal:#ff5138;--cyan:#41c1ea;--muted:#79766f;--line:#c9c1b2;color-scheme:light;font-family:"Avenir Next",Avenir,"Trebuchet MS",sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);min-height:100vh;background-image:linear-gradient(90deg,transparent 49.8%,#1616160b 50%,transparent 50.2%)}}
body:after{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.95' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.13'/%3E%3C/svg%3E")}}
header{{display:flex;justify-content:space-between;align-items:flex-start;padding:24px 4vw;border-bottom:1px solid var(--ink)}}.brand{{font:900 19px/1 monospace;letter-spacing:-.08em}}.brand b{{color:var(--signal)}}.phase{{font:700 10px/1.2 monospace;letter-spacing:.16em;text-align:right;text-transform:uppercase}}
main{{padding:clamp(30px,6vw,88px) 4vw 100px}}.hero{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(280px,.6fr);gap:5vw;align-items:end;margin-bottom:50px}}h1{{font:900 clamp(68px,12.5vw,190px)/.72 Georgia,"Times New Roman",serif;letter-spacing:-.085em;margin:0;text-transform:uppercase}}h1 em{{display:block;color:var(--signal);font-style:italic;margin-left:12vw}}.intro{{font-size:clamp(16px,1.6vw,24px);line-height:1.25;border-top:4px solid var(--ink);padding-top:18px;max-width:440px}}
#drop{{position:relative;min-height:270px;border:2px dashed var(--ink);display:grid;place-items:center;text-align:center;cursor:pointer;overflow:hidden;transition:.25s cubic-bezier(.2,.8,.2,1)}}#drop:hover,#drop.drag{{background:var(--ink);color:var(--paper);transform:rotate(-.35deg) scale(1.005)}}#drop:before{{content:"DROP";position:absolute;font:900 22vw/.7 Georgia,serif;color:var(--signal);opacity:.07;letter-spacing:-.1em}}#drop strong{{font:800 clamp(25px,4vw,54px)/.9 Georgia,serif;z-index:1}}#drop span{{display:block;font:700 11px monospace;letter-spacing:.15em;margin-top:18px;text-transform:uppercase}}input[type=file]{{display:none}}
.control{{display:flex;align-items:center;gap:18px;margin:22px 0 0;font:700 11px monospace;text-transform:uppercase;letter-spacing:.1em}}input[type=range]{{accent-color:var(--signal);width:min(320px,45vw)}}#sensitivityValue{{font-size:20px;color:var(--signal)}}
#progress,#results{{display:none}}#progress.active,#results.active{{display:block}}#progress{{margin-top:70px;border-top:1px solid var(--ink);padding-top:24px}}.progress-head{{display:flex;justify-content:space-between;font:700 12px monospace;text-transform:uppercase}}.rail{{height:18px;background:#d9d2c5;margin-top:14px;position:relative;overflow:hidden}}.rail i{{display:block;height:100%;width:4%;background:var(--signal);transition:width .7s ease}}#stage{{font:800 clamp(34px,6vw,78px)/1 Georgia,serif;margin:25px 0 0}}
#results{{margin-top:72px}}.result-head{{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end;border-bottom:1px solid var(--ink);padding-bottom:20px}}h2{{font:900 clamp(45px,8vw,110px)/.8 Georgia,serif;letter-spacing:-.07em;margin:0}}.stats{{display:flex;gap:30px;text-align:right}}.stats b{{font:900 40px/1 Georgia,serif;color:var(--signal);display:block}}.stats span{{font:700 9px monospace;text-transform:uppercase;letter-spacing:.14em}}.actions{{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}}a.button,button.reset{{appearance:none;border:1px solid var(--ink);background:transparent;color:var(--ink);padding:13px 18px;font:800 11px monospace;text-transform:uppercase;text-decoration:none;cursor:pointer}}a.button.primary{{background:var(--ink);color:var(--paper)}}a.button:hover,button.reset:hover{{background:var(--signal);border-color:var(--signal);color:white}}
.timeline{{display:flex;height:44px;margin:34px 0 20px;border:1px solid var(--ink)}}.timeline i{{display:block;border-right:1px solid var(--ink);background:var(--cyan);opacity:.65;min-width:2px;position:relative}}.timeline i:nth-child(even){{background:var(--signal)}}.timeline i:hover{{opacity:1}}.clip-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1px;background:var(--ink);border:1px solid var(--ink)}}article{{background:var(--paper);padding:12px;animation:arrive .45s both}}article video{{display:block;width:100%;aspect-ratio:9/16;object-fit:cover;background:#111}}.clip-meta{{display:flex;justify-content:space-between;align-items:flex-end;margin-top:12px}}.clip-meta b{{font:900 24px Georgia,serif}}.clip-meta span{{font:700 10px monospace;color:var(--muted)}}.clip-meta a{{font:800 10px monospace;color:var(--signal);text-transform:uppercase}}.error{{color:var(--signal);white-space:pre-wrap;font-family:monospace}}
@keyframes arrive{{from{{opacity:0;transform:translateY(20px)}}}}@media(max-width:760px){{.hero{{grid-template-columns:1fr}}h1 em{{margin-left:5vw}}.result-head{{grid-template-columns:1fr}}.stats{{text-align:left}}}}
</style></head><body><header><div class="brand"><b>///</b> CUT ROOM</div><div class="phase">Local processing<br>No uploads leave this machine</div></header><main>
<section class="hero"><h1>Find<br><em>the cut.</em></h1><p class="intro">Drop a talking-head video. The detector finds physically impossible motion, verifies it against audio, and returns every beat as a separate edit-ready clip.</p></section>
<label id="drop" for="videoInput"><div><strong>Drop your video here</strong><span>or click to choose · MP4 / MOV / M4V / WEBM</span></div></label><input id="videoInput" type="file" accept="video/*,.mp4,.mov,.m4v,.webm">
<div class="control"><label for="sensitivity">Detection sensitivity</label><input id="sensitivity" type="range" min="0" max="1" step=".01" value="{default_sensitivity}"><output id="sensitivityValue">{default_sensitivity:.2f}</output><span>Low precision ← → high recall</span></div>
<section id="progress"><div class="progress-head"><span id="filename"></span><span id="percent">0%</span></div><div class="rail"><i id="bar"></i></div><div id="stage">Reading frames…</div><p class="error" id="error"></p></section>
<section id="results"><div class="result-head"><h2>Your cut<br>is on the table.</h2><div class="stats"><div><b id="cutCount">0</b><span>Cut points</span></div><div><b id="clipCount">0</b><span>Clips</span></div></div></div><div class="actions"><a class="button primary" id="zip">Download all clips</a><a class="button" id="report" target="_blank">Open debug report</a><a class="button" id="contract" target="_blank">Prediction JSON</a><button class="reset" id="reset">Cut another video</button></div><div class="timeline" id="timeline"></div><div class="clip-grid" id="clips"></div></section>
</main><script>
const input=document.querySelector('#videoInput'),drop=document.querySelector('#drop'),sensitivity=document.querySelector('#sensitivity'),value=document.querySelector('#sensitivityValue'),progress=document.querySelector('#progress'),results=document.querySelector('#results'),bar=document.querySelector('#bar'),percent=document.querySelector('#percent'),stage=document.querySelector('#stage'),error=document.querySelector('#error');let timer;
value.value=Number(sensitivity.value).toFixed(2);sensitivity.oninput=()=>value.value=Number(sensitivity.value).toFixed(2);['dragenter','dragover'].forEach(name=>drop.addEventListener(name,e=>{{e.preventDefault();drop.classList.add('drag')}}));['dragleave','drop'].forEach(name=>drop.addEventListener(name,e=>{{e.preventDefault();drop.classList.remove('drag')}}));drop.addEventListener('drop',e=>{{const file=e.dataTransfer.files[0];if(file)process(file)}});input.onchange=()=>{{if(input.files[0])process(input.files[0])}};
const stages=['Reading frame timestamps…','Tracking face geometry…','Listening for audio splices…','Fusing boundary evidence…','Encoding exact frame ranges…','Packing your clips…'];function animate(){{let n=4,i=0;bar.style.width=n+'%';percent.textContent=n+'%';stage.textContent=stages[0];timer=setInterval(()=>{{n=Math.min(92,n+Math.max(1,(94-n)*.035));i=Math.min(stages.length-1,Math.floor(n/17));bar.style.width=n+'%';percent.textContent=Math.floor(n)+'%';stage.textContent=stages[i]}},700)}}
async function process(file){{clearInterval(timer);results.classList.remove('active');progress.classList.add('active');drop.style.display='none';error.textContent='';document.querySelector('#filename').textContent=file.name;animate();try{{const response=await fetch(`/api/process?sensitivity=${{sensitivity.value}}`,{{method:'POST',headers:{{'Content-Type':file.type||'application/octet-stream','X-Filename':encodeURIComponent(file.name)}},body:file}});const payload=await response.json();if(!response.ok)throw new Error(payload.error||'Processing failed');clearInterval(timer);bar.style.width='100%';percent.textContent='100%';stage.textContent='Cut complete.';setTimeout(()=>show(payload),350)}}catch(reason){{clearInterval(timer);error.textContent=reason.message;stage.textContent='The cut hit a snag.';drop.style.display='grid'}}}}
function show(data){{results.classList.add('active');document.querySelector('#cutCount').textContent=data.cut_count;document.querySelector('#clipCount').textContent=data.clip_count;document.querySelector('#zip').href=data.zip_url;document.querySelector('#report').href=data.report_url;document.querySelector('#contract').href=data.predictions_url;const timeline=document.querySelector('#timeline');timeline.innerHTML='';const clips=document.querySelector('#clips');clips.innerHTML='';data.clips.forEach((clip,index)=>{{const tick=document.createElement('i');tick.style.flex=clip.frame_count;tick.title=`Clip ${{clip.index}} · ${{clip.duration_sec.toFixed(3)}}s`;timeline.append(tick);const card=document.createElement('article');card.style.animationDelay=`${{index*.035}}s`;card.innerHTML=`<video controls preload="metadata" playsinline src="${{clip.url}}"></video><div class="clip-meta"><div><b>${{String(clip.index).padStart(2,'0')}}</b><br><span>${{clip.start_sec.toFixed(3)}}–${{clip.end_sec.toFixed(3)}}s · ${{clip.frame_count}}f</span></div><a href="${{clip.url}}" download>Download</a></div>`;clips.append(card)}});results.scrollIntoView({{behavior:'smooth'}})}}
document.querySelector('#reset').onclick=()=>{{progress.classList.remove('active');results.classList.remove('active');drop.style.display='grid';input.value='';window.scrollTo({{top:0,behavior:'smooth'}})}};
</script></body></html>"""


class _StudioServer(ThreadingHTTPServer):
    config: StudioConfig
    cache_dir: Path
    jobs_dir: Path
    html: bytes


class _StudioHandler(BaseHTTPRequestHandler):
    server: _StudioServer

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send((json.dumps(value) + "\n").encode(), "application/json; charset=utf-8", status)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw = range_header.removeprefix("bytes=").split(",", 1)[0]
            start_raw, end_raw = raw.split("-", 1)
            start = int(start_raw or 0)
            end = min(size - 1, int(end_raw) if end_raw else size - 1)
            if start < 0 or start > end:
                self._send(
                    b"invalid range\n", "text/plain", HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
                )
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(end - start + 1))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as handle:
            while chunk := handle.read(self.server.config.upload_chunk_bytes):
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        route = unquote(urlparse(self.path).path)
        if route == "/":
            self._send(self.server.html, "text/html; charset=utf-8")
            return
        parts = Path(route.lstrip("/")).parts
        if len(parts) >= 3 and parts[0] == "jobs":
            job_id = parts[1]
            if len(job_id) == 32 and all(character in "0123456789abcdef" for character in job_id):
                candidate = self.server.jobs_dir.joinpath(*parts[1:]).resolve()
                if candidate.is_relative_to(self.server.jobs_dir):
                    self._serve_file(candidate)
                    return
        self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/process":
            self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > self.server.config.max_upload_bytes:
            self._json(
                {"error": "video is empty or exceeds the 2 GiB local limit"}, HTTPStatus.BAD_REQUEST
            )
            return
        filename = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name
        if Path(filename).suffix.lower() not in {".mp4", ".mov", ".m4v", ".webm"}:
            self._json({"error": "use an MP4, MOV, M4V, or WEBM video"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            sensitivity = float(parse_qs(parsed.query).get("sensitivity", ["0.5"])[0])
            if not 0.0 <= sensitivity <= 1.0:
                raise ValueError("sensitivity must be between 0 and 1")
            job_id = uuid.uuid4().hex
            job_dir = self.server.jobs_dir / job_id
            job_dir.mkdir(parents=True)
            upload = job_dir / filename
            remaining = length
            with upload.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(remaining, self.server.config.upload_chunk_bytes))
                    if not chunk:
                        raise ValueError("upload ended unexpectedly")
                    output.write(chunk)
                    remaining -= len(chunk)
            detection_dir = job_dir / "detection"
            run = run_detection(
                upload,
                detection_dir,
                cache_dir=self.server.cache_dir,
                sensitivity=sensitivity,
            )
            exported = split_from_predictions(
                upload,
                run.predictions_path,
                job_dir / "clips",
                cache_dir=self.server.cache_dir,
            )
            context = ingest_video(upload, IngestConfig(cache_dir=self.server.cache_dir))
            thumbnails = prepare_thumbnails(context, LabelConfig())
            report = write_debug_report(
                run.predictions_path,
                run.normalized_signals_path,
                thumbnails,
                job_dir / "report.html",
            )
            payload = exported.to_dict()
            clips = cast(list[dict[str, object]], payload["clips"])
            for clip in clips:
                clip["url"] = f"/jobs/{job_id}/clips/{clip['filename']}"
            payload.update(
                {
                    "job_id": job_id,
                    "zip_url": f"/jobs/{job_id}/clips/{exported.zip_path.name}",
                    "report_url": f"/jobs/{job_id}/{report.name}",
                    "predictions_url": f"/jobs/{job_id}/detection/{run.predictions_path.name}",
                }
            )
            self._json(payload)
        except Exception as error:  # server boundary converts failures to API responses
            self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_studio_server(
    config: StudioConfig | None = None,
    *,
    cache_dir: str | Path = ".cutdetect/cache",
    jobs_dir: str | Path = ".cutdetect/studio/jobs",
) -> _StudioServer:
    """Create the configured local studio server without starting it."""
    settings = config or StudioConfig()
    server = _StudioServer((settings.host, settings.port), _StudioHandler)
    server.config = settings
    server.cache_dir = Path(cache_dir).expanduser().resolve()
    server.jobs_dir = Path(jobs_dir).expanduser().resolve()
    server.cache_dir.mkdir(parents=True, exist_ok=True)
    server.jobs_dir.mkdir(parents=True, exist_ok=True)
    server.html = render_studio_html(settings.default_sensitivity).encode()
    return server


def run_studio(
    *,
    config: StudioConfig | None = None,
    cache_dir: str | Path = ".cutdetect/cache",
    jobs_dir: str | Path = ".cutdetect/studio/jobs",
    open_browser: bool = True,
) -> None:
    """Start the Cut Room until interrupted."""
    settings = config or StudioConfig()
    server = create_studio_server(settings, cache_dir=cache_dir, jobs_dir=jobs_dir)
    url = f"http://{settings.host}:{server.server_port}/"
    if open_browser:

        def open_after_delay() -> None:
            time.sleep(settings.browser_open_delay_sec)
            webbrowser.open(url)

        threading.Thread(target=open_after_delay, daemon=True).start()
    print(f"Cut Room: {url}")
    print(f"Exports: {server.jobs_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
