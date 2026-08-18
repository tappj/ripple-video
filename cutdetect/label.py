# Ruff measures embedded HTML/CSS/JS source lines as Python. Intentional
# typography and asset line lengths are kept intact for readability.
# ruff: noqa: E501, RUF001
"""Local, browser-based ground-truth labeling tool."""

from __future__ import annotations

import array
import json
import subprocess
import sys
import threading
import time
import wave
import webbrowser
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from cutdetect.config import IngestConfig, LabelConfig
from cutdetect.ingest import IngestError, VideoContext, ingest_video


def _run_ffmpeg(command: Sequence[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise IngestError(f"thumbnail extraction failed: {detail}")


def prepare_thumbnails(context: VideoContext, config: LabelConfig) -> Path:
    """Extract one small JPEG for every working-domain frame in one pass."""
    directory = context.artifact_dir / "label_thumbnails"
    existing = list(directory.glob("frame_*.jpg")) if directory.is_dir() else []
    if len(existing) == context.frame_count:
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    for stale in existing:
        stale.unlink()
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-autorotate",
            "-i",
            str(context.working_video_path),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={config.thumbnail_width_px}:-2",
            "-fps_mode",
            "passthrough",
            "-q:v",
            str(config.thumbnail_jpeg_quality),
            "-start_number",
            "0",
            str(directory / "frame_%06d.jpg"),
        ]
    )
    generated = list(directory.glob("frame_*.jpg"))
    if len(generated) != context.frame_count:
        raise IngestError(
            f"expected {context.frame_count} thumbnails but ffmpeg generated {len(generated)}"
        )
    return directory


def extract_waveform(context: VideoContext, config: LabelConfig) -> dict[str, object]:
    """Return a compact peak envelope used for one-second review windows."""
    if context.audio_path is None:
        return {"sample_rate": 0, "points_per_sec": config.waveform_points, "peaks": []}
    with wave.open(str(context.audio_path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        if channels != 1 or sample_width != 2:
            raise IngestError("label waveform expects the canonical mono pcm_s16le audio")
        samples = array.array("h", audio.readframes(audio.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    samples_per_point = max(1, round(sample_rate / config.waveform_points))
    peaks = [
        max((abs(value) for value in samples[start : start + samples_per_point]), default=0)
        / 32768.0
        for start in range(0, len(samples), samples_per_point)
    ]
    return {
        "sample_rate": sample_rate,
        "points_per_sec": config.waveform_points,
        "peaks": peaks,
    }


def render_labeler_html(context: VideoContext, config: LabelConfig) -> str:
    """Render the complete labeling interface without a frontend build step."""
    metadata = json.dumps(
        {
            "frameCount": context.frame_count,
            "fps": float(context.fps),
            "columns": max(1, round(float(context.fps))),
            "reviewRadius": config.review_radius_frames,
            "waveformWindowSec": config.waveform_window_sec,
            "sourceTimes": context.original_timestamps_sec,
            "sourceName": context.source_path.name,
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cutdetect labeler</title>
<style>
:root {{ color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #090d15; color: #e5edf8; }}
header {{ position: sticky; top: 0; z-index: 4; padding: 12px 18px; background: #111827ee; border-bottom: 1px solid #334155; }}
h1 {{ margin: 0 0 8px; font-size: 18px; }}
button, select {{ color: inherit; background: #1e293b; border: 1px solid #475569; border-radius: 5px; padding: 7px 10px; }}
button:hover, button:focus {{ border-color: #38bdf8; }}
button.active {{ color: #04131c; background: #38bdf8; }}
.toolbar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.status {{ color: #93c5fd; margin-left: auto; }}
main {{ padding: 16px; }}
.hidden {{ display: none !important; }}
.second {{ display: grid; grid-template-columns: repeat(var(--columns), {config.thumbnail_width_px}px); gap: 1px; margin: 0 0 7px; align-items: start; }}
.second-label {{ grid-column: 1 / -1; color: #94a3b8; font-size: 11px; }}
.frame {{ padding: 0; border: 2px solid transparent; border-radius: 2px; background: #020617; position: relative; }}
.frame img {{ width: {config.thumbnail_width_px}px; display: block; }}
.frame span {{ position: absolute; inset: auto 1px 1px auto; padding: 1px 2px; color: white; background: #020617c7; font-size: 8px; }}
.frame.labeled {{ border-color: #fb7185; }}
.frame.unsure {{ border-color: #fbbf24; }}
#reviewStrip {{ display: flex; gap: 8px; justify-content: center; overflow-x: auto; padding: 18px 0; }}
.review-frame {{ border: 2px solid #334155; text-align: center; background: #020617; }}
.review-frame.current {{ border-color: #fb7185; }}
.review-frame img {{ height: min(44vh, 520px); display: block; }}
.review-frame div {{ padding: 5px; }}
#waveform {{ width: 100%; height: 160px; background: #020617; border: 1px solid #334155; }}
.review-actions {{ display: flex; gap: 8px; align-items: center; justify-content: center; margin: 14px 0; }}
.help {{ color: #94a3b8; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="toolbar">
    <button id="sweepTab" class="active">Sweep</button>
    <button id="reviewTab">Review</button>
    <button id="saveButton">Save labels.json</button>
    <button id="downloadButton">Download copy</button>
    <span id="counter">0 labels</span>
    <span class="status" id="status">Loading…</span>
  </div>
</header>
<main>
  <section id="sweepMode"><div id="sweep"></div></section>
  <section id="reviewMode" class="hidden">
    <div id="reviewStrip"></div>
    <canvas id="waveform"></canvas>
    <div class="review-actions">
      <button id="yesButton">Y · certain</button>
      <button id="unsureButton">U · unsure</button>
      <button id="noButton">N · not a cut</button>
      <button id="skipButton">S · skip</button>
      <select id="typeSelect">
        <option value="hard">hard</option>
        <option value="zoom_disguised">zoom_disguised</option>
        <option value="ambiguous">ambiguous</option>
      </select>
    </div>
    <p class="help">←/→ move · Y confirm · N reject · U unsure · S skip</p>
  </section>
</main>
<script>
const META = {metadata};
const labels = new Map();
let candidates = [];
let candidatePosition = 0;
let waveform = null;
const byId = id => document.getElementById(id);
const thumb = frame => `/thumbs/frame_${{String(frame).padStart(6, '0')}}.jpg`;
const clampedFrame = frame => Math.max(0, Math.min(META.frameCount - 1, frame));

function updateCounter() {{
  byId('counter').textContent = `${{labels.size}} label${{labels.size === 1 ? '' : 's'}}`;
}}
function refreshFrame(frame) {{
  const element = document.querySelector(`.frame[data-frame="${{frame}}"]`);
  if (!element) return;
  const label = labels.get(frame);
  element.classList.toggle('labeled', Boolean(label));
  element.classList.toggle('unsure', label?.confidence === 'unsure');
}}
function setLabel(frame, confidence) {{
  const previous = labels.get(frame);
  labels.set(frame, {{
    frame,
    time: META.sourceTimes[frame] ?? frame / META.fps,
    confidence,
    type: previous?.type ?? byId('typeSelect').value,
  }});
  if (!candidates.includes(frame)) candidates.push(frame);
  candidates.sort((a, b) => a - b);
  candidatePosition = candidates.indexOf(frame);
  refreshFrame(frame);
  updateCounter();
}}
function removeLabel(frame) {{
  labels.delete(frame);
  refreshFrame(frame);
  updateCounter();
}}
function buildSweep() {{
  const root = byId('sweep');
  for (let start = 0; start < META.frameCount; start += META.columns) {{
    const row = document.createElement('div');
    row.className = 'second';
    row.style.setProperty('--columns', META.columns);
    const caption = document.createElement('div');
    caption.className = 'second-label';
    caption.textContent = `${{(start / META.fps).toFixed(2)}}s · frames ${{start}}–${{Math.min(start + META.columns - 1, META.frameCount - 1)}}`;
    row.append(caption);
    for (let frame = start; frame < Math.min(start + META.columns, META.frameCount); frame++) {{
      const button = document.createElement('button');
      button.className = 'frame';
      button.dataset.frame = frame;
      button.innerHTML = `<img loading="lazy" src="${{thumb(frame)}}" alt="frame ${{frame}}"><span>${{frame}}</span>`;
      button.addEventListener('click', () => {{
        if (labels.has(frame)) removeLabel(frame); else setLabel(frame, 'certain');
        candidates = [...new Set([...candidates, frame])].sort((a, b) => a - b);
        candidatePosition = candidates.indexOf(frame);
      }});
      row.append(button);
    }}
    root.append(row);
  }}
}}
function showMode(mode) {{
  const reviewing = mode === 'review';
  byId('sweepMode').classList.toggle('hidden', reviewing);
  byId('reviewMode').classList.toggle('hidden', !reviewing);
  byId('sweepTab').classList.toggle('active', !reviewing);
  byId('reviewTab').classList.toggle('active', reviewing);
  if (reviewing) renderReview();
}}
function currentFrame() {{ return candidates[candidatePosition] ?? [...labels.keys()][0] ?? 0; }}
function renderReview() {{
  const frame = currentFrame();
  const strip = byId('reviewStrip');
  strip.innerHTML = '';
  for (let offset = -META.reviewRadius; offset <= META.reviewRadius; offset++) {{
    const shown = clampedFrame(frame + offset);
    const cell = document.createElement('div');
    cell.className = `review-frame ${{offset === 0 ? 'current' : ''}}`;
    cell.innerHTML = `<img src="${{thumb(shown)}}" alt="frame ${{shown}}"><div>${{shown}} · ${{(META.sourceTimes[shown] ?? shown / META.fps).toFixed(4)}}s</div>`;
    strip.append(cell);
  }}
  const label = labels.get(frame);
  if (label) byId('typeSelect').value = label.type;
  drawWaveform(frame);
  byId('status').textContent = `Candidate ${{candidatePosition + 1}}/${{Math.max(candidates.length, 1)}} · frame ${{frame}}`;
}}
function drawWaveform(frame) {{
  const canvas = byId('waveform');
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, canvas.clientWidth * scale);
  canvas.height = Math.max(1, canvas.clientHeight * scale);
  const context = canvas.getContext('2d');
  context.scale(scale, scale);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = '#38bdf8';
  context.beginPath();
  if (waveform?.peaks?.length) {{
    const centerTime = META.sourceTimes[frame] ?? frame / META.fps;
    const count = Math.round(waveform.points_per_sec * META.waveformWindowSec);
    const start = Math.round((centerTime - META.waveformWindowSec / 2) * waveform.points_per_sec);
    for (let index = 0; index < count; index++) {{
      const peak = waveform.peaks[start + index] ?? 0;
      const x = index * width / Math.max(count - 1, 1);
      context.moveTo(x, height / 2 - peak * height * 0.45);
      context.lineTo(x, height / 2 + peak * height * 0.45);
    }}
  }}
  context.stroke();
  context.strokeStyle = '#fb7185';
  context.beginPath(); context.moveTo(width / 2, 0); context.lineTo(width / 2, height); context.stroke();
}}
function move(delta) {{
  if (!candidates.length) return;
  candidatePosition = Math.max(0, Math.min(candidates.length - 1, candidatePosition + delta));
  renderReview();
}}
function decide(decision) {{
  const frame = currentFrame();
  if (decision === 'certain' || decision === 'unsure') setLabel(frame, decision);
  if (decision === 'no') removeLabel(frame);
  move(1);
}}
async function save() {{
  const payload = [...labels.values()].sort((a, b) => a.frame - b.frame);
  const response = await fetch('/labels', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(payload)}});
  if (!response.ok) throw new Error(await response.text());
  byId('status').textContent = `Saved ${{payload.length}} labels`;
}}
function download() {{
  const payload = JSON.stringify([...labels.values()].sort((a, b) => a.frame - b.frame), null, 2) + '\n';
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([payload], {{type: 'application/json'}}));
  link.download = 'labels.json'; link.click(); URL.revokeObjectURL(link.href);
}}
async function initialize() {{
  buildSweep();
  const [saved, wave] = await Promise.all([fetch('/labels').then(response => response.json()), fetch('/waveform').then(response => response.json())]);
  for (const label of saved) {{ labels.set(label.frame, label); candidates.push(label.frame); refreshFrame(label.frame); }}
  candidates.sort((a, b) => a - b);
  waveform = wave;
  updateCounter();
  byId('title').textContent = `cutdetect · ${{META.sourceName}} · ${{META.frameCount}} frames @ ${{META.fps.toFixed(3)}} fps`;
  byId('status').textContent = 'Ready';
}}
byId('sweepTab').onclick = () => showMode('sweep');
byId('reviewTab').onclick = () => showMode('review');
byId('yesButton').onclick = () => decide('certain');
byId('unsureButton').onclick = () => decide('unsure');
byId('noButton').onclick = () => decide('no');
byId('skipButton').onclick = () => decide('skip');
byId('saveButton').onclick = () => save().catch(error => byId('status').textContent = error);
byId('downloadButton').onclick = download;
byId('typeSelect').onchange = () => {{ const label = labels.get(currentFrame()); if (label) label.type = byId('typeSelect').value; }};
window.addEventListener('resize', () => {{ if (!byId('reviewMode').classList.contains('hidden')) drawWaveform(currentFrame()); }});
window.addEventListener('keydown', event => {{
  if (event.target.matches('select')) return;
  const key = event.key.toLowerCase();
  if (key === 'arrowleft') move(-1);
  if (key === 'arrowright') move(1);
  if (key === 'y') decide('certain');
  if (key === 'u') decide('unsure');
  if (key === 'n') decide('no');
  if (key === 's') decide('skip');
}});
initialize().catch(error => byId('status').textContent = error);
</script>
</body>
</html>"""


class _LabelServer(ThreadingHTTPServer):
    context: VideoContext
    config: LabelConfig
    labels_path: Path
    thumbnails_path: Path
    html: bytes
    waveform_json: bytes


class _LabelHandler(BaseHTTPRequestHandler):
    server: _LabelServer

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send(self.server.html, "text/html; charset=utf-8")
            return
        if route == "/labels":
            body = (
                self.server.labels_path.read_bytes()
                if self.server.labels_path.is_file()
                else b"[]\n"
            )
            self._send(body, "application/json")
            return
        if route == "/waveform":
            self._send(self.server.waveform_json, "application/json")
            return
        if route.startswith("/thumbs/frame_") and route.endswith(".jpg"):
            filename = Path(route).name
            digits = filename.removeprefix("frame_").removesuffix(".jpg")
            if digits.isdigit() and 0 <= int(digits) < self.server.context.frame_count:
                image_path = self.server.thumbnails_path / filename
                if image_path.is_file():
                    self._send(image_path.read_bytes(), "image/jpeg")
                    return
        self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/labels":
            self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > self.server.config.max_label_payload_bytes:
            self._send(b"payload too large\n", "text/plain", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
            if not isinstance(payload, list):
                raise ValueError("labels must be an array")
            validated = []
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("label must be an object")
                frame = int(item["frame"])
                confidence = str(item["confidence"])
                if frame < 0 or frame >= self.server.context.frame_count:
                    raise ValueError(f"frame out of range: {frame}")
                if confidence not in {"certain", "unsure"}:
                    raise ValueError(f"invalid confidence: {confidence}")
                validated.append(
                    {
                        "frame": frame,
                        "time": float(item["time"]),
                        "confidence": confidence,
                        "type": str(item.get("type", "hard")),
                    }
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)
            return
        validated.sort(key=lambda item: int(cast(int, item["frame"])))
        self.server.labels_path.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
        self._send(b'{"ok":true}\n', "application/json")

    def log_message(self, format: str, *args: object) -> None:
        return


def create_label_server(
    context: VideoContext,
    labels_path: str | Path,
    config: LabelConfig | None = None,
) -> _LabelServer:
    """Prepare assets and return the configured local HTTP server."""
    settings = config or LabelConfig()
    thumbnails = prepare_thumbnails(context, settings)
    server = _LabelServer((settings.host, settings.port), _LabelHandler)
    server.context = context
    server.config = settings
    server.labels_path = Path(labels_path).expanduser().resolve()
    server.labels_path.parent.mkdir(parents=True, exist_ok=True)
    server.thumbnails_path = thumbnails
    server.html = render_labeler_html(context, settings).encode()
    server.waveform_json = json.dumps(extract_waveform(context, settings)).encode()
    return server


def label_video(
    video_path: str | Path,
    *,
    labels_path: str | Path = "labels.json",
    ingest_config: IngestConfig | None = None,
    label_config: LabelConfig | None = None,
    open_browser: bool = True,
) -> None:
    """Ingest a video and run its local labeling UI until interrupted."""
    settings = label_config or LabelConfig()
    context = ingest_video(video_path, ingest_config)
    server = create_label_server(context, labels_path, settings)
    url = f"http://{settings.host}:{server.server_port}/"
    if open_browser:

        def open_after_delay() -> None:
            time.sleep(settings.browser_open_delay_sec)
            webbrowser.open(url)

        threading.Thread(
            target=open_after_delay,
            daemon=True,
        ).start()
    print(f"Labeler: {url}")
    print(f"Writing labels to: {Path(labels_path).expanduser().resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
