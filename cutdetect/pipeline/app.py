# Embedded application assets intentionally preserve readable HTML/CSS/JS lines.
# ruff: noqa: E501, RUF001
"""Local Phase E node-graph application for generation, review, and delivery."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, unquote, urlparse

from cutdetect.ingest import probe_video
from cutdetect.pipeline.audio_pipeline import (
    RUNWAY_PRESET_VOICES,
    RunwayAudioProcessor,
    validate_voice_preset,
)
from cutdetect.pipeline.capabilities import MODEL_CAPABILITIES
from cutdetect.pipeline.grouping import requires_cut_partition
from cutdetect.pipeline.orchestration import (
    DIRECT_API_ROUTE,
    GenerationGateway,
    JobState,
    PhaseCStore,
    PhaseCWorker,
    SegmentState,
    format_sse_events,
    job_status,
    prepare_phase_c_job,
)
from cutdetect.pipeline.review import ReviewService, prepare_review_proxy
from cutdetect.pipeline.ripple_ui import render_ripple_html
from cutdetect.pipeline.runway_client import (
    MODEL_ROUTER_ROUTE_PREFIX,
    JsonlCallLogger,
    PipelineError,
    RunwayDirectGateway,
    RunwayReferenceModel,
    RunwayRouterGateway,
    model_router_route,
    router_config_id_from_route,
)
from cutdetect.pipeline.stitch import stitch_job
from cutdetect.pipeline.storage import LocalDiskStorage
from cutdetect.pipeline.templates import PROMPT_TEMPLATES, UGC_CLONE_V1, UGC_PRODUCT_CLONE_V1
from cutdetect.pipeline.workflow_client import (
    PRODUCT_CLONE_WORKFLOW,
    RunwayWorkflowGateway,
    is_workflow_route,
    workflow_spec_for_model,
    workflow_spec_for_route,
)


@dataclass(frozen=True, slots=True)
class PipelineStudioConfig:
    host: str = "127.0.0.1"
    port: int = 8790
    browser_open_delay_sec: float = 0.35
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    upload_chunk_bytes: int = 1024 * 1024
    access_password: str | None = None
    detection_timeout_sec: float = 1800.0


def _boot_payload() -> dict[str, object]:
    labels = {
        "seedance2": "Seedance 2.0",
        "seedance2_5": "Seedance 2.5",
        "hailuo3": "Hailuo 3.0",
    }
    workflows = {
        model_id: workflow_spec_for_model(cast(RunwayReferenceModel, model_id))
        for model_id in MODEL_CAPABILITIES
    }
    return {
        "models": {
            model_id: {
                "label": labels[model_id],
                "ratios": ("9:16",),
                "resolutions": (workflows[model_id].resolution,),
                "defaultRatio": "9:16",
                "defaultResolution": workflows[model_id].resolution,
                "routeLabel": "Workflow API",
                "minDuration": caps.min_duration_s,
                "maxDuration": caps.max_duration_s,
                "supportsInternalCuts": caps.supports_internal_cuts,
                "notes": caps.notes,
            }
            for model_id, caps in MODEL_CAPABILITIES.items()
        },
        "productRoutes": {
            "router": {
                "label": "Seedance 2.0",
                "routeLabel": "Model Router API",
                "model": "seedance2",
                "ratio": "9:16",
                "resolution": "720p",
            },
            "workflow": {
                "label": "Hailuo 3",
                "routeLabel": "Workflow API",
                "model": "hailuo3",
                "ratio": "9:16",
                "resolution": PRODUCT_CLONE_WORKFLOW.resolution,
            },
        },
        "templates": [asdict(template) for template in PROMPT_TEMPLATES.values()],
        "voices": list(RUNWAY_PRESET_VOICES),
    }


def _health_payload() -> dict[str, str]:
    payload = {"status": "ok"}
    commit = os.environ.get("RENDER_GIT_COMMIT", "").strip()
    if commit:
        payload["commit"] = commit
    return payload


def render_pipeline_html() -> str:
    """Render the Ripple input/output workspace."""
    return render_ripple_html(_boot_payload())


def _render_legacy_pipeline_html() -> str:
    """Render the complete, dependency-free post-production desk."""
    html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cut Room · Replica Assembly</title><style>
:root{--ink:#11100d;--paper:#efe8d8;--hot:#ff4e32;--blue:#3dbfe6;--lime:#c7f36b;--dim:#777168;--rule:#bdb5a6;--ok:#1e8b5b;color-scheme:light;font-family:"Avenir Next","Trebuchet MS",sans-serif}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);background-image:linear-gradient(90deg,transparent 49.8%,#11111109 50%,transparent 50.2%);min-height:100vh}body:after{content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.14'/%3E%3C/svg%3E")}
button,input,textarea,select{font:inherit}button{cursor:pointer}.top{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;padding:15px 3vw;border-bottom:1px solid var(--ink);background:#efe8d8e8;backdrop-filter:blur(12px)}.brand{font:900 16px/1 monospace;letter-spacing:-.08em}.brand b{color:var(--hot)}.status-pill{font:800 10px monospace;letter-spacing:.12em;text-transform:uppercase;display:flex;align-items:center;gap:8px}.status-pill i{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 4px #1e8b5b20}
main{padding:55px 3vw 120px}.hero{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(300px,.55fr);gap:5vw;align-items:end;margin-bottom:55px}.kicker{font:800 10px monospace;letter-spacing:.22em;text-transform:uppercase;margin-bottom:20px}.hero h1{font:900 clamp(65px,10.6vw,160px)/.75 Georgia,"Times New Roman",serif;letter-spacing:-.085em;text-transform:uppercase;margin:0}.hero h1 em{color:var(--hot);font-weight:400}.dek{font:600 clamp(17px,1.5vw,23px)/1.25;border-top:5px solid var(--ink);padding-top:16px;margin:0}.dek small{display:block;font:700 10px/1.4 monospace;color:var(--dim);margin-top:17px;text-transform:uppercase;letter-spacing:.1em}
.graph-shell{border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);padding:25px 0;margin-bottom:54px;overflow:auto}.graph{display:grid;grid-template-columns:1.3fr 55px 1fr 55px 1.25fr 55px .8fr 55px .8fr;min-width:980px;align-items:center}.node{min-height:96px;border:1px solid var(--ink);padding:15px;background:var(--paper);position:relative;transition:.3s}.node:before{content:attr(data-step);position:absolute;right:10px;top:8px;font:900 28px Georgia;color:#1112}.node strong{font:900 20px Georgia;display:block}.node span{display:block;font:700 9px/1.4 monospace;text-transform:uppercase;color:var(--dim);margin-top:8px}.node.active{background:var(--ink);color:var(--paper);transform:translateY(-5px);box-shadow:7px 7px 0 var(--hot)}.node.active span{color:var(--paper)}.node.done{background:var(--lime)}.wire{height:1px;background:var(--ink);position:relative}.wire:after{content:"";position:absolute;right:0;top:-4px;border-left:7px solid var(--ink);border-top:4px solid transparent;border-bottom:4px solid transparent}.sources{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.sources b{font:800 9px monospace;text-transform:uppercase;border-left:5px solid var(--blue);padding:7px;background:#fff6}
.desk{display:grid;grid-template-columns:minmax(280px,.63fr) minmax(0,1.37fr);border:1px solid var(--ink)}.controls{padding:25px;border-right:1px solid var(--ink);background:#e6decd}.workspace{padding:25px;min-width:0}.section-no{font:900 10px monospace;color:var(--hot);letter-spacing:.15em}.controls h2,.workspace h2{font:900 clamp(33px,4vw,62px)/.86 Georgia;margin:8px 0 25px;letter-spacing:-.055em}.file-wells{display:grid;gap:8px}.well{border:1px dashed var(--ink);padding:13px;display:grid;grid-template-columns:38px 1fr;gap:12px;align-items:center;cursor:pointer;transition:.2s}.well:hover{background:var(--ink);color:var(--paper)}.well.ready{border-style:solid;background:var(--lime)}.well .badge{font:900 22px Georgia;color:var(--hot)}.well b{display:block;font:900 11px monospace;text-transform:uppercase}.well small{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}.well input{display:none}
.field{margin-top:17px}.field>label{display:block;font:800 9px monospace;text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px}.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}select,input[type=number],textarea{width:100%;border:1px solid var(--ink);background:var(--paper);padding:11px;border-radius:0}textarea{min-height:138px;resize:vertical;font:600 12px/1.45 monospace}.consent{display:flex;gap:10px;align-items:flex-start;margin:17px 0;font:700 10px/1.35 monospace}.consent input{accent-color:var(--hot);margin-top:2px}.cta{width:100%;border:0;background:var(--ink);color:var(--paper);padding:16px;font:900 12px monospace;text-transform:uppercase;letter-spacing:.1em;box-shadow:6px 6px 0 var(--hot);transition:.2s}.cta:hover{transform:translate(-2px,-2px);box-shadow:8px 8px 0 var(--hot)}.cta:disabled{opacity:.35;box-shadow:none;cursor:not-allowed}.secondary{border:1px solid var(--ink);background:transparent;padding:10px 13px;font:800 10px monospace;text-transform:uppercase}.secondary:hover{background:var(--hot);color:white;border-color:var(--hot)}
.empty{min-height:440px;display:grid;place-items:center;text-align:center;color:var(--dim)}.empty b{font:400 italic 52px Georgia;display:block;color:var(--ink)}.hidden{display:none!important}.notice{border-left:7px solid var(--hot);padding:13px 15px;background:#fff8;font:700 11px/1.4 monospace;margin:15px 0}.notice.ok{border-color:var(--ok)}.progress-line{height:10px;background:#d3cbbb;overflow:hidden}.progress-line i{display:block;width:8%;height:100%;background:var(--hot);animation:work 1.1s ease-in-out infinite alternate}@keyframes work{to{width:87%}}
.estimate-head,.review-head{display:flex;justify-content:space-between;gap:20px;align-items:end;border-bottom:1px solid var(--ink);padding-bottom:15px}.big-stat{font:900 52px/.8 Georgia;color:var(--hot);text-align:right}.big-stat small{display:block;font:800 9px monospace;color:var(--ink);text-transform:uppercase;margin-top:8px}.group-table{width:100%;border-collapse:collapse;margin:20px 0}.group-table th,.group-table td{text-align:left;border-bottom:1px solid var(--rule);padding:10px 5px;font:700 10px monospace}.group-table th{text-transform:uppercase;color:var(--dim)}.run-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:end}.spend{font:800 10px monospace}.spend b{font:900 24px Georgia;color:var(--hot)}
.review-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:1px;background:var(--ink);border:1px solid var(--ink);margin-top:22px}.clip{background:var(--paper);padding:13px}.clip.approved{background:#e5f0cb}.clip-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.clip-top b{font:900 27px Georgia}.clip-state{font:800 9px monospace;text-transform:uppercase;color:var(--hot)}.compare{display:grid;grid-template-columns:1fr 1fr;gap:4px}.compare figure{margin:0}.compare video{width:100%;aspect-ratio:9/16;background:#161513;object-fit:contain;display:block}.compare figcaption{font:800 8px monospace;text-transform:uppercase;margin:5px 0}.cut-rail{height:12px;background:#cbc3b5;position:relative;margin:11px 0}.cut-rail i{position:absolute;width:3px;height:100%;background:var(--hot);top:0}.clip-meta{font:700 9px/1.45 monospace;color:var(--dim)}.clip-actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:11px}.clip-actions button{border:1px solid var(--ink);background:transparent;padding:8px;font:800 9px monospace;text-transform:uppercase}.clip-actions button:hover{background:var(--ink);color:var(--paper)}.clip-actions button.primary{background:var(--ink);color:var(--paper)}.clip-actions button:disabled{opacity:.3;cursor:not-allowed}.delivery{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:22px}.delivery video{width:100%;background:#111;max-height:70vh}.delivery h3{font:900 25px Georgia;margin:0 0 8px}.download{display:inline-block;background:var(--hot);color:white;text-decoration:none;padding:12px 16px;font:900 10px monospace;text-transform:uppercase;margin-top:10px}
.recent{margin-top:25px;border-top:1px solid var(--ink);padding-top:15px}.recent h3{font:900 10px monospace;text-transform:uppercase}.recent button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--rule);background:transparent;padding:8px 0;font:700 9px monospace}.error{color:#b82110;font:800 10px/1.4 monospace;white-space:pre-wrap}
@media(max-width:900px){.hero,.desk{grid-template-columns:1fr}.controls{border-right:0;border-bottom:1px solid var(--ink)}.delivery{grid-template-columns:1fr}.hero h1{font-size:19vw}}@media(max-width:560px){main{padding-inline:15px}.top{padding-inline:15px}.row,.run-row{grid-template-columns:1fr}.workspace,.controls{padding:17px}.review-grid{grid-template-columns:1fr}}
</style></head><body>
<header class="top"><div class="brand"><b>///</b> CUT ROOM · REPLICA ASSEMBLY</div><div class="status-pill"><i></i><span>Local desk online</span></div></header>
<main><section class="hero"><div><div class="kicker">Phase E / final assembly</div><h1>Clone the<br><em>cut.</em> Keep<br>the beat.</h1></div><p class="dek">One reference performance in. Six controlled recreations out. Every cut stays visible, every paid action waits for you.<small>Your media is processed locally until you explicitly confirm a Runway generation.</small></p></section>
<section class="graph-shell"><div class="graph"><div class="node active" id="nodeInputs" data-step="01"><strong>References</strong><div class="sources"><b>Video</b><b>Face</b><b>Voice</b></div><span>Three anchors + prompt</span></div><div class="wire"></div><div class="node" id="nodeSegment" data-step="02"><strong>Segment</strong><span>Preserve every hard cut</span></div><div class="wire"></div><div class="node" id="nodeGenerate" data-step="03"><strong>Generate × N</strong><span>Submit all, then poll</span></div><div class="wire"></div><div class="node" id="nodeReview" data-step="04"><strong>Review</strong><span>Trim · approve · retry</span></div><div class="wire"></div><div class="node" id="nodeStitch" data-step="05"><strong>Stitch</strong><span>Locked until approved</span></div></div></section>
<section class="desk"><aside class="controls"><div class="section-no">001 / INPUT DESK</div><h2>Load the<br>references.</h2><div class="file-wells">
<label class="well" id="wellVideo"><span class="badge">V</span><span><b>Original UGC video</b><small id="nameVideo">Drop or choose video</small></span><input id="fileVideo" type="file" accept="video/*"></label>
<label class="well" id="wellImage"><span class="badge">I</span><span><b>Target face</b><small id="nameImage">Drop or choose image</small></span><input id="fileImage" type="file" accept="image/*"></label>
<label class="well" id="wellAudio"><span class="badge">A</span><span><b>Target voice</b><small id="nameAudio">Drop or choose audio</small></span><input id="fileAudio" type="file" accept="audio/*"></label></div>
<div class="field"><label for="model">Generation model</label><select id="model"></select></div><div class="row"><div class="field"><label for="ratio">Aspect ratio</label><select id="ratio"></select></div><div class="field"><label for="resolution">Resolution</label><select id="resolution"></select></div></div>
<div class="field"><label for="prompt">Prompt template · editable</label><textarea id="prompt"></textarea></div>
<label class="consent"><input id="consent" type="checkbox"><span>I confirm I have permission to use this video, likeness, and voice for AI generation.</span></label>
<button class="cta" id="prepare">Analyze & build the job</button><p class="error" id="error"></p>
<div class="recent"><h3>Recent local jobs</h3><div id="recentJobs">None yet.</div></div></aside>
<section class="workspace"><div class="empty" id="empty"><div><b>The bench is clear.</b><p>Load three references to reveal the generation plan.</p></div></div><div id="work" class="hidden"></div></section></section></main>
<script>const BOOT=__BOOT__;
const $=s=>document.querySelector(s),files={};let session=null,currentJob=null,eventSource=null,eventJob=null,pollTimer=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function sessionId(){return crypto.randomUUID().replaceAll('-','')}
function error(message=''){ $('#error').textContent=message }
for(const role of ['Video','Image','Audio']){$(`#file${role}`).onchange=e=>{const f=e.target.files[0];if(!f)return;files[role.toLowerCase()]=f;$(`#name${role}`).textContent=f.name;$(`#well${role}`).classList.add('ready')}}
const model=$('#model'),ratio=$('#ratio'),resolution=$('#resolution');Object.entries(BOOT.models).forEach(([id,m])=>model.add(new Option(m.label,id)));model.value='seedance2';
function modelOptions(){const caps=BOOT.models[model.value];ratio.innerHTML='';caps.ratios.forEach(v=>ratio.add(new Option(v,v)));ratio.value=model.value==='hailuo3'?'9:16':(caps.ratios.includes('720:1280')?'720:1280':caps.ratios[0]);resolution.innerHTML='';if(model.value==='hailuo3'){caps.resolutions.forEach(v=>resolution.add(new Option(v,v)));resolution.value='768P';resolution.disabled=false}else{resolution.add(new Option('Encoded by ratio',''));resolution.disabled=true}}
model.onchange=modelOptions;modelOptions();$('#prompt').value=BOOT.templates[0].body;
async function api(path,options={}){const response=await fetch(path,options);let data={};try{data=await response.json()}catch{}if(!response.ok)throw new Error(data.error||`${response.status} ${response.statusText}`);return data}
async function upload(role,file){return api(`/api/uploads/${session}/${role}`,{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream','X-Filename':encodeURIComponent(file.name)},body:file})}
$('#prepare').onclick=async()=>{error();if(!files.video||!files.image||!files.audio)return error('Load a video, face image, and voice sample first.');if(!$('#consent').checked)return error('Permission confirmation is required.');session=sessionId();$('#prepare').disabled=true;showProgress('Uploading references…','All three inputs stay local during analysis.');try{await Promise.all(Object.entries(files).map(([role,file])=>upload(role,file)));await api(`/api/sessions/${session}/prepare`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model.value,ratio:ratio.value,resolution:resolution.value,prompt:$('#prompt').value,template_id:BOOT.templates[0].id,template_version:BOOT.templates[0].version,consent:true})});pollSession()}catch(e){error(e.message);$('#prepare').disabled=false}};
function showProgress(title,detail){$('#empty').classList.add('hidden');$('#work').classList.remove('hidden');$('#work').innerHTML=`<div class="section-no">PROCESSING</div><h2>${esc(title)}</h2><div class="progress-line"><i></i></div><div class="notice">${esc(detail)}</div>`;setNodes('segment')}
async function pollSession(){try{const s=await api(`/api/sessions/${session}`);showProgress(s.stage||'Analyzing the cut…',s.message||'Building boundary-preserving groups.');if(s.status==='FAILED')throw new Error(s.error);if(s.status==='READY'){currentJob=s.job_id;await loadJob();await loadRecent();return}setTimeout(pollSession,1000)}catch(e){error(e.message);$('#prepare').disabled=false}}
function setNodes(stage){const order=['inputs','segment','generate','review','stitch'],map={inputs:'#nodeInputs',segment:'#nodeSegment',generate:'#nodeGenerate',review:'#nodeReview',stitch:'#nodeStitch'},at=order.indexOf(stage);order.forEach((n,i)=>{$(map[n]).classList.toggle('active',i===at);$(map[n]).classList.toggle('done',i<at)})}
async function loadRecent(){try{const jobs=await api('/api/jobs');$('#recentJobs').innerHTML=jobs.jobs.length?jobs.jobs.map(j=>`<button data-job="${j.job_id}">${j.job_id.slice(0,8)} · ${j.state} · ${j.model_id}</button>`).join(''):'None yet.';document.querySelectorAll('[data-job]').forEach(b=>b.onclick=()=>{stopUpdates();currentJob=b.dataset.job;loadJob()} )}catch{}}
function fileUrl(value){return value||''}
async function loadJob(forceLive=false){if(!currentJob)return;try{const data=await api(`/api/jobs/${currentJob}`);renderJob(data);syncUpdates(forceLive?'RUNNING':data.state)}catch(e){error(e.message)}}
function stopUpdates(){if(eventSource){eventSource.close();eventSource=null}eventJob=null;clearTimeout(pollTimer);pollTimer=null}
function syncUpdates(state){const live=['RUNNING','STITCHING'].includes(state);if(!live){stopUpdates();return}if(!eventSource||eventJob!==currentJob){stopUpdates();eventJob=currentJob;eventSource=new EventSource(`/api/jobs/${currentJob}/events`);eventSource.onmessage=()=>loadJob();eventSource.addEventListener('segment.state',()=>loadJob());eventSource.addEventListener('job.state',()=>loadJob())}clearTimeout(pollTimer);pollTimer=setTimeout(loadJob,2500)}
function renderJob(d){$('#empty').classList.add('hidden');$('#work').classList.remove('hidden');if(d.state==='DRAFT'||d.state==='CONFIRMED'){renderEstimate(d);setNodes('segment')}else if(['RUNNING'].includes(d.state)){renderRunning(d);setNodes('generate')}else if(['REVIEW','STITCHING'].includes(d.state)){renderReview(d);setNodes(d.state==='STITCHING'?'stitch':'review')}else if(d.state==='COMPLETE'){renderDelivery(d);setNodes('stitch')}else{renderRunning(d)}}
function renderEstimate(d){const rows=d.segments.map(s=>`<tr><td>${String(s.index+1).padStart(2,'0')}</td><td>${s.duration_sec.toFixed(2)}s</td><td>${s.requested_duration_sec}s</td><td>${s.estimated_credits}</td><td>${s.hard_cut_offsets_sec.length}</td></tr>`).join('');$('#work').innerHTML=`<div class="estimate-head"><div><div class="section-no">002 / GENERATION PLAN</div><h2>Six cuts,<br>one rhythm.</h2></div><div class="big-stat">${d.estimated_credits}<small>estimated credits</small></div></div><div class="notice">${esc(BOOT.models[d.model_id].notes)}</div><table class="group-table"><thead><tr><th>Group</th><th>Source</th><th>Output</th><th>Credits</th><th>Internal cuts</th></tr></thead><tbody>${rows}</tbody></table><div class="run-row"><button class="cta" id="runJob">Confirm & generate</button></div><p class="spend">First pass: <b>${d.estimated_credits}</b> estimated credits. Runway billing limits apply.</p>`;$('#runJob').onclick=runJob}
async function runJob(){if(!confirm(`Submit ${currentJob.slice(0,8)} to Runway?`))return;try{await api(`/api/jobs/${currentJob}/run`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});loadJob(true)}catch(e){error(e.message)}}
function renderRunning(d){const rows=d.segments.map(s=>`<tr><td>${String(s.index+1).padStart(2,'0')}</td><td>${esc(s.state)}</td><td>${s.attempt_count}</td><td>${s.estimated_credits}</td></tr>`).join('');$('#work').innerHTML=`<div class="review-head"><div><div class="section-no">003 / PARALLEL GENERATION</div><h2>The render<br>is rolling.</h2></div><div class="big-stat">${d.submitted_credits}<small>credits submitted</small></div></div><div class="progress-line"><i></i></div><table class="group-table"><thead><tr><th>Group</th><th>State</th><th>Attempts</th><th>Credits</th></tr></thead><tbody>${rows}</tbody></table><div class="notice">You can close this tab. Task IDs and progress are durable.</div>`}
function card(s,d){const ready=['READY_FOR_REVIEW','APPROVED'].includes(s.state),cuts=s.hard_cut_offsets_sec.map(t=>`<i style="left:${Math.min(100,t/s.duration_sec*100)}%"></i>`).join('');return `<article class="clip ${s.state==='APPROVED'?'approved':''}"><div class="clip-top"><b>${String(s.index+1).padStart(2,'0')}</b><span class="clip-state">${esc(s.state)}</span></div><div class="compare"><figure><video controls playsinline preload="none" src="${fileUrl(s.source_url)}"></video><figcaption>Original reference</figcaption></figure><figure>${s.output_url?`<video controls playsinline preload="metadata" src="${fileUrl(s.final_url||s.output_url)}"></video>`:'<video></video>'}<figcaption>Generated output</figcaption></figure></div><div class="cut-rail">${cuts}</div><div class="clip-meta">planned ${s.duration_sec.toFixed(2)}s · actual ${s.actual_duration_sec?s.actual_duration_sec.toFixed(2)+'s':'—'} · retry ${s.estimated_credits} credits</div><div class="clip-actions"><button ${ready?'':'disabled'} onclick="suggest(${s.index})">Suggest trim</button><button ${ready?'':'disabled'} onclick="approve(${s.index})" class="primary">Approve</button><button ${ready?'':'disabled'} onclick="regenerate(${s.index})">Regenerate</button></div></article>`}
function renderReview(d){const approved=d.segments.filter(s=>s.state==='APPROVED').length;$('#work').innerHTML=`<div class="review-head"><div><div class="section-no">004 / REVIEW GATE</div><h2>Watch every<br>seam.</h2></div><div class="big-stat">${approved}/${d.segments.length}<small>clips approved</small></div></div><div class="notice ${d.can_stitch?'ok':''}">${d.can_stitch?'All clips are locked. Final assembly is available.':'Compare each performance, inspect marked source cuts, trim only dead tails, then approve.'}</div><div class="clip-actions"><button onclick="approveAll()">Approve all reviewable</button><button class="primary" ${d.can_stitch?'':'disabled'} onclick="stitch()">Stitch final</button></div><div class="review-grid">${d.segments.map(s=>card(s,d)).join('')}</div>`}
window.suggest=async index=>{try{const s=await api(`/api/jobs/${currentJob}/segments/${index}/suggest`,{method:'POST'});if(s.end_frame===s.original_end_frame)return alert(s.reason);if(confirm(`Suggested tail trim: frame ${s.end_frame} of ${s.original_end_frame}. Apply it?`)){await api(`/api/jobs/${currentJob}/segments/${index}/trim`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start_frame:s.start_frame,end_frame:s.end_frame})});loadJob()}}catch(e){error(e.message)}};
window.approve=async index=>{try{await api(`/api/jobs/${currentJob}/segments/${index}/approve`,{method:'POST'});loadJob()}catch(e){error(e.message)}};
window.regenerate=async index=>{const prompt=window.prompt('Edit the prompt for this one clip:', $('#prompt').value);if(!prompt)return;try{await api(`/api/jobs/${currentJob}/segments/${index}/regenerate`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});await api(`/api/jobs/${currentJob}/run`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});loadJob()}catch(e){error(e.message)}};
window.approveAll=async()=>{if(!confirm('Approve and lock every currently reviewable clip?'))return;try{await api(`/api/jobs/${currentJob}/approve-all`,{method:'POST'});loadJob()}catch(e){error(e.message)}};
window.stitch=async()=>{try{await api(`/api/jobs/${currentJob}/stitch`,{method:'POST'});loadJob()}catch(e){error(e.message)}};
function renderDelivery(d){$('#work').innerHTML=`<div class="review-head"><div><div class="section-no">005 / MASTER DELIVERY</div><h2>Cut, cloned,<br>complete.</h2></div><div class="big-stat">${d.validation?.actual_frame_count||'✓'}<small>validated frames</small></div></div><div class="notice ok">Canonical profile verified · audio/video sync checked · all approvals locked.</div><div class="delivery"><div><h3>Final master</h3><video controls src="${d.final_url}"></video><a class="download" href="${d.final_url}" download>Download final MP4</a></div><div><h3>Source vs clone QC</h3><video controls muted src="${d.qc_url}"></video><a class="download" href="${d.qc_url}" download>Download QC comparison</a></div></div>`}
loadRecent();
</script></body></html>"""
    return html.replace("__BOOT__", json.dumps(_boot_payload(), separators=(",", ":")))


class _PipelineServer(ThreadingHTTPServer):
    config: PipelineStudioConfig
    root: Path
    cache_dir: Path
    html: bytes
    sessions: dict[str, dict[str, object]]
    session_lock: threading.Lock
    analysis_lock: threading.Lock
    active_jobs: set[str]
    active_lock: threading.Lock
    active_preparations: set[str]
    preparation_lock: threading.Lock


def _start_stitch_operation(server: _PipelineServer, job_id: str) -> bool:
    """Start or resume one durable stitch operation without duplicating work."""
    with server.active_lock:
        if job_id in server.active_jobs:
            return False
        server.active_jobs.add(job_id)
    storage = LocalDiskStorage(server.root)
    try:
        with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
            job = store.job(job_id)
            if job.state not in {JobState.REVIEW, JobState.STITCHING}:
                raise PipelineError(f"job {job_id} cannot stitch from {job.state.value}")
            store.set_job_state(job_id, JobState.STITCHING)
    except Exception:
        with server.active_lock:
            server.active_jobs.discard(job_id)
        raise

    def finish() -> None:
        try:
            with PhaseCStore(storage.path("orchestration.sqlite3")) as worker_store:
                try:
                    stitch_job(worker_store, storage, job_id)
                except Exception as error:
                    worker_store.record_event(
                        job_id,
                        "job.stitch_failed",
                        payload={"message": str(error)},
                    )
                    worker_store.set_job_state(job_id, JobState.REVIEW)
        finally:
            with server.active_lock:
                server.active_jobs.discard(job_id)

    threading.Thread(target=finish, daemon=True).start()
    return True


def _valid_id(value: str) -> bool:
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value)


def _session_state_path(root: Path, session_id: str) -> Path:
    return root / "staging" / session_id / "session.json"


def _persist_session(root: Path, session_id: str, state: dict[str, object]) -> None:
    path = _session_state_path(root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, default=str, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_sessions(root: Path) -> dict[str, dict[str, object]]:
    sessions: dict[str, dict[str, object]] = {}
    staging = root / "staging"
    if not staging.is_dir():
        return sessions
    for path in staging.glob("*/session.json"):
        session_id = path.parent.name
        if not _valid_id(session_id):
            continue
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            sessions[session_id] = cast(dict[str, object], value)
    return sessions


def _isolated_detection(
    video: Path,
    output_dir: Path,
    cache_dir: Path,
    *,
    timeout_sec: float,
) -> Path:
    """Run memory-intensive detection in a process that exits before encoding."""
    environment = os.environ.copy()
    environment.pop("RUNWAYML_API_SECRET", None)
    environment.pop("RIPPLE_ACCESS_PASSWORD", None)
    environment.update(
        MALLOC_ARENA_MAX="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cutdetect",
                "detect",
                str(video),
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(cache_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise PipelineError(
            f"cut detection timed out after {timeout_sec / 60:.0f} minutes"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PipelineError(f"isolated cut detection failed: {detail[-2000:]}")
    predictions = output_dir / "predictions.json"
    if not predictions.is_file():
        raise PipelineError("isolated cut detection finished without predictions")
    return predictions


def _device_hash(value: str) -> str:
    if not _valid_id(value):
        raise PipelineError("browser identity is missing; refresh Ripple and try again")
    return hashlib.sha256(value.encode()).hexdigest()


class _PipelineHandler(BaseHTTPRequestHandler):
    server: _PipelineServer

    def _owner_hash(self, query: dict[str, list[str]] | None = None) -> str:
        value = self.headers.get("X-Ripple-Device", "")
        if not value and query is not None:
            value = query.get("device", [""])[0]
        return _device_hash(value)

    @staticmethod
    def _require_job_owner(store: PhaseCStore, job_id: str, owner_hash: str) -> None:
        if not store.job_owned_by(job_id, owner_hash):
            raise PipelineError("this generation is not available on this device")

    def _authorized(self) -> bool:
        password = self.server.config.access_password
        if password is None:
            return True
        authorization = self.headers.get("Authorization", "")
        scheme, separator, encoded = authorization.partition(" ")
        if not separator or scheme.lower() != "basic":
            return False
        try:
            supplied = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return False
        username, separator, supplied_password = supplied.partition(":")
        return bool(
            separator
            and username == "ripple"
            and hmac.compare_digest(supplied_password, password)
        )

    def _require_authorization(self) -> bool:
        if self._authorized():
            return False
        body = b"Ripple access is restricted.\n"
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Ripple", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)
        return True

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send((json.dumps(value, default=str) + "\n").encode(), "application/json", status)

    def _error(self, error: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": str(error)}, status)

    def _body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise PipelineError("JSON body must be an object")
        return cast(dict[str, object], value)

    def _storage(self) -> LocalDiskStorage:
        return LocalDiskStorage(self.server.root)

    def _update_session(self, session_id: str, **values: object) -> dict[str, object]:
        with self.server.session_lock:
            state = self.server.sessions.setdefault(session_id, {"status": "UPLOADING"})
            state.update(values, updated_at=time.time())
            _persist_session(self.server.root, session_id, state)
            return dict(state)

    def _file_url(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.server.root):
            return ""
        return "/files/" + resolved.relative_to(self.server.root).as_posix()

    def _serve_file(self, path: Path) -> None:
        if not path.is_file() or not path.resolve().is_relative_to(self.server.root):
            self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start_raw, end_raw = range_header.removeprefix("bytes=").split(",", 1)[0].split("-", 1)
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
            with path.open("rb") as stream:
                stream.seek(start)
                self.wfile.write(stream.read(end - start + 1))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(self.server.config.upload_chunk_bytes):
                self.wfile.write(chunk)

    def _job_payload(self, job_id: str, owner_hash: str) -> dict[str, object]:
        storage = self._storage()
        with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
            self._require_job_owner(store, job_id, owner_hash)
            status = job_status(store, job_id)
            review = ReviewService(store=store, storage=storage).snapshot(job_id)
            segments = store.segments(job_id)
            enriched = []
            for segment in segments:
                raw = storage.path(segment.output_key)
                final = storage.path(segment.final_output_key) if segment.final_output_key else None
                review_media = None
                playback_error = None
                review_source = final if final is not None and final.is_file() else raw
                if review_source.is_file():
                    try:
                        review_media = prepare_review_proxy(review_source)
                    except PipelineError as error:
                        playback_error = str(error)
                item = next(
                    cast(dict[str, object], value)
                    for value in cast(list[object], status["segments"])
                    if cast(dict[str, object], value)["index"] == segment.index
                )
                enriched.append(
                    {
                        **item,
                        "source_url": self._file_url(segment.input_path),
                        "output_url": (
                            self._file_url(review_media)
                            if review_media is not None and review_source == raw
                            else None
                        ),
                        "final_url": (
                            self._file_url(review_media)
                            if review_media is not None and review_source != raw
                            else None
                        ),
                        "playback_error": playback_error,
                    }
                )
            job = store.job(job_id)
            payload = {
                **status,
                "approved_count": review["approved_count"],
                "can_stitch": review["can_stitch"],
                "custom_voice_fix": review["custom_voice_fix"],
                "segments": enriched,
                "final_url": (
                    self._file_url(storage.path(job.final_output_key))
                    if job.final_output_key
                    else None
                ),
                "qc_url": (
                    self._file_url(storage.path(job.qc_output_key)) if job.qc_output_key else None
                ),
            }
            worker_errors = [
                event
                for event in store.events_since(job_id)
                if event.get("event") == "job.worker_error"
            ]
            if worker_errors:
                latest = cast(dict[str, object], worker_errors[-1].get("payload", {}))
                payload["error"] = str(latest.get("message", "generation worker stopped"))
            manifest = storage.path(f"jobs/{job_id}/job.json")
            if manifest.is_file():
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(value, dict) and "validation" in value:
                    payload["validation"] = value["validation"]
            return payload

    def _events(self, job_id: str, query: dict[str, list[str]], owner_hash: str) -> None:
        storage = self._storage()
        with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
            self._require_job_owner(store, job_id, owner_hash)
        last_id = int(self.headers.get("Last-Event-ID", query.get("last_id", ["0"])[0]))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        started = time.monotonic()
        try:
            with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
                while time.monotonic() - started < 15:
                    events = store.events_since(job_id, last_id)
                    for event in events:
                        self.wfile.write("".join(format_sse_events((event,))).encode())
                        last_id = int(str(event["id"]))
                    if events:
                        self.wfile.flush()
                    time.sleep(0.5)
                self.wfile.write(b": reconnect\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        try:
            if route == "/healthz":
                self._json(_health_payload())
                return
            if self._require_authorization():
                return
            if route == "/":
                self._send(self.server.html, "text/html; charset=utf-8")
                return
            if route == "/api/jobs":
                owner_hash = self._owner_hash()
                storage = self._storage()
                database = storage.path("orchestration.sqlite3")
                if not database.is_file():
                    self._json({"jobs": []})
                    return
                with PhaseCStore(database) as store:
                    self._json(
                        {
                            "jobs": [
                                {
                                    "job_id": job.id,
                                    "state": job.state.value,
                                    "model_id": job.model_id,
                                    "created_at": job.created_at,
                                }
                                for job in store.jobs_for_device(owner_hash)[:12]
                            ]
                        }
                    )
                return
            parts = Path(route.lstrip("/")).parts
            if len(parts) == 3 and parts[:2] == ("api", "sessions") and _valid_id(parts[2]):
                owner_hash = self._owner_hash()
                with self.server.session_lock:
                    state = self.server.sessions.get(parts[2], {"status": "UPLOADING"})
                    if state.get("owner_hash") not in {None, owner_hash}:
                        raise PipelineError("this upload session belongs to another device")
                    snapshot = dict(state)
                if snapshot.get("status") == "ANALYZING" and isinstance(
                    snapshot.get("request"), dict
                ):
                    self._launch_prepare(parts[2])
                elif snapshot.get("status") == "GENERATING" and snapshot.get("job_id"):
                    self._resume_job_if_needed(str(snapshot["job_id"]))
                self._json(
                    {key: value for key, value in snapshot.items() if key != "owner_hash"}
                )
                return
            if len(parts) >= 3 and parts[:2] == ("api", "jobs") and _valid_id(parts[2]):
                query = parse_qs(parsed.query)
                owner_hash = self._owner_hash(query)
                database = self.server.root / "orchestration.sqlite3"
                with PhaseCStore(database) as owner_store:
                    self._require_job_owner(owner_store, parts[2], owner_hash)
                self._resume_job_if_needed(parts[2])
                if len(parts) == 4 and parts[3] == "events":
                    self._events(parts[2], query, owner_hash)
                elif len(parts) == 3:
                    self._json(self._job_payload(parts[2], owner_hash))
                else:
                    self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
                return
            if parts and parts[0] == "files":
                candidate = self.server.root.joinpath(*parts[1:]).resolve()
                self._serve_file(candidate)
                return
            self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._error(error)

    def _save_upload(self, session_id: str, role: str) -> None:
        if not _valid_id(session_id) or role not in {"video", "image", "audio", "product"}:
            raise PipelineError("invalid upload target")
        owner_hash = self._owner_hash()
        with self.server.session_lock:
            existing = self.server.sessions.get(session_id)
            if existing is not None and existing.get("owner_hash") not in {None, owner_hash}:
                raise PipelineError("this upload session belongs to another device")
        length = int(self.headers.get("Content-Length", "0"))
        if length < 512 or length > self.server.config.max_upload_bytes:
            limit_mib = self.server.config.max_upload_bytes // (1024 * 1024)
            raise PipelineError(
                f"asset is empty, too small, or exceeds the {limit_mib} MiB upload limit"
            )
        filename = Path(unquote(self.headers.get("X-Filename", role))).name
        allowed = {
            "video": {".mp4", ".mov", ".mkv", ".webm", ".m4v"},
            "image": {".jpg", ".jpeg", ".png", ".webp"},
            "audio": {".mp3", ".wav", ".flac", ".m4a", ".aac"},
            "product": {".jpg", ".jpeg", ".png", ".webp"},
        }
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed[role]:
            raise PipelineError(f"unsupported {role} file type: {suffix}")
        destination = self.server.root / "staging" / session_id / f"{role}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        remaining = length
        with destination.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(remaining, self.server.config.upload_chunk_bytes))
                if not chunk:
                    raise PipelineError("upload ended unexpectedly")
                output.write(chunk)
                remaining -= len(chunk)
        with self.server.session_lock:
            state = self.server.sessions.setdefault(session_id, {"status": "UPLOADING"})
            state.update(owner_hash=owner_hash)
            state[role] = str(destination)
            state["updated_at"] = time.time()
            _persist_session(self.server.root, session_id, state)
        self._json({"role": role, "filename": filename})

    def _prepare(self, session_id: str, body: dict[str, object]) -> None:
        if not _valid_id(session_id) or body.get("consent") is not True:
            raise PipelineError("permission confirmation is required")
        owner_hash = self._owner_hash()
        with self.server.session_lock:
            state = self.server.sessions.get(session_id, {})
            if state.get("owner_hash") != owner_hash:
                raise PipelineError("this upload session belongs to another device")
            video = Path(str(state.get("video", "")))
            image = Path(str(state.get("image", "")))
            audio_value = state.get("audio")
            audio = Path(str(audio_value)) if audio_value else None
            product_value = state.get("product")
            product = Path(str(product_value)) if product_value else None
            experience = str(body.get("experience", "clone"))
            if experience not in {"clone", "product"}:
                raise PipelineError("unsupported generation experience")
            if not video.is_file() or not image.is_file():
                raise PipelineError("upload a source video and target face first")
            if audio is not None and not audio.is_file():
                raise PipelineError("the optional voice reference is unavailable")
            if experience == "product" and (product is None or not product.is_file()):
                raise PipelineError("upload a product image for the product consistency test")
            voice_preset_id = (
                validate_voice_preset(str(body.get("voice_preset", "")))
                if experience == "clone"
                else None
            )
            product_route = str(body.get("product_route", "router"))
            if experience == "product" and product_route not in {"router", "workflow"}:
                raise PipelineError("unsupported product test route")
            selected_route: str | None
            selected_resolution: str | None
            if experience == "product" and product_route == "router":
                selected_model = "seedance2"
                selected_resolution = "720p"
                selected_route = model_router_route("seedance2")
            elif experience == "product":
                selected_model = "hailuo3"
                selected_resolution = PRODUCT_CLONE_WORKFLOW.resolution
                selected_route = PRODUCT_CLONE_WORKFLOW.route_id
            else:
                selected_model = str(body.get("model", "seedance2"))
                selected_resolution = (
                    str(body.get("resolution")) if body.get("resolution") else None
                )
                selected_route = None
            selected_template = (
                UGC_PRODUCT_CLONE_V1 if experience == "product" else UGC_CLONE_V1
            )
            if state.get("status") in {"ANALYZING", "GENERATING"}:
                raise PipelineError("this session is already being prepared")
            state.update(
                status="ANALYZING",
                stage="Reading source video…",
                message="Checking whether this video needs clip partitioning.",
                request={
                    "experience": experience,
                    "product_route": product_route if experience == "product" else None,
                    "model": selected_model,
                    "ratio": "9:16",
                    "resolution": selected_resolution,
                    "route_id": selected_route,
                    "prompt": str(body.get("prompt", "")),
                    "template_id": selected_template.id,
                    "template_version": selected_template.version,
                    "auto_run": body.get("auto_run") is True,
                    "voice_preset": voice_preset_id,
                },
                updated_at=time.time(),
            )
            self.server.sessions[session_id] = state
            _persist_session(self.server.root, session_id, state)

        self._launch_prepare(session_id)
        self._json({"status": "ANALYZING"}, HTTPStatus.ACCEPTED)

    def _launch_prepare(self, session_id: str) -> None:
        with self.server.preparation_lock:
            if session_id in self.server.active_preparations:
                return
            self.server.active_preparations.add(session_id)

        def work() -> None:
            try:
                self._prepare_work(session_id)
            finally:
                with self.server.preparation_lock:
                    self.server.active_preparations.discard(session_id)

        threading.Thread(target=work, daemon=True).start()

    def _prepare_work(self, session_id: str) -> None:
        acquired = False
        try:
            with self.server.session_lock:
                state = dict(self.server.sessions[session_id])
            request = state.get("request")
            if not isinstance(request, dict):
                raise PipelineError("saved preparation settings are unavailable")
            body = cast(dict[str, object], request)
            owner_hash = str(state.get("owner_hash", ""))
            video = Path(str(state.get("video", "")))
            image = Path(str(state.get("image", "")))
            audio_value = state.get("audio")
            audio = Path(str(audio_value)) if audio_value else None
            product_value = state.get("product")
            product = Path(str(product_value)) if product_value else None
            if (
                not video.is_file()
                or not image.is_file()
                or (audio and not audio.is_file())
                or (body.get("experience") == "product" and (not product or not product.is_file()))
            ):
                raise PipelineError("saved preparation inputs are unavailable")

            if not self.server.analysis_lock.acquire(blocking=False):
                self._update_session(
                    session_id,
                    stage="Queued for analysis…",
                    message="Another source is being analyzed on this instance.",
                )
                self.server.analysis_lock.acquire()
            acquired = True
            session_dir = self.server.root / "staging" / session_id
            predictions = session_dir / "detection" / "predictions.json"
            source_duration = probe_video(video).duration_sec
            needs_partition = requires_cut_partition(source_duration)
            if needs_partition and not predictions.is_file():
                self._update_session(
                    session_id,
                    stage="Detecting hard cuts…",
                    message="Running cut analysis in an isolated memory worker.",
                )
                predictions = _isolated_detection(
                    video,
                    session_dir / "detection",
                    self.server.cache_dir,
                    timeout_sec=self.server.config.detection_timeout_sec,
                )
            elif not needs_partition:
                self._update_session(
                    session_id,
                    stage="Using the full source video…",
                    message="15 seconds or less: generating once without cut detection.",
                )
            self._update_session(
                session_id,
                stage="Planning generation clips…",
                message=(
                    "Using one complete clip."
                    if not needs_partition
                    else "Balancing short sections and selecting audio pauses for long shots."
                ),
            )

            existing_job = None
            database = self.server.root / "orchestration.sqlite3"
            if database.is_file():
                with PhaseCStore(database) as store, suppress(PipelineError):
                    existing_job = store.job(session_id)
            if existing_job is None:

                def encoding_progress(index: int, total: int) -> None:
                    self._update_session(
                        session_id,
                        stage=f"Encoding source clip {index} of {total}…",
                        message=(
                            "Preparing independent browser-safe inputs with one FFmpeg thread."
                        ),
                    )

                prepared = prepare_phase_c_job(
                    video,
                    image,
                    audio,
                    product=product,
                    predictions=predictions,
                    output_root=self.server.root,
                    cache_dir=self.server.cache_dir,
                    model_id=str(body.get("model", "seedance2")),
                    ratio=str(body.get("ratio")) if body.get("ratio") else None,
                    resolution=(
                        str(body.get("resolution")) if body.get("resolution") else None
                    ),
                    route_id=str(body.get("route_id")) if body.get("route_id") else None,
                    prompt=str(body.get("prompt", "")),
                    prompt_template_id=str(body.get("template_id", "ugc_clone_v1")),
                    prompt_template_version=int(str(body.get("template_version", 1))),
                    consent_affirmed=True,
                    owner_device_hash=owner_hash,
                    job_id=session_id,
                    progress_callback=encoding_progress,
                    voice_preset_id=(
                        str(body.get("voice_preset")) if body.get("voice_preset") else None
                    ),
                )
                job = prepared.job
            else:
                job = existing_job

            auto_run = body.get("auto_run") is True
            self._update_session(
                session_id,
                status="GENERATING" if auto_run else "READY",
                stage="Generating clips…" if auto_run else "Plan ready.",
                message=(
                    f"Estimated first-pass cost: {job.estimated_credits} credits."
                    if auto_run
                    else f"Estimated at {job.estimated_credits} credits."
                ),
                estimated_credits=job.estimated_credits,
                job_id=job.id,
                model_id=job.model_id,
                error=None,
            )
            if auto_run:
                self._start_job(job.id, job.max_credits or job.estimated_credits)
        except Exception as error:
            self._update_session(
                session_id,
                status="FAILED",
                stage="Preparation stopped.",
                message="The local preparation worker could not finish.",
                error=str(error),
            )
        finally:
            if acquired:
                self.server.analysis_lock.release()

    def _resume_job_if_needed(self, job_id: str) -> None:
        database = self.server.root / "orchestration.sqlite3"
        if not database.is_file():
            return
        with PhaseCStore(database) as store:
            try:
                job = store.job(job_id)
            except PipelineError:
                return
        if job.state in {
            JobState.DRAFT,
            JobState.ESTIMATING,
            JobState.CONFIRMED,
            JobState.RUNNING,
        }:
            self._start_job(job.id, job.max_credits or job.estimated_credits)

    def _start_job(self, job_id: str, max_credits: int) -> None:
        with self.server.active_lock:
            if job_id in self.server.active_jobs:
                return
            self.server.active_jobs.add(job_id)
        storage = LocalDiskStorage(self.server.root)
        try:
            with PhaseCStore(storage.path("orchestration.sqlite3")) as resume_store:
                resumable = resume_store.job(job_id)
                if resumable.state == JobState.FAILED:
                    if not any(
                        segment.state == SegmentState.PENDING
                        for segment in resume_store.segments(job_id)
                    ):
                        raise PipelineError(
                            "this failed job has no incomplete generation step to resume"
                        )
                    # Persist before returning 202 so the browser cannot reload the
                    # stale FAILED screen while this worker is already active.
                    resume_store.set_job_state(job_id, JobState.RUNNING)
        except Exception:
            with self.server.active_lock:
                self.server.active_jobs.discard(job_id)
            raise

        def work() -> None:
            try:
                with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
                    job = store.job(job_id)
                    if job.consent_affirmed_at is None:
                        raise PipelineError("job is missing its consent record")
                    if job.state in {JobState.DRAFT, JobState.ESTIMATING, JobState.CONFIRMED}:
                        store.confirm(job_id, max_credits)
                    elif job.max_credits is not None:
                        store.raise_credit_ceiling(job_id, max_credits)
                    gateway_logger = JsonlCallLogger(
                        storage.path(f"jobs/{job_id}/runway_calls.jsonl")
                    )
                    gateway: GenerationGateway
                    if is_workflow_route(job.route_id):
                        gateway = RunwayWorkflowGateway(
                            api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                            logger=gateway_logger,
                            storage=storage,
                            spec=workflow_spec_for_route(job.route_id),
                        )
                    elif job.route_id == DIRECT_API_ROUTE:
                        gateway = RunwayDirectGateway(
                            api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                            logger=gateway_logger,
                            storage=storage,
                        )
                    elif job.route_id.startswith(MODEL_ROUTER_ROUTE_PREFIX):
                        gateway = RunwayRouterGateway(
                            api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                            logger=gateway_logger,
                            storage=storage,
                            config_id=router_config_id_from_route(job.route_id),
                            expected_model=cast(RunwayReferenceModel, job.model_id),
                        )
                    else:
                        raise PipelineError(f"unsupported generation route: {job.route_id}")
                    audio_processor = (
                        RunwayAudioProcessor(
                            api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                            logger=gateway_logger,
                            storage=storage,
                        )
                        if job.voice_preset_id is not None
                        else None
                    )
                    PhaseCWorker(
                        store=store,
                        gateway=gateway,
                        audio_processor=audio_processor,
                    ).run_until_review(job_id)
            except Exception as error:
                try:
                    with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
                        if store.job(job_id).state not in {
                            JobState.REVIEW,
                            JobState.COMPLETE,
                        }:
                            store.set_job_state(job_id, JobState.FAILED)
                        store.record_event(
                            job_id, "job.worker_error", payload={"message": str(error)}
                        )
                except Exception:
                    pass
            finally:
                with self.server.active_lock:
                    self.server.active_jobs.discard(job_id)

        threading.Thread(target=work, daemon=True).start()

    def _run_job(self, job_id: str, max_credits: int) -> None:
        self._start_job(job_id, max_credits)
        self._json({"status": "RUNNING"}, HTTPStatus.ACCEPTED)

    def do_POST(self) -> None:
        route = unquote(urlparse(self.path).path)
        parts = Path(route.lstrip("/")).parts
        try:
            if self._require_authorization():
                return
            if len(parts) == 3 and parts[:2] == ("api", "voice-previews"):
                preset = validate_voice_preset(parts[2])
                if preset is None:
                    raise PipelineError("choose a voice preset first")
                storage = self._storage()
                processor = RunwayAudioProcessor(
                    api_key=os.environ.get("RUNWAYML_API_SECRET", ""),
                    logger=JsonlCallLogger(storage.path("voice_previews/runway_calls.jsonl")),
                    storage=storage,
                )
                # Serialize first-preview creation so two devices cannot purchase the
                # same cached sample at the same time.
                with self.server.active_lock:
                    preview = processor.voice_preview(preset)
                if preview is None:
                    self._json({"status": "PENDING"}, HTTPStatus.ACCEPTED)
                else:
                    self._json({"status": "READY", "url": self._file_url(preview)})
                return
            if len(parts) == 4 and parts[:2] == ("api", "uploads"):
                self._save_upload(parts[2], parts[3])
                return
            if len(parts) == 4 and parts[:2] == ("api", "sessions") and parts[3] == "prepare":
                self._prepare(parts[2], self._body())
                return
            if len(parts) >= 4 and parts[:2] == ("api", "jobs") and _valid_id(parts[2]):
                job_id = parts[2]
                owner_hash = self._owner_hash()
                storage = self._storage()
                with PhaseCStore(storage.path("orchestration.sqlite3")) as owner_store:
                    self._require_job_owner(owner_store, job_id, owner_hash)
                if len(parts) == 4 and parts[3] == "run":
                    body = self._body()
                    self._run_job(job_id, int(str(body.get("max_credits", 0))))
                    return
                with PhaseCStore(storage.path("orchestration.sqlite3")) as store:
                    review = ReviewService(store=store, storage=storage)
                    if len(parts) == 4 and parts[3] == "approve-all":
                        review.approve_all(job_id)
                        self._json(self._job_payload(job_id, owner_hash))
                        return
                    if len(parts) == 4 and parts[3] == "stitch":
                        if not _start_stitch_operation(self.server, job_id):
                            raise PipelineError("this job already has an active operation")
                        self._json({"status": "STITCHING"}, HTTPStatus.ACCEPTED)
                        return
                    if len(parts) == 6 and parts[3] == "segments":
                        index = int(parts[4])
                        action = parts[5]
                        if action == "suggest":
                            self._json(review.suggest(job_id, index).to_dict())
                        elif action == "trim":
                            body = self._body()
                            self._json(
                                asdict(
                                    review.trim(
                                        job_id,
                                        index,
                                        start_frame=int(str(body.get("start_frame", 0))),
                                        end_frame=int(str(body["end_frame"])),
                                    )
                                )
                            )
                        elif action == "approve":
                            self._json(asdict(review.approve(job_id, index)))
                        elif action == "regenerate":
                            body = self._body()
                            self._json(
                                asdict(
                                    review.regenerate(
                                        job_id,
                                        index,
                                        prompt=str(body.get("prompt", "")),
                                        max_credits=int(str(body.get("max_credits", 0))),
                                    )
                                )
                            )
                        else:
                            raise PipelineError(f"unknown review action: {action}")
                        return
            self._send(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._error(error)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_pipeline_server(
    config: PipelineStudioConfig | None = None,
    *,
    output_root: str | Path = ".cutdetect/pipeline_studio",
    cache_dir: str | Path = ".cutdetect/cache",
) -> _PipelineServer:
    settings = config or PipelineStudioConfig()
    server = _PipelineServer((settings.host, settings.port), _PipelineHandler)
    server.config = settings
    server.root = Path(output_root).expanduser().resolve()
    server.cache_dir = Path(cache_dir).expanduser().resolve()
    server.root.mkdir(parents=True, exist_ok=True)
    server.cache_dir.mkdir(parents=True, exist_ok=True)
    server.html = render_pipeline_html().encode()
    server.sessions = _load_sessions(server.root)
    server.session_lock = threading.Lock()
    server.analysis_lock = threading.Lock()
    server.active_jobs = set()
    server.active_lock = threading.Lock()
    server.active_preparations = set()
    server.preparation_lock = threading.Lock()
    database = server.root / "orchestration.sqlite3"
    if database.is_file():
        with PhaseCStore(database) as store:
            interrupted_stitches = tuple(
                job.id for job in store.jobs() if job.state == JobState.STITCHING
            )
        for job_id in interrupted_stitches:
            _start_stitch_operation(server, job_id)
    return server


def run_pipeline_studio(
    *,
    config: PipelineStudioConfig | None = None,
    output_root: str | Path = ".cutdetect/pipeline_studio",
    cache_dir: str | Path = ".cutdetect/cache",
    open_browser: bool = True,
) -> None:
    settings = config or PipelineStudioConfig()
    server = create_pipeline_server(settings, output_root=output_root, cache_dir=cache_dir)
    url = f"http://{settings.host}:{server.server_port}/"
    if open_browser:

        def open_after_delay() -> None:
            time.sleep(settings.browser_open_delay_sec)
            webbrowser.open(url)

        threading.Thread(target=open_after_delay, daemon=True).start()
    print(f"Ripple: {url}")
    print(f"Durable jobs: {server.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
