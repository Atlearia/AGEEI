"""Explicit user-requested offline Model A test of demo video.MOV.

The API accepts fresh wall-clock sampling times only. Replayed frames are
sampled at their original spacing and explicitly named diagnostic_demo.
Recorded media times are preserved separately. These are NOT live captures,
and results are never sent to the app's warning/audio pipeline.
"""
from pathlib import Path
import base64, html, json, time, uuid
import requests

APP = Path(__file__).resolve().parents[1]
OUT = APP.parent / 'model-a-demo-results'
env = {}
for line in (APP / '.env.local').read_text().splitlines():
    if '=' in line and not line.startswith('#'):
        key, value = line.split('=', 1)
        env[key] = value.strip().strip('"').strip("'")
BASE = env['AGEII_BASE_URL'].rstrip('/')
HEADERS = {'Authorization': 'Bearer ' + env['AGEII_API_TOKEN'], 'User-Agent': 'Mozilla/5.0 Waypoint/1.0', 'Content-Type': 'application/json'}
session_id = 'diagnostic_demo_' + uuid.uuid4().hex
records = []

def esc(value): return html.escape(str(value))

def panel(frame_path, boxes, caption):
    svg = []
    for box in boxes:
        xyxy = box.get('bbox')
        if not isinstance(xyxy, list) or len(xyxy) != 4: continue
        x1,y1,x2,y2 = xyxy
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1): continue
        yellow = box.get('is_hazard') or 'danger_type' in box
        color = '#ffe14d' if yellow else '#59cfff'
        score = box.get('severity_score')
        label = str(box.get('label', 'object')) + (f' {score / 100:.2f}' if isinstance(score, (int, float)) else '')
        x,y,w,h = x1*360,y1*640,(x2-x1)*360,(y2-y1)*640
        tx,ty = max(3,min(x,235)),max(18,y-5)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="2"/><text x="{tx}" y="{ty}" fill="{color}" stroke="#111" paint-order="stroke" stroke-width="3" font-size="14" font-weight="700">{esc(label)}</text>')
    return f'<figure><div class="photo"><img src="{esc(frame_path)}"><svg viewBox="0 0 360 640">{"".join(svg)}</svg></div><figcaption>{esc(caption)}</figcaption></figure>'

def write_report():
    summary = {'source': 'human data/demo video.MOV', 'mode': 'offline prerecorded-video diagnostic, action_mode=video',
               'sampling': '6 upright SDR 360x640 JPEGs per clip, 0.4 seconds apart; playback wall-clock times satisfy API age checks, media times preserved separately',
               'motionGate': 'bypassed for this explicit diagnostic only', 'clips': records}
    (OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    cards = []
    for r in records:
        if 'response_file' not in r:
            cards.append(f'<section><h2>{r["start"]:.1f}-{r["end"]:.1f}s</h2><p>HTTP {r["http_status"]}</p></section>');continue
        raw=json.loads((OUT/r['response_file']).read_text())
        panels=[panel(r['newest_image'],raw.get('boxes',[]),f"Newest submitted frame at {r['end']:.1f}s: all detector boxes")]
        indices=sorted(set(h.get('source_frame_index') for h in raw.get('hazards',[]) if isinstance(h.get('source_frame_index'),int)))
        for index in indices:
            if index<0 or index>=6: continue
            hazards=[h for h in raw.get('hazards',[]) if h.get('source_frame_index')==index]
            panels.append(panel(r['images'][index],hazards,f"Hazards grounded in frame {index}, video {r['start']+index*.4:.1f}s"))
        descriptions=''.join(f'<li>{esc(h.get("label"))}: {esc(h.get("description"))} Severity {h.get("severity_score")}; frame {h.get("source_frame_index")}; uncertain={esc(h.get("uncertain"))}</li>' for h in raw.get('hazards',[]))
        cards.append(f'<section><h2>{r["start"]:.1f}-{r["end"]:.1f}s · {r["elapsed_ms"]/1000:.1f}s response</h2><p>{r["detections"]} object boxes · {r["hazards"]} hazards · {r["app_visible_boxes"]} would pass the app’s current box filters.</p><p>Status: {esc(raw.get("status"))}. Warning: {esc(raw.get("warning") or "none")}</p><div class="panels">{"".join(panels)}</div><ul>{descriptions}</ul><p><a href="{esc(r["response_file"])}">Raw Model A response</a></p></section>')
    doc='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model A — actual demo-video results</title>
<style>body{font:16px system-ui;background:#101115;color:#eee;max-width:1200px;margin:32px auto;padding:0 20px}h1{font-size:30px}p{line-height:1.6;color:#ccc}section{border-top:1px solid #444;padding:24px 0}.panels{display:flex;gap:18px;flex-wrap:wrap}figure{margin:0;width:270px}.photo{position:relative}img{width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%}figcaption{font-size:13px;margin:8px 0;color:#aaa}li{margin-bottom:12px;line-height:1.5}a{color:#80c8ff}</style>
<h1>Model A: actual demo-video test</h1><p>Source: demo video.MOV. These are real API outputs from six sampled clips, not the authored demo overlay. This offline test bypasses the local motion gate. Yellow = returned hazard; blue = detected object without a hazard assessment. Displayed decimal scores are severity / 100, never confidence. Empty frames are kept.</p><p>The API only accepts recent epoch timestamps, so the diagnostic uses sampling times during replay and records original video positions separately. These images are prerecorded and are not live-camera results. Earlier-frame hazards are shown on their own source image.</p>'''+''.join(cards)
    (OUT/'index.html').write_text(doc,encoding='utf-8')

for clip,start_index in enumerate([0,8,16,24,32,40],1):
    paths=[OUT/'frames'/f'frame-{start_index+i+1:03d}.jpg' for i in range(6)]
    frames=[]; begun=time.monotonic()
    for index,path in enumerate(paths):
        delay=begun+index*.4-time.monotonic()
        if delay>0:time.sleep(delay)
        sampled_ms=time.time_ns()//1_000_000
        frames.append({'image_base64':base64.b64encode(path.read_bytes()).decode(),'timestamp_ms':sampled_ms})
    payload={'session_id':session_id,'frame_id':f'diagnostic_demo_{clip}_{uuid.uuid4().hex}','action_mode':'video','frames':frames}
    started=time.monotonic()
    response=requests.post(BASE+'/v1/analyze',headers=HEADERS,json=payload,timeout=70)
    r={'clip':clip,'start':start_index*.4,'end':(start_index+5)*.4,'http_status':response.status_code,'elapsed_ms':round((time.monotonic()-started)*1000)}
    if response.ok:
        raw=response.json();name=f'clip-{clip:02d}.json';(OUT/name).write_text(json.dumps(raw,indent=2),encoding='utf-8')
        hs=raw.get('hazards',[])
        accepted=[h for h in hs if h.get('uncertain') is False and h.get('source_frame_index')==5 and h.get('source_timestamp_ms')==frames[-1]['timestamp_ms']]
        visible=[h for h in accepted if h.get('bbox') and (h.get('severity_score') is None or h.get('severity_score',0)>=30)]
        r.update({'response_file':name,'images':[str(p.relative_to(OUT)).replace('\\','/') for p in paths],
          'newest_image':str(paths[-1].relative_to(OUT)).replace('\\','/'),'status':raw.get('status'),
          'matched_frame':raw.get('frame_id')==payload['frame_id'] and raw.get('captured_at_ms')==frames[-1]['timestamp_ms'],
          'detections':len(raw.get('boxes',[])),'hazards':len(hs),'app_visible_boxes':len(visible) if raw.get('status')=='ok' else 0,
          'hazard_frame_indices':[h.get('source_frame_index') for h in hs],
          'uncertain_hazards':sum(h.get('uncertain') is not False for h in hs),
          'labels':[b.get('label') for b in raw.get('boxes',[])],
          'hazard_labels':[h.get('label') for h in hs],
          'severity_scores':[h.get('severity_score') for h in hs], 'warning':raw.get('warning')})
    else:
        r['message']='Model A request failed; raw transport/credential details omitted.'
    records.append(r);write_report()
    print(json.dumps({k:v for k,v in r.items() if k not in ['images','newest_image','response_file']}),flush=True)
