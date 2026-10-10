"""Provider integration check using fresh, real camera frames, never demo media.
Only numeric/result metadata is saved; camera images remain in browser memory.
"""
from pathlib import Path
import json, os
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
BASE = os.environ.get('WAYPOINT_URL', 'http://localhost:3000')

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=EDGE, headless=True)
    context = browser.new_context(permissions=['camera'])
    page = context.new_page()
    page.goto(BASE)
    result = page.evaluate('''async () => {
      const ready = await (await fetch('/api/live/status')).json();
      if (!ready.ready) return {ready, error: 'Model A not ready'};
      const stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      const video = document.createElement('video'); video.muted=true; video.playsInline=true;
      video.srcObject=stream; document.body.append(video); await video.play();
      const canvas=document.createElement('canvas');
      const scale=Math.min(1,640/Math.max(video.videoWidth,video.videoHeight));
      canvas.width=Math.round(video.videoWidth*scale);canvas.height=Math.round(video.videoHeight*scale);
      const ctx=canvas.getContext('2d'),frames=[];
      try {
        for(let i=0;i<6;i++){
          await new Promise(resolve=>video.requestVideoFrameCallback(resolve));
          const timestamp_ms=Date.now();ctx.drawImage(video,0,0,canvas.width,canvas.height);
          frames.push({timestamp_ms,image_base64:canvas.toDataURL('image/jpeg',.72).split(',')[1]});
          if(i<5) await new Promise(resolve=>setTimeout(resolve,400));
        }
      } finally {stream.getTracks().forEach(t=>t.stop());video.remove();}
      const stamp=frames.at(-1).timestamp_ms,frameId='frame_'+crypto.randomUUID().replaceAll('-','');
      const started=performance.now();
      const r=await fetch('/api/live/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({session_id:'camera_'+crypto.randomUUID().replaceAll('-',''),frame_id:frameId,action_mode:'video',frames})});
      const data=await r.json();
      const report={ready,httpStatus:r.status,elapsedMs:Math.round(performance.now()-started),
        captureSpanMs:stamp-frames[0].timestamp_ms,cameraSize:[canvas.width,canvas.height],
        status:data.status,matchedFrame:data.frameId===frameId&&data.capturedAt===stamp,
        hazards:data.hazards?.length,boxes:data.hazards?.filter(h=>h.box).length,
        scores:data.hazards?.map(h=>h.severity),message:data.message};
      const spoken=data.hazards?.find(h=>h.speechTicket);
      if(spoken){const a=await fetch('/api/live/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticket:spoken.speechTicket})});
        report.speechStatus=a.status;report.speechBytes=(await a.arrayBuffer()).byteLength;}
      return report;
    }''')
    result['target'] = BASE
    (ROOT/'artifacts/model-a-real-camera-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    browser.close()
    assert result.get('httpStatus') == 200 and result.get('matchedFrame'), result
