"""Record actual browser frames plus original video audio and real warning speech."""
from pathlib import Path
import base64, json, subprocess
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'screen-recordings';FRAMES=OUT/'capture-frames'
FRAMES.mkdir(parents=True,exist_ok=True)
shots=[];responses=[]
INIT=r'''
 window.recordingChunks=[];window.captureStarted=0;window.captureEnded=0;
 const AC=window.AudioContext;
 window.AudioContext=new Proxy(AC,{construct(T,args){const c=Reflect.construct(T,args);window.appAudio=c;window.mix=c.createMediaStreamDestination();return c}});
 const original=AudioNode.prototype.connect;
 AudioNode.prototype.connect=function(destination,...rest){
   const result=original.call(this,destination,...rest);
   if(window.mix && destination===this.context.destination && this!==window.mix)original.call(this,window.mix);
   return result;
 };
 window.prepareRecording=()=>{
   const v=document.querySelector('video'),c=window.appAudio;
   window.videoSource=c.createMediaElementSource(v);window.videoSource.connect(c.destination);
   window.audioRecorder=new MediaRecorder(window.mix.stream,{mimeType:'audio/webm;codecs=opus',audioBitsPerSecond:128000});
   window.audioRecorder.ondataavailable=e=>{if(e.data.size)window.recordingChunks.push(e.data)};
   v.addEventListener('playing',()=>{if(!window.captureStarted){window.captureStarted=Date.now()/1000;window.audioRecorder.start(100)}},{once:true});
   v.addEventListener('ended',()=>{window.captureEnded=Date.now()/1000;window.audioRecorder.stop()},{once:true});
 };
'''
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
    context=browser.new_context(viewport={'width':390,'height':844},device_scale_factor=1)
    context.add_init_script(INIT)
    page=context.new_page()
    page.on('response',lambda r:responses.append({'status':r.status,'track':r.headers.get('x-waypoint-track')}) if '/api/alert' in r.url else None)
    page.goto('https://waypoint-ageei-2026.vercel.app')
    page.get_by_role('button',name='Start detecting').click()
    page.evaluate('window.prepareRecording()')
    client=context.new_cdp_session(page)
    def frame(params):
        index=len(shots);file=FRAMES/f'{index:05d}.jpg'
        file.write_bytes(base64.b64decode(params['data']))
        shots.append({'file':file,'timestamp':params['metadata']['timestamp']})
        client.send('Page.screencastFrameAck',{'sessionId':params['sessionId']})
    client.on('Page.screencastFrame',frame)
    client.send('Page.startScreencast',{'format':'jpeg','quality':88,'maxWidth':390,'maxHeight':844,'everyNthFrame':1})
    page.get_by_role('button',name='Demo video 1',exact=True).click()
    page.wait_for_function('window.captureEnded>0',timeout=35000)
    page.wait_for_timeout(400)
    client.send('Page.stopScreencast')
    timing=page.evaluate('({start:window.captureStarted,end:window.captureEnded})')
    encoded=page.evaluate('''async()=>{const blob=new Blob(window.recordingChunks,{type:'audio/webm'});const buffer=new Uint8Array(await blob.arrayBuffer());let text='';for(let i=0;i<buffer.length;i+=8192)text+=String.fromCharCode(...buffer.slice(i,i+8192));return btoa(text)}''')
    (OUT/'captured-audio.webm').write_bytes(base64.b64decode(encoded))
    browser.close()
selected=[s for s in shots if timing['start']<=s['timestamp']<timing['end']]
assert len(selected)>150, len(selected)
concat=['ffconcat version 1.0']
for i,s in enumerate(selected):
    next_time=selected[i+1]['timestamp'] if i+1<len(selected) else timing['end']
    concat += ["file '"+s['file'].as_posix()+"'",f"duration {max(.001,next_time-s['timestamp']):.6f}"]
concat.append("file '"+selected[-1]['file'].as_posix()+"'")
(OUT/'frames.ffconcat').write_text('\n'.join(concat),encoding='utf-8')
offset=max(0,selected[0]['timestamp']-timing['start']);duration=timing['end']-selected[0]['timestamp']
output=OUT/'Waypoint-demo-video-1.mp4'
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-safe','0','-f','concat','-i',str(OUT/'frames.ffconcat'),'-ss',str(offset),'-i',str(OUT/'captured-audio.webm'),'-map','0:v:0','-map','1:a:0','-t',str(duration),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-r','30','-c:a','aac','-b:a','160k','-movflags','+faststart',str(output)],check=True)
report={'file':str(output),'seconds':round(duration,3),'firstFrameOffset':round(offset,3),'frames':len(selected),'voiceResponses':responses,'audio':'captured browser mix: original video + actual Qwen/ElevenLabs warnings','source':'actual app browser screencast; trimmed to video playing/end'}
(OUT/'recording-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
