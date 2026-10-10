from pathlib import Path
import json, io, math, struct, wave, os
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
EDGE=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
BASE=os.environ.get('WAYPOINT_URL','http://localhost:3000')
INIT=r'''
 window.beepStarts=0; window.startTones=0; window.voiceStarts=0; window.audioPeak=0; window.audioContexts=[]; window.phase='still'; window.phaseAt=performance.now();
 const AC=window.AudioContext;window.AudioContext=new Proxy(AC,{construct(T,args){const c=Reflect.construct(T,args);window.audioContexts.push(c);return c}});
 const connect=AudioNode.prototype.connect;
 AudioNode.prototype.connect=function(destination,...rest){
   if(this instanceof GainNode && destination===this.context.destination){
     const analyser=this.context.createAnalyser();analyser.fftSize=256;
     connect.call(this,analyser);connect.call(analyser,destination);
     const data=new Float32Array(256),timer=setInterval(()=>{analyser.getFloatTimeDomainData(data);window.audioPeak=Math.max(window.audioPeak,...data.map(Math.abs))},8);
     setTimeout(()=>clearInterval(timer),900);return destination;
   } return connect.call(this,destination,...rest);
 };
 const os=OscillatorNode.prototype.start,bs=AudioBufferSourceNode.prototype.start;
 OscillatorNode.prototype.start=function(...a){if(this.frequency.value===880)window.beepStarts++;else window.startTones++;return os.apply(this,a)};
 AudioBufferSourceNode.prototype.start=function(...a){if(this.buffer?.duration>.1)window.voiceStarts++;return bs.apply(this,a)};
 navigator.mediaDevices.getUserMedia=async()=>{
   const base=document.createElement('canvas');base.width=640;base.height=480;
   const b=base.getContext('2d');b.fillStyle='#ddd';b.fillRect(0,0,640,480);
   let seed=42;const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296};
   for(let i=0;i<600;i++){b.fillStyle=`rgb(${rand()*255},${rand()*255},${rand()*255})`;b.fillRect(rand()*640,rand()*480,4+rand()*18,4+rand()*18)}
   const subject=document.createElement('canvas');subject.width=120;subject.height=160;const sctx=subject.getContext('2d');
   sctx.fillStyle='#345';sctx.fillRect(0,0,120,160);
   for(let i=0;i<140;i++){sctx.fillStyle=`rgb(${rand()*255},${rand()*255},${rand()*255})`;sctx.fillRect(rand()*120,rand()*160,4+rand()*10,4+rand()*10)}
   const c=document.createElement('canvas');c.width=640;c.height=480;const g=c.getContext('2d');
   window.fakeCameraCanvas=c;
   const draw=()=>{const t=(performance.now()-window.phaseAt)/1000;g.setTransform(1,0,0,1,0,0);g.fillStyle='#ddd';g.fillRect(0,0,640,480);
     if(window.phase==='pan')g.translate(Math.sin(t)*35,0);
     g.drawImage(base,0,0);
     const scale=window.phase==='approach'?Math.min(2.6,1+t*1.3):1;
     g.drawImage(subject,320-60*scale,260-80*scale,120*scale,160*scale);
     requestAnimationFrame(draw)};draw();return c.captureStream(30);
 };
'''

def waveform():
    b=io.BytesIO()
    with wave.open(b,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050)
        f.writeframes(b''.join(struct.pack('<h',int(100*math.sin(2*math.pi*440*i/22050))) for i in range(6600)))
    return b.getvalue()

if __name__ == '__main__':
    import live_quick_check
