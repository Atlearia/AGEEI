"""Full playback against the deployed app with real LLM / ElevenLabs responses."""
from pathlib import Path
from urllib.parse import unquote
import json, sys
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else 'https://waypoint-ageei-2026.vercel.app'
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

def main():
    responses=[]; errors=[]; clips=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=EDGE,headless=True)
        context=browser.new_context(viewport={'width':390,'height':844},device_scale_factor=1)
        context.add_init_script('''
          window.voicePlaybacks=[];
          const start=AudioBufferSourceNode.prototype.start, stop=AudioBufferSourceNode.prototype.stop;
          AudioBufferSourceNode.prototype.start=function(...args) {
            if(this.buffer && this.buffer.duration>.1) {
              this.testPlayback={duration:this.buffer.duration, startedAt:this.context.currentTime, earlyStop:false, ended:false};
              window.voicePlaybacks.push(this.testPlayback);
              this.addEventListener('ended',()=>this.testPlayback.ended=true);
            }
            return start.apply(this,args);
          };
          AudioBufferSourceNode.prototype.stop=function(...args) {
            if(this.testPlayback && !this.testPlayback.ended)
              this.testPlayback.earlyStop=this.context.currentTime-this.testPlayback.startedAt<this.testPlayback.duration-.1;
            return stop.apply(this,args);
          };
        ''')
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        def response(r):
            if '/api/alert' in r.url:
                responses.append({'status':r.status,'demo':r.headers.get('x-waypoint-demo'),
                    'track':r.headers.get('x-waypoint-track'),'warning':unquote(r.headers.get('x-waypoint-warning','')),
                    'generation_ms':r.headers.get('x-waypoint-generation-ms')})
        page.on('response',response)
        homepage=page.goto(BASE)
        assert homepage.status==200
        for index,duration,samples in [
            (1,18.278,[(3.5,'raised-edge'),(12,'barrier')]),
            (2,106.974,[(8,'planter-edge'),(35.8,'parked-bicycles'),(55,'lamp-post'),(65,'car-ahead'),(91.5,'guardrail')])
        ]:
            print(f'Checking deployed video {index}',flush=True)
            page.get_by_role('button',name='Start detecting').click()
            page.get_by_role('button',name=f'Demo video {index}',exact=True).click()
            page.wait_for_function('document.querySelector("video")?.currentTime>.3',timeout=30000)
            for at,track in samples:
                page.wait_for_function('t=>document.querySelector("video")?.currentTime>t',arg=at,timeout=40000)
                assert page.locator(f'.overlay [data-track="{track}"]').count()==1
                scores=page.locator('.threat-score').all_text_contents()
                assert scores and all(len(s)==4 and 0<=float(s)<=1 for s in scores)
                assert page.locator('video').evaluate('(v)=>!v.muted && v.volume===1 && !v.paused && v.webkitAudioDecodedByteCount>0')
                assert page.locator('video').get_attribute('controls') is None
                page.screenshot(path=str(ROOT/f'artifacts/hosted-video-{index}-{track}.jpg'),type='jpeg',quality=75)
            page.wait_for_selector('.start-button',timeout=25000)
            assert page.locator('.viewer').is_hidden()
            assert page.locator('video').get_attribute('src') is None
            clips.append({'video':index,'duration':duration,'original_audio':'unmuted and decoded','scores':'0..1, two decimals','ended':True})
        playbacks=page.evaluate('window.voicePlaybacks')
        assert len(responses)==7,responses
        assert all(r['status']==200 for r in responses),responses
        assert len(playbacks)==7,playbacks
        assert all(a['ended'] and not a['earlyStop'] for a in playbacks),playbacks
        assert not errors,errors
        status=page.request.get(BASE+'/api/status').json()
        bad=page.request.post(BASE+'/api/alert',data={'source':'demo','demoId':'building-walk-v1','trackId':'planter-edge','mediaTime':7})
        assert bad.status==422
        result={'status':'passed','url':BASE,'clips':clips,'speech_responses':responses,'audio_playbacks':playbacks,
            'browser_errors':errors,'configuration':status,'cross_video_event_status':bad.status,'physical_phone_test':'pending'}
        (ROOT/'artifacts/hosted-two-video-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
        browser.close()

if __name__=='__main__':main()
