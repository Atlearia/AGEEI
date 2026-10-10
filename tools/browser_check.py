"""Exercise real browser video, overlay and audio playback with a mocked provider response.
The paid-provider path is verified separately against the actual /api/alert endpoint.
"""
from pathlib import Path
import io, json, math, struct, wave
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://localhost:3000'
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

def tone():
    memory=io.BytesIO()
    with wave.open(memory,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
        f.writeframes(b''.join(struct.pack('<h',int(150*math.sin(2*math.pi*440*i/22050))) for i in range(4410)))
    return memory.getvalue()

def main():
    errors=[]; requests=[]; checks=[]
    (ROOT/'artifacts').mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=EDGE,headless=True)
        context=browser.new_context(viewport={'width':390,'height':844},device_scale_factor=1)
        context.add_init_script('''
          window.playedAudio = 0;
          const originalStart = AudioBufferSourceNode.prototype.start;
          AudioBufferSourceNode.prototype.start = function(...args) {
            if (this.buffer && this.buffer.duration > .1) window.playedAudio++;
            return originalStart.apply(this, args);
          };
        ''')
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.route('**/api/status',lambda route:route.fulfill(json={'ready':True,'llmReady':True,'speechReady':True,'liveModelReady':False}))
        def speech(route):
            requests.append(route.request.post_data_json)
            route.fulfill(body=tone(),content_type='audio/wav')
        page.route('**/api/alert',speech)
        page.goto(BASE)
        page.get_by_role('button',name='Start detecting').click()
        page.screenshot(path=str(ROOT/'artifacts/02-input.jpg'),type='jpeg',quality=82)
        page.get_by_role('button',name='Demo video 1',exact=True).click()
        page.wait_for_function('document.querySelector("video")?.currentTime>.4')
        assert page.locator('.overlay .hazard-box').count()==0
        assert page.locator('video').get_attribute('controls') is None
        assert page.locator('.viewer').inner_text()==''
        assert page.locator('input[type="range"]').count()==0
        assert page.locator('video').evaluate('(v)=>!v.muted && v.volume===1')
        page.screenshot(path=str(ROOT/'artifacts/03-clear.jpg'),type='jpeg',quality=82)
        page.wait_for_function('document.querySelector("video")?.currentTime>3.3')
        assert page.locator('.overlay .hazard-box').count()==1
        page.screenshot(path=str(ROOT/'artifacts/04-curb.jpg'),type='jpeg',quality=82)
        assert len(requests)==1 and requests[0]['trackId']=='raised-edge'
        assert requests[0]['demoId']=='building-walk-v1'
        assert 0<=float(page.locator('.threat-score').text_content())<=1
        assert 2.4<requests[0]['mediaTime']<3.5
        page.get_by_role('button',name='Hide detection boxes').click()
        assert page.locator('.overlay .hazard-box').count()==0
        page.wait_for_function('document.querySelector("video")?.currentTime>10.4')
        assert len(requests)==2 and requests[1]['trackId']=='barrier'
        assert page.evaluate('window.playedAudio')==2
        assert page.locator('.overlay .hazard-box').count()==0
        page.get_by_role('button',name='Show detection boxes').click()
        page.wait_for_function('document.querySelector("video")?.currentTime>12.3')
        page.screenshot(path=str(ROOT/'artifacts/05-barrier.jpg'),type='jpeg',quality=82)
        assert page.locator('.overlay .hazard-box').count()==1
        page.wait_for_function('document.querySelector("video")?.currentTime>16.4')
        assert page.locator('.overlay .hazard-box').count()==0
        page.wait_for_selector('.start-button',timeout=10000)
        assert len(requests)==2
        checks.append('Full actual 18-second video: quiet gaps; curb and barrier boxes; one audible request per hazard; overlay off leaves voice active; automatic return at end.')
        page.screenshot(path=str(ROOT/'artifacts/01-start.jpg'),type='jpeg',quality=85)

        first_requests=requests[:]
        first_plays=page.evaluate('window.playedAudio')
        page.get_by_role('button',name='Start detecting').click()
        page.get_by_role('button',name='Demo video 2',exact=True).click()
        page.wait_for_function('document.querySelector("video")?.currentTime>.4')
        assert page.locator('video').evaluate('(v)=>!v.muted && v.volume===1 && v.webkitAudioDecodedByteCount>0')
        assert page.locator('.overlay .hazard-box').count()==0
        for at,expected in [(8,'planter-edge'),(35.8,'parked-bicycles'),(55,'lamp-post'),(65,'car-ahead'),(91.4,'guardrail')]:
            page.wait_for_function('t=>document.querySelector("video")?.currentTime>t',arg=at,timeout=40000)
            assert page.locator(f'.overlay [data-track="{expected}"]').count()==1
            scores=page.locator('.threat-score').all_text_contents()
            assert scores and all(len(s)==4 and 0<=float(s)<=1 for s in scores)
            assert page.locator('video').evaluate('(v)=>!v.muted && !v.paused && v.volume===1 && v.webkitAudioDecodedByteCount>0')
            page.screenshot(path=str(ROOT/f'artifacts/video-2-{expected}.jpg'),type='jpeg',quality=76)
        page.wait_for_function('document.querySelector("video")?.currentTime>100',timeout=20000)
        assert page.locator('.overlay .hazard-box').count()==0
        assert page.locator('.threat-score').count()==0
        assert page.locator('.viewer').inner_text()==''
        assert page.locator('video').get_attribute('controls') is None
        page.wait_for_selector('.start-button',timeout=12000)
        second_requests=requests[len(first_requests):]
        assert [r['trackId'] for r in second_requests]==['planter-edge','parked-bicycles','lamp-post','car-ahead','guardrail']
        assert all(r['demoId']=='courtyard-walk-v2' for r in second_requests)
        assert page.evaluate('window.playedAudio')-first_plays==5
        checks.append('Full 107-second second video: all five distinct hazard tracks and voice events, decimal scores, original unmuted audio continuously decoded, clear intervals, automatic return at end.')

        page.get_by_role('button',name='Start detecting').click()
        page.get_by_role('button',name='Demo video 2',exact=True).click()
        page.wait_for_function('document.querySelector("video")?.currentTime>.2')
        page.get_by_role('button',name='Mute spoken warnings').click()
        assert page.locator('video').evaluate('(v)=>!v.muted && !v.paused && v.volume===1')
        page.get_by_role('button',name='Stop detecting').click()
        checks.append('Muting spoken warnings leaves the original video audio on; stopping clears the media source.')

        for width,height in [(320,568),(430,932),(844,390)]:
            page.set_viewport_size({'width':width,'height':height})
            page.get_by_role('button',name='Start detecting').click()
            page.get_by_role('button',name='Demo video 1',exact=True).click()
            page.wait_for_function('document.querySelector("video")?.videoWidth===720')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            for selector in ['.stage','.control-panel']:
                box=page.locator(selector).bounding_box()
                assert box['y']+box['height'] <= height+1
            page.get_by_role('button',name='Stop detecting').click()
            assert page.locator('.viewer').is_hidden()
            assert page.locator('video').get_attribute('src') is None
        checks.append('Small phone, large phone, and landscape layouts; stop removes media and controls remain visible.')
        page.set_viewport_size({'width':390,'height':844})
        page.add_init_script('navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException("Denied","NotAllowedError")};')
        page.reload();page.get_by_role('button',name='Start detecting').click();page.get_by_role('button',name='Live camera Model A').click()
        page.wait_for_selector('.error-dialog[open]')
        assert 'Allow camera access' in page.locator('.error-dialog').inner_text()
        page.get_by_role('button',name='Back',exact=True).click()
        checks.append('Camera denial provides a recoverable error and returns to input selection.')

        late=browser.new_context(viewport={'width':390,'height':844})
        late.add_init_script('''
          window.pendingCameraResolve = null;
          window.testCameraTrack = null;
          navigator.mediaDevices.getUserMedia = () => new Promise(resolve => window.pendingCameraResolve = resolve);
        ''')
        page2=late.new_page(); page2.goto(BASE)
        page2.get_by_role('button',name='Start detecting').click();page2.get_by_role('button',name='Live camera Model A').click()
        page2.wait_for_function('window.pendingCameraResolve!==null')
        page2.get_by_role('button',name='Stop detecting').click()
        page2.evaluate('''() => { const canvas=document.createElement('canvas');canvas.width=100;canvas.height=100;const stream=canvas.captureStream(10);window.testCameraTrack=stream.getVideoTracks()[0];window.pendingCameraResolve(stream); }''')
        page2.wait_for_function('window.testCameraTrack.readyState==="ended"')
        checks.append('Late camera permission response after stop is discarded and its track is stopped.')
        browser.close()
    assert not errors,errors
    result={'status':'passed','checks':checks,'alert_requests':first_requests+second_requests,'browser_errors':errors,'limits':'Edge with mobile viewport; audio decoding/playback uses a short test waveform. Actual ElevenLabs is checked separately. Physical iPhone Safari remains to be tested.'}
    (ROOT/'artifacts/browser-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
