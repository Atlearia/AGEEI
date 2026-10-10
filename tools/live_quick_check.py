"""Current live behavior: all detector boxes, no motion upload gate, audible cues."""
from live_browser_check import INIT, EDGE, BASE, waveform, ROOT
from playwright.sync_api import sync_playwright
import json
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=EDGE,headless=True)
    context=browser.new_context(viewport={'width':390,'height':844});context.add_init_script(INIT)
    page=context.new_page();errors=[];pending=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.route('**/api/live/status',lambda r:r.fulfill(json={'configured':True,'ready':True}))
    page.route('**/api/live/analyze',lambda r:(pending.append(r),requests.append(r.request.post_data_json)))
    page.route('**/api/live/speech',lambda r:r.fulfill(body=waveform(),content_type='audio/wav'))
    page.goto(BASE);page.get_by_role('button',name='Start detecting').click();page.get_by_role('button',name='Live camera Model A').click()
    page.wait_for_function('window.startTones>=1 && window.audioPeak>.02',timeout=15000)
    page.wait_for_selector('.motion-indicator.pass',timeout=20000)
    page.wait_for_timeout(2800);assert len(requests)==1, {'requests':len(requests),'motion':page.locator('.motion-indicator').get_attribute('aria-label')}
    page.evaluate('window.audioPeak=0;window.phase="approach";window.phaseAt=performance.now()')
    page.wait_for_selector('.motion-indicator.alert',timeout=5000)
    page.wait_for_function('window.beepStarts>=3 && window.audioPeak>.05',timeout=2500)
    first=requests[0];stamp=first['frames'][-1]['timestamp_ms']
    def result(packet,status,boxes,hazards):
        stamp=packet['frames'][-1]['timestamp_ms']
        return {'status':status,'frameId':packet['frame_id'],'capturedAt':stamp,'expiresAt':stamp+45000,'width':640,'height':480,'latencyMs':500,'boxes':boxes,'hazards':hazards}
    pending[0].fulfill(json=result(first,'unknown',[{'id':'person','label':'person','box':[.3,.2,.3,.6],'severity':None},{'id':'curb','label':'curb','box':[.7,.6,.2,.2],'severity':.2}],[]))
    page.wait_for_selector('.analysis-preview');assert page.locator('.analysis-preview .hazard-box').count()==2
    assert page.locator('.analysis-preview text').all_text_contents()==['0.20']
    assert page.locator('.analysis-preview img').get_attribute('src')=='data:image/jpeg;base64,'+first['frames'][-1]['image_base64']
    page.evaluate('window.phase="still";window.phaseAt=performance.now()')
    page.wait_for_timeout(3200);assert len(requests)>=2
    hazard={'id':'curb','label':'curb','description':'Curb ahead','warning':'Curb ahead.','box':[.35,.5,.3,.25],'severity':.57,'direction':'ahead','dangerType':'trip_fall','speechTicket':'test-ticket'}
    pending[-1].fulfill(json=result(requests[-1],'ok',[hazard],[hazard]))
    page.wait_for_function('window.voiceStarts>=1')
    # Force a real suspended context. A single trusted tap must restore sound.
    page.evaluate('window.audioContexts[0].suspend()');page.wait_for_selector('.audio-blocked')
    page.get_by_role('button',name='Enable audio playback').click()
    page.wait_for_function('window.audioContexts[0].state==="running" && window.startTones>=2')
    page.get_by_role('button',name='Mute spoken warnings').click();before=page.evaluate('window.beepStarts')
    page.evaluate('window.phase="approach";window.phaseAt=performance.now()')
    page.wait_for_selector('.motion-indicator.alert',timeout=5000);page.wait_for_timeout(400)
    assert page.evaluate('window.beepStarts')==before
    page.get_by_role('button',name='Stop detecting').click();assert page.locator('.viewer').is_hidden()
    assert not errors,errors
    report={'status':'passed','target':BASE,'stationary_camera_approach_beeps':True,'startup_and_alarm_audio_signal':True,'unknown_assessment_shows_all_boxes':True,'unknown_scores_not_invented':True,'model_request_survives_motion_alert':True,'live_speech_at_057':True,'single_tap_restores_suspended_audio':True,'mute_and_stop':True,'browserErrors':errors}
    (ROOT/'artifacts/live-relaxed-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    browser.close()
