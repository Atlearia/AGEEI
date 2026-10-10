WAYPOINT - VERSION A

Mobile web app. Next.js on Vercel. No App Store installation required.

USE
Open https://waypoint-ageei-2026.vercel.app
Tap Start detecting, then Demo video 1, Demo video 2, or Live camera (Model A).
The supplied walks (18 seconds and about 107 seconds) play without visible playback controls.
The viewer contains only video, yellow hazard boxes, decimal threat scores, and three icon buttons:
toggle boxes, mute/unmute warnings, and stop.
Scores range from 0 to 1 and show two decimal places, e.g. 0.78. These are
threat levels, not confidence or percentages.
There are no visible object label, timeline, elapsed time, duration, source
badge, subtitles, or fake live indicator. A text dialog appears only on error.
At the end of the video the app returns to Start detecting.
Original video audio plays throughout, including during spoken warnings.
The speaker button mutes spoken warnings and live motion beeps. Stopping clears all media.

On iPhone: Safari -> Share -> Add to Home Screen -> Open as Web App.
This removes the normal browser toolbar. The screen must remain open for
detection and audio; this is not a background camera app. Internet is needed
for model/voice API calls. No offline support is claimed.

WHAT IS REAL
- The videos are the user's human data/demo video.MOV and demo video 2.MOV,
  converted from rotated HDR HEVC to upright SDR H.264/AAC MP4 for browsers.
  Video 2's original AAC audio is copied without re-encoding.
- Each audible event calls the server to turn structured hazard data into
  a short phrase with Qwen (the existing authorized account key).
- ElevenLabs turns that phrase into actual speech using its Flash model.
- Audio is decoded and played through an AudioContext unlocked on a tap.
- Repeated identical warnings can use a 30-minute process-local server cache.
  The LLM and TTS are called for a new, uncached warning.

WHAT IS AUTHORED FOR THIS DEMO
lib/demo.ts contains manually authored keyframes for a raised concrete edge
on the right and the red/white barrier ahead on the left. These are not
trained-model predictions. The assistant chose box positions and danger
levels specifically for this video, as requested.
lib/demo-2.ts contains five hazards for the second video: raised planter edge,
parked bicycles, lamp post, car ahead, and guardrail. Each has its own timed
box and threat curve. The video ID is carried through the voice request and
validated server-side so timelines cannot be mixed between videos.
Danger is a value from 0 to 1, never confidence. It appears beside each box.
Box threshold: 0.30. Speech threshold: 0.65. Speech release threshold: 0.55.
Video 1 warnings cross the threshold at approximately 2.50 and 9.95 seconds.
Video 2 warnings cross at 6, 33.7, 52, 60.5, and 89 seconds.
Boxes disappear once the relevant hazard has passed.
No warning says to step left/right or assumes a safe route.

LIVE CAMERA (CURRENT DEMO SETTINGS)
Keep the camera still. The rear camera is used without microphone access.
Local OpenCV optical flow checks for expanding objects and fast crossing.
The more sensitive prototype triggers after two observations, with an
approximate approach window of four seconds. It does not measure m/s.
A short startup tone confirms the audio path. The speaker highlights when
browser audio needs a tap. Gray motion icon = uncertain; yellow = actual
motion alert. A confirmed alert plays three local beeps without an API call.

Motion uncertainty and alerts no longer block or cancel Model A requests.
Six frames spanning about two seconds are analyzed, one request at a time.
ALL valid boxes from Model A's boxes[] are shown, including ordinary objects
and boxes with unknown severity. Unknown severity has no number; confidence
is never substituted. Geometry is normalized XYXY, converted for the viewer.
Boxes are on a larger preview of their exact analyzed snapshot because Model A
usually takes several seconds. They are not pasted onto newer camera pixels.
The model's separate hazards[] supplies assessed warning text. Live speech
threshold is now 0.50; demos retain 0.65. Model A scores are divided by 100.
Expired or mismatched results are still rejected. Stop/mute cancels audio.

These are deliberately sensitive prototype settings, with possible false
alarms. Browser tests use an approaching textured object and verify an actual
audio signal. They do not establish detection accuracy for a running person
on a physical phone. Walking with the camera is outside the stationary-camera
assumption. Model inference still runs when local motion is unknown.
The app holds camera frames in memory; provider retention is controlled by
that service. Demo videos retain their manually authored detection timelines.

VOICE TIMING
Decoded video media timestamps drive boxes and speech decisions.
requestVideoFrameCallback is used when available, with animation-frame fallback.
Boxes are linearly interpolated between source-image keyframes; object-fit
contain preserves the full source, with the same coordinate mapping for boxes.
Each hazard is announced once per run. Muting, stopping, seeking, pausing,
backgrounding, or the hazard passing cancels pending/playing warnings.
Audio that is more than 3.5 video seconds late is discarded. Failed requests
have limited retries. There is no fake browser-voice replacement for ElevenLabs.

LOCAL RUN
Node.js 22 or newer:
  npm install
  npm run dev
  Open http://localhost:3000

Production build:
  npm run build
  npm start

CONFIGURATION
.env.local is ignored by Git and deployment uploads. Never commit it.
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
LLM_API_KEY=<authorized Qwen key>
ELEVENLABS_API_KEY=<actual secret, not a key ID>
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
ELEVENLABS_MODEL_ID=eleven_flash_v2_5
AGEII_BASE_URL=<Model A API base URL from the handoff>
AGEII_API_TOKEN=<Model A bearer token>

OpenAI is also supported: use LLM_PROVIDER=openai, unset LLM_BASE_URL / LLM_MODEL,
set OPENAI_API_KEY, and optionally OPENAI_MODEL (default gpt-4.1-mini).
All secrets stay in server environment variables; none have NEXT_PUBLIC names.

DEPLOYMENT
The linked Vercel project is waypoint-ageei-2026.
  node scripts/sync-vercel-env.mjs --with-speech --with-model
  vercel --prod --yes
The helper checks project identity and sends keys only over stdin.
Video is a static asset; full video is never sent through an API function.
The alert endpoint accepts only an active demo track ID and timestamp, then
derives the data server-side. It does not accept arbitrary TTS text or prompts.

FILES
components/Waypoint.tsx   Input lifecycle, video viewer, and icon-only controls.
lib/detection.ts         Shared detection contract and source-coordinate geometry.
lib/demo.ts              Authored demo hazard timeline.
lib/demo-2.ts            Second video hazard keyframes and threat scores.
lib/alert-controller.ts  Thresholds, cancellation, deduplication, freshness checks.
lib/audio.ts             Mobile audio unlock, fetch, decoding, and playback.
server/voice.ts          LLM and ElevenLabs integrations, cache, sanitized errors.
server/model-a.ts        Model A adapter, clip validation and signed speech tickets.
lib/live.ts              Fresh camera clips, local gate, model cancellation and speech.
lib/motion.ts            Image-motion assessment and temporal alert confirmation.
lib/motion-worker.ts     Local OpenCV feature tracking, isolated from the UI thread.
lib/live-contract.ts     Exact-frame model mapping and motion-adjusted priorities.
components/AnalysisPreview.tsx  Boxes aligned to the submitted camera snapshot.
app/api/alert/route.ts   Server-only alert endpoint.
app/api/live/            Live readiness, analysis and verified speech endpoints.
app/api/status/route.ts  Boolean configuration status, no secrets.
public/media/walk.mp4    Browser-compatible copy of supplied footage.
public/media/walk-2.mp4  Browser-compatible second video with original audio.
tests/core.test.ts       Timeline/coordinate and audio race-condition tests.
tools/browser_check.py  Real video and browser lifecycle verification.
artifacts/              Local-only test reports and screenshots; not deployed.

MODEL A CONTRACT
Connected using human data/AGEII_A5000_AI_HANDOFF_WITH_ACCESS, API v1.4.
Newest-frame normalized XYXY is converted to the viewer's XYWH coordinates.
Session ID, frame ID, capture timestamp and dimensions must match exactly.
The model's default maximum result age is 45 seconds. Track identifiers are
session-scoped. Only server-signed, unexpired model warning text can reach
the live speech endpoint. Browser bundles contain no model or voice keys.

VALIDATION
  npm test
  npm run typecheck
  npm run build
  python tools/browser_check.py   (with local server on port 3000)
  python tools/live_quick_check.py
  python tools/real_camera_check.py  (requires a real camera and configured APIs)
The automated browser tests use the actual video and a short test waveform
in place of paid provider responses. Real provider/deployment validation is
recorded separately. Physical iPhone Safari must still be checked on the phone.

Production demo verification (2026-10-09): both complete videos on the public
URL produced all seven real Qwen -> ElevenLabs warnings with completed audio
playbacks, original video sound and decimal scores, without browser errors.
Generation took about 1.1-1.5 seconds. Both returned to Start at video end.
Report: artifacts/hosted-two-video-verification.json
Current validation (2026-10-09): 17 logic tests and the production build pass.
Browser checks confirm startup and alarm audio signals, stationary-camera
foreground approach, all detector boxes even when hazard reasoning is unknown,
a 0.57 spoken warning, one-tap audio recovery, mute and stop. The model request
continues through a motion alert. Physical phone/runner testing remains.

The explicitly requested Model A test on demo video.MOV sampled six clips:
all returned HTTP 200; 13 object detections total across sampled frames.
The last clip returned three curbs scored 0.43, 0.57 and 0.54. Earlier clips
had ordinary objects and uncertain/unlocalized hazards. This exposed the
original danger-only display filtering; the current adapter shows all boxes.
The offline diagnostic bypasses local motion and labels replay sampling times
separately from original media time. It is never treated as live footage.

TOOLS
python tools/live_quick_check.py  Browser motion/audio/box check (local server).
python tools/hosted_check.py     Both full demos with actual paid voice APIs.
python tools/record_demo.py      Browser screen recording with mixed audio.
python tools/model_a_demo_diagnostic.py  Explicit offline Model A investigation.
Tools need Python, Playwright and requests; recording also needs ffmpeg.
Test artifacts, recordings, source MOV files and secrets are not committed.

REFERENCE DOCUMENTATION
https://elevenlabs.io/docs/api-reference/text-to-speech/stream
https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
https://vercel.com/docs/functions/runtimes/node-js
https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback
https://docs.opencv.org/4.13.0/db/d7f/tutorial_js_lucas_kanade.html
https://visionbook.mit.edu/optical_flow.html
