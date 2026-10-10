WAYPOINT | MODEL A

Waypoint is a mobile web accessibility prototype that combines Model A visual
hazard detection with yellow detection overlays and concise spoken warnings.
It runs in a browser and can be added to a phone's home screen.

This repository contains the Model A integration and two prerecorded
demonstrations with authored hazard annotations. Model training and evaluation
scripts are documented separately in training/README.md.

FEATURES
- Live rear-camera input connected to Model A.
- Yellow boxes for valid model detections, including objects without a hazard
  assessment. Boxes appear on the exact analyzed snapshot to preserve alignment.
- Danger scores displayed from 0 to 1. Unknown scores remain unlabelled;
  detector confidence is never presented as a danger score.
- Local OpenCV motion analysis and immediate audible motion alerts.
- Spoken live warnings from Model A assessments, synthesized by ElevenLabs.
- Two prerecorded demonstrations with synchronized overlays, spoken warnings,
  and the original video audio.
- Minimal controls for input selection, overlays, audio, and stopping detection.

HOW IT WORKS
Live camera frames are sampled into short clips and sent through a server-side
adapter to Model A. The adapter validates frame identity, timestamps, dimensions,
and bounding-box coordinates before displaying a result. Model A's boxes and
hazard assessments are handled separately: valid boxes remain visible even when
the hazard assessment is uncertain.

Live warnings use assessed hazards with a danger score of at least 0.50.
Server-signed, expiring speech tickets authorize the corresponding spoken text.
Local motion alerts run independently of the model request. Model inference can
take several seconds, so returned boxes stay attached to the analyzed snapshot.

The prerecorded demonstrations use authored box trajectories and danger scores,
not Model A predictions. Structured demo events are converted into concise
warnings by a configured LLM and synthesized with ElevenLabs. The demo speech
threshold is 0.65. Playback timing, cancellation, and duplicate suppression keep
warnings synchronized with the video.

QUICK START
Requirements: Node.js 22 or newer and npm.

1. Install dependencies:
     npm install
2. Copy .env.example to .env.local and configure the services below.
3. Start the development server:
     npm run dev
4. Open http://localhost:3000 and select an input.

Production build:
  npm run build
  npm start

CONFIGURATION
Set these server-side environment variables in .env.local for local development
or in the hosting platform's environment settings for deployment:

  AGEII_BASE_URL           Model A inference API URL
  AGEII_API_TOKEN          Model A bearer token
  ELEVENLABS_API_KEY       ElevenLabs Text to Speech API key
  ELEVENLABS_VOICE_ID      Optional voice selection
  ELEVENLABS_MODEL_ID      Optional speech model selection

For demo warning generation, configure either an OpenAI-compatible provider:

  LLM_PROVIDER=openai-compatible
  LLM_BASE_URL=<provider API base URL>
  LLM_MODEL=<model name>
  LLM_API_KEY=<API key>

Or configure OpenAI:

  LLM_PROVIDER=openai
  OPENAI_API_KEY=<API key>
  OPENAI_MODEL=<optional model name>

See .env.example for defaults. Credentials stay on the server. Do not commit
.env.local or expose secret values through NEXT_PUBLIC environment variables.

DEPLOYMENT AND PHONE USE
The application supports Vercel deployment. Configure the environment variables,
link a Vercel project, and deploy with the Vercel CLI or dashboard.

Camera access requires HTTPS, except on localhost. On iPhone, open the HTTPS site
in Safari and use Share > Add to Home Screen for a standalone browser window.
Camera detection and audio require the app to remain in the foreground.
An internet connection is required for model and speech service requests.

PROJECT STRUCTURE
  components/Waypoint.tsx       Camera, video, and interface lifecycle
  components/AnalysisPreview.tsx  Detection overlay on the analyzed snapshot
  server/model-a.ts             Model A requests, validation, and speech tickets
  server/voice.ts               LLM and ElevenLabs integrations
  lib/live.ts                  Live camera sampling and analysis coordination
  lib/live-contract.ts         Model response mapping and score normalization
  lib/motion.ts                Motion assessment and alert confirmation
  lib/motion-worker.ts         OpenCV feature tracking
  lib/demo.ts, lib/demo-2.ts    Authored demonstration timelines
  lib/alert-controller.ts      Demo warning timing and cancellation
  app/api/                     Server-side analysis and speech routes
  public/media/                Browser-compatible demonstration videos
  tests/                       Automated behavior checks
  tools/                       Browser checks, recording, and model diagnostics
  training/                    RF-DETR and Gemma training, evaluation, and export

VALIDATION
  npm test
  npm run typecheck
  npm run build

With the app running locally:
  python tools/browser_check.py
  python tools/live_quick_check.py

Camera and provider checks require the relevant hardware and configured APIs.
Python browser tools use Playwright; recording tools also require ffmpeg.
Generated reports and recordings are excluded from version control.

PROTOTYPE LIMITATIONS
Local motion analysis assumes a reasonably stationary camera and does not
measure physical velocity. Camera movement, lighting, occlusion, and inference
latency can affect detections and warning timing. Sensitive motion thresholds
can produce false alerts. Automated checks cover application behavior and audio
playback; they do not establish detection accuracy or safety on a physical phone.
Camera frames are sent to the configured Model A service for analysis.

LICENSE
Application code is provided under the MIT License. Bundled third-party
components retain their respective licenses; see LICENSE and public/vendor.
