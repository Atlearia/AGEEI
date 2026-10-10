import { AudioOutput } from './audio';
import { MOTION_INTERVAL_MS, unknownMotion, type MotionResult } from './motion';
import { LIVE_SPEECH_THRESHOLD, type ContextFrame, type LiveAssessment } from './live-contract';

type Snapshot = ContextFrame & { frameId: string; width: number; height: number; url: string; motion: MotionResult };
export type LiveView = { assessment: LiveAssessment; url: string };
type Hooks = { motion(result: MotionResult): void; result(view: LiveView | null): void; status(state: string): void; error(message: string): void };

/** Local motion provides immediate beeps independently of slower model context.
 * Captures and snapshots are held only in memory; there is no camera recording.
 */
export class LiveCamera {
  private worker?: Worker;
  private workerReady = false;
  private workerBusy = false;
  private closed = false;
  private paused = false;
  private muted = false;
  private generation = 0;
  private motion = unknownMotion(Date.now());
  private motionCanvas = document.createElement('canvas');
  private contextCanvas = document.createElement('canvas');
  private lastFrame = -1;
  private lastMotion = 0;
  private lastCapture = 0;
  private lastBeep = -Infinity;
  private frames: Snapshot[] = [];
  private pendingCapture?: { timestamp: number; frameId: string; width: number; height: number };
  private request?: AbortController;
  private speech?: AbortController;
  private busy = false;
  private modelAvailable = false;
  private modelDisabled = false;
  private checkingReady = false;
  private nextReady = 0;
  private nextRequest = 0;
  private frameMax = 640;
  private assessment?: LiveAssessment;
  private speechHistory = new Map<string, number>();
  private sessionId = `camera_${crypto.randomUUID().replace(/-/g, '')}`;

  constructor(private video: HTMLVideoElement, private audio: AudioOutput, private hooks: Hooks) {}

  start() {
    this.worker = new Worker('/motion-worker.js', { name: 'waypoint-motion' });
    this.worker.onmessage = event => {
      if (this.closed) return;
      if (event.data.type === 'ready') { this.workerReady = true; return; }
      if (event.data.type === 'failed') { this.hooks.error('Motion analysis could not load. Reopen the camera.'); this.hooks.status('motion_unavailable'); return; }
      this.workerBusy = false;
      if (this.paused) return;
      const result = event.data as MotionResult;
      const captured = this.pendingCapture; this.pendingCapture = undefined;
      this.acceptMotion(Date.now() - result.timestamp <= 500 ? result : unknownMotion(Date.now(), 'motion_delayed'));
      if (captured && captured.timestamp === result.timestamp && !this.busy && this.modelAvailable) {
        const url = this.contextCanvas.toDataURL('image/jpeg', .72);
        const snapshot: Snapshot = { image_base64: url.split(',')[1], timestamp_ms: captured.timestamp, frameId: captured.frameId, width: captured.width, height: captured.height, url, motion: result };
        this.frames.push(snapshot);
        this.frames = this.frames.filter(f => captured.timestamp - f.timestamp_ms <= 3500).slice(-6);
        if (this.frames.length === 6 && captured.timestamp - this.frames[0].timestamp_ms >= 1000) {
          const frames = this.frames; this.frames = []; void this.analyze(frames);
        }
      }
    };
    this.worker.onerror = () => { this.workerBusy = false; this.workerReady = false; this.acceptMotion(unknownMotion(Date.now(), 'motion_unavailable')); this.hooks.error('Motion analysis stopped. Reopen the camera.'); };
    this.hooks.motion(this.motion); this.hooks.status('warming_up');
    void this.checkReady();
  }

  private async checkReady() {
    if (this.closed || this.modelDisabled || this.checkingReady || Date.now() < this.nextReady) return;
    this.checkingReady = true;
    try {
      const response = await fetch('/api/live/status', { cache: 'no-store', signal: AbortSignal.timeout(10000) });
      const status = await response.json();
      if (this.closed) return;
      this.modelAvailable = status.ready === true;
      this.modelDisabled = status.configured === false;
      if (!this.modelAvailable) this.hooks.error(this.modelDisabled ? 'Model A access is not configured. Local motion alerts are still available.' : 'Model A is unavailable. Local motion alerts are still available.');
    } catch { if (!this.closed) { this.modelAvailable = false; this.hooks.status('model_unavailable'); } }
    finally { this.checkingReady = false; this.nextReady = Date.now() + 5000; }
  }

  private invalidate() {
    this.generation++; this.frames = []; this.pendingCapture = undefined;
    this.request?.abort(); this.speech?.abort(); this.speech = undefined;
    this.audio.stop(); this.assessment = undefined; this.hooks.result(null);
  }

  private acceptMotion(result: MotionResult) {
    const previous = this.motion.state;
    this.motion = result;
    this.hooks.motion(result);
    if (result.state !== 'PASS') {
      // An alert interrupts speech but keeps the model request: otherwise the
      // moving person that triggered the alarm would never reach Model A.
      if (result.state === 'ALERT' && previous !== 'ALERT') { this.speech?.abort(); this.audio.stop(); }
      this.hooks.status(result.state === 'ALERT' ? 'motion_alert' : result.reason);
      if (result.state === 'ALERT' && !this.muted && Date.now() - this.lastBeep > 1800) {
        this.lastBeep = Date.now();
        const generation = this.generation;
        void this.audio.beep().catch(() => {
          if (!this.closed && !this.paused && !this.muted && generation === this.generation) this.hooks.error('Tap the speaker to enable sound.');
        });
      }
    } else if (previous !== 'PASS') { this.hooks.status(this.busy ? 'analyzing' : 'collecting'); }
  }

  frame(mediaTime: number) {
    if (this.closed || this.paused || this.video.paused || this.video.readyState < 2 || mediaTime <= this.lastFrame) return;
    this.lastFrame = mediaTime;
    const now = Date.now();
    if (this.assessment && now >= this.assessment.expiresAt) { this.speech?.abort(); this.audio.stop(); this.assessment = undefined; this.hooks.result(null); }
    if (this.motion.state !== 'UNKNOWN' && now - this.motion.timestamp > 500) this.acceptMotion(unknownMotion(now, 'motion_delayed'));
    if (!this.modelAvailable) void this.checkReady();
    if (this.workerReady && !this.workerBusy && now - this.lastMotion >= MOTION_INTERVAL_MS) {
      this.lastMotion = now; this.workerBusy = true;
      let source: CanvasImageSource = this.video;
      // Sample the exact same image for motion and context. Encoding/upload waits
      // until the worker has returned PASS for THIS captured frame.
      if (!this.busy && this.modelAvailable && !this.modelDisabled && now >= this.nextRequest && now - this.lastCapture >= 399) {
        this.lastCapture = now;
        const scale = Math.min(1, this.frameMax / Math.max(this.video.videoWidth, this.video.videoHeight));
        this.contextCanvas.width = Math.round(this.video.videoWidth * scale); this.contextCanvas.height = Math.round(this.video.videoHeight * scale);
        this.contextCanvas.getContext('2d')!.drawImage(this.video, 0, 0, this.contextCanvas.width, this.contextCanvas.height);
        source = this.contextCanvas;
        this.pendingCapture = { timestamp: now, frameId: `frame_${crypto.randomUUID().replace(/-/g, '')}`, width: this.contextCanvas.width, height: this.contextCanvas.height };
      }
      const scale = Math.min(1, 256 / Math.max(this.video.videoWidth, this.video.videoHeight));
      const canvas = this.motionCanvas;
      canvas.width = Math.round(this.video.videoWidth * scale); canvas.height = Math.round(this.video.videoHeight * scale);
      const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer;
      this.worker!.postMessage({ pixels, width: canvas.width, height: canvas.height, timestamp: now }, [pixels]);
    }
  }

  private async analyze(frames: Snapshot[]) {
    this.busy = true;
    const generation = this.generation, snapshot = frames.at(-1)!;
    const controller = new AbortController(); this.request = controller;
    const active = () => !this.closed && !this.paused && generation === this.generation && !controller.signal.aborted;
    this.hooks.status('analyzing');
    try {
      const response = await fetch('/api/live/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: this.sessionId, frame_id: snapshot.frameId, action_mode: 'video', frames: frames.map(({ image_base64, timestamp_ms }) => ({ image_base64, timestamp_ms })) }), signal: AbortSignal.any([controller.signal, AbortSignal.timeout(68000)]), cache: 'no-store' });
      if (!active()) return;
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        this.nextRequest = Date.now() + Math.min(60000, Math.max(1000, (Number(response.headers.get('Retry-After')) || 2) * 1000));
        if (response.status === 429) { this.hooks.status('model_busy'); return; }
        if (response.status === 409) { this.hooks.status('collecting'); return; }
        if (response.status === 413) { this.frameMax = 416; return; }
        if ([401, 422].includes(response.status)) this.modelDisabled = true;
        if (response.status >= 500) { this.modelAvailable = false; this.nextReady = this.nextRequest; }
        throw new Error(body.message || 'Model A could not analyze this clip.');
      }
      const result = await response.json() as LiveAssessment;
      if (!active()) return;
      if (result.frameId !== snapshot.frameId || result.capturedAt !== snapshot.timestamp_ms || result.width !== snapshot.width || result.height !== snapshot.height) throw new Error('Model A returned a different camera frame.');
      if (result.status === 'stale' || Date.now() >= result.expiresAt) { this.hooks.result(null); this.hooks.status('model_stale'); return; }
      this.assessment = result;
      this.hooks.result({ assessment: result, url: snapshot.url }); this.hooks.status('assessed');
      const hazard = result.hazards.find(h => h.severity !== null && h.severity >= LIVE_SPEECH_THRESHOLD && h.speechTicket);
      if (hazard && !this.muted) {
        const key = `${hazard.label}:${hazard.direction}:${hazard.dangerType}`;
        if (Date.now() - (this.speechHistory.get(key) || 0) >= 25000) {
          this.speechHistory.set(key, Date.now());
          void this.speak(hazard.speechTicket!, result.expiresAt, generation);
        }
      }
    } catch (e) {
      if (active()) { this.hooks.error(e instanceof Error ? e.message : 'Model A is unavailable.'); this.hooks.status('model_unavailable'); this.nextRequest = Date.now() + 3000; }
    } finally {
      if (this.request === controller) { this.request = undefined; this.busy = false; this.frames = []; }
    }
  }

  private async speak(ticket: string, expiresAt: number, generation: number) {
    this.speech?.abort(); const controller = new AbortController(); this.speech = controller;
    try {
      const clip = await this.audio.prepareLive(ticket, controller.signal);
      if (this.closed || this.paused || this.muted || this.generation !== generation || Date.now() >= expiresAt || controller.signal.aborted) return;
      await this.audio.play(clip);
    } catch (e) { if (!controller.signal.aborted && !this.closed) this.hooks.error(e instanceof Error ? e.message : 'The camera warning could not play.'); }
  }

  setMuted(muted: boolean) { this.muted = muted; if (muted) { this.speech?.abort(); this.audio.stop(); } }
  setPaused(paused: boolean) { this.paused = paused; if (paused) { this.invalidate(); this.worker?.postMessage({ type: 'reset' }); this.motion = unknownMotion(Date.now(), 'paused'); this.hooks.motion(this.motion); } }
  dispose() { this.closed = true; this.invalidate(); this.worker?.terminate(); this.worker = undefined; this.workerReady = false; this.frames = []; }
}
