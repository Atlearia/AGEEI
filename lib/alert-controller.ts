import { SPEECH_RELEASE_THRESHOLD, SPEECH_THRESHOLD, type Detection } from './detection';
import { DEMO } from './demo';

export type AlertPacket = { source: 'demo'; demoId: string; trackId: string; mediaTime: number };
export type VoicePort = { prepare: (packet: AlertPacket, signal: AbortSignal) => Promise<unknown>; play: (audio: unknown) => Promise<void>; stop: () => void };
type Job = { id: string; at: number; epoch: number; controller: AbortController };

/** Decisions are driven by media time, never by a wall-clock timeout timeline. */
export class AlertController {
  private detections: Detection[] = [];
  private time = 0;
  private epoch = 0;
  private muted = false;
  private paused = false;
  private closed = false;
  private job?: Job;
  private speakingId?: string;
  private completed = new Set<string>();
  private retryAt = new Map<string, number>();
  private attempts = new Map<string, number>();
  private voice: VoicePort;
  private onError: (message: string) => void;

  constructor(voice: VoicePort, onError: (message: string) => void, private readonly demoId = DEMO.id) { this.voice = voice; this.onError = onError; }

  update(detections: Detection[], time: number) {
    if (this.closed) return;
    if (time < this.time - 0.25) this.reset(); // Replay / loop / seek backwards.
    if (time > this.time + 1.5) this.invalidate(); // Seek forward: discard pending audio.
    this.detections = detections;
    this.time = time;
    if (this.muted || this.paused) return;
    const active = (id: string) => detections.some(d => d.id === id && d.severity >= SPEECH_RELEASE_THRESHOLD);
    if (this.job && !active(this.job.id)) this.invalidate();
    if (this.speakingId && !active(this.speakingId)) { this.voice.stop(); this.speakingId = undefined; }
    if (this.job || this.speakingId) return;
    const candidate = detections.filter(d => d.severity >= SPEECH_THRESHOLD && !this.completed.has(d.id)
      && (this.attempts.get(d.id) ?? 0) < 2 && time >= (this.retryAt.get(d.id) ?? 0))
      .sort((a, b) => b.severity - a.severity)[0];
    if (candidate) void this.request(candidate);
  }

  private async request(detection: Detection) {
    const job: Job = { id: detection.id, at: this.time, epoch: this.epoch, controller: new AbortController() };
    this.job = job;
    this.attempts.set(job.id, (this.attempts.get(job.id) ?? 0) + 1);
    try {
      const audio = await this.voice.prepare({ source: 'demo', demoId: this.demoId, trackId: job.id, mediaTime: job.at }, job.controller.signal);
      const stillCurrent = !this.closed && !this.muted && !this.paused && this.epoch === job.epoch && !job.controller.signal.aborted
        && this.time - job.at <= 3.5 && this.time >= job.at - 0.1
        && this.detections.some(d => d.id === job.id && d.severity >= SPEECH_RELEASE_THRESHOLD);
      if (!stillCurrent) return;
      this.speakingId = job.id;
      // A resolved play promise means the entire phrase has played successfully.
      await this.voice.play(audio);
      if (this.epoch === job.epoch && !this.closed && !this.muted && !this.paused) this.completed.add(job.id);
    } catch (error) {
      if (!job.controller.signal.aborted && this.epoch === job.epoch && !this.closed) {
        this.retryAt.set(job.id, this.time + 2);
        this.onError(error instanceof Error ? error.message : 'The voice connection failed.');
      }
    } finally {
      if (this.job === job) this.job = undefined;
      if (this.epoch === job.epoch && this.speakingId === job.id) this.speakingId = undefined;
    }
  }

  private invalidate() {
    this.epoch++;
    this.job?.controller.abort(); this.job = undefined;
    this.voice.stop(); this.speakingId = undefined;
  }
  setMuted(muted: boolean) { this.muted = muted; if (muted) this.invalidate(); }
  setPaused(paused: boolean) { this.paused = paused; if (paused) this.invalidate(); }
  reset() { this.invalidate(); this.completed.clear(); this.attempts.clear(); this.retryAt.clear(); }
  dispose() { this.closed = true; this.invalidate(); }
}
