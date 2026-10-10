import type { AlertPacket, VoicePort } from './alert-controller';

export class AudioOutput implements VoicePort {
  private context?: AudioContext;
  private source?: AudioBufferSourceNode;
  private finish?: () => void;
  private generation = 0;
  private beeps: OscillatorNode[] = [];
  constructor(private stateChanged: (ready: boolean) => void = () => {}) {}

  private async running() {
    const context = this.context;
    if (!context || context.state === 'closed') throw new Error('Tap the speaker to enable sound.');
    if (context.state !== 'running') {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([context.resume(), new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error('Tap the speaker to enable sound.')), 900);
        })]);
      } catch {
        this.stateChanged(false);
        throw new Error('Tap the speaker to enable sound.');
      } finally { clearTimeout(timer); }
    }
    if (context.state !== 'running') { this.stateChanged(false); throw new Error('Tap the speaker to enable sound.'); }
    this.stateChanged(true);
    return context;
  }

  /** Call directly inside a tap/click, before awaiting a network request. */
  async unlock(confirm = false) {
    const Constructor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Constructor) throw new Error('This browser cannot play spoken warnings.');
    try {
      const audioSession = (navigator as Navigator & { audioSession?: { type: string } }).audioSession;
      if (audioSession) audioSession.type = 'playback';
    } catch { /* Optional platform API; ordinary Web Audio still works. */ }
    if (!this.context || this.context.state === 'closed') {
      this.context = new Constructor({ latencyHint: 'interactive' });
      const context = this.context;
      context.onstatechange = () => this.stateChanged(context.state === 'running');
    }
    const generation = this.generation;
    await this.running();
    if (generation !== this.generation) return;
    if (confirm) { this.stop(); this.tones(1, 660, .11); return; }
    const silence = this.context.createBufferSource();
    silence.buffer = this.context.createBuffer(1, 1, 22050);
    silence.connect(this.context.destination); silence.start();
  }

  async prepare(packet: AlertPacket, signal: AbortSignal) {
    return this.fetchAudio('/api/alert', packet, signal);
  }

  async prepareLive(ticket: string, signal: AbortSignal) { return this.fetchAudio('/api/live/speech', { ticket }, signal); }

  private async fetchAudio(url: string, packet: unknown, signal: AbortSignal) {
    const result = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(packet),
      signal: AbortSignal.any([signal, AbortSignal.timeout(12000)]), cache: 'no-store'
    });
    if (!result.ok) {
      const body = await result.json().catch(() => ({}));
      throw new Error(body.message || 'The voice connection failed.');
    }
    if (!result.headers.get('Content-Type')?.startsWith('audio/')) throw new Error('The voice service did not return audio.');
    const bytes = await result.arrayBuffer();
    if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (!this.context) throw new Error('Tap to enable spoken warnings.');
    return this.context.decodeAudioData(bytes);
  }

  /** Immediate local cue: no network, model or prerecorded demo involved. */
  async beep() {
    const generation = this.generation;
    await this.running();
    if (generation !== this.generation) return false;
    this.stop();
    this.tones(3, 880, .16);
    return true;
  }

  private tones(count: number, frequency: number, volume: number) {
    const context = this.context!;
    for (let i = 0; i < count; i++) {
      const oscillator = context.createOscillator(), gain = context.createGain();
      const at = context.currentTime + i * .19;
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0, at); gain.gain.linearRampToValueAtTime(volume, at + .008);
      gain.gain.setValueAtTime(volume, at + .09); gain.gain.linearRampToValueAtTime(0, at + .12);
      oscillator.connect(gain); gain.connect(context.destination);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
      oscillator.start(at); oscillator.stop(at + .13); this.beeps.push(oscillator);
    }
  }

  async play(value: unknown) {
    const beforeResume = this.generation;
    await this.running();
    if (beforeResume !== this.generation) return;
    this.stop();
    const generation = this.generation;
    const source = this.context!.createBufferSource();
    source.buffer = value as AudioBuffer;
    source.connect(this.context!.destination);
    this.source = source;
    return new Promise<void>((resolve) => {
      this.finish = resolve;
      source.onended = () => { source.disconnect(); if (generation === this.generation) { this.source = undefined; this.finish = undefined; } resolve(); };
      source.start();
    });
  }

  stop() {
    this.generation++;
    for (const beep of this.beeps) { try { beep.stop(); } catch { /* ended */ } beep.disconnect(); }
    this.beeps = [];
    if (this.source) { try { this.source.stop(); } catch { /* already ended */ } this.source.disconnect(); this.source = undefined; }
    this.finish?.(); this.finish = undefined;
  }
  dispose() { this.stop(); if (this.context) { this.context.onstatechange = null; void this.context.close(); } this.context = undefined; }
}
