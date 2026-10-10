import test from 'node:test';
import assert from 'node:assert/strict';
import { AudioOutput } from '../lib/audio';

// Model browser interruption, not provider audio: a scheduled oscillator on a
// suspended context is silent even though start() was called successfully.
class Context {
  static instance: Context;
  state = 'suspended'; currentTime = 0; destination = {};
  onstatechange: (() => void) | null = null;
  starts = 0; resumes = 0;
  restore?: () => void;
  constructor() { Context.instance = this; }
  async resume() { this.resumes++; this.state = 'running'; this.onstatechange?.(); }
  async close() { this.state = 'closed'; }
  createBuffer() { return {}; }
  createBufferSource() { return { buffer: null, connect() {}, start() {}, stop() {}, disconnect() {} }; }
  createOscillator() {
    return { frequency: { value: 0 }, onended: null, connect() {}, disconnect() {}, stop() {}, start: () => {
      assert.equal(this.state, 'running'); this.starts++;
    } };
  }
  createGain() { return { gain: { setValueAtTime() {}, linearRampToValueAtTime() {} }, connect() {}, disconnect() {} }; }
}

test('a suspended audio context resumes before beeps, and blocked playback is reported', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'window');
  Object.defineProperty(globalThis, 'window', { configurable: true, value: { AudioContext: Context } });
  const states: boolean[] = [];
  const audio = new AudioOutput(ready => states.push(ready));
  try {
    await audio.unlock(true);
    const context = Context.instance;
    assert.equal(context.starts, 1);
    context.state = 'suspended';
    assert.equal(await audio.beep(), true);
    assert.equal(context.starts, 4); assert.equal(context.resumes, 2);
    context.state = 'suspended';
    context.resume = async () => { throw new Error('Gesture required'); };
    await assert.rejects(audio.beep(), /Tap the speaker/);
    assert.equal(context.starts, 4); assert.equal(states.at(-1), false);

    // Muting/stopping during a pending resume must not play a late alarm.
    context.resume = () => new Promise<void>(resolve => { context.restore = () => { context.state = 'running'; resolve(); }; });
    const pending = audio.beep();
    audio.stop(); context.restore!();
    assert.equal(await pending, false); assert.equal(context.starts, 4);
  } finally {
    audio.dispose();
    if (original) Object.defineProperty(globalThis, 'window', original);
    else Reflect.deleteProperty(globalThis, 'window');
  }
});
