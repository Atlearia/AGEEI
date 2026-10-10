import test from 'node:test';
import assert from 'node:assert/strict';
import { AlertController, type AlertPacket, type VoicePort } from '../lib/alert-controller';
import { demoDetectionsAt, DEMO_TRACKS, DEMOS } from '../lib/demo';
import { containRect, validDetection, type Detection } from '../lib/detection';
import { warningText, synthesize } from '../server/voice';
import { POST } from '../app/api/alert/route';

const danger: Detection = { id: 'curb', label: 'Raised curb', direction: 'right', severity: 0.8, box: [0.5, 0.4, 0.3, 0.2] };
const flush = () => new Promise(resolve => setTimeout(resolve, 0));
function deferred<T>() { let resolve!: (v: T) => void; const promise = new Promise<T>(r => resolve = r); return { promise, resolve }; }

test('actual demo has clear sections, visible dangers, distinct speech crossings, and valid source bounds', () => {
  assert.deepEqual(demoDetectionsAt(0.8), []);
  assert.deepEqual(demoDetectionsAt(7.6), []);
  assert.deepEqual(demoDetectionsAt(16.5), []);
  assert.equal(demoDetectionsAt(2)[0].severity < 0.65, true);
  assert.equal(demoDetectionsAt(3)[0].label, 'Raised curb');
  assert.equal(demoDetectionsAt(10.5)[0].label, 'Barrier');
  for (let time = 0; time < 18.3; time += 0.033) for (const d of demoDetectionsAt(time)) assert.ok(validDetection(d), `${d.id} at ${time}`);
  assert.ok(DEMO_TRACKS.every(t => t.keyframes.every((f, i) => i === 0 || f.time > t.keyframes[i - 1].time)));
});

test('contain geometry preserves exact image coordinates through portrait and landscape letterboxing', () => {
  const portrait = containRect(390, 746, 720, 1280);
  assert.equal(portrait.width, 390); assert.equal(portrait.x, 0);
  assert.ok(Math.abs(portrait.height - 693.3333333333) < 0.00001);
  assert.ok(Math.abs(portrait.y - 26.3333333333) < 0.00001);
  const r = containRect(800, 400, 720, 1280);
  assert.equal(r.height, 400); assert.equal(r.width, 225); assert.equal(r.x, 287.5);
  assert.deepEqual(containRect(0, 400, 720, 1280), { x: 0, y: 0, width: 0, height: 0 });
});

test('each video has its own valid threat timeline, including concurrent post and car hazards', () => {
  for (const demo of DEMOS) {
    for (let time = 0; time <= demo.duration; time += .033) {
      for (const detection of demoDetectionsAt(time, demo.id)) assert.ok(validDetection(detection), `${demo.id}: ${detection.id} at ${time}`);
    }
    for (const track of demo.tracks) {
      assert.ok(track.keyframes.every((f, i) => i === 0 || f.time > track.keyframes[i - 1].time));
      assert.ok(track.start >= 0 && track.end <= demo.duration);
    }
  }
  const second = DEMOS[1].id;
  assert.deepEqual(demoDetectionsAt(20, second), []);
  assert.deepEqual(demoDetectionsAt(80, second), []);
  assert.deepEqual(demoDetectionsAt(100, second), []);
  assert.equal(demoDetectionsAt(7, second)[0].id, 'planter-edge');
  assert.equal(demoDetectionsAt(35, second)[0].id, 'parked-bicycles');
  assert.deepEqual(demoDetectionsAt(59, second).map(d => d.id), ['lamp-post', 'car-ahead']);
  assert.equal(demoDetectionsAt(91, second)[0].id, 'guardrail');
  assert.deepEqual(demoDetectionsAt(7, 'unknown'), []);
});

test('voice requests identify the selected video; the server rejects events from another video', async () => {
  const packets: AlertPacket[] = [];
  const engine = new AlertController({ prepare: async packet => { packets.push(packet); return 'audio'; }, play: async () => {}, stop: () => {} }, () => assert.fail(), DEMOS[1].id);
  engine.update(demoDetectionsAt(7, DEMOS[1].id), 7); await flush();
  assert.equal(packets[0].demoId, DEMOS[1].id);
  assert.equal(packets[0].trackId, 'planter-edge');
  engine.dispose();
  for (const [body, status] of [
    [{ source: 'demo', demoId: DEMOS[0].id, trackId: 'planter-edge', mediaTime: 7 }, 422],
    [{ source: 'demo', demoId: DEMOS[1].id, trackId: 'raised-edge', mediaTime: 3 }, 422],
    [{ source: 'demo', demoId: 'unknown', trackId: 'car-ahead', mediaTime: 65 }, 400],
    [{ source: 'demo', demoId: DEMOS[0].id, trackId: 'car-ahead', mediaTime: 65 }, 400],
    [null, 400]
  ] as const) {
    const response = await POST(new Request('https://example.test/api/alert', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }));
    assert.equal(response.status, status);
  }
});

test('quiet scene and below-threshold dangers never call voice; continued presence speaks only once', async () => {
  let requested = 0, played = 0;
  const port: VoicePort = { prepare: async () => { requested++; return 'clip'; }, play: async () => { played++; }, stop: () => {} };
  const engine = new AlertController(port, () => assert.fail());
  engine.update([], 0); engine.update([{ ...danger, severity: 0.64 }], 0.5);
  assert.equal(requested, 0);
  engine.update([danger], 1); await flush();
  for (let t = 1.1; t < 10; t += .2) engine.update([danger], t);
  await flush(); assert.equal(requested, 1); assert.equal(played, 1);
  engine.dispose();
});

test('late audio is discarded after hazard disappears or the session stops', async () => {
  for (const mode of ['cleared', 'stopped', 'muted', 'paused']) {
    const pending = deferred<unknown>(); let plays = 0, signal: AbortSignal | undefined;
    const engine = new AlertController({ prepare: (_packet, s) => { signal = s; return pending.promise; }, play: async () => { plays++; }, stop: () => {} }, () => assert.fail());
    engine.update([danger], 1);
    if (mode === 'cleared') engine.update([], 1.2);
    if (mode === 'stopped') engine.dispose();
    if (mode === 'muted') engine.setMuted(true);
    if (mode === 'paused') engine.setPaused(true);
    assert.equal(signal?.aborted, true);
    pending.resolve('clip'); await flush(); assert.equal(plays, 0, mode);
    engine.dispose();
  }
});

test('request becoming stale during continuous playback never speaks; resetting replay permits warning again', async () => {
  const pending = deferred<unknown>(); let plays = 0;
  const engine = new AlertController({ prepare: () => pending.promise, play: async () => { plays++; }, stop: () => {} }, () => assert.fail());
  for (let t = 0; t < 5; t += .25) engine.update([danger], t);
  pending.resolve('clip'); await flush(); assert.equal(plays, 0);
  engine.reset(); engine.update([danger], 0); await flush(); assert.equal(plays, 1);
  engine.dispose();
});

test('LLM receives only structured facts and ElevenLabs receives its concise output with server-side authentication', async () => {
  const c = { llmKey: 'test-secret', baseUrl: 'https://example.test/v1', model: 'qwen-turbo', provider: 'openai-compatible', elevenKey: 'test-speech-secret', voiceId: 'voice', voiceModel: 'eleven_flash_v2_5' };
  const calls: { url: string; body: Record<string, unknown>; headers: Record<string, string> }[] = [];
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)), headers: init?.headers as Record<string, string> });
    return calls.length === 1 ? Response.json({ choices: [{ message: { content: 'Raised curb on your right.' } }] }) : new Response(new Uint8Array(256), { headers: { 'Content-Type': 'audio/mpeg' } });
  }) as typeof fetch;
  const text = await warningText(danger, c, fetcher);
  const audio = await synthesize(text, c, fetcher);
  assert.equal(text, 'Raised curb on your right.'); assert.equal(audio.length, 256);
  assert.equal(calls[0].headers.Authorization, 'Bearer test-secret');
  assert.equal(calls[1].headers['xi-api-key'], 'test-speech-secret');
  assert.equal(calls[1].body.text, text);
  assert.ok(!JSON.stringify(calls[0].body).includes('test-secret'));
});
