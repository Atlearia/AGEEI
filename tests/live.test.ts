import test from 'node:test';
import assert from 'node:assert/strict';
import { assessFlow, MotionGate, unknownMotion, type FlowPoint } from '../lib/motion';
import { adaptAnalysis, motionAdjustedPriority, type FrameReference } from '../lib/live-contract';
import { signSpeech, verifySpeech, validateContext } from '../server/model-a';
import { isSameOrigin } from '../server/request-guard';

test('same-origin requests use the browser host behind the Next listening address', () => {
  const request = (origin: string) => new Request('http://0.0.0.0:3000/api/live/analyze', { headers: { origin, host: 'localhost:3000' } });
  assert.equal(isSameOrigin(request('http://localhost:3000')), true);
  assert.equal(isSameOrigin(request('https://unrelated.example')), false);
  assert.equal(isSameOrigin(request('null')), false);
  assert.equal(isSameOrigin(request('http://localhost:3001')), false);
});

function field(fn: (x: number, y: number) => [number, number]) {
  const points: FlowPoint[] = [];
  for (let y = .1; y <= .9; y += .08) for (let x = .1; x <= .9; x += .08) { const [dx, dy] = fn(x, y); points.push({ x, y, dx, dy }); }
  return points;
}
test('stationary scenes, camera panning, rotation, and receding motion do not trigger an approach alert', () => {
  for (const fn of [() => [0, 0], () => [.2, -.1], (x: number, y: number) => [-(y - .5) * .4, (x - .5) * .4], (x: number, y: number) => [-(x - .5), -(y - .5)]]) {
    const result = assessFlow(field(fn as (x: number, y: number) => [number, number]), 1000, .95);
    assert.notEqual(result.state, 'ALERT'); assert.equal(result.ttc, null);
  }
  assert.equal(assessFlow(field(() => [0, 0]), 1000, .95).stationary, true);
  assert.equal(assessFlow([], 1000, 0).state, 'UNKNOWN');
  assert.equal(assessFlow(field(() => [1.2, 0]), 1000, .95).state, 'UNKNOWN');
});
test('sustained looming estimates time to contact and triggers locally; isolated noise does not', () => {
  const points = field((x, y) => x >= .3 && x <= .7 && y >= .2 && y <= .8 ? [(x - .5) * .8, (y - .5) * .8] : [0, 0]);
  const result = assessFlow(points, 1000, .95);
  assert.equal(result.state, 'ALERT'); assert.ok(Math.abs(result.ttc! - 1.25) < .01);
  const gate = new MotionGate();
  assert.equal(gate.update(result).state, 'UNKNOWN');
  assert.equal(gate.update({ ...result, timestamp: 1100 }).state, 'ALERT');
  assert.equal(gate.update({ ...result, timestamp: 1200 }).state, 'ALERT');
  assert.equal(gate.update(unknownMotion(1300, 'lost_tracking')).state, 'UNKNOWN');
  assert.equal(gate.update({ ...result, timestamp: 1400 }).state, 'UNKNOWN');
});

const frame: FrameReference = { sessionId: 'camera_test_1', frameId: 'frame_test_1', timestamp: 10000, width: 640, height: 480, timestamps: [8000, 8400, 8800, 9200, 9600, 10000] };

test('a small approaching subject is detected against a stationary background', () => {
  const points = field((x, y) => x >= .38 && x <= .62 && y >= .3 && y <= .82 ? [(x - .5) * .85, (y - .56) * .85] : [0, 0]);
  const result = assessFlow(points, 1000, .95);
  assert.equal(result.state, 'ALERT');
  assert.equal(result.reason, 'rapid_approach');
  assert.equal(assessFlow(field((x, y) => [(x - .5) * .85, (y - .5) * .85]), 1000, .95).reason, 'camera_moving');
  const receding = points.map(p => ({ ...p, dx: -p.dx, dy: -p.dy }));
  assert.equal(assessFlow(receding, 1000, .95).state, 'PASS');
  const crossing = field((x, y) => x >= .26 && x <= .5 && y >= .3 && y <= .8 ? [.32, 0] : [0, 0]);
  assert.equal(assessFlow(crossing, 1000, .95).reason, 'fast_path_crossing');
  assert.equal(assessFlow(crossing.map(p => ({ ...p, dx: -p.dx })), 1000, .95).state, 'PASS');
});
const hazard = { track_id: 2, label: 'person', bbox: [.3, .2, .7, .8], severity_score: 75, detection_confidence: .99, danger_type: 'obstacle_collision', uncertain: false, source_frame_index: 5, source_timestamp_ms: 10000, spoken_warning: 'Person ahead.' };
const raw = { session_id: frame.sessionId, frame_id: frame.frameId, captured_at_ms: frame.timestamp, image: { width: 640, height: 480, coordinate_system: 'normalized_xyxy' }, max_result_age_ms: 45000, status: 'ok', hazards: [hazard] };

test('model scores divide by 100, preserve nulls, convert xyxy, and reject stale or mismatched snapshots', () => {
  const result = adaptAnalysis(raw, frame, 11000);
  assert.equal(result.hazards[0].severity, .75);
  assert.ok(Math.abs(result.hazards[0].box![2] - .4) < 1e-9);
  assert.equal(adaptAnalysis({ ...raw, hazards: [{ ...hazard, severity_score: null }] }, frame, 11000).hazards[0].severity, null);
  assert.equal(adaptAnalysis(raw, frame, 56000).status, 'stale');
  assert.equal(adaptAnalysis({ ...raw, status: 'unknown' }, frame, 11000).hazards.length, 0);
  assert.throws(() => adaptAnalysis({ ...raw, frame_id: 'different' }, frame, 11000));
  assert.equal(adaptAnalysis({ ...raw, hazards: [{ ...hazard, uncertain: true }, { ...hazard, source_frame_index: 0 }] }, frame, 11000).hazards.length, 0);
});
test('all detector boxes survive an unknown hazard assessment without invented scores', () => {
  const r = { ...raw, status: 'unknown', boxes: [
    { label: 'person', bbox: [.2, .1, .5, .8], confidence: .99, severity_score: null },
    { label: 'curb', bbox: [.5, .5, .8, .9], severity_score: 20 }
  ] };
  const result = adaptAnalysis(r, frame, 11000);
  assert.equal(result.boxes.length, 2);
  assert.equal(result.boxes[0].severity, null);
  assert.equal(result.boxes[1].severity, .2);
  assert.equal(result.hazards.length, 0);
  assert.equal(adaptAnalysis(r, frame, 56000).boxes.length, 0);
});
test('still person collision priority drops only with reliable local motion, while static non-motion dangers remain', () => {
  const h = adaptAnalysis(raw, frame, 11000).hazards[0];
  const stationary = assessFlow(field(() => [0, 0]), 10000, .95);
  assert.equal(motionAdjustedPriority(h, stationary), .2);
  assert.equal(motionAdjustedPriority(h, unknownMotion(10000)), .75);
  assert.equal(motionAdjustedPriority({ ...h, dangerType: 'fire' }, stationary), .75);
});
test('speech tickets cannot be forged or used after their analyzed frame expires', () => {
  const previous = process.env.AGEII_API_TOKEN;
  process.env.AGEII_API_TOKEN = 'unit-test-secret-not-a-real-key';
  try {
    const ticket = signSpeech({ text: 'Person ahead.', expiresAt: 20000, frameId: frame.frameId, hazardId: 'person' });
    assert.equal(verifySpeech(ticket, 10000).text, 'Person ahead.');
    assert.throws(() => verifySpeech(ticket.slice(0, -3) + 'xxx', 10000));
    assert.throws(() => verifySpeech(ticket, 20000));
  } finally { if (previous === undefined) delete process.env.AGEII_API_TOKEN; else process.env.AGEII_API_TOKEN = previous; }
  assert.throws(() => validateContext({ action_mode: 'video', frames: [] }));
});
