import type { Box } from './detection';

export type FlowPoint = { x: number; y: number; dx: number; dy: number };
export type MotionState = 'UNKNOWN' | 'PASS' | 'ALERT';
export type MotionResult = {
  state: MotionState; timestamp: number; reason: string; quality: number;
  imageSpeed: number; stationary: boolean; ttc: number | null; score: number;
  region: Box | null; points: FlowPoint[];
};
export const MOTION_INTERVAL_MS = 100;
export const MOTION_TTC_SECONDS = 4;
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0; };

// Least-squares affine image velocity, with coordinates centered at the image center.
// Fit only moving foreground points; static scenery must not dilute object expansion.
function fit(points: FlowPoint[]) {
  const solve = (field: 'dx' | 'dy') => {
    const a = Array.from({ length: 3 }, () => [0, 0, 0, 0]);
    for (const p of points) {
      const row = [p.x - .5, p.y - .5, 1];
      for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) a[i][j] += row[i] * row[j]; a[i][3] += row[i] * p[field]; }
    }
    for (let i = 0; i < 3; i++) {
      let pivot = i;
      for (let j = i + 1; j < 3; j++) if (Math.abs(a[j][i]) > Math.abs(a[pivot][i])) pivot = j;
      [a[i], a[pivot]] = [a[pivot], a[i]];
      if (Math.abs(a[i][i]) < 1e-7) return null;
      const scale = a[i][i]; for (let j = i; j < 4; j++) a[i][j] /= scale;
      for (let k = 0; k < 3; k++) if (k !== i) { const f = a[k][i]; for (let j = i; j < 4; j++) a[k][j] -= f * a[i][j]; }
    }
    return a.map(row => row[3]);
  };
  const x = solve('dx'), y = solve('dy');
  if (!x || !y) return null;
  const residuals = points.map(p => Math.hypot(p.dx - (x[0] * (p.x - .5) + x[1] * (p.y - .5) + x[2]), p.dy - (y[0] * (p.x - .5) + y[1] * (p.y - .5) + y[2])));
  return { x, y, residual: median(residuals) };
}

export function unknownMotion(timestamp: number, reason = 'warming_up'): MotionResult {
  return { state: 'UNKNOWN', timestamp, reason, quality: 0, imageSpeed: 0, stationary: false, ttc: null, score: 0, region: null, points: [] };
}

/** Stationary-camera mode. Image-plane motion in frame units / second, NOT m/s.
 * A PASS means no rapid-motion trigger was found; it does not mean a safe route.
 */
export function assessFlow(points: FlowPoint[], timestamp: number, quality: number): MotionResult {
  if (points.length < 10 || quality < .35) return unknownMotion(timestamp, 'insufficient_tracking');
  const speed = median(points.map(p => Math.hypot(p.dx, p.dy)));
  const base: MotionResult = { state: 'PASS', timestamp, reason: 'no_motion_trigger', quality, imageSpeed: speed, stationary: speed < .012, ttc: null, score: 0, region: null, points };
  // Moving scenery on three sides is evidence that the stationary-camera
  // assumption broke. Never reinterpret a pan/zoom as an approaching object.
  const borders = [points.filter(p => p.x < .2), points.filter(p => p.x > .8), points.filter(p => p.y < .18), points.filter(p => p.y > .85)];
  const movingBorders = borders.filter(group => group.length >= 4 && median(group.map(p => Math.hypot(p.dx, p.dy))) > .035);
  if (movingBorders.length >= 3) return { ...unknownMotion(timestamp, 'camera_moving'), quality };

  const moving = points.filter(p => Math.hypot(p.dx, p.dy) > .028);

  // Overlapping central regions reduce dilution of a moving person by a static background.
  const regions: Box[] = [[.18, .15, .64, .7], [.15, .2, .4, .65], [.3, .2, .4, .65], [.45, .2, .4, .65], [.22, .32, .56, .5]];
  for (const region of regions) {
    const [x, y, w, h] = region;
    const group = moving.filter(p => p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h);
    if (group.length < 4) continue;
    const local = fit(group);
    if (!local || local.residual > .065) continue;
    const expansion = (local.x[0] + local.y[1]) / 2;
    // Require expansion in both axes, not a swinging limb or a camera translation.
    const looming = Math.min(local.x[0], local.y[1]) > .15 && expansion >= 1 / MOTION_TTC_SECONDS;
    if (looming) {
      const ttc = 1 / expansion;
      return { ...base, state: 'ALERT', reason: 'rapid_approach', stationary: false, ttc, score: Math.min(.98, .65 + (MOTION_TTC_SECONDS - ttc) * .2), region };
    }
    const residual = group.filter(p => Math.abs(p.dx) > .12);
    if (residual.length < 4 || residual.length < group.length * .55) continue;
    const dx = median(residual.map(p => p.dx));
    const cx = median(residual.map(p => p.x)), cy = median(residual.map(p => p.y));
    if (cy > .25 && cx >= .15 && cx <= .85 && Math.abs(dx) > .14 && ((cx < .5 && dx > 0) || (cx > .5 && dx < 0))) {
      return { ...base, state: 'ALERT', reason: 'fast_path_crossing', stationary: false, score: .75, region };
    }
  }
  return base;
}

/** Require sustained evidence; a single noisy frame never beeps. */
export class MotionGate {
  private hits = 0;
  private lastTime = 0;
  private held?: MotionResult;
  update(result: MotionResult): MotionResult {
    if (result.timestamp <= this.lastTime || result.timestamp - this.lastTime > 500) { this.hits = 0; this.held = undefined; }
    this.lastTime = result.timestamp;
    if (result.state === 'UNKNOWN') { this.hits = 0; this.held = undefined; return result; }
    if (result.state === 'ALERT') {
      if (++this.hits >= 2) { this.held = result; return result; }
      return { ...result, state: 'UNKNOWN', reason: 'confirming_motion' };
    }
    this.hits = 0;
    if (this.held && result.timestamp - this.held.timestamp < 650) return { ...this.held, timestamp: result.timestamp };
    this.held = undefined;
    return result;
  }
  reset() { this.hits = 0; this.lastTime = 0; this.held = undefined; }
}
