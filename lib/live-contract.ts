import type { Box, Direction } from './detection';
import type { MotionResult } from './motion';

export type ContextFrame = { image_base64: string; timestamp_ms: number };
export type ContextRequest = { session_id: string; frame_id: string; action_mode: 'video'; frames: ContextFrame[] };
export type FrameReference = { sessionId: string; frameId: string; timestamp: number; width: number; height: number; timestamps: number[] };
export type LiveHazard = { id: string; label: string; description: string; warning: string; box: Box | null; severity: number | null; direction: Direction; dangerType: string; speechTicket?: string };
export type LiveBox = { id: string; label: string; box: Box; severity: number | null };
export type LiveAssessment = { status: 'ok' | 'unknown' | 'stale'; frameId: string; capturedAt: number; expiresAt: number; width: number; height: number; boxes: LiveBox[]; hazards: LiveHazard[]; latencyMs: number };
export const LIVE_SPEECH_THRESHOLD = .5;
const record = (v: unknown): Record<string, unknown> => v && typeof v === 'object' ? v as Record<string, unknown> : {};
const line = (v: unknown, max = 160) => typeof v === 'string' ? v.trim().slice(0, max) : '';

export function normalizedXYXY(value: unknown): Box | null {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) return null;
  const [x1, y1, x2, y2] = value;
  return x2 > x1 && y2 > y1 ? [x1, y1, x2 - x1, y2 - y1] : null;
}

/** Detector boxes and hazard reasoning are independent outputs. Show all valid
 * newest-frame detector boxes even when the language assessment is unknown. */
export function adaptAnalysis(raw: unknown, frame: FrameReference, now = Date.now()): LiveAssessment {
  const r = record(raw), image = record(r.image);
  if (r.session_id !== frame.sessionId || r.frame_id !== frame.frameId || r.captured_at_ms !== frame.timestamp
    || image.width !== frame.width || image.height !== frame.height || image.coordinate_system !== 'normalized_xyxy') throw new Error('Model result does not match its captured frame.');
  const ageLimit = typeof r.max_result_age_ms === 'number' && Number.isFinite(r.max_result_age_ms) ? Math.min(45000, Math.max(0, r.max_result_age_ms)) : 0;
  const expiresAt = frame.timestamp + ageLimit;
  const status = expiresAt <= now || frame.timestamp > now + 3000 || r.status === 'stale' ? 'stale' : r.status === 'ok' ? 'ok' : 'unknown';
  const result: LiveAssessment = { status, frameId: frame.frameId, capturedAt: frame.timestamp, expiresAt, width: frame.width, height: frame.height, boxes: [], hazards: [], latencyMs: typeof r.latency_ms === 'number' ? r.latency_ms : 0 };
  if (status === 'stale') return result;
  if (Array.isArray(r.boxes)) for (const [index, item] of r.boxes.entries()) {
    const b = record(item), box = normalizedXYXY(b.bbox);
    if (!box) continue;
    const severity = typeof b.severity_score === 'number' && Number.isFinite(b.severity_score) && b.severity_score >= 0 && b.severity_score <= 100 ? b.severity_score / 100 : null;
    result.boxes.push({ id: `${frame.frameId}:box:${index}`, label: line(b.label, 60), box, severity });
  }
  if (status !== 'ok' || !Array.isArray(r.hazards)) return result;
  const joined = new Map<string, LiveHazard>();
  for (const item of r.hazards.slice(0, 80)) {
    const h = record(item);
    if (h.uncertain !== false || h.source_frame_index !== frame.timestamps.length - 1 || h.source_timestamp_ms !== frame.timestamp) continue;
    const box = normalizedXYXY(h.bbox);
    const score = typeof h.severity_score === 'number' && Number.isFinite(h.severity_score) && h.severity_score >= 0 && h.severity_score <= 100 ? h.severity_score / 100 : null;
    const id = `${frame.sessionId}:${String(h.track_id ?? 'unlocalized').slice(0, 80)}:${line(h.danger_type, 50)}`;
    const cx = box ? box[0] + box[2] / 2 : .5;
    const hazard: LiveHazard = { id, label: line(h.label, 60) || 'Obstacle', description: line(h.description, 500), warning: line(h.spoken_warning), box, severity: score, direction: cx < .35 ? 'left' : cx > .65 ? 'right' : 'ahead', dangerType: line(h.danger_type, 60) };
    const previous = joined.get(id);
    if (!previous || (hazard.severity ?? -1) > (previous.severity ?? -1)) joined.set(id, hazard);
  }
  result.hazards = [...joined.values()].sort((a, b) => (b.severity ?? -1) - (a.severity ?? -1));
  for (const hazard of result.hazards) if (hazard.box && !result.boxes.some(b => b.box.every((v, i) => Math.abs(v - hazard.box![i]) < .0001))) {
    result.boxes.push({ id: hazard.id, label: hazard.label, box: hazard.box, severity: hazard.severity });
  }
  return result;
}

/** Suppress imminent-collision priority for a demonstrably stationary person/vehicle.
 * Keep the server score for falls/fire/unknown motion. Never infer safety from no flow.
 */
export function motionAdjustedPriority(hazard: LiveHazard, motion: MotionResult): number | null {
  if (hazard.severity === null || !hazard.box || !motion.stationary || motion.quality < .7) return hazard.severity;
  if (!/^(person|car|bus|truck|train|bicycle|motorcycle|dog)$/i.test(hazard.label) || !/collision|traffic/.test(hazard.dangerType)) return hazard.severity;
  const [x, y, w, h] = hazard.box;
  const points = motion.points.filter(p => p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h);
  if (points.length < 6) return hazard.severity;
  const maxSpeed = Math.max(...points.map(p => Math.hypot(p.dx, p.dy)));
  return maxSpeed < .018 ? Math.min(hazard.severity, .2) : hazard.severity;
}
