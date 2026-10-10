export type Box = readonly [number, number, number, number];
export type Direction = 'left' | 'ahead' | 'right';
export type Detection = { id: string; label: string; severity: number; box: Box; direction: Direction };
export type Keyframe = { time: number; box: Box; severity: number };
export type Track = { id: string; label: string; direction: Direction; start: number; end: number; keyframes: readonly Keyframe[] };

export const BOX_THRESHOLD = 0.3;
export const SPEECH_THRESHOLD = 0.65;
export const SPEECH_RELEASE_THRESHOLD = 0.55;

export function interpolateTrack(track: Track, time: number): Detection | null {
  if (!Number.isFinite(time) || time < track.start || time >= track.end) return null;
  const points = track.keyframes;
  if (!points.length) return null;
  let a = points[0], b = points[points.length - 1];
  if (time <= a.time) b = a;
  else if (time >= b.time) a = b;
  else for (let i = 1; i < points.length; i++) if (time < points[i].time) { a = points[i - 1]; b = points[i]; break; }
  const f = a.time === b.time ? 0 : (time - a.time) / (b.time - a.time);
  const box = a.box.map((v, i) => v + (b.box[i] - v) * f) as unknown as Box;
  return { id: track.id, label: track.label, direction: track.direction, box, severity: a.severity + (b.severity - a.severity) * f };
}

export function detectionsAt(tracks: readonly Track[], time: number): Detection[] {
  return tracks.flatMap(track => { const d = interpolateTrack(track, time); return d && d.severity >= BOX_THRESHOLD ? [d] : []; });
}

/** Rect occupied by an object-fit:contain source, in CSS pixels. */
export function containRect(containerWidth: number, containerHeight: number, sourceWidth: number, sourceHeight: number) {
  if (Math.min(containerWidth, containerHeight, sourceWidth, sourceHeight) <= 0) return { x: 0, y: 0, width: 0, height: 0 };
  const scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight);
  const width = sourceWidth * scale, height = sourceHeight * scale;
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height };
}

export function validDetection(value: unknown): value is Detection {
  if (!value || typeof value !== 'object') return false;
  const d = value as Detection;
  return typeof d.id === 'string' && d.id.length > 0 && d.id.length <= 80 && typeof d.label === 'string' && d.label.length > 0 && d.label.length <= 60
    && Number.isFinite(d.severity) && d.severity >= 0 && d.severity <= 1
    && ['left', 'ahead', 'right'].includes(d.direction) && Array.isArray(d.box) && d.box.length === 4
    && d.box.every(Number.isFinite) && d.box[0] >= 0 && d.box[1] >= 0 && d.box[2] > 0 && d.box[3] > 0
    && d.box[0] + d.box[2] <= 1.001 && d.box[1] + d.box[3] <= 1.001;
}
