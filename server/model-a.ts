import { createHmac, timingSafeEqual } from 'node:crypto';
import { adaptAnalysis, LIVE_SPEECH_THRESHOLD, type ContextRequest, type FrameReference, type LiveAssessment } from '../lib/live-contract';
import { conciseWarning } from './voice';

export function modelConfig() {
  return { base: (process.env.AGEII_BASE_URL || 'https://kp6e6wrqu3yyin-8080.proxy.runpod.net').replace(/\/$/, ''), token: process.env.AGEII_API_TOKEN || '' };
}
export class ModelError extends Error { constructor(message: string, public status = 502, public retryAfter = 2) { super(message); } }
const headers = () => ({ Authorization: `Bearer ${modelConfig().token}`, 'User-Agent': 'Mozilla/5.0 Waypoint/1.0' });

function jpegSize(encoded: string) {
  const b = Buffer.from(encoded, 'base64');
  if (b[0] !== 255 || b[1] !== 216) throw new ModelError('Camera frames must be JPEG images.', 422);
  for (let i = 2; i + 9 < b.length;) {
    if (b[i] !== 255) break;
    const marker = b[i + 1];
    const length = b.readUInt16BE(i + 2);
    if (length < 2 || i + length + 2 > b.length) break;
    if ([192, 193, 194].includes(marker)) return { height: b.readUInt16BE(i + 5), width: b.readUInt16BE(i + 7) };
    i += length + 2;
  }
  throw new ModelError('The camera image could not be read.', 422);
}

export function validateContext(value: unknown, now = Date.now()): { payload: ContextRequest; frame: FrameReference } {
  const r = value as ContextRequest;
  const id = /^[A-Za-z0-9_-]{8,80}$/;
  if (!r || typeof r !== 'object' || typeof r.session_id !== 'string' || typeof r.frame_id !== 'string' || !id.test(r.session_id) || !id.test(r.frame_id) || r.action_mode !== 'video'
    || !Array.isArray(r.frames) || r.frames.length < 4 || r.frames.length > 8
    || Object.keys(r).some(k => !['session_id', 'frame_id', 'action_mode', 'frames'].includes(k))) throw new ModelError('Invalid camera clip.', 422);
  let size: { width: number; height: number } | undefined;
  let last = -Infinity;
  for (const f of r.frames) {
    if (!f || Object.keys(f).some(k => !['image_base64', 'timestamp_ms'].includes(k))
      || typeof f.image_base64 !== 'string' || f.image_base64.length > 700000 || !/^[A-Za-z0-9+/]+={0,2}$/.test(f.image_base64)
      || !Number.isSafeInteger(f.timestamp_ms) || f.timestamp_ms <= last || f.timestamp_ms > now + 3000 || f.timestamp_ms < now - 45000) throw new ModelError('Camera frames are invalid or expired. Capture a fresh clip.', 422);
    const dimensions = jpegSize(f.image_base64);
    if (Math.min(dimensions.width, dimensions.height) < 16 || Math.max(dimensions.width, dimensions.height) > 2048 || dimensions.width * dimensions.height > 4000000
      || (size && (size.width !== dimensions.width || size.height !== dimensions.height))) throw new ModelError('Camera frame dimensions do not match.', 422);
    size = dimensions; last = f.timestamp_ms;
  }
  const span = last - r.frames[0].timestamp_ms;
  if (span < 1000 || span > 3500) throw new ModelError('Capture a clip spanning one to three seconds.', 422);
  return { payload: r, frame: { sessionId: r.session_id, frameId: r.frame_id, timestamp: last, width: size!.width, height: size!.height, timestamps: r.frames.map(f => f.timestamp_ms) } };
}

export async function modelReady() {
  if (!modelConfig().token) return false;
  try { const r = await fetch(modelConfig().base + '/ready', { headers: headers(), signal: AbortSignal.timeout(8000), cache: 'no-store' }); return r.ok && (await r.json()).status === 'ok'; } catch { return false; }
}

export async function analyzeContext(payload: ContextRequest, frame: FrameReference, signal?: AbortSignal): Promise<LiveAssessment> {
  const c = modelConfig();
  if (!c.token) throw new ModelError('Model A access is not configured.', 503);
  const response = await fetch(c.base + '/v1/analyze', { method: 'POST', headers: { ...headers(), 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(65000)]) : AbortSignal.timeout(65000), cache: 'no-store' });
  if (!response.ok) {
    const retryAfter = Math.min(60, Math.max(1, Number(response.headers.get('Retry-After')) || 2));
    const messages: Record<number, string> = { 401: 'Model A access needs attention.', 403: 'Model A access needs attention.', 409: 'Capture order changed. Starting a fresh clip.', 413: 'Camera clip is too large. Reducing image size.', 422: 'Model A could not accept this camera clip.', 429: 'Model A is busy. Waiting for a fresh clip.' };
    throw new ModelError(messages[response.status] || 'Model A is temporarily unavailable.', response.status === 403 ? 401 : response.status, retryAfter);
  }
  const result = adaptAnalysis(await response.json(), frame);
  for (const hazard of result.hazards) {
    if (hazard.severity === null || hazard.severity < LIVE_SPEECH_THRESHOLD || !hazard.warning) continue;
    try { hazard.speechTicket = signSpeech({ text: conciseWarning(hazard.warning), expiresAt: result.expiresAt, frameId: result.frameId, hazardId: hazard.id }); } catch { /* An unusable phrase is never spoken. */ }
  }
  return result;
}

type SpeechTicket = { text: string; expiresAt: number; frameId: string; hazardId: string };
function signature(payload: string) { return createHmac('sha256', modelConfig().token).update(payload).digest('base64url'); }
export function signSpeech(ticket: SpeechTicket) {
  const payload = Buffer.from(JSON.stringify(ticket)).toString('base64url');
  return payload + '.' + signature(payload);
}
export function verifySpeech(token: unknown, now = Date.now()): SpeechTicket {
  if (!modelConfig().token || typeof token !== 'string' || token.length > 2000) throw new ModelError('Invalid speech request.', 400);
  const [payload, signed, extra] = token.split('.');
  const expected = signature(payload || '');
  if (extra || !signed || signed.length !== expected.length || !timingSafeEqual(Buffer.from(signed), Buffer.from(expected))) throw new ModelError('Invalid speech request.', 400);
  const ticket = JSON.parse(Buffer.from(payload, 'base64url').toString()) as SpeechTicket;
  if (!Number.isFinite(ticket.expiresAt) || ticket.expiresAt <= now || ticket.expiresAt > now + 48000) throw new ModelError('This camera warning has expired.', 410);
  ticket.text = conciseWarning(ticket.text);
  return ticket;
}
