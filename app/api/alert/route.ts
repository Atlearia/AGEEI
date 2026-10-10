import { demoDetectionsAt, DEMO, getDemo } from '../../../lib/demo';
import { SPEECH_THRESHOLD } from '../../../lib/detection';
import { createWarningAudio, VoiceError } from '../../../server/voice';
import { isSameOrigin } from '../../../server/request-guard';
export const runtime = 'nodejs';
export const maxDuration = 20;
const buckets = new Map<string, { time: number; count: number }>();

export async function POST(request: Request) {
  if (!isSameOrigin(request)) return Response.json({ message: 'Invalid origin.' }, { status: 403 });
  if (!request.headers.get('content-type')?.includes('application/json')) return Response.json({ message: 'JSON required.' }, { status: 415 });
  try {
    const text = await request.text();
    if (text.length > 2048) return Response.json({ message: 'Request too large.' }, { status: 413 });
    const body = JSON.parse(text);
    // Older open clients without a demo ID still refer to the original video.
    const demo = getDemo(body?.demoId ?? DEMO.id);
    // A client can request only an active, authored demo hazard. No arbitrary prompts or TTS text.
    if (!body || !demo || body.source !== 'demo' || typeof body.trackId !== 'string' || typeof body.mediaTime !== 'number'
      || !Number.isFinite(body.mediaTime) || body.mediaTime < 0 || body.mediaTime > demo.duration) {
      return Response.json({ message: 'Invalid detection event.' }, { status: 400 });
    }
    const detection = demoDetectionsAt(body.mediaTime, demo.id).find(d => d.id === body.trackId && d.severity >= SPEECH_THRESHOLD);
    if (!detection) return Response.json({ message: 'No audible danger at this time.' }, { status: 422 });
    const ip = request.headers.get('x-forwarded-for')?.split(',')[0] || 'local';
    const now = Date.now();
    if (buckets.size > 1000) for (const [key, b] of buckets) if (now - b.time > 60000) buckets.delete(key);
    const bucket = buckets.get(ip);
    if (bucket && now - bucket.time < 60000) {
      if (++bucket.count > 24) return Response.json({ message: 'Please wait a moment before trying again.' }, { status: 429, headers: { 'Retry-After': '60' } });
    } else buckets.set(ip, { time: now, count: 1 });
    const clip = await createWarningAudio(detection);
    return new Response(clip.audio.slice().buffer as ArrayBuffer, { headers: {
      'Content-Type': 'audio/mpeg', 'Content-Length': String(clip.audio.byteLength), 'Cache-Control': 'private, no-store',
      'X-Waypoint-Demo': demo.id, 'X-Waypoint-Track': detection.id, 'X-Waypoint-Warning': encodeURIComponent(clip.text),
      'X-Waypoint-Generation-Ms': String(clip.generatedMs)
    } });
  } catch (error) {
    if (error instanceof SyntaxError) return Response.json({ message: 'Invalid JSON.' }, { status: 400 });
    if (error instanceof VoiceError) return Response.json({ message: error.message, code: error.code }, { status: error.status });
    // Never return provider errors, request headers, credentials, or raw upstream bodies.
    return Response.json({ message: 'The voice connection timed out. Try again.', code: 'voice_unavailable' }, { status: 502 });
  }
}
