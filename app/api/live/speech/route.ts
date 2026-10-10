import { ModelError, verifySpeech } from '../../../../server/model-a';
import { synthesize, VoiceError } from '../../../../server/voice';
import { guardRequest } from '../../../../server/request-guard';
export const runtime = 'nodejs';
export const maxDuration = 20;
export async function POST(request: Request) {
  const rejected = guardRequest(request, 15); if (rejected) return rejected;
  try {
    const text = await request.text();
    if (text.length > 2500) return Response.json({ message: 'Request too large.' }, { status: 413 });
    const ticket = verifySpeech(JSON.parse(text).ticket);
    const audio = await synthesize(ticket.text);
    if (Date.now() >= ticket.expiresAt) throw new ModelError('This camera warning has expired.', 410);
    return new Response(audio.slice().buffer as ArrayBuffer, { headers: { 'Content-Type': 'audio/mpeg', 'Cache-Control': 'no-store', 'X-Waypoint-Warning': encodeURIComponent(ticket.text) } });
  } catch (e) {
    if (e instanceof ModelError || e instanceof VoiceError) return Response.json({ message: e.message }, { status: e.status });
    return Response.json({ message: 'The camera warning could not play.' }, { status: 502 });
  }
}
