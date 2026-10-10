import { analyzeContext, ModelError, validateContext } from '../../../../server/model-a';
import { guardRequest } from '../../../../server/request-guard';
export const runtime = 'nodejs';
export const maxDuration = 75;
export async function POST(request: Request) {
  const rejected = guardRequest(request, 15); if (rejected) return rejected;
  if (!request.headers.get('content-type')?.includes('application/json')) return Response.json({ message: 'JSON required.' }, { status: 415 });
  try {
    const body = await request.text();
    if (body.length > 4000000) return Response.json({ message: 'Camera clip is too large.' }, { status: 413 });
    const { payload, frame } = validateContext(JSON.parse(body));
    return Response.json(await analyzeContext(payload, frame, request.signal), { headers: { 'Cache-Control': 'no-store' } });
  } catch (e) {
    if (e instanceof ModelError) return Response.json({ message: e.message }, { status: e.status, headers: { 'Retry-After': String(e.retryAfter), 'Cache-Control': 'no-store' } });
    if (e instanceof SyntaxError) return Response.json({ message: 'Invalid camera request.' }, { status: 400 });
    return Response.json({ message: 'Model A did not return a usable result. Retrying with a fresh clip.' }, { status: 502 });
  }
}
