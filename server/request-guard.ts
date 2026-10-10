const buckets = new Map<string, { at: number; count: number }>();
export function isSameOrigin(request: Request) {
  const origin = request.headers.get('origin');
  if (!origin) return true;
  try {
    const url = new URL(request.url);
    // Next can use the internal listening address in request.url. The Host
    // header retains the browser-facing address (including its port).
    const expected = new URL(`${url.protocol}//${request.headers.get('host') || url.host}`);
    return new URL(origin).origin === expected.origin;
  } catch { return false; }
}
export function guardRequest(request: Request, limit: number) {
  if (!isSameOrigin(request)) return Response.json({ message: 'Invalid origin.' }, { status: 403 });
  const key = new URL(request.url).pathname + ':' + (request.headers.get('x-forwarded-for')?.split(',')[0] || 'local');
  const now = Date.now(), bucket = buckets.get(key);
  if (buckets.size > 1000) for (const [k, b] of buckets) if (now - b.at > 60000) buckets.delete(k);
  if (bucket && now - bucket.at < 60000) { if (++bucket.count > limit) return Response.json({ message: 'Please wait a moment.' }, { status: 429, headers: { 'Retry-After': '10' } }); }
  else buckets.set(key, { at: now, count: 1 });
  return null;
}
