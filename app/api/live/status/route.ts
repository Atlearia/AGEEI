import { modelConfig, modelReady } from '../../../../server/model-a';
export const dynamic = 'force-dynamic';
export async function GET() {
  const configured = Boolean(modelConfig().token);
  return Response.json({ configured, ready: configured && await modelReady() }, { headers: { 'Cache-Control': 'no-store' } });
}
