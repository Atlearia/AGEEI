import { configurationStatus } from '../../../server/voice';
import { modelConfig } from '../../../server/model-a';
export const dynamic = 'force-dynamic';
export function GET() { return Response.json({ ...configurationStatus(), liveModelConfigured: Boolean(modelConfig().token) }, { headers: { 'Cache-Control': 'no-store' } }); }
