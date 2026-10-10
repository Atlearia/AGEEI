import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { adaptAnalysis } from '../lib/live-contract';
const directory = join(process.cwd(), '..', 'model-a-demo-results');
const results = [];
for (let clip = 1; clip <= 6; clip++) {
  const raw = JSON.parse(readFileSync(join(directory, `clip-${String(clip).padStart(2, '0')}.json`), 'utf8'));
  // Replay the adapter at response-receipt time. This is an offline report,
  // never a current warning; actual stored responses have now expired.
  const adapted = adaptAnalysis(raw, { sessionId: raw.session_id, frameId: raw.frame_id, timestamp: raw.captured_at_ms,
    width: raw.image.width, height: raw.image.height, timestamps: raw.frame_timestamps_ms }, raw.captured_at_ms + raw.age_ms);
  results.push({ clip, status: adapted.status, boxes: adapted.hazards.filter(h => h.box && (h.severity === null || h.severity >= .3)).length,
    scores: adapted.hazards.map(h => h.severity), audible: adapted.hazards.filter(h => h.severity !== null && h.severity >= .65).length });
}
writeFileSync(join(directory, 'app-adapter-check.json'), JSON.stringify(results, null, 2));
console.log(JSON.stringify(results));
if (results.at(-1)!.boxes !== 3 || results.at(-1)!.audible !== 0) throw new Error('Unexpected actual-response adapter result');
