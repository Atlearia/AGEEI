import { createHash } from 'node:crypto';
import type { Detection } from '../lib/detection';

export class VoiceError extends Error {
  constructor(public code: string, message: string, public status = 502) { super(message); this.name = 'VoiceError'; }
}
type VoiceConfig = { llmKey: string; baseUrl: string; model: string; provider: string; elevenKey: string; voiceId: string; voiceModel: string };
export function voiceConfig(): VoiceConfig {
  return {
    llmKey: process.env.LLM_API_KEY || process.env.OPENAI_API_KEY || '',
    baseUrl: (process.env.LLM_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, ''),
    model: process.env.LLM_MODEL || process.env.OPENAI_MODEL || 'gpt-4.1-mini',
    provider: process.env.LLM_PROVIDER || 'openai',
    elevenKey: process.env.ELEVENLABS_API_KEY || '',
    voiceId: process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb',
    voiceModel: process.env.ELEVENLABS_MODEL_ID || 'eleven_flash_v2_5'
  };
}
export function configurationStatus() {
  const c = voiceConfig();
  return { ready: Boolean(c.llmKey && c.elevenKey), llmReady: Boolean(c.llmKey), speechReady: Boolean(c.elevenKey) };
}

const PROMPT = 'You turn structured hazard detections into spoken warnings for a blind pedestrian. Return exactly one concise English phrase, ideally 3 to 7 words, at most 10 words. State the physical hazard and its position. Use only the supplied facts. Do not invent distances, safe routes, or movement instructions. Do not say AI, detection, confidence, severity, numbers, explanations, greetings, or quotation marks. Examples: Raised curb on your right. Barrier ahead, on your left. Output only the phrase.';
export function conciseWarning(raw: unknown): string {
  if (typeof raw !== 'string') throw new VoiceError('invalid_text', 'The warning service returned no speech.');
  const clean = raw.trim().replace(/^["'“”]+|["'“”]+$/g, '').replace(/\s+/g, ' ');
  if (!clean || clean.length > 140 || clean.split(/\s+/).length > 12 || /[<>\n\r{}]/.test(clean)) {
    throw new VoiceError('invalid_text', 'The warning service returned an unusable warning.');
  }
  return clean;
}

export async function warningText(detection: Detection, c = voiceConfig(), fetcher: typeof fetch = fetch) {
  if (!c.llmKey) throw new VoiceError('llm_not_configured', 'The warning service is not configured yet.', 503);
  const input = JSON.stringify({ hazard: detection.label, position: detection.direction, dangerLevel: Number(detection.severity.toFixed(2)), box: detection.box, coordinateSystem: 'normalized source image x,y,width,height; x grows to the right' });
  const openai = c.provider === 'openai';
  const response = await fetcher(c.baseUrl + (openai ? '/responses' : '/chat/completions'), {
    method: 'POST', headers: { Authorization: `Bearer ${c.llmKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(openai
      ? { model: c.model, instructions: PROMPT, input, max_output_tokens: 72, temperature: 0, store: false }
      : { model: c.model, messages: [{ role: 'system', content: PROMPT }, { role: 'user', content: input }], max_tokens: 72, temperature: 0 }),
    signal: AbortSignal.timeout(7000)
  });
  if (!response.ok) throw new VoiceError('llm_request_failed', 'The warning service could not respond. Try again.');
  const body = await response.json();
  const raw = openai
    ? body.output?.filter((item: { type: string }) => item.type === 'message').flatMap((item: { content: { type: string; text: string }[] }) => item.content || []).filter((part: { type: string }) => part.type === 'output_text').map((part: { text: string }) => part.text).join(' ')
    : body.choices?.[0]?.message?.content;
  return conciseWarning(raw);
}

export async function synthesize(text: string, c = voiceConfig(), fetcher: typeof fetch = fetch) {
  if (!c.elevenKey) throw new VoiceError('speech_not_configured', 'The voice service is not configured yet.', 503);
  const response = await fetcher(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(c.voiceId)}/stream?output_format=mp3_44100_128`, {
    method: 'POST', headers: { 'xi-api-key': c.elevenKey, 'Content-Type': 'application/json', Accept: 'audio/mpeg' },
    body: JSON.stringify({ text, model_id: c.voiceModel, language_code: 'en', voice_settings: { stability: 0.65, similarity_boost: 0.75, use_speaker_boost: false, speed: 1.08 } }),
    signal: AbortSignal.timeout(8000)
  });
  if (!response.ok) throw new VoiceError('speech_request_failed', 'The voice service could not play this warning. Try again.');
  const audio = new Uint8Array(await response.arrayBuffer());
  if (audio.byteLength < 100 || audio.byteLength > 1024 * 1024) throw new VoiceError('invalid_audio', 'The voice service returned unusable audio.');
  return audio;
}

type Clip = { text: string; audio: Uint8Array; generatedMs: number };
const cache = new Map<string, { expires: number; clip: Promise<Clip> }>();
export function createWarningAudio(d: Detection): Promise<Clip> {
  const c = voiceConfig();
  if (!c.llmKey || !c.elevenKey) throw new VoiceError('not_configured', 'Voice setup is not finished yet.', 503);
  // Key identity is hashed, never logged. Audio is cached server-side for repeated demo runs.
  const key = createHash('sha256').update(JSON.stringify([d.label, d.direction, c.model, c.baseUrl, c.voiceId, c.voiceModel, c.llmKey, c.elevenKey])).digest('hex');
  const cached = cache.get(key);
  if (cached && cached.expires > Date.now()) return cached.clip;
  if (cache.size >= 64) cache.delete(cache.keys().next().value!);
  const clip = (async () => {
    const started = performance.now();
    const text = await warningText(d, c);
    const audio = await synthesize(text, c);
    return { text, audio, generatedMs: Math.round(performance.now() - started) };
  })();
  cache.set(key, { expires: Date.now() + 30 * 60 * 1000, clip });
  void clip.catch(() => { if (cache.get(key)?.clip === clip) cache.delete(key); });
  return clip;
}
