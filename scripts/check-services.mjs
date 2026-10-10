// Deliberately outputs configuration presence and statuses, never credentials.
const llmKey = process.env.LLM_API_KEY || process.env.OPENAI_API_KEY;
const speechKey = process.env.ELEVENLABS_API_KEY;
console.log(JSON.stringify({ llmConfigured: Boolean(llmKey), elevenLabsConfigured: Boolean(speechKey) }));
if (!speechKey) process.exit(1);
const r = await fetch('https://api.elevenlabs.io/v1/text-to-speech/' + (process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb') + '/stream?output_format=mp3_44100_128', {
  method: 'POST', headers: { 'xi-api-key': speechKey, 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Raised curb on your right.', model_id: process.env.ELEVENLABS_MODEL_ID || 'eleven_flash_v2_5' }), signal: AbortSignal.timeout(15000)
});
if (r.ok) console.log(JSON.stringify({ status: r.status, audioBytes: (await r.arrayBuffer()).byteLength }));
else { const body = await r.json().catch(() => ({})); console.log(JSON.stringify({ status: r.status, error: body.detail?.status || 'request_failed' })); process.exitCode = 1; }
