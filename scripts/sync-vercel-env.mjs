// Run locally only. Values go to Vercel over stdin, never command arguments or logs.
import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join } from 'node:path';
process.loadEnvFile('.env.local');
const link = JSON.parse(await readFile('.vercel/project.json', 'utf8'));
if (link.projectName !== 'waypoint-ageei-2026') throw new Error('Unexpected Vercel project; refusing to modify its environment.');
const cli = join(process.env.APPDATA, 'npm/node_modules/vercel/dist/vc.js');
const includeSpeech = process.argv.includes('--with-speech');
const names = ['LLM_PROVIDER', 'LLM_BASE_URL', 'LLM_MODEL', 'LLM_API_KEY', 'ELEVENLABS_VOICE_ID', 'ELEVENLABS_MODEL_ID'];
if (includeSpeech) names.push('ELEVENLABS_API_KEY');
if (process.argv.includes('--with-model')) names.push('AGEII_BASE_URL', 'AGEII_API_TOKEN');
for (const name of names) {
  const value = process.env[name];
  if (!value) continue;
  const code = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, 'env', 'add', name, 'production', '--force', '--yes', '--sensitive'], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    child.on('error', reject);
    // Discard raw CLI output so credentials cannot be echoed into a transcript.
    child.stdout.resume(); child.stderr.resume();
    child.on('close', resolve); child.stdin.end(value);
  });
  if (code !== 0) throw new Error(`Could not save ${name} (exit ${code}).`);
  console.log(`Saved ${name} to production.`);
}
