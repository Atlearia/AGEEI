import { build } from 'esbuild';
await build({ entryPoints: ['lib/motion-worker.ts'], outfile: 'public/motion-worker.js', bundle: true, format: 'iife', platform: 'browser', target: 'es2022', minify: true });
