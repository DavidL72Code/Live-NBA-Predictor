import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = resolve(root, 'dist');
// The backend runs on Render. This fallback only applies when
// NBA_WINPROB_PUBLIC_API_BASE is unset — a preview build, or a local `node
// scripts/build-vercel-frontend.mjs` — so it must point somewhere that exists.
// It previously named a Hugging Face Space that was never deployed, and every
// such build failed at runtime with Hugging Face's 404 page as the fetch body.
const DEFAULT_API_BASE = 'https://live-nba-predictor.onrender.com';
const apiBase = (
  process.env.NBA_WINPROB_PUBLIC_API_BASE || DEFAULT_API_BASE
).replace(/\/+$/, '');

mkdirSync(outDir, { recursive: true });
copyFileSync(resolve(root, 'src/nba_winprob/ui/index.html'), resolve(outDir, 'index.html'));
copyFileSync(resolve(root, 'src/nba_winprob/ui/rd.html'), resolve(outDir, 'rd.html'));

writeFileSync(
  resolve(outDir, 'config.js'),
  `window.NBA_WINPROB_CONFIG = ${JSON.stringify({ apiBase }, null, 2)};\n`,
);
