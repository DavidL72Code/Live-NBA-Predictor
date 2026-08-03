import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = resolve(root, 'dist');
const apiBase = (
  process.env.NBA_WINPROB_PUBLIC_API_BASE ||
  'https://davidl72code-swoosh-ai.hf.space'
).replace(/\/+$/, '');

mkdirSync(outDir, { recursive: true });
copyFileSync(resolve(root, 'src/nba_winprob/ui/index.html'), resolve(outDir, 'index.html'));
copyFileSync(resolve(root, 'src/nba_winprob/ui/rd.html'), resolve(outDir, 'rd.html'));

writeFileSync(
  resolve(outDir, 'config.js'),
  `window.NBA_WINPROB_CONFIG = ${JSON.stringify({ apiBase }, null, 2)};\n`,
);
