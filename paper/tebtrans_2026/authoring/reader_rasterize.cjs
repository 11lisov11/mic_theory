const fs = require('node:fs');
const path = require('node:path');
const sharp = require('C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
(async () => {
  const root = path.join(__dirname, 'reader_revision_20261029', 'assets');
  for (const n of [1, 2, 3]) {
    await sharp(fs.readFileSync(path.join(root, `figure_${n}.svg`)), {density: 400})
      .flatten({background: '#ffffff'})
      .jpeg({quality: 98, chromaSubsampling: '4:4:4'})
      .toFile(path.join(root, `figure_${n}.jpg`));
  }
})().catch(e => { console.error(e); process.exit(1); });
