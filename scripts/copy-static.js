const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const dist = path.join(root, 'dist');

if (!fs.existsSync(dist)) fs.mkdirSync(dist, { recursive: true });

fs.copyFileSync(path.join(root, 'index.html'), path.join(dist, 'index.html'));
console.log('OK  dist/index.html');

['content.js', 'app.js'].forEach(name => {
  const from = path.join(root, 'src', name);
  if (!fs.existsSync(from)) throw new Error('Falta ' + from);
  fs.copyFileSync(from, path.join(dist, name));
  console.log('OK  dist/' + name);
});