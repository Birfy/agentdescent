// Load the rendered client bundle the way dsh's module loader does, give it a
// real React, and check it registers into a slot and renders run rows.
// argv[2] = lib/client.js, argv[3] = a react module path.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const react = (await import(process.argv[3])).default ?? (await import(process.argv[3]));
const reactDomServer = await import(process.argv[4]);

const out = {};
let loaded = null;
globalThis.window = {
  __ModuleLoader__: { load: (spec) => { loaded = spec; } },
};
// the bundle calls fetch() on mount; a stub keeps the render deterministic
globalThis.fetch = async () => ({ json: async () => RUNS });
const RUNS = [
  { run_id: '20260905-1', state: 'running', round: 2, rounds: 6, best_reward: 0.5,
    calls: 12, kind: 'skill_dir', target: '/tmp/csv-total' },
  { run_id: '20260905-2', state: 'done', round: 6, rounds: 6, best_reward: 1.0,
    calls: 40, kind: 'plugin', target: '/tmp/my-plugin' },
];

new Function('window', readFileSync(process.argv[2], 'utf8'))(globalThis.window);
if (!loaded) throw new Error('the bundle never called window.__ModuleLoader__.load');
out.id = loaded.id;

const require = (name) => {
  if (name === 'react') return react;
  throw new Error('bundle asked for an unexpected module: ' + name);
};
const mod = loaded.factory(require);
out.inject = mod.inject;
out.apply = typeof mod.apply;

// a stub slots service in the shape dsh-client-ui-jobs uses
let registration = null;
const ctx = {
  slots: {
    inject(slotName, fn) { out.injectedSlot = slotName; fn(); },
    register(spec, Component) { registration = { spec, Component }; },
  },
};
mod.apply(ctx);
if (!registration) throw new Error('apply() registered no slot component');
out.slot = registration.spec;

// render it for real, with runs already resolved
const html = reactDomServer.renderToStaticMarkup(react.createElement(registration.Component));
out.rendersTrigger = html.includes('evolve');
out.htmlLength = html.length;
process.stdout.write(JSON.stringify(out));
