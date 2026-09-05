// Drive the rendered Cordis plugin through dsh's own skill-name validator.
// argv[2] = the plugin's lib/index.js, argv[3] = the installed @deepseek-ai/dsh-skill entry.
import { pathToFileURL } from 'node:url';

const plugin = await import(pathToFileURL(process.argv[2]));
const skillMod = await import(pathToFileURL(process.argv[3]));

const out = { name: plugin.name, inject: plugin.inject, apply: typeof plugin.apply };
let registered = null, disposed = false;

const ctx = {
  skills: {
    register(skill) {
      if (!skillMod.isSkillName(skill.name)) throw new Error('registry rejects name: ' + skill.name);
      if (!skill.description) throw new Error('empty description');
      if (typeof skill.content !== 'string' || !skill.content) throw new Error('empty content');
      registered = skill;
      return () => { disposed = true; };
    },
  },
  effect(gen) {
    const { value } = gen().next();
    this._disposers = (this._disposers || []).concat(value);
  },
  logger: { info() {} },
};

plugin.apply(ctx);
out.registered = registered && {
  name: registered.name, source: registered.source,
  descriptionLength: registered.description.length,
  contentLength: registered.content.length,
  contentHead: registered.content.slice(0, 16),
};
(ctx._disposers || []).forEach((d) => d());
out.disposed = disposed;
process.stdout.write(JSON.stringify(out));
