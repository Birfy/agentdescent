// Execute the shipped script with a minimal DOM and manually advanced timers.
// This is deterministic interaction coverage, not a rendered-browser test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(process.env.EXPERIENCE_JS || 'docs/assets/experience.js', 'utf8');

function demo() {
  class Element {
    constructor(dataset = {}) {
      this.dataset = dataset; this.children = []; this.listeners = {};
      this.textContent = ''; this.value = ''; this.hidden = false; this.disabled = false;
      this.classList = {toggle() {}};
    }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    click() { if (!this.disabled) this.listeners.click?.({currentTarget: this}); }
    append(...children) { this.children.push(...children); }
    replaceChildren() { this.children = []; }
    getAttribute() { return null; }
    setAttribute() {}
  }
  const elements = Object.fromEntries(['demo-chat-body', 'demo-approve', 'demo-run',
    'demo-status', 'demo-tool-line', 'demo-path'].map(id => [id, new Element()]));
  const presets = ['skill', 'prompt', 'code'].map(demoKind => new Element({demoKind}));
  let now = 0;
  const timers = [];
  vm.runInNewContext(source, {
    document: {
      querySelector: selector => elements[selector.slice(1)] || null,
      querySelectorAll: selector => selector === '.demo-presets button' ? presets : [],
      createElement: () => new Element(),
    },
    setTimeout: (callback, delay) => timers.push({callback, at: now + delay}),
  });
  async function advance(ms) {
    const end = now + ms;
    while (true) {
      timers.sort((a, b) => a.at - b.at);
      if (!timers.length || timers[0].at > end) break;
      const timer = timers.shift(); now = timer.at; timer.callback();
      await Promise.resolve();
    }
    now = end;
    await Promise.resolve();
  }
  return {
    start: elements['demo-run'], approve: elements['demo-approve'],
    status: elements['demo-status'], path: elements['demo-path'], advance,
    select: kind => presets.find(button => button.dataset.demoKind === kind).click(),
    messages: () => elements['demo-chat-body'].children.map(article => {
      const body = article.children[1];
      return {label: body.children[0].textContent, text: body.children[1].textContent};
    }),
  };
}

test('approval gates and repeated clicks preserve the complete sample workflow', async () => {
  const d = demo();
  d.approve.click(); assert.deepEqual(d.messages(), []);
  d.start.click(); d.start.click(); d.approve.click();
  await d.advance(1040);
  assert.equal(d.messages().length, 3);
  assert.equal(d.status.textContent, 'AWAITING APPROVAL');
  await d.advance(5000); assert.equal(d.messages().length, 3);
  d.approve.click(); d.approve.click();
  await d.advance(1480);
  assert.equal(d.status.textContent, 'DIFF READY');
  assert.equal(d.messages().length, 7);
  await d.advance(5000); assert.equal(d.messages().length, 7);
  d.approve.click(); d.approve.click(); await d.advance(350);
  assert.equal(d.status.textContent, 'COMPLETE');
  assert.equal(d.messages().length, 9);
  d.approve.click(); assert.equal(d.messages().length, 9);
});

for (const delay of [0, 380, 930, 1480, 1481]) {
  test(`restart cancels stale run/apply callbacks after ${delay}ms`, async () => {
    const d = demo(); d.start.click(); await d.advance(1040); d.approve.click();
    await d.advance(delay);
    if (delay === 1481) d.approve.click(); // Interrupt the apply animation too.
    d.select('code'); d.start.click(); await d.advance(5000);
    assert.equal(d.status.textContent, 'AWAITING APPROVAL');
    assert.deepEqual(d.messages().map(m => m.label), ['YOU', 'AGENTDESCENT · DOCTOR', 'AGENTDESCENT · PLAN']);
    assert.match(d.messages()[0].text, /agent code/);
    assert.equal(d.approve.hidden, false);
    d.approve.click(); await d.advance(1480);
    assert.equal(d.status.textContent, 'DIFF READY');
    assert.match(d.messages().at(-1).text, /Rank retrieved evidence/);
  });
}

for (const initial of ['skill', 'prompt', 'code']) {
  test(`preset edits cannot change the ${initial} plan or approved diff`, async () => {
    const examples = {skill: 'missing appendix pages', prompt: 'cite the source', code: 'Rank retrieved evidence'};
    const d = demo(); d.select(initial); d.path.value = './custom-artifact'; d.start.click();
    d.select('prompt'); await d.advance(1040);
    d.select('code'); d.approve.click(); await d.advance(380);
    d.select('skill'); await d.advance(1100);
    assert.match(d.messages()[0].text, /custom-artifact/);
    assert.ok(d.messages().at(-1).text.includes(examples[initial]));
    d.approve.click(); await d.advance(350);
    d.select('prompt'); d.start.click(); await d.advance(1040); d.approve.click(); await d.advance(1480);
    assert.ok(d.messages().at(-1).text.includes(examples.prompt));
  });
}

test('restart while awaiting approval replaces the old plan', async () => {
  const d = demo(); d.start.click(); await d.advance(1040);
  d.select('prompt'); d.start.click(); d.approve.click(); await d.advance(1040);
  assert.equal(d.messages().length, 3);
  d.approve.click(); await d.advance(1480);
  assert.match(d.messages().at(-1).text, /cite the source/);
});
