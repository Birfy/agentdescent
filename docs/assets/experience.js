(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const copy = async (text, button) => {
    try {
      await navigator.clipboard.writeText(text);
      const before = button.textContent;
      button.textContent = 'Copied ✓';
      setTimeout(() => { button.textContent = before; }, 1300);
    } catch (_) {
      button.title = 'Clipboard unavailable; select the adjacent text to copy.';
    }
  };
  $$('[data-copy]').forEach((button) => button.addEventListener('click', () => copy(button.dataset.copy, button)));

  const page = $('.experience');
  const themeButton = $('.theme-toggle');
  if (page && themeButton) {
    let theme = 'light';
    try { theme = localStorage.getItem('agentdescent-theme-v2') || 'light'; } catch (_) { /* storage may be unavailable */ }
    const setTheme = (next) => {
      page.dataset.theme = next;
      themeButton.innerHTML = next === 'dark' ? '☀ <span>Light mode</span>' : '☾ <span>Dark mode</span>';
      themeButton.setAttribute('aria-label', next === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      try { localStorage.setItem('agentdescent-theme-v2', next); } catch (_) { /* storage may be unavailable */ }
    };
    setTheme(theme);
    themeButton.addEventListener('click', () => setTheme(page.dataset.theme === 'dark' ? 'light' : 'dark'));
  }

  // Keep Material documentation pages in the same color mode as the landing pages.
  if (!page) {
    const paletteOptions = $$('input[data-md-color-scheme]');
    let savedTheme;
    try { savedTheme = localStorage.getItem('agentdescent-theme-v2'); } catch (_) { /* storage may be unavailable */ }
    if (savedTheme === 'dark' || savedTheme === 'light') {
      const scheme = savedTheme === 'dark' ? 'slate' : 'default';
      const option = paletteOptions.find((input) => input.dataset.mdColorScheme === scheme);
      if (option) { option.checked = true; document.body.dataset.mdColorScheme = scheme; }
    }
    paletteOptions.forEach((input) => input.addEventListener('change', () => {
      const next = input.dataset.mdColorScheme === 'slate' ? 'dark' : 'light';
      try { localStorage.setItem('agentdescent-theme-v2', next); } catch (_) { /* storage may be unavailable */ }
    }));
  }

  const hosts = {
    claude: {name: 'CLAUDE CODE', label: 'Claude Code', setup: 'agentdescent install claude-code\nclaude --plugin-dir ~/.agentdescent/plugins/claude-code', help: 'Start Claude Code with the plugin directory. In the session, /plugin lists it.'},
    codex: {name: 'CODEX', label: 'Codex', setup: 'codex plugin marketplace add Birfy/agentdescent\ncodex plugin add agentdescent@agentdescent\ncodex mcp list', help: 'The marketplace plugin includes the AgentDescent MCP server. Confirm that codex mcp list shows it.'},
    opencode: {name: 'OPENCODE', label: 'OpenCode', setup: 'agentdescent install opencode\nopencode mcp list', help: 'Restart OpenCode after installation and check that AgentDescent is connected.'},
    dsh: {name: 'DEEPSEEK HARNESS', label: 'DeepSeek Harness', setup: 'agentdescent install dsh\ndsh --profile web --dump-config | grep agentdescent', help: 'Restart DeepSeek Harness and check that its configuration contains AgentDescent.'}
  };
  const oneClickCommand = 'curl -fsSL https://raw.githubusercontent.com/Birfy/agentdescent/main/scripts/install.sh | bash';
  const pipCommand = "python3 -m pip install --upgrade 'agentdescent[mcp] @ git+https://github.com/Birfy/agentdescent.git@main'";
  const hostOpen = {
    claude: {label:'LAUNCH CLAUDE CODE', command:'claude --plugin-dir ~/.agentdescent/plugins/claude-code', help:'The installer writes a local plugin directory. Launch Claude Code with it.'},
    codex: {label:'VERIFY CODEX', command:'codex mcp list', help:'Restart Codex, then check that the agentdescent MCP server is enabled.'},
    opencode: {label:'VERIFY OPENCODE', command:'opencode mcp list', help:'Restart OpenCode, then check that agentdescent is connected.'},
    dsh: {label:'VERIFY DEEPSEEK HARNESS', command:'dsh --profile web --dump-config | grep agentdescent', help:'Restart DeepSeek Harness, then confirm its config contains agentdescent.'}
  };
  let installMode = 'script';
  let selectedHost = 'claude';
  const renderHost = () => {
    const display = installMode === 'script' ? hostOpen[selectedHost] : {
      label: `CONNECT ${hosts[selectedHost].name}`,
      command: hosts[selectedHost].setup,
      help: hosts[selectedHost].help
    };
    if ($('#qs3-host-label')) $('#qs3-host-label').textContent = display.label;
    if ($('#qs3-host-command')) $('#qs3-host-command').textContent = display.command;
    if ($('#qs3-host-help')) $('#qs3-host-help').textContent = display.help;
  };
  $$('.qs3-tabbar button').forEach((button) => button.addEventListener('click', () => {
    installMode = button.dataset.installMode;
    $$('.qs3-tabbar button').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); });
    $('#qs3-install-command').textContent = installMode === 'script' ? oneClickCommand : pipCommand;
    $('#qs3-install-note').textContent = installMode === 'script'
      ? 'Installs AgentDescent and connects the agent CLIs already on your PATH.'
      : 'Installs current main with pip. Use the host setup command in step 02.';
    renderHost();
  }));
  $$('.qs3-host-tabs button').forEach((button) => button.addEventListener('click', () => {
    selectedHost = button.dataset.host;
    $$('.qs3-host-tabs button').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); });
    renderHost();
  }));
  $('#qs3-copy-install')?.addEventListener('click', (event) => copy($('#qs3-install-command').textContent, event.currentTarget));
  $('#qs3-copy-host')?.addEventListener('click', (event) => copy($('#qs3-host-command').textContent, event.currentTarget));
  $('#qs3-copy-prompt')?.addEventListener('click', (event) => copy($('#qs3-prompt-text').textContent, event.currentTarget));
  $$('.host-tabs button').forEach((button) => button.addEventListener('click', () => {
    $$('.host-tabs button').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); });
    const host = hosts[button.dataset.host];
    $('#qs-host-command').textContent = host.setup;
    $('#qs-host-help').textContent = host.help;
  }));
  $('#qs-host-copy')?.addEventListener('click', (event) => copy($('#qs-host-command').textContent, event.currentTarget));

  const recipes = {
    base: {
      kicker: 'DEFAULT',
      title: 'Start with one artifact.',
      copy: 'The same entry point handles the task, reward, actor and artifact strategy. Unspecified policy slots use their defaults.',
      code: 'evolve(tasks, reward, agent=agent,\n       strategy=FileTree(files),\n       policies=Policies())',
      note: 'Illustrative configuration · supply your own tasks, reward, agent and files.'
    },
    search: {
      kicker: 'SELECTION + SAMPLING',
      title: 'Search a wider frontier.',
      copy: 'A beam keeps several candidates in play while difficulty-weighted sampling spends more work on useful tasks. The artifact and reward stay the same.',
      code: 'evolve(tasks, reward, agent=agent,\n       strategy=FileTree(files),\n       policies=Policies(\n           selection=Beam(4),\n           task_sampler=DifficultyWeighted()))',
      note: 'Change only the decisions this algorithm needs.'
    },
    merge: {
      kicker: 'CONFLICT + FUSION',
      title: 'Change how diffs combine.',
      copy: 'Reflective merge installs a paired conflict resolver and fusion policy. Compatible edits can be combined before the acceptance gate.',
      code: 'evolve(tasks, reward, agent=agent,\n       strategy=FileTree(files),\n       policies=Policies(\n           **reflective_merge(completion)))',
      note: 'The same loop still handles workers, evaluation and versioning.'
    },
    custom: {
      kicker: 'OPTIMIZER EXIT',
      title: 'Bring your own algorithm.',
      copy: 'If your method needs a population, an archive or state that individual policy slots do not keep, supply an aggregator factory.',
      code: 'evolve(tasks, reward, agent=agent,\n       strategy=my_strategy,\n       aggregator_factory=my_optimizer_factory)',
      note: 'Use a custom Strategy for your artifact and an optimizer for its search rule.'
    }
  };
  $$('.recipe-options button').forEach((button) => button.addEventListener('click', () => {
    $$('.recipe-options button').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); });
    const recipe = recipes[button.dataset.recipe];
    $('#recipe-kicker').textContent = recipe.kicker;
    $('#recipe-title').textContent = recipe.title;
    $('#recipe-copy').textContent = recipe.copy;
    $('#recipe-code').textContent = recipe.code;
    $('#recipe-note').textContent = recipe.note;
  }));

  const artifacts = {
    prompt: {kicker: 'SINGLE SLOT STRATEGY', title: 'Evolve one instruction.', copy: 'Replace a single value and compare candidates against the tasks you care about.', filename: 'prompt.md', preview: 'You are a helpful assistant.\n\n+ Respond with only the requested answer.\n+ Omit extra explanation and restatement.', link: 'quickstart-skill/', linkText: 'Read the prompt quickstart'},
    skill: {kicker: 'FILE TREE STRATEGY', title: 'Grow a skill library.', copy: 'Treat paths in a folder as independent keys. Workers can edit separate files, then the aggregator fuses compatible changes.', filename: 'skills/pdf-audit/SKILL.md', preview: '# PDF audit\n\n+ Verify every section before reporting.\n+ Cite the page for each finding.\n\nreferences/rules.md  ·  unchanged', link: 'quickstart-directory/', linkText: 'Read the directory quickstart'},
    code: {kicker: 'PROGRAM + HARNESS STRATEGIES', title: 'Change executable behavior.', copy: 'Evolve a program or agent harness while keeping its evaluation tasks and protected files outside the search.', filename: 'agent/solver.py', preview: 'def solve(task):\n    context = retrieve(task)\n+   context = rank_evidence(context)\n    return answer(context, task)', link: 'quickstart-agent-code/', linkText: 'Read the code quickstart'},
    custom: {kicker: 'YOUR STRATEGY', title: 'Represent your own artifact.', copy: 'Define initial state, rendering and proposal-to-diff conversion. The same engine can then evaluate and merge your representation.', filename: 'my_strategy.py', preview: 'class MyStrategy:\n    def initial(self): ...\n    def render(self, state): ...\n    def to_diff(self, state, proposal, author, base_version, target): ...', link: 'strategies/', linkText: 'Write a custom Strategy'}
  };
  $$('.artifact-picker button').forEach((button) => button.addEventListener('click', () => {
    $$('.artifact-picker button').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); });
    const artifact = artifacts[button.dataset.artifact];
    $('#artifact-kicker').textContent = artifact.kicker;
    $('#artifact-title').textContent = artifact.title;
    $('#artifact-copy').textContent = artifact.copy;
    $('#artifact-filename').textContent = artifact.filename;
    $('#artifact-preview').textContent = artifact.preview;
    $('#artifact-link').href = artifact.link;
    $('#artifact-link').innerHTML = `${artifact.linkText} <span>↗</span>`;
  }));

  const demoBody = $('#demo-chat-body');
  if (!demoBody) return;
  const kinds = {
    skill: {path:'~/.claude/skills/pdf-audit', noun:'skill', example:'Add a check for missing appendix pages'},
    prompt: {path:'./prompts/answer.md', noun:'prompt', example:'Answer concisely and cite the source'},
    code: {path:'./agent/solver.py', noun:'agent code', example:'Rank retrieved evidence before answering'}
  };
  let activeHost = 'claude';
  let activeKind = 'skill';
  let phase = 0;
  const approve = $('#demo-approve');
  const start = $('#demo-run');
  const status = $('#demo-status');
  const toolLine = $('#demo-tool-line');
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  function addMessage(kind, label, text, detail) {
    const article = document.createElement('div');
    article.className = `demo-message ${kind}`;
    const icon = document.createElement('span'); icon.className = 'message-icon'; icon.textContent = kind === 'user' ? '●' : '✳';
    const body = document.createElement('div'); body.className = 'message-content';
    const small = document.createElement('small'); small.textContent = label;
    const paragraph = document.createElement('p'); paragraph.textContent = text;
    body.append(small, paragraph);
    if (detail) { const extra = document.createElement('p'); extra.className = 'message-detail'; extra.textContent = detail; body.append(extra); }
    article.append(icon, body); demoBody.append(article); demoBody.scrollTop = demoBody.scrollHeight;
  }
  function setActive(selector, selected) { $$(selector).forEach((button) => { const active = button === selected; button.classList.toggle('active', active); if (button.getAttribute('role') === 'tab') button.setAttribute('aria-selected', String(active)); }); }
  $$('.demo-hosts button').forEach((button) => button.addEventListener('click', () => {
    activeHost = button.dataset.demoHost; setActive('.demo-hosts button', button);
    $('#demo-window-title').textContent = `${hosts[activeHost].name} / AGENTDESCENT`;
  }));
  $$('.demo-presets button').forEach((button) => button.addEventListener('click', () => {
    activeKind = button.dataset.demoKind; setActive('.demo-presets button', button);
    $('#demo-path').value = kinds[activeKind].path;
  }));
  start.addEventListener('click', async () => {
    if (start.disabled) return;
    start.disabled = true; phase = 0; approve.hidden = true; demoBody.replaceChildren();
    const path = $('#demo-path').value.trim() || kinds[activeKind].path;
    const kind = kinds[activeKind];
    status.textContent = 'PLANNING'; toolLine.textContent = 'Calling doctor and plan…';
    addMessage('user', 'YOU', `Improve the ${kind.noun} at ${path}. Show the plan and call count before starting.`);
    await pause(460);
    addMessage('tool', 'AGENTDESCENT · DOCTOR', 'Checked the agent, provider configuration and run store.', 'No files changed.');
    await pause(580);
    addMessage('tool', 'AGENTDESCENT · PLAN', 'Sample plan: 12 cases · 4 workers · up to 48 agent calls.', 'Please review the plan and cost before starting.');
    status.textContent = 'AWAITING APPROVAL'; toolLine.textContent = 'Plan ready · no run started';
    approve.textContent = 'Approve sample run ↗'; approve.hidden = false; phase = 1; start.disabled = false;
  });
  approve.addEventListener('click', async () => {
    approve.hidden = true;
    if (phase === 1) {
      phase = 0; status.textContent = 'RUNNING'; toolLine.textContent = 'start → status → show';
      addMessage('user', 'YOU', 'The plan looks good. Start the sample run.');
      await pause(380); addMessage('tool', 'AGENTDESCENT · START', 'Started a background evolution run.');
      await pause(550); addMessage('tool', 'AGENTDESCENT · STATUS', 'Workers proposed edits; the aggregator evaluated the candidate.');
      await pause(550); addMessage('tool', 'AGENTDESCENT · SHOW', `Proposed diff: + ${kinds[activeKind].example}.`, 'The real artifact remains untouched until you approve apply.');
      status.textContent = 'DIFF READY'; toolLine.textContent = 'Diff ready · waiting for apply approval';
      approve.textContent = 'Approve sample apply ↗'; approve.hidden = false; phase = 2;
    } else if (phase === 2) {
      phase = 0; addMessage('user', 'YOU', 'Apply the sample result.');
      await pause(350); addMessage('tool', 'AGENTDESCENT · APPLY', 'Sample change applied; the original is backed up.', 'This page is only a simulation and changed no files.');
      status.textContent = 'COMPLETE'; toolLine.textContent = 'Sample workflow complete · no real files changed';
    }
  });
})();
