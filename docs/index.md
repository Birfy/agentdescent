---
description: Explore AgentDescent, the open-source framework for parallel self-evolving agents. Try an interactive merge simulation and start with an offline demo.
---

<div class="experience exp-home">
  <nav class="exp-nav" aria-label="Page navigation">
    <a class="exp-brand" href="./"><span class="brand-mark">A<span>↘</span></span><span>AgentDescent</span></a>
    <div class="exp-nav-links"><a href="#idea">The idea</a><a href="#lab">Interactive lab</a><a href="#inside">What evolves</a><a href="install/" class="nav-start">Get started <span>↗</span></a></div>
  </nav>

  <section class="exp-hero">
    <div class="hero-noise" aria-hidden="true"></div>
    <div class="hero-content">
      <div class="eyebrow"><span class="pulse-dot"></span> OPEN SOURCE FRAMEWORK · PYTHON 3.9+</div>
      <h1>Agents get better.<br><em>Together.</em></h1>
      <p>Give Claude Code, Codex or another agent a task: improve this skill. AgentDescent plans the run, explores changes in parallel and returns a diff for you to review.</p>
      <div class="hero-ctas"><a href="install/" class="exp-button primary">Use it in your agent <span>↗</span></a><a href="#lab" class="exp-button ghost">See it inside your agent <span>↓</span></a></div>
      <div class="hero-command"><span>$</span><code>pip install agentdescent</code><button class="exp-copy" data-copy="pip install agentdescent" aria-label="Copy install command">Copy</button></div>
    </div>
    <div class="hero-product" aria-label="Illustration of parallel workers proposing edits to a shared versioned artifact">
      <div class="product-head"><span><i></i><i></i><i></i></span><b>AGENTDESCENT / EVOLUTION RUN</b><small>● LIVE LOOP</small></div>
      <div class="product-inner"><div class="product-title"><span>01 / PARALLEL EXPLORATION</span><b>Three workers. Three ideas.</b></div>
        <div class="product-tracks"><div><span class="track-avatar">01</span><b>Improve answer format</b><small>+ rules.md</small><i>✓</i></div><div><span class="track-avatar">02</span><b>Add citation behavior</b><small>+ citations.md</small><i>✓</i></div><div><span class="track-avatar">03</span><b>Refine reasoning</b><small>~ prompt.md</small><i>✓</i></div></div>
        <div class="product-connector"><span></span><b>⤨</b><span></span></div><div class="product-merge"><span class="merge-emblem">✳</span><div><small>CONFLICT-AWARE AGGREGATOR</small><strong>Merge → verify → commit</strong></div><span class="merge-arrow">↗</span></div>
        <div class="product-bottom"><span>v.08</span><b>Shared library updated</b><small>3 compatible edits accepted</small></div>
      </div>
      <div class="product-glow" aria-hidden="true"></div>
    </div>
    <a class="hero-scroll" href="#idea">SCROLL TO EXPLORE <span>↓</span></a>
  </section>

  <section class="metric-ribbon"><div><strong>20</strong><span>published methods<br>with runnable ports</span></div><div><strong>6.8×</strong><span>median speedup in the<br>reported GEPA setting*</span></div><div><strong>40%</strong><span>fewer model calls in the<br>reported merge setting*</span></div><a href="results/">* See measured settings &amp; limitations <span>↗</span></a></section>

  <section class="exp-section idea" id="idea"><div class="section-label">01 / THE CORE IDEA</div><div class="idea-grid"><h2>Diffs are the gradients.<br><span>But diffs don’t add.</span></h2><div><p>AgentDescent treats prompts, skills and code as evolving artifacts. Workers explore a shared snapshot and propose edits with evidence. The aggregator decides which edits can coexist, verifies the result and writes an accepted version to the ledger.</p><a class="inline-link" href="architecture/">Follow a diff through the system <span>↗</span></a></div></div><div class="idea-rail"><div><span>01</span><b>Explore</b><small>N workers try tasks in parallel</small></div><i>→</i><div><span>02</span><b>Propose</b><small>Diffs carry evidence</small></div><i>→</i><div><span>03</span><b>Resolve</b><small>Fuse or handle conflicts</small></div><i>→</i><div><span>04</span><b>Commit</b><small>Versioned, inspectable result</small></div></div></section>

  <section class="exp-lab" id="lab"><div class="lab-heading"><div><div class="section-label">02 / INTERACTIVE AGENT DEMO</div><h2>Stay in your agent.<br><em>Improve what it uses.</em></h2><p>See the plugin workflow as it appears inside Claude Code, Codex, OpenCode or DeepSeek Harness.</p></div><span class="lab-tag"><span class="pulse-dot"></span> TRY A SAMPLE CONVERSATION</span></div>
    <div class="agent-demo"><div class="demo-config"><span class="panel-overline">PICK YOUR WORKSPACE</span><div class="control-title">01 &nbsp; Your agent</div><div class="demo-hosts" role="tablist" aria-label="Agent for demo"><button class="active" role="tab" aria-selected="true" data-demo-host="claude">Claude Code</button><button role="tab" aria-selected="false" data-demo-host="codex">Codex</button><button role="tab" aria-selected="false" data-demo-host="opencode">OpenCode</button><button role="tab" aria-selected="false" data-demo-host="dsh">DeepSeek Harness</button></div><div class="control-title demo-target-title">02 &nbsp; What to improve</div><div class="demo-presets"><button class="active" data-demo-kind="skill">Skill directory</button><button data-demo-kind="prompt">Prompt</button><button data-demo-kind="code">Agent code</button></div><label class="demo-path-label" for="demo-path">Artifact path</label><input id="demo-path" value="~/.claude/skills/pdf-audit" spellcheck="false"><p class="control-help">Change the path to personalize the sample conversation.</p><button class="exp-button primary lab-run" id="demo-run">Start sample conversation <span>▶</span></button><div class="lab-config-note"><span>◇</span><p>Illustrative browser demo. No agent runs, model calls or file changes.</p></div></div>
      <div class="demo-chat"><div class="lab-display-top"><div><span class="window-dots"><i></i><i></i><i></i></span><b id="demo-window-title">CLAUDE CODE / AGENTDESCENT</b></div><span id="demo-status">READY</span></div><div class="demo-chat-body" id="demo-chat-body" aria-live="polite"><div class="demo-welcome"><span>✳</span><strong>Your agent can run the whole loop.</strong><small>Choose a host, then start the sample conversation.</small></div></div><div class="demo-chat-footer"><div><span class="tool-light"></span><span id="demo-tool-line">AgentDescent tools ready · doctor → plan → start → show → apply</span></div><button id="demo-approve" hidden>Approve sample apply ↗</button></div></div></div><div class="lab-under"><span>The agent shows the plan and diff before changing your files.</span><a href="install/">Set up your agent ↗</a></div>
  </section>

  <section class="exp-section inside" id="inside"><div class="section-label">03 / YOUR ARTIFACT, YOUR RULES</div><div class="inside-heading"><h2>What should your<br>agent learn?</h2><p>One engine, multiple representations. The key space determines which concurrent edits can merge.</p></div><div class="artifact-picker" role="tablist" aria-label="Artifact types"><button class="active" role="tab" aria-selected="true" data-artifact="prompt">01 &nbsp; Prompt</button><button role="tab" aria-selected="false" data-artifact="skill">02 &nbsp; Skill directory</button><button role="tab" aria-selected="false" data-artifact="code">03 &nbsp; Agent code</button></div><div class="artifact-view"><div><span class="artifact-kicker" id="artifact-kicker">SINGLE SLOT STRATEGY</span><h3 id="artifact-title">Evolve one instruction.</h3><p id="artifact-copy">Start from a prompt and let the loop search for a stronger version against your tasks.</p><a class="inline-link" id="artifact-link" href="quickstart-skill/">Read the prompt quickstart <span>↗</span></a></div><div class="artifact-code"><div><span>ARTIFACT PREVIEW</span><small id="artifact-filename">prompt.md</small></div><pre id="artifact-preview">You are a helpful assistant.

+ Respond with only the requested answer.
+ Omit extra explanation and restatement.</pre></div></div></section>

  <section class="exp-section architecture-teaser"><div class="section-label">04 / BUILT FOR REAL SYSTEMS</div><h2>One loop.<br>Many ways to make it yours.</h2><div class="feature-grid"><a href="agents/"><span>↗</span><h3>Bring your agent</h3><p>Use a callable, an OpenAI-compatible endpoint, Claude or a coding agent.</p><small>AGENTS &amp; LLMS →</small></a><a href="policies/"><span>⤨</span><h3>Shape the decisions</h3><p>Plug in policies for sampling, conflicts, acceptance and promotion.</p><small>POLICIES →</small></a><a href="governance/"><span>◎</span><h3>Keep the boundary</h3><p>Separate evolving artifacts from the evaluator and control what gets written.</p><small>GOVERNANCE →</small></a></div></section>

  <section class="exp-cta"><div><span class="section-label">YOUR FIRST RUN IS THREE COMMANDS AWAY</span><h2>See the loop for yourself.</h2><p>No API key needed for the offline demo.</p></div><a class="exp-button light" href="install/">Open Quickstart <span>↗</span></a></section>
  <footer class="exp-footer"><span>AgentDescent <small>© Open source · MIT</small></span><div><a href="https://github.com/Birfy/agentdescent">GitHub ↗</a><a href="api/">API</a><a href="results/">Results</a></div></footer>
</div>
