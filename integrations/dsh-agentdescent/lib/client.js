window.__ModuleLoader__.load({
  id: 'dsh-agentdescent',
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;
    const React = require('react');

    // Where `agentdescent serve` listens. Read-only and loopback-only; it
    // answers with CORS only to a loopback Origin, which is why this panel can
    // read it from the dsh web page and a random website cannot.
    const PANEL_URL = "http://127.0.0.1:8787/";
    const POLL_MS = 5000;

    const LIVE = { running: 1, created: 1 };

    function useRuns() {
      const [state, setState] = React.useState({ runs: [], error: null, loaded: false });
      React.useEffect(() => {
        let alive = true;
        const tick = () => {
          fetch(PANEL_URL + 'api/runs', { cache: 'no-store' })
            .then((r) => r.json())
            .then((runs) => { if (alive) setState({ runs: runs, error: null, loaded: true }); })
            .catch(() => {
              // Not running is the ordinary case, not an error to shout about:
              // the panel just says how to start it.
              if (alive) setState({ runs: [], error: 'offline', loaded: true });
            });
        };
        tick();
        const id = setInterval(tick, POLL_MS);
        return () => { alive = false; clearInterval(id); };
      }, []);
      return state;
    }

    const S = {
      root: { position: 'relative' },
      trigger: {
        minHeight: 28, cursor: 'pointer', background: 'none', border: 0, borderRadius: 6,
        display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 6px',
        fontSize: 12, lineHeight: '18px', color: 'var(--dsw-alias-label-tertiary, #6b7280)',
      },
      dot: (live) => ({
        width: 6, height: 6, borderRadius: 3,
        background: live ? 'var(--dsw-alias-label-accent, #1d4ed8)'
                         : 'var(--dsw-alias-label-tertiary, #9ca3af)',
      }),
      menu: {
        position: 'absolute', top: 'calc(100% + 5px)', left: 0, zIndex: 100,
        width: 380, maxWidth: 'calc(100vw - 32px)', maxHeight: 420, overflow: 'auto',
        padding: 8, borderRadius: 12, background: 'var(--dsw-specific-menu, #fff)',
        boxShadow: 'var(--dsw-elevation-prominent, 0 8px 24px rgba(0,0,0,.18))',
        color: 'var(--dsw-alias-label-primary, #111)', fontSize: 12,
      },
      row: { display: 'flex', gap: 8, alignItems: 'baseline', padding: '4px 2px' },
      id: { fontFamily: 'var(--dsw-font-mono, ui-monospace, monospace)', fontSize: 11 },
      dim: { color: 'var(--dsw-alias-label-tertiary, #6b7280)' },
      num: { fontVariantNumeric: 'tabular-nums' },
      target: {
        color: 'var(--dsw-alias-label-tertiary, #6b7280)', flex: 1, minWidth: 0,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      },
    };

    function Row(props) {
      const r = props.run;
      const rounds = r.rounds ? r.round + '/' + r.rounds : String(r.round);
      const best = (r.best_reward === null || r.best_reward === undefined)
        ? '-' : r.best_reward.toFixed(3);
      return React.createElement('div', { style: S.row },
        React.createElement('span', { style: S.dot(LIVE[r.state] === 1) }),
        React.createElement('span', { style: S.id }, r.run_id),
        React.createElement('span', { style: S.dim }, r.state),
        React.createElement('span', { style: S.num }, rounds),
        React.createElement('span', { style: S.num }, best),
        React.createElement('span', { style: S.target, title: r.target || '' },
          (r.kind || '?') + ': ' + (r.target || '?')));
    }

    /** Header action: how many runs are live, and the list behind a click. */
    function AgentDescentRuns() {
      const [open, setOpen] = React.useState(false);
      const state = useRuns();
      const live = state.runs.filter((r) => LIVE[r.state] === 1).length;

      const body = state.error
        ? React.createElement('div', { style: S.dim },
            'No run panel. Start it with: agentdescent serve')
        : state.runs.length === 0
          ? React.createElement('div', { style: S.dim }, 'No runs yet.')
          : state.runs.slice(0, 40).map((r) =>
              React.createElement(Row, { key: r.run_id, run: r }));

      return React.createElement('div', { style: S.root },
        React.createElement('button', {
          type: 'button', style: S.trigger, onClick: () => setOpen(!open),
          title: 'AgentDescent runs',
        },
          React.createElement('span', { style: S.dot(live > 0) }),
          'evolve',
          live > 0 ? React.createElement('span', { style: S.num }, ' ' + live) : null),
        open ? React.createElement('div', { style: S.menu }, body) : null);
    }

    const inject = ['slots'];

    function apply(ctx) {
      ctx.slots.inject('conversation.session.header.actions', () => ctx.slots.register({
        name: 'conversation.session.header.actions',
        id: 'agentdescent-runs',
        order: 30,
      }, AgentDescentRuns));
    }

    exports.apply = apply;
    exports.inject = inject;
    exports.AgentDescentRuns = AgentDescentRuns;
    return module.exports;
  },
});
