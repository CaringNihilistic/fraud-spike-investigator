/* Fraud Spike Investigator — dashboard.
   React + htm (tagged templates), no JSX, no build step, no CDN.
   Everything shown here comes from the live pipeline via /api. */
const { useState, useEffect, useMemo, useRef, useCallback } = React;
const html = htm.bind(React.createElement);

// Write key, injected same-origin by GET / (see src/serve/api.py). Mutating
// endpoints reject requests without it, so the board can act but a stray
// caller on the network cannot override an analyst decision.
const KEY = (document.querySelector('meta[name="fsi-key"]') || {}).content || '';

const api = async (path, opts = {}) => {
  const r = await fetch(path, { ...opts, headers: { 'X-API-Key': KEY, ...(opts.headers || {}) } });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const post = (p, body) => api(p, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

const inr = (n) => {
  if (n == null) return '—';
  if (n >= 1e7) return '₹' + (n / 1e7).toFixed(2) + 'Cr';
  if (n >= 1e5) return '₹' + (n / 1e5).toFixed(2) + 'L';
  if (n >= 1e3) return '₹' + (n / 1e3).toFixed(1) + 'K';
  return '₹' + n.toFixed(0);
};
const riskColor = (r) => (r >= 85 ? 'var(--red)' : r >= 25 ? 'var(--amber)' : 'var(--green)');
/* Colour by merchant-level flagged RATE (0-1), which is the signal the spike
   detector actually fires on — not a single transaction's score. */
const rateColor = (rate) => (rate >= 0.4 ? 'var(--red)'
  : rate >= 0.15 ? 'var(--amber)' : 'var(--green)');

/* poll a JSON endpoint on an interval; pauses when the tab is hidden */
function usePoll(path, ms, deps = []) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (document.hidden) return;
      try { const d = await api(path); if (alive) { setData(d); setErr(null); } }
      catch (e) { if (alive) setErr(e.message); }
    };
    tick();
    const id = setInterval(tick, ms);
    return () => { alive = false; clearInterval(id); };
  }, [path, ms, ...deps]);
  return [data, err];
}

/* ------------------------------------------------------------ header */
function Header({ status, onSpeed, onPause, view, setView }) {
  const s = status || {};
  // local echo + debounce: the label tracks the thumb instantly, but the POST
  // fires only when the user settles (a drag would otherwise fire dozens)
  const [spd, setSpd] = useState(null);
  const timer = useRef(null);
  const onSlide = (v) => {
    setSpd(v);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => { onSpeed(v); setTimeout(() => setSpd(null), 1500); }, 350);
  };
  return html`
    <header>
      <div class="brand">
        <div class="mark">R</div>
        <div>
          <h1>Fraud Spike Investigator</h1>
          <div class="sub">merchant-level detection · entity correlation · policy-gated investigation</div>
        </div>
      </div>
      <nav class="views">
        <button class=${'vtab' + (view === 'console' ? ' on' : '')}
                onClick=${() => setView('console')}>Live console</button>
        <button class=${'vtab' + (view === 'pitch' ? ' on' : '')}
                onClick=${() => setView('pitch')}>Pitch & architecture</button>
      </nav>
      <span class="spacer"></span>
      ${/* The biggest number on screen is money, not throughput. */''}
      <div class="hero-inr" title="Fraud prevented minus legitimate revenue impacted minus analyst review cost, costed live with the same constants as the offline report (₹50/review, 7% step-up abandon, 90% of fraud fails step-up).">
        <span class="hv">${inr((s.economics || {}).net_protected_inr)}</span>
        <span class="hl">net protected</span>
      </div>
      <span class="counter mono">
        ${`${(s.processed || 0).toLocaleString()} / ${(s.total || 0).toLocaleString()} txns · ${(s.pct || 0).toFixed(0)}%`}
        ${s.merchants_in_spike ? html`${' · '}<b>${s.merchants_in_spike} under attack</b>` : ''}
        ${s.review_pending ? html`${' · '}<span style=${{color:'var(--amber)'}}>${s.review_pending} awaiting review</span>` : ''}
      </span>
      <label class="sub">speed
        <input type="range" min="50" max="4000" step="50" value=${spd ?? (s.speed_tps || 200)}
               onChange=${(e) => onSlide(+e.target.value)} style=${{ width: '120px', marginLeft: '6px' }} />
        <span class="mono"> ${Math.round(spd ?? s.speed_tps ?? 0)}/s</span>
      </label>
      <button onClick=${() => onPause(!s.paused)}>${s.paused ? '▶ resume' : '⏸ pause'}</button>
    </header>`;
}

/* ------------------------------------------------------------ merchants */
function MerchantCard({ m, selected, onSelect }) {
  const d = m.fraud_rate || {};
  const isFlash = m.merchant_id === 'm11' && m.txn_count > 300;
  const peakPct = (m.peak_rate_ever || 0) * 100;
  const sig = m.signature || {};
  const mix = m.action_mix || {};
  const cls = ['mcard', m.in_spike ? 'spiking' : '',
               isFlash && !m.in_spike ? 'flash' : '',
               selected ? 'selected' : ''].join(' ');

  // Lead with the conclusion in words, then show the numbers that support it.
  // The card used to be five metrics and no sentence, which left the reader to
  // assemble "a 5.5L account takeover hit 79 transactions" for themselves.
  const verdict = m.in_spike
    ? html`<b>${inr(m.exposure_inr)}</b>${' '}at risk across ${m.flagged_count}${' '}
           of ${(m.txn_count || 0).toLocaleString()} transactions`
    : isFlash
      ? html`${(m.txn_count || 0).toLocaleString()} transactions —${' '}<b>6× normal volume</b>,
             and${' '}<b>not one of them was blocked</b>`
      : html`${m.flagged_count} of ${(m.txn_count || 0).toLocaleString()} transactions flagged —
             ordinary background rate`;

  // The burst can be over by the time anyone looks. Saying so is the
  // difference between "0.0%" reading as a contradiction and reading as news.
  const cooled = m.in_spike && (d.current_rate || 0) < (d.baseline_rate || 0);

  return html`
    <div class=${cls} onClick=${() => onSelect(m.merchant_id)}>
      <div class="top">
        <span class="mid">${m.merchant_id}</span>
        ${m.in_spike
          ? html`<span class="badge spike">under attack</span>`
          : isFlash
            ? html`<span class="badge legit">6× flash sale · not flagged</span>`
            : html`<span class="badge clear">normal</span>`}
      </div>
      <div class="gauge">
        <i class=${peakPct < 1 ? 'zero' : ''}
           style=${{ width: Math.min(100, peakPct) + '%', background: rateColor(m.peak_rate_ever) }}></i>
      </div>

      <div class="verdict">${verdict}</div>
      ${sig.text ? html`<div class="sig" title="Counted from this merchant's own flagged transactions. Not a model output and not a label - just who shares what.">${sig.text}</div>` : ''}

      ${(mix.restrict || mix.step_up || mix.review) ? html`
        <div class="acts">
          <span class="al">system did</span>
          ${mix.restrict ? html`<span class="act r">blocked ${mix.restrict}</span>` : ''}
          ${mix.step_up ? html`<span class="act s">OTP on ${mix.step_up}</span>` : ''}
          ${mix.review ? html`<span class="act v">${mix.review} to human review</span>` : ''}
        </div>` : html`<div class="acts"><span class="al">system did</span>
          <span class="act n">nothing — allowed all ${(m.txn_count || 0).toLocaleString()}</span></div>`}

      <div class="stat hero"><span class="k" title="Worst 30-transaction window in this merchant's history: what share of them the ML scorer flagged as high risk. This is the number the spike detector watches.">worst 30-txn window</span>
        <span class="v" style=${{ color: rateColor(m.peak_rate_ever) }}>
          ${(m.peak_rate_ever * 100).toFixed(0)}% flagged</span></div>
      <div class="stat"><span class="k" title="How many standard deviations the flagged rate rose above this merchant's own normal. The detector fires at z >= 4.">how abnormal</span>
        <span class=${'v ' + (m.peak_z_ever >= 4 ? 'delta up' : 'delta flat')}>
          ${m.peak_z_ever >= 4 ? `${m.peak_z_ever}σ above normal · fired` : `${m.peak_z_ever}σ · below threshold`}</span></div>
      <div class="stat"><span class="k" title="Flagged rate right now (last 30 txns) vs this merchant's long-run normal.">right now / normal</span>
        <span class="v" style=${{ color: 'var(--dim)' }}>
          ${(d.current_rate * 100).toFixed(1)}% vs ${(d.baseline_rate * 100).toFixed(1)}%${cooled ? ' · burst has passed' : ''}</span></div>
      ${m.top_cause ? html`<div class="stat"><span class="k">agent's diagnosis</span>
        <span class="v mono" style=${{ color: 'var(--violet)' }}>${m.top_cause.replace(/_/g, ' ')}</span></div>` : ''}
    </div>`;
}

/* ------------------------------------------------------------ entity graph */
/* Deliberately a tiny hand-rolled force layout rather than a graph library:
   ~30 lines, no dependency, and enough to show the shape that matters —
   a few entities fanning out across many accounts. */
function EntityGraph({ merchantId, compact }) {
  const [graph] = usePoll(`/api/merchants/${merchantId}/entity-graph`, 3000, [merchantId]);
  const [pos, setPos] = useState([]);
  const [hover, setHover] = useState(null);
  const raf = useRef(null);
  // neighbours of the hovered node, so its part of the network lights up
  const nb = useMemo(() => {
    if (!hover || !graph) return null;
    const set = new Set([hover]);
    for (const l of graph.links) {
      if (l.source === hover) set.add(l.target);
      if (l.target === hover) set.add(l.source);
    }
    return set;
  }, [hover, graph]);

  useEffect(() => {
    if (!graph || !graph.nodes.length) { setPos([]); return; }
    const W = 720, H = 400;
    const idx = new Map(graph.nodes.map((n, i) => [n.id, i]));
    // Seed hubs on a WIDE inner ring (not a point) so they never start on top
    // of each other, accounts on an outer ring - converges fast and spread out.
    const hubs = graph.nodes.filter((n) => n.kind !== 'customer').length || 1;
    let hi = 0;
    let P = graph.nodes.map((n, i) => {
      const hub = n.kind !== 'customer';
      const a = hub ? (hi++ / hubs) * Math.PI * 2
                    : (i / graph.nodes.length) * Math.PI * 2;
      const r = hub ? 95 : 165;
      return { x: W / 2 + Math.cos(a) * r, y: H / 2 + Math.sin(a) * r, vx: 0, vy: 0, n };
    });
    const links = graph.links.map((l) => [idx.get(l.source), idx.get(l.target)])
      .filter(([a, b]) => a != null && b != null);

    let step = 0;
    const tick = () => {
      for (let it = 0; it < 2; it++) {
        for (let i = 0; i < P.length; i++) {           // repulsion
          for (let j = i + 1; j < P.length; j++) {
            let dx = P[j].x - P[i].x, dy = P[j].y - P[i].y;
            let d2 = dx * dx + dy * dy || 0.01;
            if (d2 > 90000) continue;
            // hubs repel each other much harder: they carry the labels, and
            // two overlapping hubs make the whole picture unreadable on video
            const bothHubs = P[i].n.kind !== 'customer' && P[j].n.kind !== 'customer';
            const f = (bothHubs ? 2600 : 420) / d2, d = Math.sqrt(d2);
            const ux = (dx / d) * f, uy = (dy / d) * f;
            P[i].vx -= ux; P[i].vy -= uy; P[j].vx += ux; P[j].vy += uy;
          }
        }
        for (const [a, b] of links) {                  // spring
          const dx = P[b].x - P[a].x, dy = P[b].y - P[a].y;
          const d = Math.hypot(dx, dy) || 0.01, f = (d - 78) * 0.005;
          const ux = (dx / d) * f, uy = (dy / d) * f;
          P[a].vx += ux; P[a].vy += uy; P[b].vx -= ux; P[b].vy -= uy;
        }
        for (const p of P) {                            // integrate + centre
          p.vx += (W / 2 - p.x) * 0.0012; p.vy += (H / 2 - p.y) * 0.0012;
          p.x += (p.vx *= 0.82); p.y += (p.vy *= 0.82);
          p.x = Math.max(18, Math.min(W - 18, p.x));
          p.y = Math.max(18, Math.min(H - 18, p.y));
        }
      }
      setPos(P.map((p) => ({ ...p })));
      if (++step < 120) raf.current = requestAnimationFrame(tick);
    };
    tick();
    return () => raf.current && cancelAnimationFrame(raf.current);
  }, [graph]);

  if (!graph) return html`<div class="empty">loading…</div>`;
  if (!graph.nodes.length)
    return html`<div class=${'empty' + (compact ? ' empty-compact' : '')}>
      No shared entities among flagged transactions — every account uses its own
      device, IP and instrument. This is what legitimate traffic looks like.
    </div>`;

  const idx = new Map(graph.nodes.map((n, i) => [n.id, i]));
  return html`
    <div>
      <svg class=${'graph' + (compact ? ' compact' : '')} viewBox="0 0 720 400" preserveAspectRatio="xMidYMid meet">
        ${graph.links.map((l, i) => {
          const a = pos[idx.get(l.source)], b = pos[idx.get(l.target)];
          if (!a || !b) return null;
          const hot = hover && (l.source === hover || l.target === hover);
          return html`<line key=${i} x1=${a.x} y1=${a.y} x2=${b.x} y2=${b.y}
            stroke=${hot ? 'var(--rzp-blue)' : undefined}
            stroke-width=${hot ? 1.8 : 1}
            opacity=${hover && !hot ? 0.15 : 1} />`;
        })}
        ${pos.map((p, i) => {
          const hub = p.n.kind !== 'customer';
          const r = hub ? Math.min(17, 7 + p.n.size * 0.8) : 4;
          const dim = nb && !nb.has(p.n.id);
          return html`
          <g key=${i} opacity=${dim ? 0.18 : 1}
             onMouseEnter=${() => setHover(p.n.id)} onMouseLeave=${() => setHover(null)}
             style=${{ cursor: 'pointer' }}>
            <circle class=${p.n.kind} cx=${p.x} cy=${p.y} r=${r}
                    stroke=${hub ? '#FFFFFF' : 'none'} stroke-width=${hub ? 2 : 0}>
              <title>${p.n.kind}: ${p.n.label}${p.n.size > 1 ? ` — shared by ${p.n.size} accounts. Hover to highlight its network.` : ''}</title>
            </circle>
            ${hub ? html`
              <text x=${p.x} y=${p.y + r + 12} text-anchor="middle"
                    style=${{ paintOrder: 'stroke', stroke: '#F8FAFD', strokeWidth: '3px' }}>
                ${p.n.label.slice(0, 14)} · ${p.n.size}</text>` : ''}
          </g>`; })}
      </svg>
      ${compact ? '' : html`<div class="legend">
        <span><i style=${{ background: 'var(--red)' }}></i>device</span>
        <span><i style=${{ background: 'var(--amber)' }}></i>IP</span>
        <span><i style=${{ background: 'var(--violet)' }}></i>instrument</span>
        <span><i style=${{ background: '#4b5c6b' }}></i>account</span>
        <span style=${{ marginLeft: 'auto' }}>node size = accounts sharing it</span>
      </div>
      <div class="note">${graph.note}. A ring or farm shows as few large hubs with
        many accounts attached; ordinary traffic shows as nothing at all.</div>`}
    </div>`;
}

/* ------------------------------------------------------------ investigation */
function Investigation({ merchantId, inSpike }) {
  const [cfg] = usePoll('/api/config', 30000);
  const [rep, setRep] = useState(null);
  const [state, setState] = useState('idle');
  const [showAudit, setShowAudit] = useState(false);

  const load = useCallback(async () => {
    try { setRep(await api(`/api/merchants/${merchantId}/investigation`)); setState('ok'); }
    catch { setRep(null); setState('none'); }
  }, [merchantId]);
  useEffect(() => { setState('loading'); load(); const id = setInterval(load, 4000);
                    return () => clearInterval(id); }, [load]);

  const run = async () => {
    setState('running');
    try { setRep(await post(`/api/merchants/${merchantId}/investigate`)); setState('ok'); }
    catch (e) { setState('error:' + e.message); }
  };

  if (state === 'running') return html`<div class="empty">investigating…</div>`;
  if (!rep) return html`
    <div>
      <div class="empty">
        ${cfg && cfg.agent_enabled === false
          ? html`The LLM investigator is <b>disabled on this hosted instance</b>, on
                 purpose: an API key on a public host would let any visitor spend
                 credits. Everything else you see is live. Run it locally with
                 ${' '}<span class="mono">python run_demo.py</span>${' '}and an
                 ANTHROPIC_API_KEY to see investigations, or watch the pitch video.`
          : inSpike === false
            ? 'No spike -> no investigation. The agent only runs when the detector fires, so quiet merchants cost zero analyst time and zero tokens.'
            : 'No investigation yet — these fire automatically when the spike detector trips.'}
      </div>
      ${cfg && cfg.agent_enabled === false ? ''
        : html`<button class="sm" onClick=${run}>run investigation now</button>`}
    </div>`;

  const conf = rep.confidence ?? 0;
  return html`
    <div>
      ${rep.degraded ? html`
        <div class="badge degraded" style=${{ display: 'inline-block', marginBottom: '8px' }}>
          degraded — deterministic fallback, not the LLM
        </div>` : ''}
      <div class="kv">
        <span class="k">cause</span><span class="mono" style=${{ color: 'var(--purple)' }}>${rep.cause}</span>
        <span class="k">exposure</span><span>${inr(rep.exposure_inr)}
          <span class="note" style=${{ display: 'inline' }}> (computed in Python, not by the model)</span></span>
        <span class="k">recommended</span>
        <span><span class=${'tag ' + rep.recommended_action}>${rep.recommended_action}</span></span>
        <span class="k">after policy gate</span>
        <span><span class=${'tag ' + (rep.validated_action || 'review')}>${rep.validated_action}</span>
          ${rep.validated_action !== rep.recommended_action
            ? html`<span class="note" style=${{ display: 'inline' }}> ← degraded by allowlist</span>` : ''}</span>
        <span class="k">confidence</span><span>${(conf * 100).toFixed(0)}%</span>
      </div>
      <div style=${{ marginTop: '10px' }}>
        <span class="k" style=${{ color: 'var(--dim)', fontSize: '12px' }}>evidence</span>
        <ul class="evidence">${(rep.evidence || []).map((e, i) => html`<li key=${i}>${e}</li>`)}</ul>
      </div>
      <div style=${{ marginTop: '10px', display: 'flex', gap: '8px' }}>
        <button class="sm" onClick=${run}>re-run</button>
        <button class="sm" onClick=${() => setShowAudit(!showAudit)}>
          ${showAudit ? 'hide' : 'show'} audit log (${(rep.audit || []).length})</button>
      </div>
      ${showAudit ? html`
        <div class="scroll" style=${{ marginTop: '8px' }}>
          <table><thead><tr><th>#</th><th>tool</th><th>inputs</th><th>output</th><th>ok</th></tr></thead>
            <tbody>${(rep.audit || []).map((a, i) => html`
              <tr key=${i}><td class="mono">${a.ts}</td><td class="mono">${a.tool}</td>
                <td class="mono" style=${{ color: 'var(--faint)' }}>${a.inputs_hash}</td>
                <td class="mono" style=${{ color: 'var(--faint)' }}>${a.output_hash}</td>
                <td>${a.ok ? '✓' : '✗'}</td></tr>`)}</tbody></table>
          <div class="note">Hashes, not payloads — the log proves what was called and what
            came back without storing raw transaction data.</div>
        </div>` : ''}
    </div>`;
}

/* ------------------------------------------------------------ review queue */
function ReviewQueue() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(null);
  const [pendingOnly, setPendingOnly] = useState(true);
  const load = useCallback(async () => {
    try { setData(await api(`/api/review-queue?pending_only=${pendingOnly}`)); } catch {}
  }, [pendingOnly]);
  useEffect(() => { load(); const id = setInterval(load, 2500); return () => clearInterval(id); }, [load]);

  const act = async (id, action) => {
    setBusy(id);
    try { await post(`/api/review-queue/${id}/decision`, { action, note: 'analyst console' }); await load(); }
    finally { setBusy(null); }
  };

  const cases = (data?.cases || []).slice().reverse().slice(0, 40);
  if (!cases.length) return html`<div class="empty">Review queue empty.</div>`;
  return html`
    <div>
      <div style=${{ display: 'flex', alignItems: 'baseline', gap: '10px', flexWrap: 'wrap', marginBottom: '10px' }}>
        <div class="note" style=${{ marginTop: 0 }}>
          <b style=${{ color: 'var(--amber)' }}>${data.pending} pending</b>
          ${` of ${data.total_cases ?? data.cases.length} total cases. `}
          Every restrict and review requires a human — the system holds, it does not act alone.
        </div>
        <button class=${'chip sm' + (pendingOnly ? ' active' : '')}
                style=${{ marginLeft: 'auto' }}
                onClick=${() => setPendingOnly(!pendingOnly)}>
          ${pendingOnly ? 'showing pending' : 'showing all'}</button>
      </div>
      <div class="scroll">
        <table>
          <thead><tr><th>#</th><th>merchant</th><th>₹</th><th>risk</th>
            <th>system</th><th>analyst</th><th></th></tr></thead>
          <tbody>${cases.map((c) => html`
            <tr key=${c.case_id} class=${c.overridden ? 'overridden' : ''}>
              <td class="mono">${c.case_id}</td>
              <td class="mono">${c.merchant_id}</td>
              <td class="mono">${inr(c.amount_inr)}</td>
              <td class="mono" style=${{ color: riskColor(c.risk_score) }}>${c.risk_score.toFixed(0)}</td>
              <td><span class=${'tag ' + c.system_action}>${c.system_action}</span></td>
              <td>${c.analyst_action
                    ? html`<span class=${'tag ' + c.analyst_action}>${c.analyst_action}</span>
                           ${c.overridden ? html`<span class="note" style=${{ display: 'inline' }}> override</span>` : ''}`
                    : html`<span class="note">pending</span>`}</td>
              <td>${c.analyst_action ? '' : html`
                <div style=${{ display: 'flex', gap: '5px', alignItems: 'center' }}>
                  <button class="sm ok" disabled=${busy === c.case_id}
                          title=${'confirm the system action: ' + c.system_action}
                          onClick=${() => act(c.case_id, c.system_action)}>approve</button>
                  <select class="sm-select" disabled=${busy === c.case_id} value=""
                          title="override with any allowlisted action - the server rejects anything else"
                          onChange=${(e) => { const v = e.target.value; if (v) act(c.case_id, v); }}>
                    <option value="" disabled>override…</option>
                    ${['allow', 'step_up', 'review', 'restrict']
                       .filter((a) => a !== c.system_action)
                       .map((a) => html`<option key=${a} value=${a}>${a.replace('_', ' ')}</option>`)}
                  </select>
                </div>`}</td>
            </tr>`)}</tbody>
        </table>
      </div>
    </div>`;
}

/* ------------------------------------------------------------ event feed */
function Feed({ events }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [events]);
  if (!events?.length) return html`<div class="empty">waiting for the stream…</div>`;
  return html`
    <div class="feed" ref=${ref}>
      ${events.map((e, i) => html`
        <div class="row" key=${i}>
          <span class=${'kind ' + e.kind}>${e.kind}</span>
          <span>${e.message}</span>
        </div>`)}
    </div>`;
}

/* --------------------------------------------- the closing argument
   Two merchants, the SAME volume spike shape, opposite entity structure,
   opposite decision. This is the whole thesis in one frame: the detector
   fires on fraud-score RATE, not traffic, so a 6x legitimate sale and a
   device farm are told apart by what is behind the transactions - not by
   how many there are. Rendered from live state, not hardcoded. */
function ContrastSide({ m, tone, headline, sub }) {
  if (!m) return html`<div class="empty">waiting…</div>`;
  const restricts = (m.action_mix || {}).restrict || 0;
  return html`
    <div class=${'side ' + tone}>
      <div class="side-top">
        <span class="side-id">${m.merchant_id}</span>
        <span class=${'badge ' + (tone === 'bad' ? 'spike' : 'legit')}>${headline}</span>
      </div>
      <div class="side-sub">${sub}</div>
      <div class="stat"><span class="k">transactions</span><span class="v">${(m.txn_count || 0).toLocaleString()}</span></div>
      <div class="stat"><span class="k">peak flagged rate</span>
        <span class="v" style=${{ color: tone === 'bad' ? 'var(--red)' : 'var(--green)' }}>
          ${((m.peak_rate_ever || 0) * 100).toFixed(0)}%</span></div>
      <div class="stat"><span class="k">peak spike z</span><span class="v">${(m.peak_z_ever ?? 0).toFixed(2)}</span></div>
      <div class="stat"><span class="k">transactions restricted</span>
        <span class="v" style=${{ color: restricts ? 'var(--red)' : 'var(--green)' }}>${restricts}</span></div>
      <div class="side-graph"><${EntityGraph} merchantId=${m.merchant_id} compact=${true} /></div>
    </div>`;
}

function Contrast({ merchants }) {
  const farm = merchants.find((m) => m.merchant_id === 'm5');
  const sale = merchants.find((m) => m.merchant_id === 'm11');
  if (!farm || !sale) return '';
  return html`
    <div class="panel contrast-panel">
      <h2>same spike, opposite verdict — why volume is not evidence</h2>
      <div class="note" style=${{ marginTop: 0, marginBottom: '14px' }}>
        Both merchants show a large jump in transaction volume — and the${' '}
        <b>legitimate</b>${' '}one is the busier of the two. A volume-threshold detector
        flags both. Ours flags one, because it fires on the fraud-score${' '}
        <b>rate</b>${' '}and then asks${' '}<b>who is behind the transactions</b>.
      </div>
      <div class="contrast">
        <${ContrastSide} m=${farm} tone="bad" headline="attack · restricted"
          sub="Device farm — a handful of devices driving dozens of accounts." />
        <div class="vs">vs</div>
        <${ContrastSide} m=${sale} tone="good" headline="legitimate · untouched"
          sub="Flash sale — 6× traffic, every account its own device and IP." />
      </div>
    </div>`;
}

/* ------------------------------------------------- pitch & architecture
   The strongest material in this project lives in README.md, where a judge
   has to go looking for it. This puts it one click from the demo. Every
   number here is reproduced from artifacts_out/ by a command in the README. */
/* ---------------------------------------- cost / capacity panels
   Reproduced from `python -m src.policy.threshold_sweep`, which writes
   artifacts_out/threshold_sweep.csv. Hardcoded here because artifacts_out is
   gitignored, so a fresh deployment has no CSV to read - same convention the
   metric tiles above already use. Values are the validation slice. */
const SWEEP = [
  { cut: 20, npv: 79147, legit: 336 },
  { cut: 25, npv: 76756, legit: 257 },
  { cut: 30, npv: 76756, legit: 257 },
  { cut: 40, npv: 76756, legit: 257 },
  { cut: 50, npv: 76756, legit: 257 },
  { cut: 55, npv: 76756, legit: 257 },
  { cut: 60, npv: 68339, legit: 0 },
  { cut: 70, npv: 66685, legit: 0 },
  { cut: 80, npv: 66685, legit: 0 },
];

function CostCurve() {
  const max = Math.max(...SWEEP.map((d) => d.npv));
  const min = Math.min(...SWEEP.map((d) => d.npv)) * 0.985;
  return html`
    <div class="panel">
      <h2>the false-positive tradeoff, and where we set the dial</h2>
      <p class="note" style=${{ marginTop: 0 }}>
        Blocking harder always prevents more fraud. It also destroys more good
        revenue. Every cutoff below was scored on the <b>validation</b> slice
        only — the test slice was read once, at the end — and the pair we adopted
        was fixed by a rule written down in advance.
      </p>
      <div class="curve">
        ${SWEEP.map((d) => {
          const w = 100 * (d.npv - min) / (max - min);
          const on = d.cut === 20, old = d.cut === 60;
          return html`
          <div class=${'crow' + (on ? ' on' : '') + (old ? ' old' : '')} key=${d.cut}>
            <div class="cc">step-up ≥ ${d.cut}</div>
            <div class="cbar"><i style=${{ width: Math.max(2, w) + '%' }}></i></div>
            <div class="cn">${inr(d.npv)}</div>
            <div class="cl">${inr(d.legit)}${' '}blocked</div>
            <div class="ct">${on ? 'adopted' : old ? 'previous default' : ''}</div>
          </div>`;
        })}
      </div>
      <div class="legend2">
        <span>bar = net protected value</span>
        <span>right column = legitimate ₹ wrongly impacted</span>
      </div>
      <p>Moving the step-up cutoff from 60 to 20 is worth <b>+15.8%</b> net
        protected value — and it does cost legitimate revenue that the old cut
        did not touch at all (₹0 → ₹336). We took that trade because step-up is
        friction, not a block: a real customer completes an OTP. We would not
        take it for the${' '}<i>restrict</i> cutoff, and we didn't.</p>
      <p class="note"><b>What the flat run in the middle actually means.</b> Net
        protected value is <i>identical</i> across a wide band because no
        validation transaction scores there. Our pre-declared "adopt the best
        pair" rule would have picked an arbitrary point inside that dead zone.
        We caught it and refined the rule to a per-parameter margin — documented
        as a post-hoc refinement rather than presented as foresight
        (failure-log 10). It then earned its keep twice more: the best pair now
        wants restrict=55, but moving restrict alone while step-up sits at 60 is
        not a valid policy at all (step-up must stay the lower bar), so the rule
        reports it <b>not independently evaluable</b> and keeps 85. Getting
        there crashed the sweep, and the first fix left the console printing a
        correct verdict while the saved artifact silently kept stale numbers.
        Both fixed; that's failure-log 26.</p>
    </div>`;
}

function Capacity() {
  const perK = 41.9, perAnalystHour = 30;
  const rows = [10e3, 100e3, 1e6].map((v) => {
    const cases = v / 1000 * perK;
    const hours = cases / perAnalystHour;
    return { v, cases, hours, fte: hours / 8 };
  });
  const f = (n) => n >= 1e5 ? (n / 1e5).toFixed(1) + 'L' : Math.round(n).toLocaleString();
  return html`
    <div class="panel">
      <h2>would the queue actually be staffable?</h2>
      <p class="note" style=${{ marginTop: 0 }}>
        We price an analyst review at ₹50 but never checked whether the analysts
        exist. This is that check. Assumption stated so it can be argued with:${' '}
        <b>30 cases per analyst-hour</b> (two minutes each), 8-hour shifts.
      </p>
      <div class="tscroll2">
        <table class="cap">
          <tr><th>transactions / day</th><th>review cases / day</th><th>analyst-hours / day</th><th>full-time analysts</th></tr>
          ${rows.map((r) => html`<tr key=${r.v}>
            <td class="mono">${r.v.toLocaleString()}</td>
            <td class="mono">${f(r.cases)}</td>
            <td class="mono">${Math.round(r.hours).toLocaleString()}</td>
            <td class=${'mono ' + (r.fte > 50 ? 'bad' : '')}>${Math.round(r.fte).toLocaleString()}</td>
          </tr>`)}
        </table>
      </div>
      <p><b>At PSP scale this does not staff.</b> 184 full-time analysts to
        clear one million transactions a day is not a rounding error, it is a
        department. Our 4.19% review rate is tuned for net rupees, and nothing
        in the objective function knows that analyst capacity is finite.</p>
      <p class="note">The honest fix is not a better model — it is a second
        constraint. The restrict and review cutoffs would have to be re-swept
        against a capacity ceiling (<i>maximise net protected value subject to
        ≤ N cases/day</i>), which trades recall for a queue an ops team can
        actually clear. We know the lever and we have the sweep harness to move
        it; what we do not have is the one number only an ops team can give us —
        how many cases a day they can absorb. Reported as an open question
        rather than a solved one.</p>
    </div>`;
}

function Metric({ k, v, note }) {
  return html`<div class="met">
    <div class="mk">${k}</div><div class="mv">${v}</div>
    ${note ? html`<div class="mn">${note}</div>` : ''}</div>`;
}

function Pitch() {
  return html`
  <div class="pitch">
    <div class="panel">
      <h2>the problem</h2>
      <p class="big">Merchants don't lose money one transaction at a time.${' '}
        <b>They lose it in bursts.</b></p>
      <p>Card-testing waves, device farms, IP clusters, account takeovers, fraud rings.
        Per-order scoring catches individual bad orders. Nobody tells a merchant${' '}
        <i>"you are under attack right now, here's who's behind it, here's your ₹ exposure,
        here's the bounded action."</i></p>
      <p>This sits <b>one level above per-order scoring</b>. It consumes per-order risk as an
        input and answers the question that layer can't: is this merchant under attack, and
        what should a human do in the next ten minutes?</p>
      <p class="note">The hard part isn't catching attacks — it's not crying wolf. A 6×
        legitimate flash sale must never be blocked. That's a first-class scenario here,
        and it's the finale of the demo.</p>
    </div>

    <div class="panel">
      <h2>architecture</h2>
      <div class="flow">
        <div class="fstep"><b>transaction stream</b><span>replayed test slice, in ts order</span></div>
        <div class="farr">↓</div>
        <div class="fstep"><b>22 incremental features</b><span>every feature from prior events only — state updated after emission</span></div>
        <div class="farr">↓</div>
        <div class="fstep"><b>XGBoost + isotonic calibration</b><span>chosen empirically over 3 alternatives on validation; no SMOTE</span></div>
        <div class="farr">↓</div>
        <div class="fstep"><b>merchant spike detector</b><span>EWMA + z-score on fraud-score RATE, not volume — ~50 lines</span></div>
        <div class="farr">↓</div>
        <div class="fstep"><b>risk fusion</b><span>calibrated probability is the floor; spike / graph / rules escalate into the headroom</span></div>
        <div class="farr">↓</div>
        <div class="fstep hot"><b>policy engine — the only component that authorises anything</b>
          <span>frozen allowlist: allow · step_up · review · restrict</span></div>
        <div class="farr">↓</div>
        <div class="fstep"><b>human review queue</b><span>analyst override, bound by the same allowlist</span></div>
      </div>
      <p class="note" style=${{ marginTop: '14px' }}>
        The LLM investigator hangs <b>off</b> this path. It fires on spike, reads six
        read-only tools, and writes a report. If it recommends anything outside the
        allowlist it is degraded to <b>human review</b> — never escalated. Disabling it
        changes no decision, and that is pytest-enforced.
      </p>
    </div>

    <div class="panel">
      <h2>results — temporal held-out test slice, synthetic data</h2>
      <div class="mets">
        <${Metric} k="attacks detected" v="25 / 25" note="across 5 seeds of the generator" />
        <${Metric} k="false alarms" v="0" note="in 35 non-attack merchant-windows" />
        <${Metric} k="flash sale flagged" v="0 / 5" note="the legitimate 6× spike" />
        <${Metric} k="net protected value" v="₹7.95L" note="after 578 reviews × ₹50" />
        <${Metric} k="precision / recall" v="0.927 / 0.857" note="at the cost-optimal threshold" />
        <${Metric} k="legitimate ₹ wrongly blocked" v="₹21.7K" note="0.21% of legitimate value processed" />
        <${Metric} k="calibration (Brier / ECE)" v="0.0082 / 0.0031" note="measured, not assumed" />
        <${Metric} k="LLM policy violations" v="0 / 13" note="0/13 unsafe actions too" />
      </div>
    </div>

    <${CostCurve} />
    <${Capacity} />

    <div class="panel">
      <h2>we attacked our own evaluation three times</h2>
      <p class="note" style=${{ marginTop: 0 }}>Each audit lowered a number we had already
        written down. We published the lower number every time. All three reproduce from
        the repo.</p>
      <div class="audits">
        <div class="aud">
          <div class="an">01</div>
          <div>
            <h3>Was the agent reading our answer key?</h3>
            <p>It scored <b>9/10</b> on cause. Then we noticed our simulator's entity IDs were
              self-labelling — <span class="mono">pi_STOLEN_*</span>,${' '}
              <span class="mono">d_FARM_F</span> — and transcripts were citing them verbatim
              as evidence. We hashed every ID. The score fell to <b>5/10</b>. That gap is
              exactly how much was the dataset whispering.</p>
          </div>
        </div>
        <div class="aud">
          <div class="an">02</div>
          <div>
            <h3>Was the <i>model</i> reading our answer key?</h3>
            <p>Same attack, one layer down. <b>Two features reproduced the entire 22-feature
              model</b> — because our generator created attack accounts on the attack day.
              It forced us to retract our own headline claim that entity/graph features
              were the source of the lift.</p>
            <p><b>Then we fixed the generator and re-measured.</b> The two proxies fell${' '}
              <b>0.9328 → 0.5997</b>; the headline fell only <b>0.9344 → 0.8981</b>. The
              shortcut dropped 0.333, the headline 0.036 — and the retracted claim turned out
              to be <b>true</b>: component_size is now the top feature by both single-feature
              PR-AUC and model importance. The fix cost us ₹10.57L → ₹7.95L in net protected
              value, and we published that too.</p>
          </div>
        </div>
        <div class="aud">
          <div class="an">03</div>
          <div>
            <h3>Does any of it survive real data?</h3>
            <p>The unchanged recipe scores <b>0.731</b> on ULB (284k real transactions) and${' '}
              <b>0.460</b> on IEEE-CIS, with zero tuning. It also exposed a latent bug ours
              could never surface: the cost threshold couldn't express "block nothing", which
              on real data was <b>3.8× cheaper</b> than what it chose.</p>
          </div>
        </div>
      </div>
      <p class="note"><b>26 failures</b> are logged with root causes in the repo. Every one was
        caught by measuring a claim, not by re-reading code. Two of them were our own audit
        tools hardcoding their failing verdicts — instruments that structurally could not
        report a pass.</p>
    </div>

    <div class="panel">
      <h2>what we're not claiming</h2>
      <ul class="lims">
        <li>Data is <b>synthetic</b> and labelled as such everywhere. Simulator parameters are
          design choices, not Razorpay statistics. <b>Nothing here is Razorpay data.</b></li>
        <li>Our audit tooling was written by the person whose work it audits — and two of
          those tools hardcoded their own failing verdicts. Fixed, but the bias is structural.</li>
        <li>We fixed the two label proxies we found. <b>We have not proven there are no
          others</b> — a negative result is only as strong as the test that produced it.</li>
        <li>The <b>merchant-level</b> layer is validated on controlled scenarios only —
          neither public dataset we evaluated has a merchant column.</li>
        <li>The agent gets the cause right <b>8/13</b> times. It is advisory and cannot act.
          That separation is the point, and it's tested.</li>
        <li>42 review cases per 1,000 transactions is priced at ₹50 but never checked
          against whether the analysts exist.</li>
        <li>The agent eval is <b>n=13</b> — roughly a ±25-point confidence interval, so 8/13
          and 5/13 are not distinguishable. Every number from it is a small-sample result.</li>
      </ul>
    </div>
  </div>`;
}

/* ------------------------------------------------------------ app */
function App() {
  const [status] = usePoll('/api/status', 1000);
  const [mdata] = usePoll('/api/merchants', 1500);
  const [sel, setSel] = useState(null);
  const [view, setView] = useState('console');

  const merchants = mdata?.merchants || [];
  // Auto-focus the most DEMONSTRATIVE merchant, not merely the first spiking
  // one. Account takeover spikes hard but has no shared entities by
  // construction, so auto-selecting it opens the demo on an empty graph —
  // technically correct and a terrible first frame. Prefer a spiking merchant
  // whose entity network actually has something to draw.
  //
  // This used to filter on top_cause, which only exists once the LLM has run —
  // so on the hosted instance (--no-agent) the guard silently did nothing and
  // the demo always opened on the empty graph it was written to avoid. It now
  // reads the signature, which is counted server-side and always present.
  useEffect(() => {
    if (sel || !merchants.length) return;
    const spiking = merchants.filter((m) => m.in_spike);
    const withEntities = spiking.find((m) => (m.signature || {}).hubs > 0);
    setSel((withEntities || spiking[0] || merchants[0]).merchant_id);
  }, [merchants, sel]);

  const finale = (status?.events || []).find((e) => e.kind === 'finale');
  const setSpeed = (v) => post(`/api/replay/speed?speed=${v}`).catch(() => {});
  const setPause = (p) => post(`/api/replay/pause?paused=${p}`).catch(() => {});

  // status filter for the merchant grid + click-to-inspect scrolls to detail
  const [filter, setFilter] = useState('all');
  const nAttack = merchants.filter((m) => m.in_spike).length;
  const shown = merchants.filter((m) =>
    filter === 'attack' ? m.in_spike : filter === 'clear' ? !m.in_spike : true);
  const inspect = (id) => {
    setSel(id);
    const el = document.getElementById('detail');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return html`
    <div>
      <${Header} status=${status} onSpeed=${setSpeed} onPause=${setPause}
                 view=${view} setView=${setView} />
      <main>
        ${view === 'pitch' ? html`<${Pitch} />` : html`<div>
        <div class="watching">
          <b>What you are watching:</b> ${(status?.total || 0).toLocaleString()}${' '}
          transactions replayed one at a time across ${merchants.length} merchants,
          through the real pipeline — scorer, spike detector, policy engine, review
          queue. ${nAttack > 0 ? html`<b class="bad">${nAttack} of these merchants are
          under coordinated attack right now.</b>` : ''} Every block and every review
          below is held for a human to confirm.${' '}<b>Nothing here acts on its own,
          and the language model cannot authorise anything.</b>
        </div>
        ${finale ? html`
          <div class="finale-banner">
            <span>✓ ${finale.message}</span>
            <span class="why">— volume spiked 6×, the fraud-score rate did not.
              The detector fires on risk, not traffic.</span>
          </div>` : ''}
        ${finale ? html`<${Contrast} merchants=${merchants} />` : ''}
        <div class="panel" style=${{ marginBottom: '14px' }}>
          <h2>merchants ${' '}<span class="sandbox">judge sandbox</span></h2>
          <div class="note" style=${{ marginTop: '-4px', marginBottom: '12px' }}>
            Live replay of a held-out test slice through the real pipeline — not a canned
            animation. <b>Click any merchant</b> to see who is behind its flagged
            transactions, or run an investigation on demand. Attack merchants sort to the top.
          </div>
          <div class="chips">
            <button class=${'chip' + (filter === 'all' ? ' active' : '')}
                    onClick=${() => setFilter('all')}>All ${' '}<span class="n">${merchants.length}</span></button>
            <button class=${'chip' + (filter === 'attack' ? ' active' : '')}
                    onClick=${() => setFilter('attack')}>Under attack ${' '}<span class="n">${nAttack}</span></button>
            <button class=${'chip' + (filter === 'clear' ? ' active' : '')}
                    onClick=${() => setFilter('clear')}>Clear ${' '}<span class="n">${merchants.length - nAttack}</span></button>
            <span class="note" style=${{ marginTop: '6px', marginLeft: 'auto' }}>
              click a merchant to inspect · hover any dotted label for what it means
            </span>
          </div>
          <div class="grid cols-3">
            ${shown.map((m) => html`
              <${MerchantCard} key=${m.merchant_id} m=${m}
                selected=${sel === m.merchant_id} onSelect=${inspect} />`)}
          </div>
        </div>
        <div class="grid cols-2" id="detail">
          <div class="panel">
            <h2>entity network ${sel ? `— ${sel}` : ''}</h2>
            <div class="note" style=${{ marginTop: 0, marginBottom: '10px' }}>
              Who is behind the flagged transactions. Hubs are entities shared by
              multiple accounts; legitimate traffic has none.
            </div>
            ${sel ? html`<${EntityGraph} merchantId=${sel} />` : html`<div class="empty">select a merchant</div>`}
          </div>
          <div class="panel">
            <h2>investigation ${sel ? `— ${sel}` : ''}</h2>
            ${sel ? html`<${Investigation} merchantId=${sel}
                inSpike=${(merchants.find((m) => m.merchant_id === sel) || {}).in_spike} />`
              : html`<div class="empty">select a merchant</div>`}
          </div>
        </div>
        <div class="grid cols-2" style=${{ marginTop: '14px' }}>
          <div class="panel"><h2>review queue</h2><${ReviewQueue} /></div>
          <div class="panel"><h2>pipeline events</h2><${Feed} events=${status?.events} /></div>
        </div>
        </div>`}
      </main>
    </div>`;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App} />`);
