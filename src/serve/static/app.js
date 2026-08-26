/* Fraud Spike Investigator — dashboard.
   React + htm (tagged templates), no JSX, no build step, no CDN.
   Everything shown here comes from the live pipeline via /api. */
const { useState, useEffect, useMemo, useRef, useCallback } = React;
const html = htm.bind(React.createElement);

const api = async (path, opts) => {
  const r = await fetch(path, opts);
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
function Header({ status, onSpeed, onPause }) {
  const s = status || {};
  return html`
    <header>
      <div class="brand">
        <div class="mark">R</div>
        <div>
          <h1>Fraud Spike Investigator</h1>
          <div class="sub">merchant-level detection · entity correlation · policy-gated investigation</div>
        </div>
      </div>
      <span class="spacer"></span>
      <span class="counter mono">
        ${`${(s.processed || 0).toLocaleString()} / ${(s.total || 0).toLocaleString()} txns · ${(s.pct || 0).toFixed(0)}%`}
        ${s.merchants_in_spike ? html`${' · '}<b>${s.merchants_in_spike} under attack</b>` : ''}
        ${s.review_pending ? html`${' · '}<span style=${{color:'var(--amber)'}}>${s.review_pending} awaiting review</span>` : ''}
      </span>
      <label class="sub">speed
        <input type="range" min="50" max="4000" step="50" value=${s.speed_tps || 200}
               onChange=${(e) => onSpeed(+e.target.value)} style=${{ width: '120px', marginLeft: '6px' }} />
        <span class="mono"> ${Math.round(s.speed_tps || 0)}/s</span>
      </label>
      <button onClick=${() => onPause(!s.paused)}>${s.paused ? '▶ resume' : '⏸ pause'}</button>
    </header>`;
}

/* ------------------------------------------------------------ merchants */
function MerchantCard({ m, selected, onSelect }) {
  const d = m.fraud_rate || {};
  const isFlash = m.merchant_id === 'm11' && m.txn_count > 300;
  const peakPct = (m.peak_rate_ever || 0) * 100;
  const cls = ['mcard', m.in_spike ? 'spiking' : '',
               isFlash && !m.in_spike ? 'flash' : '',
               selected ? 'selected' : ''].join(' ');
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
      <div class="stat hero"><span class="k">peak flagged rate</span>
        <span class="v" style=${{ color: rateColor(m.peak_rate_ever) }}>
          ${(m.peak_rate_ever * 100).toFixed(0)}%</span></div>
      <div class="stat"><span class="k">spike z-score</span>
        <span class=${'v ' + (m.peak_z_ever >= 4 ? 'delta up' : 'delta flat')}>
          ${m.peak_z_ever >= 4 ? `z=${m.peak_z_ever} · fired` : `z=${m.peak_z_ever} · below threshold`}</span></div>
      <div class="stat"><span class="k">now / baseline</span>
        <span class="v" style=${{ color: 'var(--dim)' }}>
          ${(d.current_rate * 100).toFixed(1)}% vs ${(d.baseline_rate * 100).toFixed(1)}%</span></div>
      <div class="stat"><span class="k">₹ exposure</span><span class="v">${inr(m.exposure_inr)}</span></div>
      <div class="stat"><span class="k">txns at risk</span>
        <span class="v">${m.flagged_count} / ${m.txn_count}</span></div>
      ${m.top_cause ? html`<div class="stat"><span class="k">cause</span>
        <span class="v mono" style=${{ color: 'var(--violet)' }}>${m.top_cause}</span></div>` : ''}
    </div>`;
}

/* ------------------------------------------------------------ entity graph */
/* Deliberately a tiny hand-rolled force layout rather than a graph library:
   ~30 lines, no dependency, and enough to show the shape that matters —
   a few entities fanning out across many accounts. */
function EntityGraph({ merchantId }) {
  const [graph] = usePoll(`/api/merchants/${merchantId}/entity-graph`, 3000, [merchantId]);
  const [pos, setPos] = useState([]);
  const raf = useRef(null);

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
    return html`<div class="empty">
      No shared entities among flagged transactions — every account uses its own
      device, IP and instrument. This is what legitimate traffic looks like.
    </div>`;

  const idx = new Map(graph.nodes.map((n, i) => [n.id, i]));
  return html`
    <div>
      <svg class="graph" viewBox="0 0 720 400" preserveAspectRatio="xMidYMid meet">
        ${graph.links.map((l, i) => {
          const a = pos[idx.get(l.source)], b = pos[idx.get(l.target)];
          return a && b ? html`<line key=${i} x1=${a.x} y1=${a.y} x2=${b.x} y2=${b.y} />` : null;
        })}
        ${pos.map((p, i) => {
          const hub = p.n.kind !== 'customer';
          const r = hub ? Math.min(17, 7 + p.n.size * 0.8) : 4;
          return html`
          <g key=${i}>
            <circle class=${p.n.kind} cx=${p.x} cy=${p.y} r=${r}
                    stroke=${hub ? 'rgba(255,255,255,.55)' : 'none'} stroke-width=${hub ? 1.5 : 0}>
              <title>${p.n.kind}: ${p.n.label}${p.n.size > 1 ? ` — ${p.n.size} accounts` : ''}</title>
            </circle>
            ${hub ? html`
              <text x=${p.x} y=${p.y + r + 12} text-anchor="middle"
                    style=${{ paintOrder: 'stroke', stroke: '#050933', strokeWidth: '3px' }}>
                ${p.n.label.slice(0, 14)} · ${p.n.size}</text>` : ''}
          </g>`; })}
      </svg>
      <div class="legend">
        <span><i style=${{ background: 'var(--red)' }}></i>device</span>
        <span><i style=${{ background: 'var(--amber)' }}></i>IP</span>
        <span><i style=${{ background: 'var(--purple)' }}></i>instrument</span>
        <span><i style=${{ background: '#4b5c6b' }}></i>account</span>
        <span style=${{ marginLeft: 'auto' }}>node size = accounts sharing it</span>
      </div>
      <div class="note">${graph.note}. A ring or farm shows as few large hubs with
        many accounts attached; ordinary traffic shows as nothing at all.</div>
    </div>`;
}

/* ------------------------------------------------------------ investigation */
function Investigation({ merchantId, inSpike }) {
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
        ${inSpike === false
          ? 'No spike -> no investigation. The agent only runs when the detector fires, so quiet merchants cost zero analyst time and zero tokens.'
          : 'No investigation yet — these fire automatically when the spike detector trips.'}
      </div>
      <button class="sm" onClick=${run}>run investigation now</button>
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
  const load = useCallback(async () => {
    try { setData(await api('/api/review-queue?pending_only=false')); } catch {}
  }, []);
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
      <div class="note" style=${{ marginTop: 0, marginBottom: '10px' }}>
        <b style=${{ color: 'var(--amber)' }}>${data.pending} pending</b>
        ${` of ${data.total_cases ?? data.cases.length} total cases. `}
        Every restrict and review requires a human — the system holds, it does not act alone.
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
                <div style=${{ display: 'flex', gap: '4px' }}>
                  <button class="sm ok" disabled=${busy === c.case_id}
                          onClick=${() => act(c.case_id, c.system_action)}>approve</button>
                  <button class="sm" disabled=${busy === c.case_id}
                          onClick=${() => act(c.case_id, 'allow')}>override→allow</button>
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

/* ------------------------------------------------------------ app */
function App() {
  const [status] = usePoll('/api/status', 1000);
  const [mdata] = usePoll('/api/merchants', 1500);
  const [sel, setSel] = useState(null);

  const merchants = mdata?.merchants || [];
  // Auto-focus the most DEMONSTRATIVE merchant, not merely the first spiking
  // one. Account takeover spikes hard but has no shared entities by
  // construction, so auto-selecting it opens the demo on an empty graph —
  // technically correct and a terrible first frame. Prefer a spiking merchant
  // whose entity network actually has something to draw.
  useEffect(() => {
    if (sel || !merchants.length) return;
    const spiking = merchants.filter((m) => m.in_spike);
    const withEntities = spiking.find((m) => m.top_cause &&
      ['device_farm', 'fraud_ring', 'ip_cluster', 'card_testing'].includes(m.top_cause));
    setSel((withEntities || spiking[0] || merchants[0]).merchant_id);
  }, [merchants, sel]);

  const finale = (status?.events || []).find((e) => e.kind === 'finale');
  const setSpeed = (v) => post(`/api/replay/speed?speed=${v}`).catch(() => {});
  const setPause = (p) => post(`/api/replay/pause?paused=${p}`).catch(() => {});

  return html`
    <div>
      <${Header} status=${status} onSpeed=${setSpeed} onPause=${setPause} />
      <main>
        ${finale ? html`
          <div class="finale-banner">
            <span>✓ ${finale.message}</span>
            <span class="why">— volume spiked 6×, the fraud-score rate did not.
              The detector fires on risk, not traffic.</span>
          </div>` : ''}
        <div class="panel" style=${{ marginBottom: '14px' }}>
          <h2>merchants</h2>
          <div class="grid cols-3">
            ${merchants.map((m) => html`
              <${MerchantCard} key=${m.merchant_id} m=${m}
                selected=${sel === m.merchant_id} onSelect=${setSel} />`)}
          </div>
        </div>
        <div class="grid cols-2">
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
      </main>
    </div>`;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App} />`);
