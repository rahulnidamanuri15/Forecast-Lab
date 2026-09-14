// Execute the actual dashboard JavaScript, not a second implementation of it.
// Minimal DOM adapter: these tests cover rendering/data isolation, not CSS layout.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const html = readFileSync(require('node:path').join(__dirname, '..', 'index.html'), 'utf8');
const source = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)]
  .map(m => m[1]).find(s => s.includes('function initNowcasts'));

function dashboard(responses) {
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], {
    textContent: '', innerHTML: '', style: {}, hidden: true,
    classList: { add() {}, remove() {}, toggle() {} },
    setAttribute() {}, removeAttribute() {}, addEventListener() {},
  }]));
  const requested = [];
  const context = vm.createContext({
    window: { API_BASE: 'http://localhost' },
    document: { readyState: 'loading', addEventListener() {}, querySelectorAll: () => [],
      getElementById(id) { assert.ok(nodes.has(id), `missing DOM id ${id}`); return nodes.get(id); } },
    console, setTimeout, clearTimeout, AbortController,
    fetch: async url => {
      const path = url.replace('http://localhost', '');
      requested.push(path);
      assert.ok(Object.hasOwn(responses, path), `unexpected API request ${path}`);
      const data = responses[path];
      return { ok: data !== 404, status: data === 404 ? 404 : 200, json: async () => data };
    },
  });
  vm.runInContext(source, context);
  return { context, nodes, requested };
}

for (const [prefix, base, valueKey, actualKey] of [
  ['pm25', '', 'predicted_pm2_5', 'actual_pm2_5'],
  ['el', '/electricity', 'predicted_demand_mw', 'actual_demand_mw'],
]) {
  function responses(forecast = 404) {
    return {
      [`${base}/forecast?model=lightgbm&source=nowcast`]: forecast,
      [`${base}/predictions?model=lightgbm&limit=15&source=nowcast`]: { predictions: [] },
      [`${base}/diagnostics`]: { model_bundle: { status: 'ok', artifact_format: 'bundle',
        feature_count: 15, bundle_version: 1,
        training_window: { first: '2030-01-01', last: '2030-01-02', rows: 2 } } },
    };
  }

  test(`${prefix}: top nowcast block is a banner only, rows returned for the canonical tables`, async () => {
    const api = responses({ source: 'nowcast', forecast_date: '2030-01-02',
      created_at: '2030-01-02T12:00:00Z', cutoff: '2030-01-02T00:00:00Z',
      status: 'verified', [prefix === 'el' ? 'forecast_demand_mw' : 'forecast_pm2_5']: 50 });
    api[`${base}/predictions?model=lightgbm&limit=15&source=nowcast`].predictions = [
      { source: 'nowcast', forecast_date: '<unsafe>', created_at: 'issued-time',
        model: 'lightgbm', [valueKey]: 50, [actualKey]: 53, error: 3,
        ...(prefix === 'el' ? { error_pct: 5 } : {}) },
      { source: 'nowcast', forecast_date: 'pending-day', model: 'lightgbm',
        [valueKey]: 52, [actualKey]: null },
      { source: 'verified', forecast_date: 'advance-must-not-render' },
      { source: 'backtest', forecast_date: 'backtest-must-not-render' },
    ];
    const { context, nodes, requested } = dashboard(api);
    const result = await context.initNowcasts(prefix, { evaluation: [
      { model: 'lightgbm', nowcast: { scored_count: 1, pending_count: 1, mae: 3, rmse: 4 },
        verified: { mae: 9999 }, backtest: { mae: 8888 } },
    ] });
    assert.match(nodes.get(`${prefix}-nowcast-latest`).textContent, /DELAYED ESTIMATE.*scored.*cutoff/);
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /bundle.*training 2030-01-01 to 2030-01-02/);
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /Not historical per-prediction/);
    // Single-table rule: no duplicate evaluation/ledger tables on top. The rows
    // are returned so the canonical Latest-Scored-Day + ledger below render them.
    assert.equal(result.hasNowcast, true);
    assert.equal(result.rows.length, 2);
    assert.ok(result.rows.every(r => r.source === 'nowcast'));
    const records = nodes.get(`${prefix}-nowcast-records`).innerHTML;
    assert.equal(records, '');
    assert.doesNotMatch(records, /Nowcast evaluation|Nowcast ledger|must-not-render/);
    // No per-model leaderboard round-trip here: the canonical board below covers
    // that shape via its own fallback fetch only when advance is empty.
    assert.doesNotMatch(requested.join('\n'), /leaderboard\?source=nowcast/);
    assert.equal(nodes.get(`${prefix}-nowcast-section`).style.display, '');
    const filtered = context.publishedOnly({ predictions: [
      { source: 'verified', provenance_consistent: true }, { source: 'nowcast' }, { source: 'backtest' },
      { source: 'verified', provenance_consistent: false },
    ] });
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0].source, 'verified');
  });

  test(`${prefix}: expected empty nowcast record still shows model diagnostics`, async () => {
    const { context, nodes } = dashboard(responses());
    const result = await context.initNowcasts(prefix, { evaluation: [] });
    assert.equal(nodes.get(`${prefix}-nowcast-latest`).textContent, 'No delayed estimate on record.');
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /Current deployment: bundle/);
    assert.equal(nodes.get(`${prefix}-nowcast-records`).innerHTML, '');
    assert.equal(result.hasNowcast, false);
    assert.equal(result.rows.length, 0);
    assert.equal(nodes.get(`${prefix}-nowcast-section`).style.display, 'none');
  });

  test(`${prefix}: artifact corruption is visible`, async () => {
    const api = responses();
    api[`${base}/diagnostics`] = { model_bundle: { status: 'error' } };
    const { context, nodes } = dashboard(api);
    await context.initNowcasts(prefix, null);
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /MODEL ARTIFACT ERROR/);
  });

  test(`${prefix}: nowcast-only deployment renders nowcasts in the canonical tables`, async () => {
    const api = responses({ source: 'nowcast', forecast_date: '2030-01-02',
      created_at: '2030-01-02T12:00:00Z', cutoff: '2030-01-02T00:00:00Z',
      status: 'pending', [prefix === 'el' ? 'forecast_demand_mw' : 'forecast_pm2_5']: 50 });
    api[`${base}/predictions?model=lightgbm&limit=15&source=nowcast`].predictions = [
      { source: 'nowcast', forecast_date: '2030-01-02', created_at: '2030-01-02T12:00:00Z',
        model: 'lightgbm', [valueKey]: 50, [actualKey]: 53, error: 3,
        ...(prefix === 'el' ? { error_pct: 5 } : {}) },
      { source: 'nowcast', forecast_date: '2030-01-01', created_at: '2030-01-01T12:00:00Z',
        model: 'lightgbm', [valueKey]: 52, [actualKey]: null, error: null,
        ...(prefix === 'el' ? { error_pct: null } : {}) },
    ];
    Object.assign(api, {
      [`${base}/health`]: { status: 'ok', stale_days: 2, source_lag_expected: true },
      [`${base}/forecast?model=lightgbm&source=daily`]: 404,
      [`${base}/history?days=30`]: { historical_data: [] },
      [`${base}/evaluation`]: { evaluation: [
        { model: 'lightgbm', nowcast: { scored_count: 10, pending_count: 1, mae: 777, rmse: 888 } },
      ] },
      [`${base}/predictions?model=lightgbm&limit=15&source=daily`]: { predictions: [] },
      [`${base}/leaderboard`]: 404,
      [`${base}/leaderboard?source=nowcast`]: { source: 'nowcast', leaderboard: [
        { model: 'lightgbm', mae: 777, rmse: 888, sample_size: 1, as_of: '2030-01-02',
          description: 'delayed', ...(prefix === 'el' ? { mape: 1.5 } : {}) },
      ] },
    });
    const { context, nodes, requested } = dashboard(api);
    await (prefix === 'el' ? context.initElectricity() : context.initDashboard());
    assert.match(nodes.get(prefix === 'el' ? 'el-hero-date' : 'hero-forecast-date').textContent,
      /No advance forecast on record/);
    // No duplicate top tables: the banner stays, the rows live below.
    assert.equal(nodes.get(`${prefix}-nowcast-records`).innerHTML, '');
    assert.equal(nodes.get(`${prefix}-nowcast-section`).style.display, '');
    // Canonical ledger shows the delayed rows with provenance labelling.
    const ledgerId = prefix === 'el' ? 'el-history-table-body' : 'history-table-body';
    const ledgerHtml = nodes.get(ledgerId).innerHTML;
    assert.match(ledgerHtml, /2030-01-02/);
    assert.match(ledgerHtml, /Delayed/);
    assert.doesNotMatch(ledgerHtml, /No prediction published/);
    // Canonical board shows the delayed latest-scored-day.
    const boardId = prefix === 'el' ? 'el-leaderboard-table-body' : 'leaderboard-table-body';
    assert.match(nodes.get(boardId).innerHTML, /lightgbm/);
    // Subtitles say what the rows are: never presented as advance forecasts.
    assert.match(nodes.get(`${prefix}-ledger-subtitle`).textContent, /Delayed estimates/);
    assert.match(nodes.get(`${prefix}-board-subtitle`).textContent, /delayed-estimate|nowcast/i);
    // Headline metrics fall back to the delayed record instead of em-dashes.
    const metric = nodes.get(prefix === 'el' ? 'el-metric-lgb-mae' : 'metric-lgb-mae');
    assert.match(String(metric.textContent), /777/);
    // One fallback board fetch only: 6 advance + 3 nowcast + 1 fallback.
    assert.equal(requested.length, 10);
    assert.ok(requested.some(p => p === `${base}/leaderboard?source=nowcast`));
  });

  test(`${prefix}: advance rows are never merged with nowcasts`, async () => {
    const api = responses({ source: 'nowcast', forecast_date: '2030-01-02',
      created_at: '2030-01-02T12:00:00Z', cutoff: '2030-01-02T00:00:00Z',
      status: 'pending', [prefix === 'el' ? 'forecast_demand_mw' : 'forecast_pm2_5']: 999 });
    api[`${base}/predictions?model=lightgbm&limit=15&source=nowcast`].predictions = [
      { source: 'nowcast', forecast_date: 'nowcast-must-not-merge', model: 'lightgbm',
        [valueKey]: 999, [actualKey]: null },
    ];
    const advanceRow = prefix === 'el'
      ? { source: 'verified', provenance_consistent: true, forecast_date: 'advance-day',
          model: 'lightgbm', predicted_demand_mw: 10, actual_demand_mw: 12, error: 2, error_pct: 16,
          created_at: '2030-01-01T00:00:00Z' }
      : { source: 'verified', provenance_consistent: true, forecast_date: 'advance-day',
          model: 'lightgbm', predicted_pm2_5: 10, actual_pm2_5: 12, error: 2,
          created_at: '2030-01-01T00:00:00Z' };
    Object.assign(api, {
      [`${base}/health`]: { status: 'ok', stale_days: 0, source_lag_expected: true },
      [`${base}/forecast?model=lightgbm&source=daily`]: { source: 'verified',
        forecast_date: 'advance-day', created_at: '2030-01-01T00:00:00Z',
        [prefix === 'el' ? 'forecast_demand_mw' : 'forecast_pm2_5']: 10, status: 'pending' },
      [`${base}/history?days=30`]: { historical_data: [] },
      [`${base}/evaluation`]: { evaluation: [
        { model: 'lightgbm', verified: { scored_count: 1, mae: 2, rmse: 2 } },
      ] },
      [`${base}/predictions?model=lightgbm&limit=15&source=daily`]: { predictions: [advanceRow] },
      [`${base}/leaderboard`]: { source: 'verified', leaderboard: [
        { model: 'lightgbm', mae: 2, rmse: 2, sample_size: 1, as_of: 'advance-day',
          description: 'advance', ...(prefix === 'el' ? { mape: 1 } : {}) },
      ] },
    });
    const { context, nodes, requested } = dashboard(api);
    await (prefix === 'el' ? context.initElectricity() : context.initDashboard());
    const ledgerId = prefix === 'el' ? 'el-history-table-body' : 'history-table-body';
    const ledgerHtml = nodes.get(ledgerId).innerHTML;
    assert.match(ledgerHtml, /advance-day/);
    assert.doesNotMatch(ledgerHtml, /nowcast-must-not-merge/);
    assert.doesNotMatch(ledgerHtml, /Delayed/);
    assert.equal(nodes.get(`${prefix}-nowcast-records`).innerHTML, '');
    assert.doesNotMatch(requested.join('\n'), /leaderboard\?source=nowcast/);
  });
}
