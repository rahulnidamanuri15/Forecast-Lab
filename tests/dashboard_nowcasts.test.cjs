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

  test(`${prefix}: separate scored/pending nowcasts, metrics, and authoritative bundle`, async () => {
    const api = responses({ source: 'nowcast', forecast_date: '2030-01-02',
      created_at: '2030-01-02T12:00:00Z', cutoff: '2030-01-02T00:00:00Z',
      status: 'verified', [prefix === 'el' ? 'forecast_demand_mw' : 'forecast_pm2_5']: 50 });
    api[`${base}/predictions?model=lightgbm&limit=15&source=nowcast`].predictions = [
      { source: 'nowcast', forecast_date: '<unsafe>', created_at: 'issued-time',
        [valueKey]: 50, [actualKey]: 53, error: 3 },
      { source: 'nowcast', forecast_date: 'pending-day', [valueKey]: 52, [actualKey]: null },
      { source: 'verified', forecast_date: 'advance-must-not-render' },
      { source: 'backtest', forecast_date: 'backtest-must-not-render' },
    ];
    const { context, nodes, requested } = dashboard(api);
    await context.initNowcasts(prefix, { evaluation: [
      { model: 'lightgbm', nowcast: { scored_count: 1, pending_count: 1, mae: 3, rmse: 4 },
        verified: { mae: 9999 }, backtest: { mae: 8888 } },
    ] });
    assert.match(nodes.get(`${prefix}-nowcast-latest`).textContent, /DELAYED ESTIMATE.*scored.*cutoff/);
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /bundle.*training 2030-01-01 to 2030-01-02/);
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /Not historical per-prediction/);
    const records = nodes.get(`${prefix}-nowcast-records`).innerHTML;
    assert.match(records, /&lt;unsafe&gt;/);
    assert.match(records, /Delayed estimate · pending/);
    assert.match(records, /Delayed estimate · scored/);
    // The duplicate "latest scored day" subtable was removed: the canonical
    // advance leaderboard below already covers that shape, and the ledger's
    // scored row carries the same day-level error when n=1.
    assert.doesNotMatch(records, /leaderboard · latest scored day/);
    assert.doesNotMatch(requested.join('\n'), /leaderboard\?source=nowcast/);
    assert.equal(nodes.get(`${prefix}-nowcast-section`).style.display, '');
    assert.doesNotMatch(records, /9999|8888|must-not-render|<unsafe>|undefined/);
    const filtered = context.publishedOnly({ predictions: [
      { source: 'verified', provenance_consistent: true }, { source: 'nowcast' }, { source: 'backtest' },
      { source: 'verified', provenance_consistent: false },
    ] });
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0].source, 'verified');
  });

  test(`${prefix}: expected empty nowcast record still shows model diagnostics`, async () => {
    const { context, nodes } = dashboard(responses());
    await context.initNowcasts(prefix, { evaluation: [] });
    assert.equal(nodes.get(`${prefix}-nowcast-latest`).textContent, 'No delayed estimate on record.');
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /Current deployment: bundle/);
    assert.doesNotMatch(nodes.get(`${prefix}-nowcast-records`).innerHTML, /undefined/);
    assert.equal(nodes.get(`${prefix}-nowcast-section`).style.display, 'none');
  });

  test(`${prefix}: artifact corruption is visible`, async () => {
    const api = responses();
    api[`${base}/diagnostics`] = { model_bundle: { status: 'error' } };
    const { context, nodes } = dashboard(api);
    await context.initNowcasts(prefix, null);
    assert.match(nodes.get(`${prefix}-bundle`).textContent, /MODEL ARTIFACT ERROR/);
  });

  test(`${prefix}: nowcast-only deployment does not fill advance KPIs`, async () => {
    const api = responses();
    Object.assign(api, {
      [`${base}/health`]: { status: 'ok', stale_days: 2, source_lag_expected: true },
      [`${base}/forecast?model=lightgbm&source=daily`]: 404,
      [`${base}/history?days=30`]: { historical_data: [] },
      [`${base}/evaluation`]: { evaluation: [
        { model: 'lightgbm', nowcast: { scored_count: 10, mae: 777, rmse: 888 } },
      ] },
      [`${base}/predictions?model=lightgbm&limit=15&source=daily`]: { predictions: [] },
      [`${base}/leaderboard`]: 404,
    });
    const { context, nodes } = dashboard(api);
    await (prefix === 'el' ? context.initElectricity() : context.initDashboard());
    assert.match(nodes.get(prefix === 'el' ? 'el-hero-date' : 'hero-forecast-date').textContent,
      /No advance forecast on record/);
    const metric = nodes.get(prefix === 'el' ? 'el-metric-lgb-mae' : 'metric-lgb-mae');
    assert.equal(metric.textContent, '');
    assert.match(nodes.get(`${prefix}-nowcast-records`).innerHTML, /777/);
  });
}
