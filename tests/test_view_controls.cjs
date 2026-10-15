// Run with: node --test tests/test_view_controls.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
    const unchanged = Symbol('no_update');
    const meta = {period_id: 1, channels: ['EHZ'], default_time_range: ['a', 'b']};
    const figure = {data: [{}], layout: {meta}};
    const graph = {_fullLayout: {meta, xaxis: {range: ['2026-02-02 00:00:00', '2026-02-02 00:05:00']}}};
    const context = {window: {dash_clientside: {no_update: unchanged}},
        document: {querySelector: () => graph}};
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/shakepi/assets/view_controls.js'), 'utf8'), context);
    const api = context.window.dash_clientside.shakepi;
    return {figure, graph, unchanged, api, use: () => api.useVisibleRange(1, figure, 1, ['EHZ'])};
}

test('initial view, zoom, reset and numeric UTC bounds are read live', () => {
    const s = setup();
    assert.equal(s.use()[0], '2026-02-02T00:00:00.000Z');
    s.graph._fullLayout.xaxis.range = ['2026-02-02 00:01:00', '2026-02-02 00:02:00'];
    assert.equal(s.use()[0], '2026-02-02T00:01:00.000Z');
    s.graph._fullLayout.xaxis.range = ['2026-02-02 00:00:00', '2026-02-02 00:05:00'];
    assert.equal(s.use()[1], '2026-02-02T00:05:00.000Z');
    s.graph._fullLayout.xaxis.range = [1770000000000, 1770000300000];
    assert.equal(s.use()[0], new Date(1770000000000).toISOString());
});

test('missing, stale and invalid views preserve both search fields', () => {
    const s = setup();
    for (const bounds of [null, [], ['bad', '2026-02-02'], [2, 1], [1, 1], [0, Infinity]]) {
        s.graph._fullLayout.xaxis.range = bounds;
        const result = s.use();
        assert.equal(result[0], s.unchanged);
        assert.equal(result[1], s.unchanged);
        assert.ok(result[2]);
    }
    assert.equal(s.api.useVisibleRange(1, s.figure, 2, ['EHZ'])[0], s.unchanged);
    assert.equal(s.api.useVisibleRange(1, s.figure, 1, ['ENZ'])[0], s.unchanged);
    s.graph._fullLayout.meta = {};
    assert.equal(s.use()[0], s.unchanged);
});

test('buttons require a populated figure matching the selection', () => {
    const s = setup();
    assert.equal(s.api.viewButtonsDisabled(s.figure, 1, ['EHZ'])[0], false);
    for (const [figure, period, channels] of [
        [{data: []}, 1, ['EHZ']], [s.figure, null, ['EHZ']],
        [s.figure, 2, ['EHZ']], [s.figure, 1, ['ENZ']]
    ]) assert.equal(s.api.viewButtonsDisabled(figure, period, channels)[0], true);
});
