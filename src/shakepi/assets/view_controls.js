/* Read the rendered axes at click time, rather than retaining zoom events. */
(function () {
    "use strict";

    function matchesSelection(figure, period, channels) {
        const meta = figure?.layout?.meta;
        return Boolean(figure?.data?.length && meta && period != null &&
            meta.period_id === period && channels?.length &&
            JSON.stringify(meta.channels) === JSON.stringify([...channels].sort()));
    }

    function utcMilliseconds(value) {
        // Plotly date axes use milliseconds for numeric bounds, and may emit
        // timezone-free strings even though these recordings are in UTC.
        if (typeof value === "number") return value;
        if (typeof value !== "string" || !value.trim()) return NaN;
        let text = value.trim().replace(" ", "T");
        if (/^\d{4}-\d{2}-\d{2}$/.test(text)) text += "T00:00:00";
        if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(text)) text += "Z";
        return Date.parse(text);
    }

    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.shakepi = {
        viewButtonsDisabled: function (figure, period, channels) {
            const disabled = !matchesSelection(figure, period, channels);
            return [disabled, disabled];
        },
        useVisibleRange: function (clicks, figure, period, channels) {
            const unchanged = window.dash_clientside.no_update;
            const preserve = message => [unchanged, unchanged, message];
            if (!clicks) return [unchanged, unchanged, unchanged];
            if (!matchesSelection(figure, period, channels)) {
                return preserve("Load a waveform for the selected recording and channels first.");
            }
            const graph = document.querySelector("#overview-graph .js-plotly-plot");
            const layout = graph?._fullLayout;
            // A new figure can arrive before Plotly finishes drawing it.
            if (!layout || JSON.stringify(layout.meta) !== JSON.stringify(figure.layout.meta)) {
                return preserve("The waveform is updating. Try again when it finishes.");
            }
            // Waveform and spectrogram time axes are matched by Plotly, so
            // this range also reflects zooming on any linked subplot.
            const range = layout.xaxis?.range;
            if (!Array.isArray(range) || range.length !== 2) {
                return preserve("No visible time range is available. Reset the view and try again.");
            }
            const start = utcMilliseconds(range[0]);
            const end = utcMilliseconds(range[1]);
            if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end ||
                Math.abs(start) > 8640000000000000 || Math.abs(end) > 8640000000000000) {
                return preserve("The visible time range is invalid. Reset the view and try again.");
            }
            return [new Date(start).toISOString(), new Date(end).toISOString(),
                "Search times updated to the visible range."];
        }
    };
})();
