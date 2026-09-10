"use strict";

/* Front end for the city weather comparison.
 *
 * All the number-crunching happens server side in /api/weather (see weatherlib/).
 * This file turns the per-city payloads into the same set of Plotly charts the
 * old Streamlit app drew: monthly climate normals, daytime temperature, the
 * monthly-averages "prediction" table, and the current-year-vs-baseline
 * temperature and rainfall comparisons.
 */

const PALETTE = [
  { line: "blue", fill: "rgba(0,0,255,0.1)" },
  { line: "red", fill: "rgba(255,0,0,0.1)" },
  { line: "green", fill: "rgba(0,128,0,0.1)" },
];
const ANOMALY = { warm: "rgba(214,39,40,0.45)", cool: "rgba(31,119,180,0.45)" };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const PLOT_CONFIG = { responsive: true, displayModeBar: false };

const $ = (sel) => document.querySelector(sel);
const isDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
const ink = () => (isDark() ? "#e6e8eb" : "#222222");

let cityData = [];

/* ------------------------------------------------------------------ layout */

function baseLayout({ title, xTitle, yTitle, height = 400, hovermode } = {}) {
  const grid = "rgba(128,128,128,0.15)";
  const zero = "rgba(128,128,128,0.35)";
  return {
    title: title ? { text: title, font: { size: 15 } } : undefined,
    height,
    hovermode: hovermode || "closest",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: isDark() ? "#c9ced6" : "#1a1d21", size: 12 },
    margin: { l: 55, r: 20, t: title ? 50 : 20, b: 60 },
    xaxis: { title: xTitle, gridcolor: grid, zerolinecolor: zero },
    yaxis: { title: yTitle, gridcolor: grid, zerolinecolor: zero },
    legend: { orientation: "h", y: -0.22, font: { size: 11 } },
  };
}

function chartBlock(parent, heading, caption) {
  const block = document.createElement("div");
  block.className = "chart-block";
  const h = document.createElement("h3");
  h.textContent = heading;
  block.appendChild(h);
  if (caption) {
    const p = document.createElement("p");
    p.className = "caption";
    p.textContent = caption;
    block.appendChild(p);
  }
  const chart = document.createElement("div");
  chart.className = "chart";
  block.appendChild(chart);
  parent.appendChild(block);
  return chart;
}

function note(parent, text) {
  const p = document.createElement("p");
  p.className = "caption";
  p.textContent = text;
  parent.appendChild(p);
}

/* ----------------------------------------------------------------- fetching */

async function fetchCity(name, start, end) {
  const url = `/api/weather?city=${encodeURIComponent(name)}` +
              `&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`;
  const res = await fetch(url);
  let body;
  try {
    body = await res.json();
  } catch (_) {
    throw new Error(`Request failed (${res.status}).`);
  }
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status}).`);
  return body;
}

function setStatus(banners) {
  const box = $("#status");
  box.innerHTML = "";
  for (const b of banners) {
    const div = document.createElement("div");
    div.className = `banner ${b.cls}`;
    div.textContent = b.text;
    box.appendChild(div);
  }
}

/* -------------------------------------------------------------- form submit */

function readForm() {
  const names = [$("#city1").value.trim(), $("#city2").value.trim()];
  const useThird = $("#add-third").checked && $("#city3").value.trim();
  if (useThird) names.push($("#city3").value.trim());
  return { start: $("#start-date").value, end: $("#end-date").value, names };
}

async function onSubmit(event) {
  event.preventDefault();
  const { start, end, names } = readForm();

  if (!start || !end) {
    setStatus([{ cls: "error", text: "Please pick a start and end date." }]);
    return;
  }
  if (start >= end) {
    setStatus([{ cls: "error", text: "End date must be after start date." }]);
    return;
  }
  if (names.some((n) => !n)) {
    setStatus([{ cls: "error", text: "Please enter a name for every city." }]);
    return;
  }

  $("#submit-btn").disabled = true;
  setStatus([{ cls: "loading", text: `Fetching weather for ${names.join(", ")}…` }]);
  ["#resolved1", "#resolved2", "#resolved3"].forEach((s) => { $(s).textContent = ""; $(s).classList.remove("error"); });

  const settled = await Promise.allSettled(names.map((n) => fetchCity(n, start, end)));

  const errors = [];
  const ok = [];
  settled.forEach((result, i) => {
    const target = $(`#resolved${i + 1}`);
    if (result.status === "fulfilled") {
      const d = result.value;
      ok.push(d);
      if (target) {
        target.classList.remove("error");
        target.textContent = `resolved as ${d.lat.toFixed(4)}, ${d.lon.toFixed(4)}`;
      }
    } else {
      errors.push(`${names[i]}: ${result.reason.message}`);
      if (target) {
        target.classList.add("error");
        target.textContent = result.reason.message;
      }
    }
  });

  $("#submit-btn").disabled = false;

  if (errors.length || ok.length < 2) {
    setStatus(errors.map((text) => ({ cls: "error", text })));
    return;
  }

  cityData = ok.map((d, i) => ({ ...d, color: PALETTE[i].line, fill: PALETTE[i].fill }));

  const warnings = [];
  cityData.forEach((c) => (c.warnings || []).forEach((w) => warnings.push(`${c.city}: ${w}`)));
  setStatus(warnings.map((text) => ({ cls: "info", text })));

  $("#welcome").hidden = true;
  $("#results").hidden = false;
  renderAll();
}

/* --------------------------------------------------------------- rendering */

function renderAll() {
  renderMap();
  renderPlots();
  setupAveragesTab();
  renderCurrent();
}

function renderMap() {
  const dark = isDark();
  Plotly.newPlot("map", [{
    type: "scattergeo",
    mode: "markers+text",
    lat: cityData.map((c) => c.lat),
    lon: cityData.map((c) => c.lon),
    text: cityData.map((c) => c.city),
    textposition: "top center",
    textfont: { color: dark ? "#c9ced6" : "#1a1d21" },
    marker: { size: 10, color: cityData.map((c) => c.color) },
    hoverinfo: "text",
  }], {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    geo: {
      scope: "world",
      projection: { type: "natural earth" },
      fitbounds: "locations",
      showland: true,
      landcolor: dark ? "#2a2f38" : "#e9ecef",
      showcountries: true,
      countrycolor: "rgba(128,128,128,0.4)",
      showcoastlines: true,
      coastlinecolor: "rgba(128,128,128,0.4)",
      bgcolor: "rgba(0,0,0,0)",
    },
  }, PLOT_CONFIG);
}

/* ---- Plots tab -------------------------------------------------------- */

function hasColumn(metric) {
  return cityData.some((c) => c.monthly.some((row) => row[metric] != null));
}

function plotMetric(container, { metric, p10, p90, title, unit, reference, referenceLabel }) {
  const traces = [];
  for (const c of cityData) {
    const rows = c.monthly.filter((r) => r[metric] != null);
    if (!rows.length) {
      note(container, `${title} data not available for ${c.city}.`);
      continue;
    }
    const x = rows.map((r) => r.time);
    const hasBand = p10 && p90 && rows.every((r) => r[p10] != null && r[p90] != null);
    if (hasBand) {
      traces.push({
        x, y: rows.map((r) => r[p90]), mode: "lines", line: { width: 0 },
        showlegend: false, hoverinfo: "skip",
      });
      traces.push({
        x, y: rows.map((r) => r[p10]), mode: "lines", fill: "tonexty",
        fillcolor: c.fill, line: { width: 0 }, name: `${c.city} 10th–90th pct`,
      });
    }
    traces.push({
      x, y: rows.map((r) => r[metric]), mode: "lines+markers",
      name: `${c.city} mean`, line: { color: c.color, width: 2 },
    });
    if (reference && rows.every((r) => r[reference] != null)) {
      traces.push({
        x, y: rows.map((r) => r[reference]), mode: "lines",
        name: `${c.city} ${referenceLabel}`,
        line: { color: c.color, width: 1, dash: "dot" },
      });
    }
  }
  if (!traces.length) {
    note(container, `No ${title.toLowerCase()} data available for the selected cities.`);
    return;
  }
  Plotly.newPlot(container, traces,
    baseLayout({ title: `${title} (10th–90th percentile range)`, xTitle: "Month", yTitle: unit }),
    PLOT_CONFIG);
}

function plotExtremes(container) {
  const traces = [];
  for (const c of cityData) {
    const rows = c.monthly.filter(
      (r) => r.temperature_2m_min_absolute != null && r.temperature_2m_max_absolute != null);
    if (!rows.length) {
      note(container, `Temperature extremes not available for ${c.city}.`);
      continue;
    }
    const x = rows.map((r) => r.time);
    traces.push({
      x, y: rows.map((r) => r.temperature_2m_min_absolute), mode: "lines+markers",
      name: `${c.city} coldest`, line: { color: c.color, dash: "dot" },
    });
    traces.push({
      x, y: rows.map((r) => r.temperature_2m_max_absolute), mode: "lines+markers",
      name: `${c.city} hottest`, line: { color: c.color, dash: "dash" },
    });
  }
  if (!traces.length) return;
  Plotly.newPlot(container, traces,
    baseLayout({ title: "Absolute temperature extremes by month", xTitle: "Month", yTitle: "°C" }),
    PLOT_CONFIG);
}

function renderRecords(parent) {
  const block = document.createElement("div");
  block.className = "chart-block";
  const h = document.createElement("h3");
  h.textContent = "🌡️ Record temperature extremes";
  block.appendChild(h);
  const cap = document.createElement("p");
  cap.className = "caption";
  cap.textContent = "The single hottest and coldest days (by daily average temperature) across the whole selected range.";
  block.appendChild(cap);

  const rows = cityData
    .filter((c) => c.records)
    .map((c) => ({ city: c.city, ...c.records }));

  if (!rows.length) {
    note(block, "No record temperature data available for the selected cities.");
    parent.appendChild(block);
    return;
  }

  const table = document.createElement("table");
  table.className = "data";
  table.innerHTML =
    "<tr><th>City</th><th>Record low (°C)</th><th>Coldest day</th>" +
    "<th>Record high (°C)</th><th>Hottest day</th></tr>" +
    rows.map((r) =>
      `<tr><td>${r.city}</td><td>${r.record_low}</td><td>${r.record_low_date}</td>` +
      `<td>${r.record_high}</td><td>${r.record_high_date}</td></tr>`).join("");
  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  scroll.appendChild(table);
  block.appendChild(scroll);

  const chart = document.createElement("div");
  chart.className = "chart";
  block.appendChild(chart);
  parent.appendChild(block);

  const traces = [];
  for (const r of rows) {
    traces.push({
      type: "bar", x: [r.city], y: [r.record_high], name: `${r.city} record high`,
      marker: { color: "red" }, text: [r.record_high_date], hovertemplate: "%{y} °C<br>%{text}<extra></extra>",
    });
    traces.push({
      type: "bar", x: [r.city], y: [r.record_low], name: `${r.city} record low`,
      marker: { color: "blue" }, text: [r.record_low_date], hovertemplate: "%{y} °C<br>%{text}<extra></extra>",
    });
  }
  Plotly.newPlot(chart, traces,
    baseLayout({ title: "All-time temperature records", xTitle: "City", yTitle: "°C"}),
    PLOT_CONFIG);
}

function renderPlots() {
  const root = $("#plots-charts");
  root.innerHTML = "";

  plotMetric(chartBlock(root, "🌡️ Monthly temperature (°C)",
    "Mean temperature for each month; the shaded band is the 10th–90th percentile of daily values."),
    { metric: "temperature_2m_mean", p10: "temperature_2m_mean_p10", p90: "temperature_2m_mean_p90",
      title: "Temperature", unit: "°C" });

  const daytimeBlock = chartBlock(root, "☀️ Monthly daytime temperature (°C)",
    "Only the hours between local sunrise and sunset. The dotted line is the all-day (24h) mean — the gap is how much the nights pull the average down.");
  if (hasColumn("daytime_temperature")) {
    plotMetric(daytimeBlock, {
      metric: "daytime_temperature", p10: "daytime_temperature_p10", p90: "daytime_temperature_p90",
      title: "Daytime temperature", unit: "°C",
      reference: "temperature_2m_mean", referenceLabel: "all-day mean",
    });
  } else {
    note(daytimeBlock, "Daytime temperature data not available for the selected cities.");
  }

  plotMetric(chartBlock(root, "💧 Average humidity (%)",
    "Mean relative humidity for each month, with the 10th–90th percentile band."),
    { metric: "relative_humidity_2m_mean", p10: "relative_humidity_2m_mean_p10",
      p90: "relative_humidity_2m_mean_p90", title: "Humidity", unit: "%" });

  plotMetric(chartBlock(root, "🌧️ Monthly precipitation (mm)",
    "Mean total precipitation per month; the band is the 10th–90th percentile of the monthly totals."),
    { metric: "precipitation_sum", p10: "precipitation_sum_p10", p90: "precipitation_sum_p90",
      title: "Precipitation", unit: "mm" });

  plotMetric(chartBlock(root, "🌞 Monthly sunshine hours",
    "Mean total sunshine hours per month, with the 10th–90th percentile band."),
    { metric: "sunshine_hours", p10: "sunshine_hours_p10", p90: "sunshine_hours_p90",
      title: "Sunshine", unit: "hours" });

  plotExtremes(chartBlock(root, "🌡️ Temperature extremes (°C)",
    "The absolute highest and lowest temperatures recorded in each calendar month over the selected period."));

  renderRecords(root);
}

/* ---- Monthly Averages tab ------------------------------------------- */

const PREDICTION_METRICS = [
  { label: "Temperature mean", unit: "°C", mean: "temperature_2m_mean",
    p10: "temperature_2m_mean_p10", p90: "temperature_2m_mean_p90" },
  { label: "Humidity mean", unit: "%", mean: "relative_humidity_2m_mean",
    p10: "relative_humidity_2m_mean_p10", p90: "relative_humidity_2m_mean_p90" },
  { label: "Precipitation (mm, avg monthly)", unit: "", mean: "precipitation_sum" },
  { label: "Sunshine (hours, avg monthly)", unit: "", mean: "sunshine_hours" },
];

function setupAveragesTab() {
  const input = $("#prediction-date");
  const today = new Date();
  const tomorrow = new Date(today.getTime() + 86400000);
  input.value = tomorrow.toISOString().slice(0, 10);
  input.onchange = renderAverages;
  renderAverages();
}

function renderAverages() {
  const value = $("#prediction-date").value;
  if (!value) return;
  const date = new Date(value + "T00:00:00");
  const month = date.getMonth() + 1;
  $("#averages-heading").textContent =
    `Prediction for ${date.toLocaleDateString("en-US", { month: "long", day: "numeric" })}`;

  const headCells = ["City"];
  for (const m of PREDICTION_METRICS) {
    headCells.push(m.unit ? `${m.label} (${m.unit})` : m.label);
    if (m.p10) headCells.push(m.unit ? `${m.label} 10th–90th (${m.unit})` : `${m.label} 10th–90th`);
  }

  const bodyRows = cityData.map((c) => {
    const row = c.monthly.find((r) => r.month === month);
    const cells = [c.city];
    for (const m of PREDICTION_METRICS) {
      const mean = row ? row[m.mean] : null;
      cells.push(mean != null ? mean.toFixed(1) : "No data");
      if (m.p10) {
        const lo = row ? row[m.p10] : null;
        const hi = row ? row[m.p90] : null;
        cells.push(lo != null && hi != null ? `${lo.toFixed(1)}–${hi.toFixed(1)}` : "No data");
      }
    }
    return cells;
  });

  const html =
    "<tr>" + headCells.map((h) => `<th>${h}</th>`).join("") + "</tr>" +
    bodyRows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("");
  $("#averages-table").innerHTML = `<div class="table-scroll"><table class="data">${html}</table></div>`;
}

/* ---- Comparison to this year tab ----------------------------------- */

function centeredRolling(values, window) {
  if (window <= 1) return values.slice();
  const n = values.length;
  const lead = Math.floor(window / 2);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - window + 1 + lead);
    const hi = Math.min(n - 1, i + lead);
    let sum = 0;
    let count = 0;
    for (let j = lo; j <= hi; j++) {
      if (values[j] == null || Number.isNaN(values[j])) continue;
      sum += values[j];
      count += 1;
    }
    out[i] = count ? sum / count : null;
  }
  return out;
}

const doyKey = (m, d) => m * 100 + d;
const parseISO = (s) => new Date(s + "T00:00:00");
const shortDate = (dt) => dt.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
const signed = (x, digits) => (x >= 0 ? "+" : "") + x.toFixed(digits);

function buildTempComparison(c) {
  const climo = new Map(c.currentYear.daytimeClimatology.map((r) => [doyKey(r.m, r.d), r.avg]));
  const merged = [];
  for (const point of c.currentYear.daytimeThisYear) {
    if (point.t == null) continue;
    const dt = parseISO(point.date);
    const avg = climo.get(doyKey(dt.getMonth() + 1, dt.getDate()));
    if (avg == null) continue;
    merged.push({ date: dt, actual: point.t, average: avg });
  }
  merged.sort((a, b) => a.date - b.date);
  return merged;
}

function tempSummaryRow(city, merged) {
  const anomalies = merged.map((r) => r.actual - r.average);
  let maxI = 0;
  let minI = 0;
  anomalies.forEach((a, i) => {
    if (a > anomalies[maxI]) maxI = i;
    if (a < anomalies[minI]) minI = i;
  });
  const mean = anomalies.reduce((s, a) => s + a, 0) / anomalies.length;
  return [
    city,
    String(merged.length),
    String(anomalies.filter((a) => a > 0).length),
    String(anomalies.filter((a) => a < 0).length),
    signed(mean, 2),
    `${shortDate(merged[maxI].date)} (${signed(anomalies[maxI], 1)} °C)`,
    `${shortDate(merged[minI].date)} (${signed(anomalies[minI], 1)} °C)`,
  ];
}

function plotTempComparison(container, city, merged, baselineLabel, year, smoothing) {
  const dates = merged.map((r) => r.date);
  const actual = centeredRolling(merged.map((r) => r.actual), smoothing);
  const average = centeredRolling(merged.map((r) => r.average), smoothing);
  const above = actual.map((v, i) => (v > average[i] ? v : average[i]));
  const below = actual.map((v, i) => (v < average[i] ? v : average[i]));

  const hidden = { x: dates, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" };
  const traces = [
    { ...hidden, y: average },
    { x: dates, y: above, mode: "lines", fill: "tonexty", fillcolor: ANOMALY.warm,
      line: { width: 0 }, name: "Warmer than average", hoverinfo: "skip" },
    { ...hidden, y: average },
    { x: dates, y: below, mode: "lines", fill: "tonexty", fillcolor: ANOMALY.cool,
      line: { width: 0 }, name: "Cooler than average", hoverinfo: "skip" },
    { x: dates, y: average, mode: "lines", name: `${baselineLabel} average`,
      line: { color: "gray", width: 2, dash: "dash" } },
    { x: dates, y: actual, mode: "lines", name: String(year),
      line: { color: ink(), width: 1.5 } },
  ];
  Plotly.newPlot(container, traces,
    baseLayout({ title: `${city} — daytime temperature vs average`, xTitle: "Date", yTitle: "°C",
                 hovermode: "x unified" }),
    PLOT_CONFIG);
}

function buildPrecipComparison(c) {
  const climo = new Map(c.currentYear.precipClimatology.map((r) => [doyKey(r.m, r.d), r.avg]));
  const merged = [];
  for (const point of c.currentYear.precipThisYear) {
    if (point.acc == null) continue;
    const dt = parseISO(point.date);
    const avg = climo.get(doyKey(dt.getMonth() + 1, dt.getDate()));
    if (avg == null) continue;
    merged.push({ date: dt, actual: point.acc, average: avg });
  }
  merged.sort((a, b) => a.date - b.date);
  return merged;
}

function renderCurrent() {
  const root = $("#current-content");
  root.innerHTML = "";
  const year = cityData[0].currentYear.year;
  $('.tab[data-tab="current"]').textContent = `🌡️ Comparison to ${year}`;

  const baselineLabel = cityData[0].currentYear.baselineLabel;
  const hasBaselineRange = cityData.some((c) => c.currentYear.hasBaselineRange);

  /* --- daytime temperature anomaly --- */
  const tempBlock = document.createElement("div");
  tempBlock.className = "chart-block";
  tempBlock.innerHTML =
    `<h3>☀️ Daytime temperature: ${year} so far vs the ${baselineLabel} average</h3>` +
    `<p class="caption">Each day of ${year} against the average daytime temperature for that ` +
    `calendar day across ${baselineLabel}. Red = warmer than usual, blue = cooler. Daylight hours only.</p>`;
  root.appendChild(tempBlock);

  const comparable = cityData.filter(
    (c) => c.currentYear.daytimeThisYear && c.currentYear.daytimeClimatology);

  if (!comparable.length) {
    note(tempBlock, hasBaselineRange
      ? `No ${year} daytime data available yet for the selected cities.`
      : `The selected range leaves no earlier years to compare ${year} against. Pick a start date before ${year}.`);
  } else {
    const sliderRow = document.createElement("div");
    sliderRow.className = "slider-row";
    sliderRow.innerHTML =
      `<label for="smoothing">Smoothing (days)</label>` +
      `<input type="range" id="smoothing" min="1" max="31" value="1" />` +
      `<output id="smoothing-out">1</output>`;
    tempBlock.appendChild(sliderRow);

    const summaryHost = document.createElement("div");
    tempBlock.appendChild(summaryHost);
    const chartHost = document.createElement("div");
    tempBlock.appendChild(chartHost);

    const merges = comparable.map((c) => ({ c, merged: buildTempComparison(c) }))
      .filter((x) => x.merged.length);

    const rows = merges.map(({ c, merged }) => tempSummaryRow(c.city, merged));
    if (rows.length) {
      const head = ["City", "Days compared", "Warmer than average", "Cooler than average",
                    "Mean difference (°C)", "Biggest warm day", "Biggest cool day"];
      summaryHost.innerHTML =
        `<div class="table-scroll"><table class="data"><tr>` +
        head.map((h) => `<th>${h}</th>`).join("") + "</tr>" +
        rows.map((r) => "<tr>" + r.map((v) => `<td>${v}</td>`).join("") + "</tr>").join("") +
        "</table></div>";
    }

    const draw = () => {
      const smoothing = Number($("#smoothing").value);
      $("#smoothing-out").textContent = String(smoothing);
      chartHost.innerHTML = "";
      for (const { c, merged } of merges) {
        const div = document.createElement("div");
        div.className = "chart";
        chartHost.appendChild(div);
        plotTempComparison(div, c.city, merged, baselineLabel, year, smoothing);
      }
    };
    $("#smoothing").oninput = draw;
    draw();
  }

  /* --- accumulated precipitation --- */
  const precipBlock = document.createElement("div");
  precipBlock.className = "chart-block";
  precipBlock.innerHTML =
    `<h3>🌧️ Accumulated precipitation: ${year} so far vs the ${baselineLabel} average</h3>` +
    `<p class="caption">Rainfall added up from 1 January. The dashed line is how much a typical ` +
    `${baselineLabel} year had by the same date, so the gap is the running surplus or shortfall.</p>`;
  root.appendChild(precipBlock);

  const precipComparable = cityData.filter(
    (c) => c.currentYear.precipThisYear && c.currentYear.precipClimatology);

  if (!precipComparable.length) {
    note(precipBlock, hasBaselineRange
      ? `No ${year} precipitation data available yet for the selected cities.`
      : `No earlier years in the selected range to compare ${year} rainfall against.`);
    return;
  }

  const precipMerges = precipComparable.map((c) => ({ c, merged: buildPrecipComparison(c) }))
    .filter((x) => x.merged.length);

  const precipRows = precipMerges.map(({ c, merged }) => {
    const latest = merged[merged.length - 1];
    const share = latest.average ? `${Math.round((latest.actual / latest.average) * 100)}%` : "n/a";
    return [
      c.city,
      latest.actual.toFixed(0),
      latest.average.toFixed(0),
      signed(latest.actual - latest.average, 0),
      share,
    ];
  });
  if (precipRows.length) {
    const head = ["City", `${year} so far (mm)`, "Typical by this date (mm)",
                  "Difference (mm)", "Share of typical"];
    const host = document.createElement("div");
    host.innerHTML =
      `<div class="table-scroll"><table class="data"><tr>` +
      head.map((h) => `<th>${h}</th>`).join("") + "</tr>" +
      precipRows.map((r) => "<tr>" + r.map((v) => `<td>${v}</td>`).join("") + "</tr>").join("") +
      "</table></div>";
    precipBlock.appendChild(host);
  }

  const chart = document.createElement("div");
  chart.className = "chart";
  chart.style.height = "450px";
  precipBlock.appendChild(chart);

  const traces = [];
  for (const { c, merged } of precipMerges) {
    const dates = merged.map((r) => r.date);
    traces.push({
      x: dates, y: merged.map((r) => r.average), mode: "lines",
      name: `${c.city} typical`, line: { color: c.color, width: 1.5, dash: "dash" },
    });
    traces.push({
      x: dates, y: merged.map((r) => r.actual), mode: "lines",
      name: `${c.city} ${year}`, line: { color: c.color, width: 2.5 },
    });
  }
  Plotly.newPlot(chart, traces,
    baseLayout({ title: "Accumulated precipitation since 1 January", xTitle: "Date", yTitle: "mm",
                 height: 450, hovermode: "x unified" }),
    PLOT_CONFIG);
}

/* --------------------------------------------------------------- wiring up */

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      document.querySelectorAll(".tab-panel").forEach((p) =>
        p.classList.toggle("is-active", p.dataset.panel === name));
      // Plotly needs a resize nudge for charts drawn while their panel was hidden.
      window.dispatchEvent(new Event("resize"));
    });
  });
}

$("#add-third").addEventListener("change", (e) => {
  $("#city3-wrap").hidden = !e.target.checked;
});
$("#settings-form").addEventListener("submit", onSubmit);
$("#end-date").max = new Date().toISOString().slice(0, 10);
setupTabs();
