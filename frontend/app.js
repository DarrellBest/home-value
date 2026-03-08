"use strict";

function formatMoney(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "$-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);
}

function formatPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(2) + "%";
}

function formatDate(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toLocaleDateString();
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function buildTableRows(rows, dateKey) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return '<tr><td colspan="4">No data</td></tr>';
  }
  return rows
    .map(
      (row) =>
        `<tr>
          <td>${row.address || "-"}</td>
          <td>${formatMoney(row.price)}</td>
          <td>${formatDate(row[dateKey])}</td>
          <td>${row.source || "-"}</td>
        </tr>`
    )
    .join("");
}

function renderMlDetails(ml) {
  if (!ml) return;

  if (ml.intervals) {
    const p90 = ml.intervals.prediction90;
    const c80 = ml.intervals.confidence80;
    if (p90)
      setText(
        "pred-interval-90",
        formatMoney(p90.low) + " - " + formatMoney(p90.high)
      );
    if (c80)
      setText(
        "conf-interval-80",
        formatMoney(c80.low) + " - " + formatMoney(c80.high)
      );
  }

  if (ml.stats) {
    setText("ml-stddev", formatMoney(ml.stats.weightedStdDev));
    setText(
      "ml-ess",
      ml.stats.effectiveSampleSize != null
        ? ml.stats.effectiveSampleSize.toFixed(2)
        : "-"
    );
    setText(
      "ml-cv",
      ml.stats.coeffOfVariation != null
        ? ml.stats.coeffOfVariation.toFixed(4)
        : "-"
    );
    setText(
      "ml-seasonal",
      ml.stats.seasonalFactor != null
        ? ml.stats.seasonalFactor.toFixed(3)
        : "-"
    );
  }

  const table = document.getElementById("comp-scores-table");
  if (table && Array.isArray(ml.compScores)) {
    table.innerHTML = ml.compScores.slice(0, 10)
      .map((c) => {
        const simPct = (c.similarityScore * 100).toFixed(1);
        const barWidth = Math.round(c.similarityScore * 100);
        return `<tr>
            <td>${c.address || "-"}</td>
            <td>${formatMoney(c.rawPrice)}</td>
            <td>${formatMoney(c.adjustedPrice)}</td>
            <td>
              <div class="sim-bar-wrap">
                <div class="sim-bar" style="width:${barWidth}%"></div>
                <span class="sim-label">${simPct}%</span>
              </div>
            </td>
            <td>${c.isListing ? "Listing" : "Sale"}</td>
          </tr>`;
      })
      .join("");
  }
}

function renderSources(sources) {
  const list = document.getElementById("source-list");
  if (!list) return;

  if (!Array.isArray(sources) || sources.length === 0) {
    list.innerHTML =
      '<div class="source-item"><span>No source telemetry</span></div>';
    return;
  }

  list.innerHTML = sources
    .map((src) => {
      const ok = !!src.ok;
      const statusLabel = ok ? "LIVE" : "FALLBACK";
      const statusClass = ok ? "ok" : "bad";
      const valueLabel = Number.isFinite(src.extractedValue)
        ? formatMoney(src.extractedValue)
        : "-";
      const reason = ok
        ? "Signals: " + (src.signalCount || 0)
        : src.error || "Unavailable";

      return `<div class="source-item">
          <div>
            <div>${src.label}</div>
            <div class="source-meta">${reason}</div>
          </div>
          <div style="text-align: right;">
            <div class="source-status ${statusClass}">${statusLabel}</div>
            <div class="source-meta">${valueLabel}</div>
          </div>
        </div>`;
    })
    .join("");
}

let kfoldChart = null;
let residualChart = null;

function renderComponentMetrics(cm) {
  if (!cm) return;

  function setComp(prefix, data) {
    if (!data) return;
    setText(prefix + "-mape", data.mape != null ? data.mape.toFixed(1) + "%" : "-");
    setText(prefix + "-rmse", data.rmse != null ? formatMoney(data.rmse) : "-");
    setText(prefix + "-mae", data.mae != null ? formatMoney(data.mae) : "-");
    setText(prefix + "-r2", data.r2 != null ? data.r2.toFixed(4) : "-");
    setText(prefix + "-n", data.n_evaluated != null ? data.n_evaluated + " sales" : "-");

    // Color-code MAPE: green <5%, yellow 5-10%, red >10%
    var el = document.getElementById(prefix + "-mape");
    if (el && data.mape != null) {
      if (data.mape < 5) el.style.color = "#00ff88";
      else if (data.mape < 10) el.style.color = "#ffaa00";
      else el.style.color = "#ff5555";
    }
  }

  setComp("cm-comp", cm.comp);
  setComp("cm-cat", cm.catboost);
  setComp("cm-blend", cm.blend);
}

function renderTrainingStats(ts, validationDetails) {
  if (!ts) return;

  renderComponentMetrics(ts.componentMetrics);

  setText("ts-sample-count", ts.trainingDataCount != null ? ts.trainingDataCount + " sales" : "-");
  setText("ts-rmse", ts.testAccuracy?.rmse != null ? "\u00B1" + formatMoney(ts.testAccuracy.rmse) : "-");
  setText("ts-mae", ts.testAccuracy?.mae != null ? formatMoney(ts.testAccuracy.mae) : "-");
  setText("ts-r2", ts.r2Score != null ? ts.r2Score.toFixed(4) : "-");
  setText("ts-cv-folds", ts.cvFolds != null ? ts.cvFolds + " folds" : "-");
  setText("ts-avg-sim", ts.avgSimilarityScore != null ? (ts.avgSimilarityScore * 100).toFixed(1) + "%" : "-");
  setText("ts-mape", ts.mape != null ? ts.mape.toFixed(1) + "%" : "-");
  setText("ts-median-ae", ts.medianAE != null ? formatMoney(ts.medianAE) : "-");

  const ra = ts.residualAnalysis;
  setText("ts-coverage", ra?.predictionIntervalCoverage != null ? ra.predictionIntervalCoverage.toFixed(1) + "%" : "-");

  if (ts.dataFreshness) {
    const oldest = ts.dataFreshness.oldest || "?";
    const newest = ts.dataFreshness.newest || "?";
    setText("ts-freshness", oldest + " \u2192 " + newest);
  }

  setText("ts-last-trained", formatDate(ts.lastTrainedAt));

  // Source breakdown tags
  const breakdownEl = document.getElementById("ts-source-breakdown");
  if (breakdownEl && ts.dataSourceBreakdown) {
    const entries = Object.entries(ts.dataSourceBreakdown);
    breakdownEl.innerHTML = entries.map(function (entry) {
      return '<span class="ml-source-tag"><span class="tag-label">' + entry[0] + '</span><span class="tag-count">' + entry[1] + '</span></span>';
    }).join("");
  }

  // Feature importance bars
  const fiEl = document.getElementById("ts-feature-importance");
  if (fiEl && ts.featureImportance) {
    const entries = Object.entries(ts.featureImportance);
    const maxVal = Math.max.apply(null, entries.map(function (e) { return e[1]; }));
    fiEl.innerHTML = entries.map(function (entry) {
      var pct = maxVal > 0 ? Math.round((entry[1] / maxVal) * 100) : 0;
      return '<div class="fi-row">' +
        '<span class="fi-label">' + entry[0] + '</span>' +
        '<div class="fi-bar-wrap"><div class="fi-bar" style="width:' + pct + '%"></div></div>' +
        '<span class="fi-value">' + entry[1].toFixed(1) + '%</span>' +
        '</div>';
    }).join("");
  }

  renderKFoldCV(ts.kFoldCV, ts.overfittingWarning);
  renderTrainTestSplit(ts.trainTestSplit);
  renderResidualAnalysis(ts.residualAnalysis, validationDetails || []);
}

function renderKFoldCV(kf, overfitWarning) {
  if (!kf || !kf.nFolds) return;

  setText("kf-nfolds", kf.nFolds + "-fold");
  setText("kf-mean-rmse", kf.mean?.rmse != null ? formatMoney(kf.mean.rmse) + " \u00B1" + formatMoney(kf.std?.rmse || 0) : "-");
  setText("kf-mean-mae", kf.mean?.mae != null ? formatMoney(kf.mean.mae) + " \u00B1" + formatMoney(kf.std?.mae || 0) : "-");
  setText("kf-mean-r2", kf.mean?.r2 != null ? kf.mean.r2.toFixed(2) + " \u00B1" + (kf.std?.r2 || 0).toFixed(2) : "-");
  setText("kf-mean-mape", kf.mean?.median_ape != null ? kf.mean.median_ape.toFixed(1) + "%" : "-");

  const overfitEl = document.getElementById("kf-overfit");
  if (overfitEl) {
    if (overfitWarning === "high") {
      overfitEl.textContent = "HIGH";
      overfitEl.style.color = "#ff5555";
    } else if (overfitWarning === "moderate") {
      overfitEl.textContent = "MODERATE";
      overfitEl.style.color = "#ffaa00";
    } else {
      overfitEl.textContent = "LOW";
      overfitEl.style.color = "#00ff88";
    }
  }

  // K-fold bar chart
  const canvas = document.getElementById("kfold-chart");
  if (!canvas || typeof Chart === "undefined" || !kf.folds || kf.folds.length === 0) return;

  if (kfoldChart) kfoldChart.destroy();

  const labels = kf.folds.map(function (_, i) { return "Fold " + (i + 1); });
  const rmseData = kf.folds.map(function (f) { return f.rmse; });
  const maeData = kf.folds.map(function (f) { return f.mae; });

  kfoldChart = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "RMSE",
          data: rmseData,
          backgroundColor: "rgba(170, 136, 255, 0.5)",
          borderColor: "#aa88ff",
          borderWidth: 1,
        },
        {
          label: "MAE",
          data: maeData,
          backgroundColor: "rgba(0, 204, 255, 0.5)",
          borderColor: "#00ccff",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#9a6e42", font: { family: "Share Tech Mono", size: 10 } },
        },
        tooltip: {
          callbacks: {
            label: function (ctx) { return ctx.dataset.label + ": " + formatMoney(ctx.parsed.y); },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#9a6e42", font: { family: "Share Tech Mono", size: 10 } },
          grid: { color: "rgba(170,136,255,0.08)" },
        },
        y: {
          ticks: {
            color: "#9a6e42",
            font: { family: "Share Tech Mono", size: 10 },
            callback: function (v) { return "$" + (v / 1000).toFixed(0) + "k"; },
          },
          grid: { color: "rgba(170,136,255,0.08)" },
        },
      },
    },
  });
}

function renderTrainTestSplit(tt) {
  if (!tt) return;

  setText("tt-train-size", tt.trainSize != null ? tt.trainSize + " samples" : "-");
  setText("tt-test-size", tt.testSize != null ? tt.testSize + " samples" : "-");

  const tr = tt.trainMetrics || {};
  const te = tt.testMetrics || {};
  setText("tt-train-rmse", tr.rmse != null ? formatMoney(tr.rmse) : "-");
  setText("tt-train-mae", tr.mae != null ? formatMoney(tr.mae) : "-");
  setText("tt-train-r2", tr.r2 != null ? tr.r2.toFixed(4) : "-");
  setText("tt-train-mape", tr.mape != null ? tr.mape.toFixed(1) + "%" : "-");
  setText("tt-test-rmse", te.rmse != null ? formatMoney(te.rmse) : "-");
  setText("tt-test-mae", te.mae != null ? formatMoney(te.mae) : "-");
  setText("tt-test-r2", te.r2 != null ? te.r2.toFixed(4) : "-");
  setText("tt-test-mape", te.mape != null ? te.mape.toFixed(1) + "%" : "-");

  // Generalization gap indicator
  const gapPct = tt.generalizationGapPct;
  setText("tt-gen-gap", tt.generalizationGap != null ? formatMoney(tt.generalizationGap) : "-");

  const fill = document.getElementById("tt-gen-gap-fill");
  const label = document.getElementById("tt-gen-gap-label");
  if (fill && label && gapPct != null) {
    const clampedPct = Math.min(gapPct, 40);
    fill.style.width = (clampedPct / 40 * 100) + "%";
    label.textContent = gapPct.toFixed(1) + "%";
    if (gapPct < 10) {
      fill.style.background = "#00ff88";
      label.style.color = "#00ff88";
    } else if (gapPct < 20) {
      fill.style.background = "#ffaa00";
      label.style.color = "#ffaa00";
    } else {
      fill.style.background = "#ff5555";
      label.style.color = "#ff5555";
    }
  }
}

function renderResidualAnalysis(ra, validationDetails) {
  if (!ra) return;

  const biasEl = document.getElementById("ra-bias");
  if (biasEl && ra.bias != null) {
    // ra.bias is a number (dollar amount), determine if overestimate or underestimate
    const biasType = ra.bias > 1000 ? "overestimate" : ra.bias < -1000 ? "underestimate" : "neutral";
    biasEl.textContent = biasType.toUpperCase() + " (" + formatMoney(ra.bias) + ")";
    biasEl.style.color = biasType === "overestimate" ? "#ffaa00" : biasType === "underestimate" ? "#00ccff" : "#00ff88";
  }
  setText("ra-mean", ra.bias != null ? formatMoney(ra.bias) : "-");
  setText("ra-std", ra.std != null ? formatMoney(ra.std) : "-");
  setText("ra-outliers", ra.outlierCount != null ? ra.outlierCount + " flagged" : "-");
  setText("ra-coverage", ra.predictionIntervalCoverage != null ? ra.predictionIntervalCoverage.toFixed(1) + "%" : "-");

  // Residual scatter chart: predicted vs actual
  const canvas = document.getElementById("residual-chart");
  if (!canvas || typeof Chart === "undefined" || !validationDetails || validationDetails.length === 0) return;

  if (residualChart) residualChart.destroy();

  // Convert validation details to residuals format
  const outlierAddresses = (ra.outliers || []).map(function(o) { return o.address; });
  const residuals = validationDetails.map(function(v) {
    return {
      actual: v.actual,
      predicted: v.predicted,
      isOutlier: outlierAddresses.includes(v.address)
    };
  });

  const normal = residuals.filter(function (r) { return !r.isOutlier; });
  const outliers = residuals.filter(function (r) { return r.isOutlier; });

  // Perfect prediction line
  const allActuals = residuals.map(function (r) { return r.actual; });
  const minPrice = Math.min.apply(null, allActuals);
  const maxPrice = Math.max.apply(null, allActuals);

  residualChart = new Chart(canvas.getContext("2d"), {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Perfect Prediction",
          data: [{ x: minPrice, y: minPrice }, { x: maxPrice, y: maxPrice }],
          type: "line",
          borderColor: "rgba(255, 153, 0, 0.4)",
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Predictions",
          data: normal.map(function (r) { return { x: r.actual, y: r.predicted }; }),
          backgroundColor: "rgba(0, 204, 255, 0.6)",
          borderColor: "#00ccff",
          pointRadius: 4,
        },
        {
          label: "Outliers",
          data: outliers.map(function (r) { return { x: r.actual, y: r.predicted }; }),
          backgroundColor: "rgba(255, 85, 85, 0.7)",
          borderColor: "#ff5555",
          pointRadius: 6,
          pointStyle: "triangle",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#9a6e42", font: { family: "Share Tech Mono", size: 10 } },
        },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              if (ctx.dataset.label === "Perfect Prediction") return null;
              return "Actual: " + formatMoney(ctx.parsed.x) + " | Predicted: " + formatMoney(ctx.parsed.y);
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Actual Price", color: "#9a6e42", font: { family: "Share Tech Mono", size: 10 } },
          ticks: {
            color: "#9a6e42",
            font: { family: "Share Tech Mono", size: 10 },
            callback: function (v) { return "$" + (v / 1000).toFixed(0) + "k"; },
          },
          grid: { color: "rgba(255,153,0,0.08)" },
        },
        y: {
          title: { display: true, text: "Predicted Price", color: "#9a6e42", font: { family: "Share Tech Mono", size: 10 } },
          ticks: {
            color: "#9a6e42",
            font: { family: "Share Tech Mono", size: 10 },
            callback: function (v) { return "$" + (v / 1000).toFixed(0) + "k"; },
          },
          grid: { color: "rgba(255,153,0,0.08)" },
        },
      },
    },
  });
}

function render(data) {
  setText("estimate-value", formatMoney(data.valuation?.estimate));
  setText(
    "estimate-range",
    "Range: " +
      formatMoney(data.valuation?.confidenceRange?.low) +
      " - " +
      formatMoney(data.valuation?.confidenceRange?.high)
  );
  setText("confidence-pct", formatPct(data.valuation?.confidencePct));
  setText("valuation-method", data.valuation?.methodology || "-");
  setText("last-refresh", formatDate(data.valuation?.lastRefreshedAt));

  setText(
    "current-balance",
    formatMoney(data.financials?.ltv?.currentBalance)
  );
  setText("ltv-ratio", formatPct(data.financials?.ltv?.ratioPct));
  setText("pmi-target", formatMoney(data.financials?.pmi?.thresholdHomeValue));
  setText("pmi-gap", formatMoney(data.financials?.pmi?.deltaToThreshold));
  setText("property-address", data.property?.address || "-");
  setText(
    "property-sqft",
    data.property?.squareFeet?.toLocaleString?.() || "-"
  );
  setText("property-year", data.property?.yearBuilt || "-");
  setText("market-median", formatMoney(data.market?.medianPrice));
  setText("market-yoy", formatPct(data.market?.yoyChangePct));

  const salesTable = document.getElementById("sales-table");
  if (salesTable) {
    salesTable.innerHTML = buildTableRows(
      data.comparables?.recentSales?.slice(0, 10),
      "closedDate"
    );
  }

  const listingTable = document.getElementById("listing-table");
  if (listingTable) {
    listingTable.innerHTML = buildTableRows(
      data.comparables?.currentListings?.slice(0, 10),
      "listedDate"
    );
  }

  const estimate = data.valuation?.estimate;
  const balance = data.financials?.ltv?.currentBalance;
  const equity = (estimate && balance) ? estimate - balance : null;
  setText("equity-value", equity != null ? formatMoney(equity) : "$-");

  renderSources(data.sources);
  renderMlDetails(data.valuation?.mlDetails);
  renderTrainingStats(data.trainingStats, data.valuation?.mlDetails?.validation?.details || []);

  const status = document.getElementById("hv-refresh-status");
  if (status) {
    status.textContent =
      "Updated " +
      new Date(data.meta?.generatedAt || Date.now()).toLocaleTimeString();
  }
}

async function loadData(force) {
  const status = document.getElementById("hv-refresh-status");
  if (status) status.textContent = force ? "Refreshing..." : "Loading...";

  const endpoint = force ? "/api/valuation?refresh=1" : "/api/valuation";
  const response = await fetch(endpoint);
  if (!response.ok) {
    throw new Error("HTTP " + response.status);
  }

  const data = await response.json();
  render(data);
}

function setupRefreshButton() {
  const btn = document.getElementById("manual-refresh-btn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await fetch("/api/refresh", { method: "POST" });
      await loadData(true);
    } catch (err) {
      const status = document.getElementById("hv-refresh-status");
      if (status) status.textContent = "Refresh error: " + err.message;
    } finally {
      btn.disabled = false;
    }
  });
}

let historyChart = null;

async function loadHistory() {
  try {
    const resp = await fetch("/api/valuation/history?weeks=52");
    if (!resp.ok) return;
    const data = await resp.json();
    renderHistoryChart(data);
  } catch (err) {
    console.warn("History load failed:", err);
  }
}

function renderHistoryChart(data) {
  const canvas = document.getElementById("history-chart");
  if (!canvas || typeof Chart === "undefined") return;

  const history = data.history || [];
  if (history.length === 0) {
    canvas.parentElement.innerHTML =
      '<div style="text-align:center;padding:40px;color:var(--hv-muted)">No historical data yet. Valuations will appear after weekly updates.</div>';
    return;
  }

  const labels = history.map((h) => {
    const d = new Date(h.created_at);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  });
  const estimates = history.map((h) => h.estimate);
  const lows = history.map((h) => h.confidence_low);
  const highs = history.map((h) => h.confidence_high);
  const ltvLine = history.map(() => data.ltvThreshold);

  if (historyChart) historyChart.destroy();

  const ctx = canvas.getContext("2d");
  historyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Confidence High",
          data: highs,
          borderColor: "transparent",
          backgroundColor: "rgba(170, 136, 255, 0.15)",
          fill: "+1",
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: "Confidence Low",
          data: lows,
          borderColor: "transparent",
          backgroundColor: "rgba(170, 136, 255, 0.15)",
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: "Estimate",
          data: estimates,
          borderColor: "#ff9900",
          backgroundColor: "rgba(255, 153, 0, 0.1)",
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: "#ff9900",
          tension: 0.3,
          fill: false,
        },
        {
          label: "80% LTV Threshold",
          data: ltvLine,
          borderColor: "#ff5555",
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              if (ctx.dataset.label === "Confidence High" || ctx.dataset.label === "Confidence Low") return null;
              return ctx.dataset.label + ": " + formatMoney(ctx.parsed.y);
            },
          },
          filter: function (item) {
            return item.dataset.label !== "Confidence High" && item.dataset.label !== "Confidence Low";
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#9a6e42", font: { family: "Share Tech Mono", size: 10 } },
          grid: { color: "rgba(255,153,0,0.08)" },
        },
        y: {
          ticks: {
            color: "#9a6e42",
            font: { family: "Share Tech Mono", size: 10 },
            callback: (v) => "$" + (v / 1000).toFixed(0) + "k",
          },
          grid: { color: "rgba(255,153,0,0.08)" },
        },
      },
    },
  });
}

(async function init() {
  setupRefreshButton();
  try {
    await Promise.all([loadData(false), loadHistory()]);
  } catch (err) {
    const status = document.getElementById("hv-refresh-status");
    if (status) status.textContent = "Load error: " + err.message;
  }

  setInterval(() => {
    loadData(false).catch(() => {});
  }, 5 * 60 * 1000);

  setInterval(() => {
    loadHistory().catch(() => {});
  }, 30 * 60 * 1000);
})();
