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
    table.innerHTML = ml.compScores
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
      data.comparables?.recentSales,
      "closedDate"
    );
  }

  const listingTable = document.getElementById("listing-table");
  if (listingTable) {
    listingTable.innerHTML = buildTableRows(
      data.comparables?.currentListings,
      "listedDate"
    );
  }

  renderSources(data.sources);
  renderMlDetails(data.valuation?.mlDetails);

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

(async function init() {
  setupRefreshButton();
  try {
    await loadData(false);
  } catch (err) {
    const status = document.getElementById("hv-refresh-status");
    if (status) status.textContent = "Load error: " + err.message;
  }

  setInterval(() => {
    loadData(false).catch(() => {});
  }, 5 * 60 * 1000);
})();
