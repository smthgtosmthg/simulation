/* ══════════════════════════════════════════════════════════════
   Active Inference Dashboard — Visualization
   Polls /api/state, /api/history, /api/qr_state
   Renders belief map + charts + camera feeds + QR detection
   ══════════════════════════════════════════════════════════════ */

const POLL_MS = 500;
const DRONE_COLORS = ['#22d3ee', '#34d399', '#a78bfa', '#fbbf24', '#f472b6', '#fb923c'];
let MAP_CELL_PX = 10; // pixels per grid cell on the map canvas (recalculated dynamically)

let paused = false;
let lastStep = -1;
let lastQrTimestamp = 0;

// Camera tab state per drone: 'annotated' | 'raw'
const camTabState = {};

// ── Chart.js instances ──
let chartEntropy, chartCoverage, chartFE, chartIG, chartInnovation, chartQrRate;

// ════════════════════════════════════════════════════
// Initialization
// ════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    document.getElementById('btnPause').addEventListener('click', togglePause);
    poll();                     // first fetch immediately
    setInterval(poll, POLL_MS); // then periodic
});

function togglePause() {
    paused = !paused;
    document.getElementById('btnPause').textContent = paused ? '▶' : '⏸';
}

// ════════════════════════════════════════════════════
// Polling
// ════════════════════════════════════════════════════

async function poll() {
    if (paused) return;
    try {
        const [stateRes, histRes, qrRes, ns3Res] = await Promise.all([
            fetch('/api/state'),
            fetch('/api/history'),
            fetch('/api/qr_state'),
            fetch('/api/ns3'),
        ]);
        if (!stateRes.ok || !histRes.ok) { setOffline(); return; }
        const state = await stateRes.json();
        const history = await histRes.json();
        const qrState = qrRes.ok ? await qrRes.json() : null;
        const ns3Data = ns3Res && ns3Res.ok ? await ns3Res.json() : null;

        const isNew = state.step !== lastStep;
        lastStep = state.step;

        if (isNew) {
            setOnline();
            renderMap(state);
            updateKPIs(state);
            updateBadges(state);
            updateDroneTable(state, qrState);
            updateCharts(history, state);
            updateResilienceEvents(state);
            // ── New tabs (rendent même si invisibles, négligeable en CPU) ──
            updateResilienceTab(state, history);
            updateNs3Tab(state, ns3Data);
        }

        // QR state updates independently (from decoder thread)
        if (qrState && qrState.timestamp !== lastQrTimestamp) {
            lastQrTimestamp = qrState.timestamp;
            updateQrKPIs(qrState);
            updateCameraFeeds(qrState, state);
            updateQrChart(qrState);
        }
    } catch {
        setOffline();
    }
}

function setOnline() {
    document.querySelector('.pulse').classList.add('live');
    document.getElementById('statusText').textContent = 'Live';
}
function setOffline() {
    document.querySelector('.pulse').classList.remove('live');
    document.getElementById('statusText').textContent = 'Offline';
}

// ════════════════════════════════════════════════════
// Belief Map Canvas
// ════════════════════════════════════════════════════

function beliefToRGB(p) {
    // 0 (free) → green, 0.5 (unknown) → dark slate, 1 (occupied) → red
    let r, g, b;
    if (p <= 0.5) {
        const t = p / 0.5;
        r = lerp(16, 15, t);
        g = lerp(185, 23, t);
        b = lerp(129, 42, t);
    } else {
        const t = (p - 0.5) / 0.5;
        r = lerp(15, 239, t);
        g = lerp(23, 68, t);
        b = lerp(42, 68, t);
    }
    return [Math.round(r), Math.round(g), Math.round(b)];
}

function lerp(a, b, t) { return a + (b - a) * t; }

function renderMap(state) {
    const canvas = document.getElementById('canvasMap');
    const ctx = canvas.getContext('2d');
    const env = state.environment;
    const belief = state.fused_belief;

    if (!belief || belief.length === 0) return;

    const gw = env.grid_width;
    const gh = env.grid_height;
    const res = env.grid_resolution;

    // ── Effective bounds: crop to detected walls (real environment) ──
    const eb = env.effective_bounds;
    const pad = 2; // grid-cell padding around walls
    let cx1 = 0, cy1 = 0, cx2 = gw, cy2 = gh;
    if (eb && eb.x2 > eb.x1 && eb.y2 > eb.y1) {
        cx1 = Math.max(0, eb.x1 - pad);
        cy1 = Math.max(0, eb.y1 - pad);
        cx2 = Math.min(gw, eb.x2 + pad);
        cy2 = Math.min(gh, eb.y2 + pad);
    }
    const cropW = cx2 - cx1;
    const cropH = cy2 - cy1;

    // Dynamic cell size: fit container while keeping aspect ratio
    const wrap = document.getElementById('mapWrap');
    const maxW = wrap.clientWidth - 32;
    const maxH = window.innerHeight * 0.68;
    MAP_CELL_PX = Math.max(4, Math.min(Math.floor(maxW / cropW), Math.floor(maxH / cropH)));
    const cell = MAP_CELL_PX;

    canvas.width = cropW * cell;
    canvas.height = cropH * cell;

    // 1) Draw belief grid — cropped + Y-FLIPPED
    const tmp = document.createElement('canvas');
    tmp.width = cropW;
    tmp.height = cropH;
    const tctx = tmp.getContext('2d');
    const img = tctx.createImageData(cropW, cropH);

    for (let y = cy1; y < cy2; y++) {
        for (let x = cx1; x < cx2; x++) {
            const p = belief[y] ? belief[y][x] : 0.5;
            const [r, g, b] = beliefToRGB(p);
            const localX = x - cx1;
            const localY = y - cy1;
            const flippedY = cropH - 1 - localY;
            const idx = (flippedY * cropW + localX) * 4;
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255;
        }
    }
    tctx.putImageData(img, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);

    // 2) Draw obstacle outlines (cropped + Y-flipped)
    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.lineWidth = 1;
    for (const obs of (env.obstacles || [])) {
        if (obs.type === 'box') {
            const ox = ((obs.x / res) - cx1) * cell;
            const oy = canvas.height - ((obs.y / res) - cy1) * cell - (obs.h / res) * cell;
            ctx.strokeRect(ox, oy, (obs.w / res) * cell, (obs.h / res) * cell);
        } else if (obs.type === 'cylinder') {
            const ocx = ((obs.x / res) - cx1) * cell;
            const ocy = canvas.height - ((obs.y / res) - cy1) * cell;
            const or2 = (obs.radius / res) * cell;
            ctx.beginPath();
            ctx.arc(ocx, ocy, or2, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    // 3) Draw drone trails + positions (cropped + Y-flipped)
    const ch = canvas.height;
    for (const drone of (state.drones || [])) {
        const color = DRONE_COLORS[drone.id % DRONE_COLORS.length];
        const trail = drone.trail || [];

        // trail
        if (trail.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.35;
            for (let i = 0; i < trail.length; i++) {
                const px = ((trail[i][0] / res) - cx1) * cell;
                const py = ch - ((trail[i][1] / res) - cy1) * cell;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        // drone dot
        const dx = ((drone.x / res) - cx1) * cell;
        const dy = ch - ((drone.y / res) - cy1) * cell;
        ctx.beginPath();
        ctx.arc(dx, dy, cell * 0.8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // heading arrow
        const hLen = cell * 1.5;
        const hx = dx + Math.cos(drone.heading) * hLen;
        const hy = dy - Math.sin(drone.heading) * hLen;
        ctx.beginPath();
        ctx.moveTo(dx, dy);
        ctx.lineTo(hx, hy);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // label
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillText(`D${drone.id}`, dx + cell, dy - cell * 0.5);
    }
}

// ════════════════════════════════════════════════════
// KPI Updates
// ════════════════════════════════════════════════════

function updateKPIs(state) {
    const m = state.metrics || {};
    const r = state.resilience || {};
    const drones = state.drones || [];
    const activeCount = drones.filter(d => d.active !== false).length;
    const phase = m.resilience_phase || r.phase || 'normal';

    document.getElementById('kpiStep').textContent = state.step || 0;
    document.getElementById('kpiCoverage').textContent = (m.exploration_pct || 0).toFixed(1) + '%';
    document.getElementById('kpiEntropy').textContent = (m.mean_entropy || 0).toFixed(3);
    document.getElementById('kpiDrones').textContent = drones.length;
    document.getElementById('kpiInfoGain').textContent = (m.step_info_gain || 0).toFixed(4);
    document.getElementById('kpiActiveDrones').textContent = `${activeCount}/${drones.length}`;
    document.getElementById('kpiInnovation').textContent = (m.innovation_ema || 0).toFixed(4);

    const phaseEl = document.getElementById('kpiPhase');
    phaseEl.textContent = phase.toUpperCase();
    phaseEl.setAttribute('data-phase', phase);
}

function updateBadges(state) {
    const m = state.metrics || {};
    const r = state.resilience || {};
    const phase = m.resilience_phase || r.phase || 'normal';

    document.getElementById('badgeStep').textContent = `Step ${state.step || 0}`;
    document.getElementById('badgeCoverage').textContent = `${(m.exploration_pct || 0).toFixed(1)}%`;
    document.getElementById('badgeEntropy').textContent = `H = ${(m.mean_entropy || 0).toFixed(3)}`;

    const badgePhase = document.getElementById('badgePhase');
    badgePhase.textContent = phase.toUpperCase();
    badgePhase.setAttribute('data-phase', phase);

    // Planner / arch badges (Tâches 1 et 2)
    const planner = state.planner || 'aif';
    const archCfg = state.arch_configured || 'centralized';
    const archEff = state.arch_effective || archCfg;
    const bp = document.getElementById('badgePlanner');
    if (bp) bp.textContent = `planner: ${planner}`;
    const ba = document.getElementById('badgeArch');
    if (ba) {
        ba.textContent = archEff === archCfg ? `arch: ${archCfg}` : `arch: ${archCfg}→${archEff}`;
    }
}

// ════════════════════════════════════════════════════
// QR KPI Updates
// ════════════════════════════════════════════════════

function updateQrKPIs(qr) {
    if (!qr) return;
    document.getElementById('kpiQrTotal').textContent = qr.total_processed || 0;
    document.getElementById('kpiQrDecoded').textContent = qr.total_decoded || 0;
    document.getElementById('kpiQrFailed').textContent = qr.total_failed || 0;
    document.getElementById('kpiQrRate').textContent = (qr.success_rate || 0) + '%';

    const cache = qr.cache || {};
    document.getElementById('kpiQrCacheHits').textContent = cache.hits || 0;

    // Last QR data from any drone
    const drones = qr.drones || {};
    let lastData = '—';
    for (const [, dr] of Object.entries(drones)) {
        if (dr.last_data) lastData = dr.last_data;
    }
    const dataEl = document.getElementById('kpiQrData');
    dataEl.textContent = lastData.length > 20 ? lastData.slice(0, 20) + '…' : lastData;
    dataEl.title = lastData;

    // Header badges
    document.getElementById('badgeQrDecoded').textContent = `✓ ${qr.total_decoded || 0}`;
    document.getElementById('badgeQrFailed').textContent = `✗ ${qr.total_failed || 0}`;
    document.getElementById('badgeQrRate').textContent = `${qr.success_rate || 0}%`;
}

// ════════════════════════════════════════════════════
// Camera Feeds
// ════════════════════════════════════════════════════

function updateCameraFeeds(qrState, simState) {
    const container = document.getElementById('cameraFeeds');
    const drones = qrState.drones || {};
    const droneIds = Object.keys(drones).sort((a, b) => +a - +b);

    if (droneIds.length === 0) return;

    // Check if we need a full rebuild (drone set changed)
    const existingIds = Array.from(container.querySelectorAll('.cam-card'))
        .map(el => el.dataset.droneId).sort();
    const needRebuild = existingIds.join(',') !== droneIds.join(',');

    if (needRebuild) {
        // Full build — only on first load or when drones change
        let html = '';
        for (const id of droneIds) {
            html += _buildCamCardHtml(id, drones[id]);
        }
        container.innerHTML = html;
    } else {
        // Incremental update — no DOM rebuild, just update src + text
        for (const id of droneIds) {
            _updateCamCardInPlace(id, drones[id]);
        }
    }
}

function _buildCamCardHtml(id, dr) {
    const color = DRONE_COLORS[+id % DRONE_COLORS.length];
    const status = dr.last_status || 'none';
    const tab = camTabState[id] || 'annotated';

    const rawSrc = `/api/frame/latest_drone_${id}.jpg?t=${Date.now()}`;
    const annSrc = `/api/frame/annotated_drone_${id}.jpg?t=${Date.now()}`;
    const imgSrc = tab === 'raw' ? rawSrc : annSrc;

    const statusLabel = {
        'success': 'DECODED', 'failed': 'FAILED',
        'cached': 'CACHED', 'none': 'WAITING',
    }[status] || status.toUpperCase();

    const statusCls = {
        'success': 'success', 'failed': 'failed',
        'cached': 'cached', 'none': 'none',
    }[status] || 'none';

    const qrData = dr.last_data || '—';
    const method = dr.last_method || '—';
    const okCount = dr.decoded_count || 0;
    const failCount = dr.failed_count || 0;
    const rate = dr.success_rate !== undefined ? dr.success_rate : 0;

    return `
    <div class="cam-card" data-status="${status}" data-drone-id="${id}" id="camCard_${id}">
        <div class="cam-card-header">
            <span class="cam-drone-label">
                <span style="color:${color};font-size:1.1rem">●</span>
                Drone ${id}
            </span>
            <span class="cam-status-badge ${statusCls}" id="camBadge_${id}">${statusLabel}</span>
        </div>
        <div class="cam-tabs">
            <div class="cam-tab ${tab === 'annotated' ? 'active' : ''}"
                 onclick="switchCamTab('${id}', 'annotated')">📊 Annotated</div>
            <div class="cam-tab ${tab === 'raw' ? 'active' : ''}"
                 onclick="switchCamTab('${id}', 'raw')">📷 Raw</div>
        </div>
        <div class="cam-frame-wrap">
            <img id="camImg_${id}" src="${imgSrc}" alt="Drone ${id} camera"
                 onerror="this.style.display='none';this.nextElementSibling.style.display=''"
                 onload="this.style.display='';this.nextElementSibling.style.display='none'">
            <div class="cam-frame-placeholder">No frame available</div>
        </div>
        <div class="cam-card-footer">
            <div>
                <span class="cam-qr-data" id="camQrData_${id}" title="${qrData}">${qrData.length > 24 ? qrData.slice(0, 24) + '…' : qrData}</span>
                <span class="cam-method" id="camMethod_${id}">(${method})</span>
            </div>
            <div class="cam-stats">
                <span class="cam-stat-ok" id="camOk_${id}">✓${okCount}</span>
                <span class="cam-stat-fail" id="camFail_${id}">✗${failCount}</span>
                <span class="cam-stat-rate" id="camRate_${id}">${rate}%</span>
            </div>
        </div>
    </div>`;
}

function _updateCamCardInPlace(id, dr) {
    const status = dr.last_status || 'none';
    const tab = camTabState[id] || 'annotated';

    // Update image src without replacing the element
    const imgEl = document.getElementById(`camImg_${id}`);
    if (imgEl) {
        const rawSrc = `/api/frame/latest_drone_${id}.jpg?t=${Date.now()}`;
        const annSrc = `/api/frame/annotated_drone_${id}.jpg?t=${Date.now()}`;
        const newSrc = tab === 'raw' ? rawSrc : annSrc;
        // Only update if the base URL changed (avoid unnecessary reloads)
        const currentBase = imgEl.src.split('?')[0];
        const newBase = newSrc.split('?')[0];
        // Always update the timestamp to get fresh frames
        if (imgEl.src === '' || imgEl.naturalWidth === 0 || currentBase.endsWith(newBase.split('/').pop())) {
            imgEl.src = newSrc;
        } else {
            imgEl.src = newSrc;
        }
    }

    // Update status badge
    const statusLabel = {
        'success': 'DECODED', 'failed': 'FAILED',
        'cached': 'CACHED', 'none': 'WAITING',
    }[status] || status.toUpperCase();
    const statusCls = {
        'success': 'success', 'failed': 'failed',
        'cached': 'cached', 'none': 'none',
    }[status] || 'none';

    const badge = document.getElementById(`camBadge_${id}`);
    if (badge) {
        badge.textContent = statusLabel;
        badge.className = `cam-status-badge ${statusCls}`;
    }

    const card = document.getElementById(`camCard_${id}`);
    if (card) card.setAttribute('data-status', status);

    // Update footer text
    const qrData = dr.last_data || '—';
    const method = dr.last_method || '—';
    const dataEl = document.getElementById(`camQrData_${id}`);
    if (dataEl) {
        dataEl.textContent = qrData.length > 24 ? qrData.slice(0, 24) + '…' : qrData;
        dataEl.title = qrData;
    }
    const methEl = document.getElementById(`camMethod_${id}`);
    if (methEl) methEl.textContent = `(${method})`;

    const okEl = document.getElementById(`camOk_${id}`);
    if (okEl) okEl.textContent = `✓${dr.decoded_count || 0}`;
    const failEl = document.getElementById(`camFail_${id}`);
    if (failEl) failEl.textContent = `✗${dr.failed_count || 0}`;
    const rateEl = document.getElementById(`camRate_${id}`);
    if (rateEl) rateEl.textContent = `${dr.success_rate !== undefined ? dr.success_rate : 0}%`;
}

function switchCamTab(droneId, tab) {
    camTabState[droneId] = tab;
}

// ════════════════════════════════════════════════════
// Drone Table (with QR columns)
// ════════════════════════════════════════════════════

function updateDroneTable(state, qrState) {
    const tbody = document.getElementById('droneTableBody');
    tbody.innerHTML = '';
    const qrDrones = (qrState && qrState.drones) ? qrState.drones : {};

    for (const d of (state.drones || [])) {
        const color = DRONE_COLORS[d.id % DRONE_COLORS.length];
        const isActive = d.active !== false;
        const statusCls = isActive ? 'active' : 'landed';
        const statusTxt = isActive ? 'ACTIVE' : 'LANDED';
        const innov = (d.innovation !== undefined) ? d.innovation.toFixed(4) : '—';

        // QR detection info
        const qrd = qrDrones[String(d.id)] || {};
        const qrStatus = qrd.last_status || '—';
        const qrRate = qrd.success_rate !== undefined ? qrd.success_rate + '%' : '—';

        let qrStatusHtml = `<span style="color:var(--text-dim)">${qrStatus}</span>`;
        if (qrStatus === 'success') qrStatusHtml = `<span style="color:var(--green)">✓ DECODED</span>`;
        else if (qrStatus === 'failed') qrStatusHtml = `<span style="color:var(--red)">✗ FAILED</span>`;
        else if (qrStatus === 'cached') qrStatusHtml = `<span style="color:var(--yellow)">◎ CACHED</span>`;

        const row = document.createElement('tr');
        if (!isActive) row.style.opacity = '0.5';
        row.innerHTML = `
            <td><span style="color:${color};font-weight:700">●</span> ${d.id}</td>
            <td><span class="drone-status ${statusCls}">${statusTxt}</span></td>
            <td>(${d.x.toFixed(1)}, ${d.y.toFixed(1)})</td>
            <td>${d.action}</td>
            <td>${d.free_energy.toFixed(3)}</td>
            <td>${d.info_gain.toFixed(3)}</td>
            <td>${innov}</td>
            <td>${d.total_distance.toFixed(1)} m</td>
            <td>${d.local_entropy.toFixed(3)}</td>
            <td>${qrStatusHtml}</td>
            <td>${qrRate}</td>
        `;
        tbody.appendChild(row);
    }
}

// ════════════════════════════════════════════════════
// Charts
// ════════════════════════════════════════════════════

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: {
        legend: { display: false, labels: { color: '#94a3b8', font: { size: 11 } } },
    },
    scales: {
        x: {
            ticks: { color: '#64748b', maxTicksLimit: 10, font: { size: 10 } },
            grid: { color: 'rgba(51,65,85,.3)' },
        },
        y: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(51,65,85,.3)' },
        },
    },
};

function initCharts() {
    chartEntropy = new Chart(document.getElementById('chartEntropy'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Mean Entropy', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.1)', fill: true, tension: .3, pointRadius: 0, borderWidth: 2 }] },
        options: { ...CHART_DEFAULTS },
    });

    chartCoverage = new Chart(document.getElementById('chartCoverage'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Coverage %', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.1)', fill: true, tension: .3, pointRadius: 0, borderWidth: 2 }] },
        options: { ...CHART_DEFAULTS, scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 100 } } },
    });

    chartFE = new Chart(document.getElementById('chartFE'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...CHART_DEFAULTS, plugins: { legend: { display: true, labels: { color: '#94a3b8', font: { size: 10 } } } } },
    });

    chartIG = new Chart(document.getElementById('chartIG'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Step IG', data: [], backgroundColor: 'rgba(167,139,250,.5)', borderColor: '#a78bfa', borderWidth: 1 }] },
        options: { ...CHART_DEFAULTS },
    });

    chartInnovation = new Chart(document.getElementById('chartInnovation'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'Innovation Mean', data: [], borderColor: '#f472b6', backgroundColor: 'rgba(244,114,182,.1)', fill: false, tension: .3, pointRadius: 0, borderWidth: 1.5 },
                { label: 'Innovation EMA', data: [], borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,.1)', fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
            ]
        },
        options: { ...CHART_DEFAULTS, plugins: { legend: { display: true, labels: { color: '#94a3b8', font: { size: 10 } } } } },
    });

    chartQrRate = new Chart(document.getElementById('chartQrRate'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'Success Rate %', data: [], borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,.15)', fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
            ]
        },
        options: { ...CHART_DEFAULTS, scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 100 } } },
    });
}

function updateCharts(history, state) {
    if (!history || history.length === 0) return;

    // Downsample if too many points
    const maxPts = 200;
    const step = history.length > maxPts ? Math.ceil(history.length / maxPts) : 1;
    const sampled = history.filter((_, i) => i % step === 0 || i === history.length - 1);

    const labels = sampled.map(h => h.step);

    // Entropy
    chartEntropy.data.labels = labels;
    chartEntropy.data.datasets[0].data = sampled.map(h => h.mean_entropy);
    chartEntropy.update();

    // Coverage
    chartCoverage.data.labels = labels;
    chartCoverage.data.datasets[0].data = sampled.map(h => h.exploration_pct);
    chartCoverage.update();

    // Info Gain (bar)
    chartIG.data.labels = labels;
    chartIG.data.datasets[0].data = sampled.map(h => h.step_info_gain);
    chartIG.update();

    // Free Energy per drone
    const numDrones = (state.drones || []).length;
    if (chartFE.data.datasets.length !== numDrones) {
        chartFE.data.datasets = [];
        for (let i = 0; i < numDrones; i++) {
            chartFE.data.datasets.push({
                label: `Drone ${i}`,
                data: [],
                borderColor: DRONE_COLORS[i % DRONE_COLORS.length],
                borderWidth: 1.5,
                pointRadius: 0,
                tension: .3,
                fill: false,
            });
        }
    }
    chartFE.data.labels = labels;
    for (let i = 0; i < numDrones; i++) {
        chartFE.data.datasets[i].data = sampled.map(h =>
            (h.free_energies && h.free_energies[i] !== undefined) ? h.free_energies[i] : 0
        );
    }
    chartFE.update();

    // Innovation (mean + EMA)
    chartInnovation.data.labels = labels;
    chartInnovation.data.datasets[0].data = sampled.map(h => h.innovation_mean || 0);
    chartInnovation.data.datasets[1].data = sampled.map(h => h.innovation_ema || 0);
    chartInnovation.update();
}

// ── QR Detection Chart (success rate over time from detection history) ──
const qrRateHistory = []; // cumulative for chart

function updateQrChart(qrState) {
    if (!qrState) return;
    const hist = qrState.history || [];
    if (hist.length === 0) return;

    // Compute rolling success rate (window = last 20 detections)
    const windowSize = 20;
    const total = qrState.total_processed || 1;

    // Add current overall rate as a data point
    qrRateHistory.push({
        idx: total,
        rate: qrState.success_rate || 0,
    });

    // Keep last 200 points
    if (qrRateHistory.length > 200) {
        qrRateHistory.splice(0, qrRateHistory.length - 200);
    }

    chartQrRate.data.labels = qrRateHistory.map(p => p.idx);
    chartQrRate.data.datasets[0].data = qrRateHistory.map(p => p.rate);
    chartQrRate.update();
}

// ════════════════════════════════════════════════════
// Resilience Events
// ════════════════════════════════════════════════════

function updateResilienceEvents(state) {
    const r = state.resilience || {};
    const events = r.events || [];
    const emptyEl = document.getElementById('eventsEmpty');
    const listEl = document.getElementById('eventsList');

    if (events.length === 0) {
        emptyEl.style.display = '';
        listEl.innerHTML = '';
        return;
    }
    emptyEl.style.display = 'none';

    // Build HTML (newest first)
    const reversed = [...events].reverse();
    listEl.innerHTML = reversed.map(ev => {
        const evType = ev.type || ev.event || 'evt';
        let iconCls = 'stress';
        let icon = '⚠';
        if (evType === 'recovery_reached' || evType === 'recovered') { iconCls = 'recovered'; icon = '✓'; }
        else if (evType === 'stress_resolved' || evType === 'resolved') { iconCls = 'resolved'; icon = '✓'; }
        const cause = ev.cause ? `<span class="event-cause">${ev.cause}</span>` : '';
        const detail = ev.entropy !== undefined ? `<span class="event-detail">H=${ev.entropy} innov=${ev.innovation || '—'}</span>` : '';
        return `<div class="event-item">
            <span class="event-icon ${iconCls}">${icon}</span>
            <span class="event-step">step ${ev.step}</span>
            <span>${evType}</span>
            ${cause}
            ${detail}
        </div>`;
    }).join('');
}

// ════════════════════════════════════════════════════
// ══════════  TABS + RESILIENCE + NS-3 + RUNS  ════════
// ════════════════════════════════════════════════════

const PHASE_COLORS = {
    normal:   'rgba(16,185,129,0.0)',  // transparent: pas de bande
    recovery: 'rgba(239,68,68,0.18)',
    durable:  'rgba(251,191,36,0.18)',
};

// ── Plugin Chart.js : bandes verticales colorées selon resilience_phase ──
const phaseBandsPlugin = {
    id: 'phaseBands',
    beforeDatasetsDraw(chart, args, opts) {
        const phases = (opts && opts.phases) || [];
        if (!phases.length) return;
        const { ctx, scales: { x }, chartArea } = chart;
        if (!x || !chartArea) return;
        ctx.save();
        for (const band of phases) {
            const color = PHASE_COLORS[band.phase] || 'transparent';
            if (color === 'transparent' || color === PHASE_COLORS.normal) continue;
            const sx = x.getPixelForValue(band.start);
            const ex = x.getPixelForValue(band.end);
            if (!Number.isFinite(sx) || !Number.isFinite(ex)) continue;
            ctx.fillStyle = color;
            ctx.fillRect(sx, chartArea.top, ex - sx, chartArea.bottom - chartArea.top);
        }
        ctx.restore();
    }
};
if (typeof Chart !== 'undefined') Chart.register(phaseBandsPlugin);

// ── Phase bands extraction from history ──
function extractPhaseBands(history) {
    if (!history || !history.length) return [];
    const bands = [];
    let curPhase = history[0].resilience_phase || 'normal';
    let start = history[0].step;
    for (let i = 1; i < history.length; i++) {
        const ph = history[i].resilience_phase || 'normal';
        if (ph !== curPhase) {
            bands.push({ phase: curPhase, start, end: history[i].step });
            curPhase = ph;
            start = history[i].step;
        }
    }
    bands.push({ phase: curPhase, start, end: history[history.length - 1].step });
    return bands;
}

let chartResEntropy, chartResCoverage, chartResInnovation, chartResActive;

function initResilienceCharts() {
    const baseOpts = JSON.parse(JSON.stringify(CHART_DEFAULTS));
    baseOpts.plugins = {
        legend: { display: true, labels: { color: '#94a3b8', font: { size: 10 } } },
        phaseBands: { phases: [] },
    };

    chartResEntropy = new Chart(document.getElementById('chartResEntropy'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'mean entropy', data: [], borderColor: '#3b82f6',
              backgroundColor: 'rgba(59,130,246,.08)', fill: true,
              tension: .3, pointRadius: 0, borderWidth: 2 },
        ]},
        options: JSON.parse(JSON.stringify(baseOpts)),
    });

    chartResCoverage = new Chart(document.getElementById('chartResCoverage'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'coverage %', data: [], borderColor: '#10b981',
              backgroundColor: 'rgba(16,185,129,.08)', fill: true,
              tension: .3, pointRadius: 0, borderWidth: 2 },
        ]},
        options: (function(){
            const o = JSON.parse(JSON.stringify(baseOpts));
            o.scales.y.min = 0; o.scales.y.max = 100;
            return o;
        })(),
    });

    chartResInnovation = new Chart(document.getElementById('chartResInnovation'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'mean', data: [], borderColor: '#f472b6',
              backgroundColor: 'rgba(244,114,182,.08)', fill: false,
              tension: .3, pointRadius: 0, borderWidth: 1.5 },
            { label: 'EMA', data: [], borderColor: '#fbbf24',
              backgroundColor: 'rgba(251,191,36,.12)', fill: true,
              tension: .3, pointRadius: 0, borderWidth: 2 },
            { label: 'spike (EMA + kσ)', data: [], borderColor: '#ef4444',
              borderDash: [4, 3], fill: false,
              tension: 0, pointRadius: 0, borderWidth: 1.2 },
        ]},
        options: JSON.parse(JSON.stringify(baseOpts)),
    });

    chartResActive = new Chart(document.getElementById('chartResActive'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'active drones', data: [], borderColor: '#22d3ee',
              backgroundColor: 'rgba(34,211,238,.12)', fill: true,
              tension: 0, pointRadius: 0, borderWidth: 2, stepped: true },
        ]},
        options: JSON.parse(JSON.stringify(baseOpts)),
    });
}

function updateResilienceTab(state, history) {
    if (!chartResEntropy) return;  // not initialized yet
    if (!history || !history.length) return;

    const labels = history.map(h => h.step);
    const ent = history.map(h => h.mean_entropy || 0);
    const cov = history.map(h => h.exploration_pct || 0);
    const innM = history.map(h => h.innovation_mean || 0);
    const innE = history.map(h => h.innovation_ema || 0);
    // Spike threshold approx : on n'a pas k·σ par step, on dessine EMA + 2·|EMA-mean|
    const spike = history.map(h => {
        const e = h.innovation_ema || 0;
        const m = h.innovation_mean || 0;
        return e + 2.0 * Math.abs(m - e);
    });
    const active = history.map(h => h.active_drones || 0);
    const bands = extractPhaseBands(history);

    [chartResEntropy, chartResCoverage, chartResInnovation, chartResActive].forEach(c => {
        if (!c.options.plugins.phaseBands) c.options.plugins.phaseBands = {};
        c.options.plugins.phaseBands.phases = bands;
    });

    chartResEntropy.data.labels = labels;
    chartResEntropy.data.datasets[0].data = ent;
    chartResEntropy.update();

    chartResCoverage.data.labels = labels;
    chartResCoverage.data.datasets[0].data = cov;
    chartResCoverage.update();

    chartResInnovation.data.labels = labels;
    chartResInnovation.data.datasets[0].data = innM;
    chartResInnovation.data.datasets[1].data = innE;
    chartResInnovation.data.datasets[2].data = spike;
    chartResInnovation.update();

    chartResActive.data.labels = labels;
    chartResActive.data.datasets[0].data = active;
    chartResActive.update();

    // KPI cards
    const r = state.resilience || {};
    const phase = (state.metrics && state.metrics.resilience_phase) || r.phase || 'normal';
    const pv = document.getElementById('resPhaseValue');
    if (pv) {
        pv.textContent = phase.toUpperCase();
        pv.setAttribute('data-phase', phase);
    }
    const stressT0 = document.getElementById('resStressT0');
    if (stressT0) stressT0.textContent = (r.stress_t0 != null && r.stress_t0 >= 0) ? r.stress_t0 : '—';
    const durable = document.getElementById('resDurable');
    if (durable) durable.textContent = r.durable_count || 0;
    const cause = document.getElementById('resCause');
    if (cause) cause.textContent = r.cause || '—';

    // Events timeline
    const evList = document.getElementById('resEventsList');
    if (evList) {
        const events = r.events || [];
        if (!events.length) {
            evList.innerHTML = '<div class="runs-empty">No event yet</div>';
        } else {
            const sorted = [...events].sort((a, b) => (b.step || 0) - (a.step || 0));
            evList.innerHTML = sorted.map(ev => {
                const t = ev.type || ev.event || '?';
                return `<div class="event-item">
                    <span class="event-step">step ${ev.step}</span>
                    <span>${t}</span>
                    ${ev.cause ? `<span class="event-cause">${ev.cause}</span>` : ''}
                </div>`;
            }).join('');
        }
    }
}

// ══════════ NS-3 TAB ══════════

let ns3TimeChart;
const ns3HistoryByPair = {};  // "i-j" -> [{step, lat}, ...]
const NS3_MAX_HIST = 200;

function initNs3Charts() {
    const opts = JSON.parse(JSON.stringify(CHART_DEFAULTS));
    opts.plugins = { legend: { display: true, labels: { color: '#94a3b8', font: { size: 10 } } } };
    ns3TimeChart = new Chart(document.getElementById('ns3TimeSeries'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: opts,
    });
}

function pairKey(a, b) {
    const i = Math.min(a, b), j = Math.max(a, b);
    return `${i}-${j}`;
}

function pairColor(k) {
    // deterministic color from string hash
    let h = 0;
    for (const c of k) h = (h * 31 + c.charCodeAt(0)) | 0;
    const colors = ['#22d3ee', '#34d399', '#a78bfa', '#fbbf24', '#f472b6', '#fb923c', '#ef4444', '#3b82f6'];
    return colors[Math.abs(h) % colors.length];
}

function updateNs3Tab(state, ns3Data) {
    const net = (state && state.network) || {};
    const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setText('ns3Mode', state.ns3_mode || net.ns3_mode || 'none');
    setText('ns3Cloud', net.cloud_link_active ? 'UP' : 'DOWN');
    setText('ns3QueueSize', net.queue_size != null ? net.queue_size : 0);
    setText('ns3Dropped', net.msg_dropped != null ? net.msg_dropped : 0);
    setText('ns3Sent', net.msg_sent != null ? net.msg_sent : 0);
    setText('ns3Delivered', net.msg_delivered != null ? net.msg_delivered : 0);

    const cloudEl = document.getElementById('ns3Cloud');
    if (cloudEl) {
        cloudEl.style.color = net.cloud_link_active ? 'var(--green)' : 'var(--red)';
    }

    // Pairs : prendre depuis ns3Data si dispo (CSV NS-3) sinon depuis network.ns3_pairs
    let pairs = [];
    if (ns3Data && Array.isArray(ns3Data.pairs)) pairs = ns3Data.pairs;
    else if (Array.isArray(net.ns3_pairs)) pairs = net.ns3_pairs;

    // Table
    const tbody = document.getElementById('ns3TableBody');
    if (tbody) {
        if (!pairs.length) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No NS-3 data — start with --ns3 wifi|5g</td></tr>';
        } else {
            const cuts = new Set((net.cut_pairs || []).map(s => s.split('-').map(Number).sort().join('-')));
            tbody.innerHTML = pairs.map(p => {
                const a = p.a, b = p.b;
                const k = pairKey(a, b);
                const cut = cuts.has(k) || net.all_drone_links_cut;
                return `<tr>
                    <td>${a} ↔ ${b}</td>
                    <td>${(p.latency_ms || 0).toFixed(2)}</td>
                    <td>${(p.jitter_ms || 0).toFixed(2)}</td>
                    <td>${p.rx_packets != null ? p.rx_packets : '—'}</td>
                    <td>${cut ? '<span style="color:var(--red)">CUT</span>' : '<span style="color:var(--green)">OK</span>'}</td>
                </tr>`;
            }).join('');
        }
    }

    // Heatmap
    const nDrones = (state.drones || []).length;
    renderNs3Heatmap(pairs, nDrones);

    // Time-series : append current step latencies per pair
    if (state.step != null && pairs.length) {
        const labelSet = new Set();
        for (const p of pairs) {
            const k = pairKey(p.a, p.b);
            if (!ns3HistoryByPair[k]) ns3HistoryByPair[k] = [];
            ns3HistoryByPair[k].push({ step: state.step, lat: p.latency_ms || 0 });
            if (ns3HistoryByPair[k].length > NS3_MAX_HIST) {
                ns3HistoryByPair[k].splice(0, ns3HistoryByPair[k].length - NS3_MAX_HIST);
            }
            labelSet.add(state.step);
        }
        if (ns3TimeChart) {
            const allSteps = new Set();
            Object.values(ns3HistoryByPair).forEach(h => h.forEach(p => allSteps.add(p.step)));
            const sortedSteps = [...allSteps].sort((a, b) => a - b);
            ns3TimeChart.data.labels = sortedSteps;
            ns3TimeChart.data.datasets = Object.keys(ns3HistoryByPair).map(k => {
                const m = new Map(ns3HistoryByPair[k].map(p => [p.step, p.lat]));
                return {
                    label: k,
                    data: sortedSteps.map(s => m.has(s) ? m.get(s) : null),
                    borderColor: pairColor(k),
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                    spanGaps: true,
                };
            });
            ns3TimeChart.update();
        }
    }
}

function renderNs3Heatmap(pairs, nDrones) {
    const canvas = document.getElementById('ns3Heatmap');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const n = Math.max(nDrones || 0,
        ...pairs.map(p => Math.max(p.a, p.b) + 1));
    if (!n || !pairs.length) {
        canvas.width = canvas.parentElement.clientWidth || 320;
        canvas.height = 200;
        ctx.fillStyle = '#0a0f1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '13px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No NS-3 data', canvas.width / 2, canvas.height / 2);
        return;
    }

    const M = Array.from({ length: n }, () => Array(n).fill(null));
    let maxLat = 0;
    for (const p of pairs) {
        const lat = p.latency_ms || 0;
        M[p.a][p.b] = lat;
        M[p.b][p.a] = lat;
        if (lat > maxLat) maxLat = lat;
    }
    if (maxLat <= 0) maxLat = 1.0;

    const wrap = canvas.parentElement;
    const size = Math.min(wrap.clientWidth || 320, 320);
    canvas.width = size;
    canvas.height = size;
    const cell = Math.floor((size - 30) / n);
    const off = 25;

    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
        ctx.fillText(`D${i}`, off + i * cell + cell / 2, 14);
        ctx.fillText(`D${i}`, 12, off + i * cell + cell / 2 + 3);
    }

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const v = M[i][j];
            const x = off + j * cell;
            const y = off + i * cell;
            if (v == null || i === j) {
                ctx.fillStyle = '#1e293b';
                ctx.fillRect(x, y, cell - 1, cell - 1);
            } else {
                const t = Math.min(1, v / maxLat);
                // colormap : bleu → violet → rouge
                const r = Math.round(40 + 200 * t);
                const g = Math.round(60 - 30 * t);
                const b = Math.round(200 - 180 * t);
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(x, y, cell - 1, cell - 1);
                ctx.fillStyle = '#fff';
                ctx.font = `${Math.max(8, cell/4)}px JetBrains Mono, monospace`;
                ctx.textAlign = 'center';
                ctx.fillText(v.toFixed(1), x + cell / 2, y + cell / 2 + 3);
            }
        }
    }
}

// ══════════ RUNS TAB ══════════

let _runsCache = [];

async function loadRunsList() {
    const listEl = document.getElementById('runsList');
    if (!listEl) return;
    listEl.innerHTML = '<div class="runs-empty">Loading…</div>';
    try {
        const r = await fetch('/api/runs');
        if (!r.ok) throw new Error('runs not available');
        const data = await r.json();
        const runs = data.runs || [];
        _runsCache = runs;
        if (!runs.length) {
            listEl.innerHTML = '<div class="runs-empty">No runs yet. Run scripts/run_all_experiments.sh.</div>';
            return;
        }
        listEl.innerHTML = runs.map(run => {
            const cfg = run.config || {};
            const tag = cfg.run_tag || run.tag || '?';
            const planner = cfg.planner || '?';
            const arch = cfg.arch || '?';
            const ns3 = cfg.ns3_mode || 'none';
            return `<div class="run-item" data-tag="${run.tag}">
                <div class="run-name">${run.tag}</div>
                <div class="run-meta">
                    <span class="badge">${planner}</span>
                    <span class="badge">${arch}</span>
                    <span class="badge">${ns3}</span>
                </div>
                <div class="run-time">${run.mtime || ''}</div>
            </div>`;
        }).join('');
        listEl.querySelectorAll('.run-item').forEach(el => {
            el.addEventListener('click', () => loadRunGallery(el.dataset.tag));
        });
    } catch (e) {
        listEl.innerHTML = `<div class="runs-empty">Error loading runs: ${e.message}</div>`;
    }
}

function loadRunGallery(tag) {
    const run = _runsCache.find(r => r.tag === tag);
    if (!run) return;
    const titleEl = document.getElementById('runDetailTitle');
    const metaEl  = document.getElementById('runDetailMeta');
    const galleryEl = document.getElementById('runsGallery');
    if (titleEl) titleEl.textContent = run.tag;
    if (metaEl) {
        const c = run.config || {};
        metaEl.textContent = `${c.planner || '?'} · ${c.arch || '?'} · ns3=${c.ns3_mode || 'none'}`;
    }
    const images = run.images || [];
    if (!images.length) {
        galleryEl.innerHTML = '<div class="runs-empty">No images for this run.</div>';
        return;
    }
    galleryEl.innerHTML = images.map(img => {
        const src = `/api/run/${encodeURIComponent(run.tag)}/img/${encodeURIComponent(img)}`;
        return `<div class="run-img-card">
            <div class="run-img-name">${img}</div>
            <img class="run-img" src="${src}" alt="${img}"
                 onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'runs-empty',textContent:'(missing) '+'${img}'}))">
        </div>`;
    }).join('');
}

// ══════════ TAB SWITCHING ══════════

document.addEventListener('DOMContentLoaded', () => {
    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    // Init resilience + ns-3 charts (separate from main initCharts to avoid double)
    if (typeof Chart !== 'undefined') {
        try { initResilienceCharts(); } catch (e) { console.warn('initResilienceCharts:', e); }
        try { initNs3Charts(); } catch (e) { console.warn('initNs3Charts:', e); }
    }
    const refresh = document.getElementById('btnRefreshRuns');
    if (refresh) refresh.addEventListener('click', loadRunsList);
    // Load runs list on first display
    loadRunsList();
});

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.tab-panel').forEach(p =>
        p.classList.toggle('active', p.id === `tab-${tab}`));
    // Re-fetch runs when switching to Runs tab (cheap)
    if (tab === 'runs') loadRunsList();
}
