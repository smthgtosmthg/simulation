/* ══════════════════════════════════════════════════════════════
   Active Inference Dashboard — Visualization
   Polls /api/state and /api/history, renders belief map + charts
   ══════════════════════════════════════════════════════════════ */

const POLL_MS = 500;
const DRONE_COLORS = ['#22d3ee', '#34d399', '#a78bfa', '#fbbf24', '#f472b6', '#fb923c'];
let MAP_CELL_PX = 10; // pixels per grid cell on the map canvas (recalculated dynamically)

let paused = false;
let lastStep = -1;

// ── Chart.js instances ──
let chartEntropy, chartCoverage, chartFE, chartIG;

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
        const [stateRes, histRes] = await Promise.all([
            fetch('/api/state'),
            fetch('/api/history'),
        ]);
        if (!stateRes.ok || !histRes.ok) { setOffline(); return; }
        const state   = await stateRes.json();
        const history = await histRes.json();

        if (state.step === lastStep) return; // no new data
        lastStep = state.step;

        setOnline();
        renderMap(state);
        updateKPIs(state);
        updateBadges(state);
        updateDroneTable(state);
        updateCharts(history, state);
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

    canvas.width  = cropW * cell;
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
            img.data[idx]     = r;
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
    document.getElementById('kpiStep').textContent      = state.step || 0;
    document.getElementById('kpiCoverage').textContent   = (m.exploration_pct || 0).toFixed(1) + '%';
    document.getElementById('kpiEntropy').textContent    = (m.mean_entropy || 0).toFixed(3);
    document.getElementById('kpiDrones').textContent     = (state.drones || []).length;
    document.getElementById('kpiInfoGain').textContent   = (m.step_info_gain || 0).toFixed(4);
}

function updateBadges(state) {
    const m = state.metrics || {};
    document.getElementById('badgeStep').textContent     = `Step ${state.step || 0}`;
    document.getElementById('badgeCoverage').textContent  = `${(m.exploration_pct || 0).toFixed(1)}%`;
    document.getElementById('badgeEntropy').textContent   = `H = ${(m.mean_entropy || 0).toFixed(3)}`;
}

function updateDroneTable(state) {
    const tbody = document.getElementById('droneTableBody');
    tbody.innerHTML = '';
    for (const d of (state.drones || [])) {
        const color = DRONE_COLORS[d.id % DRONE_COLORS.length];
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><span style="color:${color};font-weight:700">●</span> ${d.id}</td>
            <td>(${d.x.toFixed(1)}, ${d.y.toFixed(1)})</td>
            <td>${d.action}</td>
            <td>${d.free_energy.toFixed(3)}</td>
            <td>${d.info_gain.toFixed(3)}</td>
            <td>${d.total_distance.toFixed(1)} m</td>
            <td>${d.local_entropy.toFixed(3)}</td>
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
            grid:  { color: 'rgba(51,65,85,.3)' },
        },
        y: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid:  { color: 'rgba(51,65,85,.3)' },
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
}
