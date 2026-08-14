let currentExperimentResults = [];
const API_BASE = window.API_BASE_URL || localStorage.getItem('API_BASE_URL') || '';

document.addEventListener('DOMContentLoaded', () => {
    initPlatform();
    loadDashboardData();

    document.getElementById('btn-recommend').addEventListener('click', () => {
        loadDashboardData();
    });

    document.getElementById('workload-select').addEventListener('change', (e) => {
        const desc = document.getElementById('workload-desc');
        if (e.target.value === 'short_generation') {
            desc.textContent = '5 deterministic prompts testing interactive latency (~15 input tokens).';
        } else {
            desc.textContent = 'Document-grounded technical passage (~650 input tokens) testing prefill & KV-cache.';
        }
        loadDashboardData();
    });

    document.getElementById('objective-select').addEventListener('change', () => {
        loadDashboardData();
    });

    document.getElementById('quant-filter').addEventListener('change', () => {
        filterAndRenderTable();
    });

    document.getElementById('thread-filter').addEventListener('change', () => {
        filterAndRenderTable();
    });

    document.getElementById('context-filter').addEventListener('change', () => {
        filterAndRenderTable();
    });
});

async function initPlatform() {
    try {
        const res = await fetch(`${API_BASE}/api/platform`);
        if (res.ok) {
            const data = await res.json();
            document.getElementById('val-provider').textContent = data.provider.toUpperCase();
            document.getElementById('val-arch').textContent = data.architecture;
            document.getElementById('val-cpu').textContent = data.cpu || 'AMD64 Family 23';
            document.getElementById('val-cores').textContent = `${data.physical_cores}P / ${data.logical_cores}L`;
            document.getElementById('val-ram').textContent = `${data.ram_gb} GB`;
        }
    } catch (e) {
        console.error('Failed to load platform info', e);
    }
}

async function loadDashboardData() {
    const workload = document.getElementById('workload-select').value;
    const objective = document.getElementById('objective-select').value;

    document.getElementById('rec-obj-badge').textContent = `${objective.toUpperCase()} OPTIMIZED`;

    try {
        // 1. Fetch Recommendation
        const recRes = await fetch(`${API_BASE}/api/optimize/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workload, objective })
        });

        if (recRes.ok) {
            const rec = await recRes.json();
            if (rec.status === 'success') {
                const cfg = rec.recommended_configuration;
                const quant = cfg.quantization || 'Q4_K_M';
                document.getElementById('rec-config-hero').textContent = `${quant} • Threads=${cfg.threads}, Context=${cfg.context_size}`;
                document.getElementById('rec-reason').textContent = rec.reason;
                document.getElementById('rec-lat').textContent = `${rec.metrics.mean_latency_ms.toFixed(2)} ms`;
                document.getElementById('rec-tps').textContent = `${rec.metrics.mean_tokens_per_second.toFixed(2)} tok/s`;

                const latImp = rec.baseline_improvement.latency_pct;
                const tpsImp = rec.baseline_improvement.throughput_pct;
                document.getElementById('rec-lat-imp').textContent = `${latImp >= 0 ? '+' : ''}${latImp.toFixed(2)}%`;
                document.getElementById('rec-tps-imp').textContent = `${tpsImp >= 0 ? '+' : ''}${tpsImp.toFixed(2)}%`;

                renderParetoCards(rec.pareto_configurations);
            }
        }

        // 2. Fetch Latest Experiment for Table & Scatter Plot
        const expRes = await fetch(`${API_BASE}/api/optimization/latest?workload_type=${workload}`);
        if (expRes.ok) {
            const exp = await expRes.json();
            if (exp.results) {
                currentExperimentResults = exp.results;
                filterAndRenderTable();
            }
        }
    } catch (e) {
        console.error('Error fetching dashboard data:', e);
    }
}

function filterAndRenderTable() {
    const qFilter = document.getElementById('quant-filter').value;
    const tFilter = document.getElementById('thread-filter').value;
    const cFilter = document.getElementById('context-filter').value;

    let filtered = currentExperimentResults;
    if (qFilter !== 'ALL') {
        filtered = filtered.filter(r => (r.configuration.quantization || 'Q4_K_M') === qFilter);
    }
    if (tFilter !== 'ALL') {
        filtered = filtered.filter(r => String(r.configuration.threads) === tFilter);
    }
    if (cFilter !== 'ALL') {
        filtered = filtered.filter(r => String(r.configuration.context_size) === cFilter);
    }

    renderTable(filtered);
    renderScatterPlot(filtered);
}

function renderParetoCards(paretoList) {
    const container = document.getElementById('pareto-list');
    container.innerHTML = '';

    if (!paretoList || paretoList.length === 0) {
        container.innerHTML = '<div class="text-muted">No Pareto points available.</div>';
        return;
    }

    paretoList.forEach(p => {
        const item = document.createElement('div');
        item.className = 'pareto-item';
        const q = p.quantization || 'Q4_K_M';
        const sizeStr = p.model_size_mb ? ` (${p.model_size_mb.toFixed(1)}MB)` : '';
        item.innerHTML = `
            <span class="p-config">${q} T=${p.threads}, C=${p.context_size}${sizeStr}</span>
            <span class="p-stats">${p.mean_latency_ms.toFixed(1)}ms | ${p.mean_tokens_per_second.toFixed(1)} tok/s</span>
        `;
        container.appendChild(item);
    });
}

function renderTable(results) {
    const tbody = document.getElementById('matrix-tbody');
    tbody.innerHTML = '';

    document.getElementById('table-count-badge').textContent = `${results.length} Configurations`;

    results.forEach((r, idx) => {
        const tr = document.createElement('tr');
        if (r.pareto_optimal) tr.className = 'pareto-row';

        const cfg = r.configuration;
        const res = r.results;
        const quant = cfg.quantization || 'Q4_K_M';
        const cfgId = r.configuration_id || cfg.configuration_id || `cfg_${quant}_T${cfg.threads}_C${cfg.context_size}`;
        const sizeMb = (res.model_size_mb !== undefined ? `${res.model_size_mb.toFixed(1)} MB` : '--');
        const loadMs = (res.load_time_ms !== undefined ? `${res.load_time_ms.toFixed(0)} ms` : '--');
        const scoreVal = r.score !== undefined ? r.score : 0;
        const scorePct = Math.min(scoreVal * 100, 100);

        // Determine quantization color
        let quantColor = '#00d4ff';
        if (quant === 'Q5_K_M') quantColor = '#7c5cfc';
        else if (quant === 'Q8_0') quantColor = '#22d3a7';

        tr.innerHTML = `
            <td>${idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : '#' + (idx + 1)}</td>
            <td><code style="font-size: 11px; opacity: 0.8;">${cfgId}</code></td>
            <td><strong style="color: ${quantColor};">${quant}</strong></td>
            <td>${cfg.threads}</td>
            <td>${cfg.context_size}</td>
            <td>${sizeMb}</td>
            <td>${loadMs}</td>
            <td>${res.mean_latency_ms.toFixed(2)} ms</td>
            <td>${res.p95_latency_ms.toFixed(2)} ms</td>
            <td><strong>${res.mean_tokens_per_second.toFixed(2)}</strong> tok/s</td>
            <td>${r.pareto_optimal ? '<span class="tag-pareto">✦ PARETO</span>' : '<span class="tag-dominated">Dominated</span>'}</td>
            <td>
                <div class="score-cell">
                    <span>${scoreVal.toFixed(4)}</span>
                    <div class="score-bar-track">
                        <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
                    </div>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function renderScatterPlot(results) {
    const canvas = document.getElementById('paretoChart');
    const ctx = canvas.getContext('2d');

    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = 900 * dpr;
    canvas.height = 360 * dpr;
    canvas.style.width = '900px';
    canvas.style.height = '360px';
    ctx.scale(dpr, dpr);

    const w = 900;
    const h = 360;

    ctx.clearRect(0, 0, w, h);

    if (!results || results.length === 0) return;

    const padLeft = 65, padRight = 45, padTop = 30, padBottom = 55;

    const latencies = results.map(r => r.results.mean_latency_ms);
    const tpsList = results.map(r => r.results.mean_tokens_per_second);

    const minLat = Math.min(...latencies) * 0.9;
    const maxLat = Math.max(...latencies) * 1.1;
    const minTps = Math.min(...tpsList) * 0.85;
    const maxTps = Math.max(...tpsList) * 1.15;

    // Draw Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;

    // Horizontal gridlines
    const gridYCount = 5;
    ctx.font = '11px JetBrains Mono';
    for (let i = 0; i <= gridYCount; i++) {
        const val = minTps + (i / gridYCount) * (maxTps - minTps);
        const y = h - padBottom - (i / gridYCount) * (h - padTop - padBottom);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();
        ctx.fillStyle = '#556178';
        ctx.fillText(val.toFixed(1), 10, y + 4);
    }

    // Vertical gridlines
    const gridXCount = 6;
    for (let i = 0; i <= gridXCount; i++) {
        const val = minLat + (i / gridXCount) * (maxLat - minLat);
        const x = padLeft + (i / gridXCount) * (w - padLeft - padRight);
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, h - padBottom);
        ctx.stroke();
        ctx.fillStyle = '#556178';
        ctx.fillText(`${val.toFixed(0)}ms`, x - 18, h - padBottom + 20);
    }

    // Axis Labels
    ctx.font = '12px Outfit';
    ctx.fillStyle = '#8b99b0';
    ctx.fillText('Latency (ms) — Lower is Better ◄', w / 2 - 100, h - 8);

    ctx.save();
    ctx.translate(14, h / 2 + 80);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Throughput (tok/s) — Higher is Better ▲', 0, 0);
    ctx.restore();

    // Sort: draw dominated first, then Pareto on top
    const sortedResults = [...results].sort((a, b) => (a.pareto_optimal ? 1 : 0) - (b.pareto_optimal ? 1 : 0));

    // Plot Points
    sortedResults.forEach(r => {
        const lat = r.results.mean_latency_ms;
        const tps = r.results.mean_tokens_per_second;
        const sizeMb = r.results.model_size_mb || 468.0;
        const isPareto = r.pareto_optimal;
        const quant = r.configuration.quantization || 'Q4_K_M';

        const radius = 5 + ((sizeMb - 468.0) / (645.0 - 468.0)) * 6;

        const x = padLeft + ((lat - minLat) / (maxLat - minLat)) * (w - padLeft - padRight);
        const y = h - padBottom - ((tps - minTps) / (maxTps - minTps)) * (h - padTop - padBottom);

        ctx.save();

        if (isPareto) {
            // Outer glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 3);
            gradient.addColorStop(0, 'rgba(0, 212, 255, 0.2)');
            gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');
            ctx.beginPath();
            ctx.arc(x, y, radius * 3, 0, 2 * Math.PI);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Main point
            ctx.beginPath();
            ctx.arc(x, y, radius + 2, 0, 2 * Math.PI);
            const pointGradient = ctx.createRadialGradient(x - 1, y - 1, 0, x, y, radius + 2);
            pointGradient.addColorStop(0, '#4ff0ff');
            pointGradient.addColorStop(1, '#00a8d4');
            ctx.fillStyle = pointGradient;
            ctx.shadowColor = '#00d4ff';
            ctx.shadowBlur = 16;
            ctx.fill();
            ctx.shadowBlur = 0;

            // White border ring
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Label
            ctx.font = '10px JetBrains Mono';
            ctx.fillStyle = '#e8edf5';
            ctx.shadowColor = 'rgba(0,0,0,0.7)';
            ctx.shadowBlur = 4;
            ctx.fillText(`${quant} T=${r.configuration.threads} C=${r.configuration.context_size}`, x + 12, y - 8);
            ctx.shadowBlur = 0;
        } else {
            // Quantization-based coloring
            let color = 'rgba(0, 212, 255, 0.25)';
            if (quant === 'Q5_K_M') color = 'rgba(124, 92, 252, 0.25)';
            else if (quant === 'Q8_0') color = 'rgba(34, 211, 167, 0.25)';

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();

            // Subtle border
            let borderColor = 'rgba(0, 212, 255, 0.15)';
            if (quant === 'Q5_K_M') borderColor = 'rgba(124, 92, 252, 0.15)';
            else if (quant === 'Q8_0') borderColor = 'rgba(34, 211, 167, 0.15)';
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.restore();
    });

    // Draw legend in chart
    const legendY = padTop + 8;
    const legendX = w - padRight - 180;
    ctx.font = '10px Outfit';

    // Q4_K_M
    ctx.beginPath();
    ctx.arc(legendX, legendY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
    ctx.fill();
    ctx.fillStyle = '#8b99b0';
    ctx.fillText('Q4_K_M', legendX + 10, legendY + 3);

    // Q5_K_M
    ctx.beginPath();
    ctx.arc(legendX + 65, legendY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(124, 92, 252, 0.5)';
    ctx.fill();
    ctx.fillStyle = '#8b99b0';
    ctx.fillText('Q5_K_M', legendX + 75, legendY + 3);

    // Q8_0
    ctx.beginPath();
    ctx.arc(legendX + 130, legendY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(34, 211, 167, 0.5)';
    ctx.fill();
    ctx.fillStyle = '#8b99b0';
    ctx.fillText('Q8_0', legendX + 140, legendY + 3);
}
