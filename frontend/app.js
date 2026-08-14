document.addEventListener('DOMContentLoaded', () => {
    initPlatform();
    loadDashboardData();

    document.getElementById('btn-recommend').addEventListener('click', () => {
        loadDashboardData();
    });

    document.getElementById('workload-select').addEventListener('change', (e) => {
        const desc = document.getElementById('workload-desc');
        if (e.target.value === 'short_generation') {
            desc.textContent = '5 deterministic prompts testing short-response interactive latency (~15 input tokens).';
        } else {
            desc.textContent = 'Document-grounded technical passage (~650 input tokens) testing prefill & KV-cache.';
        }
        loadDashboardData();
    });

    document.getElementById('objective-select').addEventListener('change', () => {
        loadDashboardData();
    });
});

async function initPlatform() {
    try {
        const res = await fetch('/api/platform');
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
        const recRes = await fetch('/api/optimize/recommend', {
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
        const expRes = await fetch(`/api/optimization/latest?workload_type=${workload}`);
        if (expRes.ok) {
            const exp = await expRes.json();
            if (exp.results) {
                renderTable(exp.results);
                renderScatterPlot(exp.results);
            }
        }
    } catch (e) {
        console.error('Error fetching dashboard data:', e);
    }
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

    results.forEach(r => {
        const tr = document.createElement('tr');
        if (r.pareto_optimal) tr.className = 'pareto-row';

        const cfg = r.configuration;
        const res = r.results;
        const quant = cfg.quantization || 'Q4_K_M';
        const sizeMb = (res.model_size_mb !== undefined ? `${res.model_size_mb.toFixed(1)} MB` : '--');
        const loadMs = (res.load_time_ms !== undefined ? `${res.load_time_ms.toFixed(0)} ms` : '--');

        tr.innerHTML = `
            <td><strong class="text-cyan">${quant}</strong></td>
            <td><strong>T=${cfg.threads}, C=${cfg.context_size}</strong></td>
            <td>${sizeMb}</td>
            <td>${loadMs}</td>
            <td>${res.mean_latency_ms.toFixed(2)} ms</td>
            <td>${res.median_latency_ms.toFixed(2)} ms</td>
            <td>${res.p95_latency_ms.toFixed(2)} ms</td>
            <td><strong>${res.mean_tokens_per_second.toFixed(2)}</strong> tok/s</td>
            <td>${r.pareto_optimal ? '<span class="tag-pareto">PARETO OPTIMAL</span>' : '<span class="tag-dominated">Dominated</span>'}</td>
            <td>${(r.score !== undefined ? r.score.toFixed(4) : '--')}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderScatterPlot(results) {
    const canvas = document.getElementById('paretoChart');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const padLeft = 60, padRight = 40, padTop = 30, padBottom = 50;

    const latencies = results.map(r => r.results.mean_latency_ms);
    const tpsList = results.map(r => r.results.mean_tokens_per_second);

    const minLat = Math.min(...latencies) * 0.9;
    const maxLat = Math.max(...latencies) * 1.1;
    const minTps = Math.min(...tpsList) * 0.85;
    const maxTps = Math.max(...tpsList) * 1.15;

    // Draw Grid & Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    // Horizontal gridlines (TPS)
    const gridYCount = 4;
    ctx.font = '11px JetBrains Mono';
    ctx.fillStyle = '#64748b';
    for (let i = 0; i <= gridYCount; i++) {
        const val = minTps + (i / gridYCount) * (maxTps - minTps);
        const y = h - padBottom - (i / gridYCount) * (h - padTop - padBottom);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();
        ctx.fillText(val.toFixed(1), 15, y + 4);
    }

    // Vertical gridlines (Latency)
    const gridXCount = 5;
    for (let i = 0; i <= gridXCount; i++) {
        const val = minLat + (i / gridXCount) * (maxLat - minLat);
        const x = padLeft + (i / gridXCount) * (w - padLeft - padRight);
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, h - padBottom);
        ctx.stroke();
        ctx.fillText(`${val.toFixed(0)}ms`, x - 18, h - padBottom + 20);
    }

    // Axis Labels
    ctx.font = '12px Outfit';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Latency (ms) — Lower is Better ◄', w / 2 - 80, h - 10);

    ctx.save();
    ctx.translate(14, h / 2 + 60);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Throughput (tok/s) — Higher is Better ▲', 0, 0);
    ctx.restore();

    // Plot Points
    results.forEach(r => {
        const lat = r.results.mean_latency_ms;
        const tps = r.results.mean_tokens_per_second;
        const isPareto = r.pareto_optimal;
        const quant = r.configuration.quantization || 'Q4_K_M';

        const x = padLeft + ((lat - minLat) / (maxLat - minLat)) * (w - padLeft - padRight);
        const y = h - padBottom - ((tps - minTps) / (maxTps - minTps)) * (h - padTop - padBottom);

        ctx.beginPath();
        if (isPareto) {
            ctx.arc(x, y, 7, 0, 2 * Math.PI);
            ctx.fillStyle = '#00e5ff';
            ctx.shadowColor = '#00e5ff';
            ctx.shadowBlur = 12;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Label Pareto Point
            ctx.font = '10px JetBrains Mono';
            ctx.fillStyle = '#f0f4f8';
            ctx.fillText(`${quant} T=${r.configuration.threads}, C=${r.configuration.context_size}`, x + 10, y - 6);
        } else {
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
            ctx.fill();
        }
    });
}
