document.addEventListener('DOMContentLoaded', function () {
    fetchAnalytics();
});

async function fetchAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();

        if (data.status === 'error') {
            alert(data.message);
            window.location.href = '/';
            return;
        }

        // Update Stat Cards
        document.getElementById('totalVolume').innerText = data.summary.total_volume.toLocaleString();
        document.getElementById('avgHourly').innerText = data.summary.average_hourly;
        document.getElementById('peakHour').innerText = data.summary.peak_hour + ":00";

        // Update Performance Metrics
        if (data.summary.metrics) {
            document.getElementById('rmseScore').innerText = data.summary.metrics.rmse;
            document.getElementById('r2Score').innerText = data.summary.metrics.r2_score;
        }

        // Render Charts
        renderHourlyChart(data.charts.hourly.labels, data.charts.hourly.data);
        renderJunctionChart(data.charts.junction.labels, data.charts.junction.data);

        // Render Pie Chart if element exists (Demo page)
        if (document.getElementById('congestionChart')) {
            renderCongestionChart(data.charts.congestion.labels, data.charts.congestion.data);
        }

    } catch (error) {
        console.error('Error fetching analytics:', error);
    }
}

function renderHourlyChart(labels, data) {
    const ctx = document.getElementById('hourlyChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels.map(l => l + ":00"),
            datasets: [{
                label: 'Avg Vehicles',
                data: data,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function renderJunctionChart(labels, data) {
    const ctx = document.getElementById('junctionChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => 'Junction ' + l),
            datasets: [{
                label: 'Avg Vehicles',
                data: data,
                backgroundColor: '#10b981',
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function renderCongestionChart(labels, data) {
    const ctx = document.getElementById('congestionChart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'], // Green, Orange, Red
                borderColor: '#1e293b',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { size: 12 } }
                }
            }
        }
    });
}
