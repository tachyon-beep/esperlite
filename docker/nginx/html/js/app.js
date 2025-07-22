// Esper Demo Portal - Real-time Application

class EsperDemoApp {
    constructor() {
        this.charts = {};
        this.logsPaused = false;
        this.logBuffer = [];
        this.metricsHistory = {
            loss: [],
            accuracy: [],
            gpu: [],
            time: []
        };
        
        this.init();
    }

    init() {
        this.setupTabs();
        this.setupCharts();
        this.setupEventListeners();
        this.startPolling();
        this.connectWebSocket();
        this.updateUptime();
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                
                // Update active states
                tabButtons.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));
                
                button.classList.add('active');
                document.getElementById(tabName).classList.add('active');
            });
        });
    }

    setupCharts() {
        // Training progress chart
        const trainingCtx = document.getElementById('training-chart');
        if (trainingCtx) {
            this.charts.training = new Chart(trainingCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#1a73e8',
                        backgroundColor: 'rgba(26, 115, 232, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#ea4335',
                        backgroundColor: 'rgba(234, 67, 53, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Training Accuracy',
                        data: [],
                        borderColor: '#34a853',
                        backgroundColor: 'rgba(52, 168, 83, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e7e9ea' }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#2f3336' },
                            ticks: { color: '#8b98a5' }
                        },
                        y: {
                            position: 'left',
                            grid: { color: '#2f3336' },
                            ticks: { color: '#8b98a5' },
                            title: {
                                display: true,
                                text: 'Loss',
                                color: '#8b98a5'
                            }
                        },
                        y1: {
                            position: 'right',
                            grid: { drawOnChartArea: false },
                            ticks: { color: '#8b98a5' },
                            title: {
                                display: true,
                                text: 'Accuracy (%)',
                                color: '#8b98a5'
                            }
                        }
                    }
                }
            });
        }

        // GPU utilization chart
        const gpuCtx = document.getElementById('gpu-chart');
        if (gpuCtx) {
            this.charts.gpu = new Chart(gpuCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'GPU Utilization',
                        data: [],
                        borderColor: '#34a853',
                        backgroundColor: 'rgba(52, 168, 83, 0.2)',
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
                        x: {
                            display: false
                        },
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: '#2f3336' },
                            ticks: { 
                                color: '#8b98a5',
                                callback: value => value + '%'
                            }
                        }
                    }
                }
            });
        }

        // Resources chart
        const resourcesCtx = document.getElementById('resources-chart');
        if (resourcesCtx) {
            this.charts.resources = new Chart(resourcesCtx, {
                type: 'bar',
                data: {
                    labels: ['CPU', 'Memory', 'GPU Memory', 'Disk I/O'],
                    datasets: [{
                        label: 'Usage %',
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            'rgba(26, 115, 232, 0.6)',
                            'rgba(52, 168, 83, 0.6)',
                            'rgba(251, 188, 4, 0.6)',
                            'rgba(234, 67, 53, 0.6)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { color: '#2f3336' },
                            ticks: { color: '#8b98a5' }
                        },
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: '#2f3336' },
                            ticks: { color: '#8b98a5' }
                        }
                    }
                }
            });
        }
    }

    setupEventListeners() {
        // Log controls
        const logFilter = document.getElementById('log-filter');
        if (logFilter) {
            logFilter.addEventListener('change', () => this.filterLogs());
        }

        const clearLogsBtn = document.getElementById('clear-logs');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => this.clearLogs());
        }

        const pauseLogsBtn = document.getElementById('pause-logs');
        if (pauseLogsBtn) {
            pauseLogsBtn.addEventListener('click', () => this.toggleLogsPause());
        }
    }

    async startPolling() {
        // Poll for status updates
        setInterval(() => this.updateStatus(), 2000);
        
        // Poll for training updates
        setInterval(() => this.updateTraining(), 1000);
        
        // Poll for kernel stats
        setInterval(() => this.updateKernelStats(), 5000);
        
        // Initial updates
        this.updateStatus();
        this.updateTraining();
        this.updateKernelStats();
    }

    async updateStatus() {
        try {
            const response = await fetch('http://localhost:8889/api/status');
            const data = await response.json();
            
            // Update service status
            Object.entries(data.services).forEach(([service, status]) => {
                const item = document.querySelector(`[data-service="${service}"]`);
                if (item) {
                    const badge = item.querySelector('.status-badge');
                    badge.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);
                    badge.className = `status-badge ${status.status}`;
                }
            });
            
            // Update GPU info
            if (data.gpu.available) {
                document.getElementById('gpu-status').textContent = data.gpu.device_name;
                document.getElementById('gpu-name').textContent = data.gpu.device_name;
                
                const utilization = data.gpu.utilization || 0;
                document.getElementById('gpu-util').style.width = utilization + '%';
                document.getElementById('gpu-util-text').textContent = utilization + '%';
                
                const memoryPercent = (data.gpu.memory_allocated / data.gpu.memory_reserved * 100).toFixed(1);
                document.getElementById('gpu-mem').style.width = memoryPercent + '%';
                document.getElementById('gpu-mem-text').textContent = 
                    `${data.gpu.memory_allocated.toFixed(1)} / ${data.gpu.memory_reserved.toFixed(1)} GB`;
                
                // Update GPU chart
                this.updateGPUChart(utilization);
            }
            
            // Update connection status
            document.getElementById('connection-status').textContent = 'Connected';
            document.getElementById('connection-status').style.color = '#34a853';
            
        } catch (error) {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('connection-status').style.color = '#ea4335';
        }
    }

    async updateTraining() {
        try {
            const response = await fetch('http://localhost:8889/api/training');
            const data = await response.json();
            
            // Update epoch progress
            const progress = (data.current_epoch / data.total_epochs) * 100;
            const circumference = 2 * Math.PI * 50;
            const offset = circumference - (progress / 100 * circumference);
            
            const progressFill = document.getElementById('epoch-progress');
            if (progressFill) {
                progressFill.style.strokeDashoffset = offset;
            }
            
            document.getElementById('current-epoch').textContent = data.current_epoch;
            document.getElementById('total-epochs').textContent = data.total_epochs;
            
            // Update metrics
            document.getElementById('current-loss').textContent = data.current_loss.toFixed(4);
            document.getElementById('current-accuracy').textContent = (data.current_accuracy * 100).toFixed(1) + '%';
            document.getElementById('learning-rate').textContent = data.learning_rate.toExponential(2);
            document.getElementById('adaptation-count').textContent = data.adaptations_count;
            
            // Update training chart
            this.updateTrainingChart(data);
            
        } catch (error) {
            console.error('Failed to update training:', error);
        }
    }

    async updateKernelStats() {
        try {
            const response = await fetch('http://localhost:8889/api/kernels');
            const data = await response.json();
            
            document.getElementById('total-kernels').textContent = data.total_kernels;
            document.getElementById('active-kernels').textContent = data.active_kernels;
            document.getElementById('blueprints').textContent = data.total_blueprints;
            document.getElementById('cache-rate').textContent = (data.cache_hit_rate * 100).toFixed(0) + '%';
            
        } catch (error) {
            console.error('Failed to update kernel stats:', error);
        }
    }

    updateTrainingChart(data) {
        if (!this.charts.training) return;
        
        const chart = this.charts.training;
        const maxPoints = 50;
        
        // Add new data point
        chart.data.labels.push(`Epoch ${data.current_epoch}`);
        chart.data.datasets[0].data.push(data.current_loss);
        chart.data.datasets[1].data.push(data.val_loss || data.current_loss * 1.1);
        chart.data.datasets[2].data.push(data.current_accuracy * 100);
        
        // Keep only last N points
        if (chart.data.labels.length > maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        chart.update('none');
    }

    updateGPUChart(utilization) {
        if (!this.charts.gpu) return;
        
        const chart = this.charts.gpu;
        const maxPoints = 60;
        
        // Add timestamp
        const now = new Date();
        chart.data.labels.push(now.toLocaleTimeString());
        chart.data.datasets[0].data.push(utilization);
        
        // Keep only last N points
        if (chart.data.labels.length > maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none');
    }

    connectWebSocket() {
        // For now, poll logs instead of streaming
        setInterval(async () => {
            try {
                const response = await fetch('http://localhost:8889/api/logs');
                const data = await response.json();
                data.logs.forEach(log => this.addLogEntry(log));
            } catch (error) {
                console.error('Failed to fetch logs:', error);
            }
        }, 5000);
    }

    addLogEntry(log) {
        const container = document.getElementById('log-container');
        if (!container) return;
        
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        
        const time = new Date(log.timestamp).toLocaleTimeString();
        const service = log.service || log.channel?.split(':')[1] || 'system';
        const message = log.message || log.data || '';
        
        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-service">${service}</span>
            <span class="log-message">${message}</span>
        `;
        
        container.appendChild(entry);
        
        // Auto-scroll
        container.scrollTop = container.scrollHeight;
        
        // Limit log entries
        while (container.children.length > 500) {
            container.removeChild(container.firstChild);
        }
    }

    filterLogs() {
        const filter = document.getElementById('log-filter').value;
        const entries = document.querySelectorAll('.log-entry');
        
        entries.forEach(entry => {
            const service = entry.querySelector('.log-service').textContent;
            if (!filter || service === filter) {
                entry.style.display = 'flex';
            } else {
                entry.style.display = 'none';
            }
        });
    }

    clearLogs() {
        const container = document.getElementById('log-container');
        container.innerHTML = '';
    }

    toggleLogsPause() {
        this.logsPaused = !this.logsPaused;
        const btn = document.getElementById('pause-logs');
        btn.innerHTML = this.logsPaused ? 
            '<i class="fas fa-play"></i> Resume' : 
            '<i class="fas fa-pause"></i> Pause';
    }

    updateUptime() {
        const startTime = Date.now();
        
        setInterval(() => {
            const uptime = Date.now() - startTime;
            const hours = Math.floor(uptime / 3600000);
            const minutes = Math.floor((uptime % 3600000) / 60000);
            const seconds = Math.floor((uptime % 60000) / 1000);
            
            document.getElementById('uptime').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.esperApp = new EsperDemoApp();
});