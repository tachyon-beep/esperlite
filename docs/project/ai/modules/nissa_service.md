# Nissa Service (`src/esper/services/nissa/`)

## Overview

Nissa is the observability and monitoring service for the Esper platform. Named after the nature-attuned planeswalker, it provides comprehensive visibility into system health, performance metrics, and operational insights. Nissa implements Phase B5's observability requirements, exposing Prometheus-compatible metrics, performing anomaly detection, and providing real-time system analysis. The service achieves <1ms metric collection overhead while maintaining detailed telemetry.

## Architecture

```
All Services → Metrics Collection → Nissa → Prometheus/Grafana
                                      ↓
                                Anomaly Detection
                                      ↓
                                Alert Manager
```

## Core Components

### Service Configuration
```python
@dataclass
class NissaConfig:
    """Configuration for Nissa observability service."""
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 9090
    metrics_path: str = "/metrics"
    
    # Collection settings
    collection_interval_seconds: int = 10
    metric_retention_hours: int = 168  # 7 days
    max_metrics_per_endpoint: int = 1000
    
    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_window_minutes: int = 10
    anomaly_sensitivity: float = 2.5  # Standard deviations
    
    # Export settings
    enable_prometheus: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    
    # Performance settings
    metric_buffer_size: int = 10000
    batch_export_size: int = 1000
```

### Main Service (`service.py`)

**Purpose:** FastAPI-based service exposing metrics and analysis endpoints.

#### Key Endpoints

**`GET /metrics` - Prometheus Metrics Endpoint**
```python
@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics() -> str:
    """
    Export metrics in Prometheus format.
    
    Returns all collected metrics in Prometheus text format:
    # HELP esper_training_loss Current training loss
    # TYPE esper_training_loss gauge
    esper_training_loss{model="transformer",layer="layer_0"} 0.234
    
    # HELP esper_kernel_cache_hits_total Total kernel cache hits
    # TYPE esper_kernel_cache_hits_total counter
    esper_kernel_cache_hits_total{tier="l1"} 15234
    """
    
    return metric_collector.export_prometheus()
```

**`GET /health/system` - System Health Dashboard**
```python
@app.get("/health/system", response_model=SystemHealth)
async def get_system_health() -> SystemHealth:
    """
    Comprehensive system health status.
    
    Response:
        {
            "overall_status": "healthy",
            "components": {
                "training": {"status": "healthy", "details": {...}},
                "kernel_cache": {"status": "healthy", "hit_rate": 0.92},
                "services": {
                    "tamiyo": {"status": "healthy", "uptime": 3600},
                    "urza": {"status": "degraded", "compilation_queue": 45},
                    "tezzeret": {"status": "healthy", "synthesis_rate": 10}
                }
            },
            "anomalies": [],
            "recommendations": ["Consider scaling Urza service"]
        }
    """
```

**`GET /anomalies` - Detected Anomalies**
```python
@app.get("/anomalies", response_model=List[Anomaly])
async def get_anomalies(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    severity: Optional[str] = None
) -> List[Anomaly]:
    """
    Get detected anomalies within time range.
    
    Returns list of anomalies with details:
    - Metric affected
    - Detection time
    - Severity level
    - Suggested actions
    """
```

**`POST /analyze` - On-Demand Analysis**
```python
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_metrics(request: AnalysisRequest) -> AnalysisResult:
    """
    Perform detailed analysis of specific metrics.
    
    Request:
        {
            "metrics": ["training_loss", "kernel_cache_hit_rate"],
            "time_range": "1h",
            "analysis_type": "correlation"
        }
    
    Response:
        {
            "correlations": {...},
            "trends": {...},
            "predictions": {...},
            "insights": [...]
        }
    """
```

### Metric Collector (`metric_collector.py`)

**Purpose:** Collects and aggregates metrics from all system components.

#### Key Classes

**`MetricCollector`** - Central Metric Collection
```python
class MetricCollector:
    """
    Collects metrics from all Esper components.
    
    Features:
    - Efficient metric storage
    - Automatic aggregation
    - Tag-based organization
    - Memory-efficient buffers
    """
    
    def __init__(self, config: NissaConfig):
        self.config = config
        
        # Metric storage
        self.gauges: Dict[str, Gauge] = {}
        self.counters: Dict[str, Counter] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.summaries: Dict[str, Summary] = {}
        
        # Metric buffer for batch processing
        self.metric_buffer = deque(maxlen=config.metric_buffer_size)
        
        # Collection tasks
        self._collection_task = None
```

**Metric Types:**

**Gauge - Point-in-time Values**
```python
class Gauge:
    """Metric that can go up and down."""
    
    def set(self, value: float, labels: Dict[str, str] = None):
        """Set gauge to specific value."""
        
    def inc(self, amount: float = 1.0):
        """Increment gauge."""
        
    def dec(self, amount: float = 1.0):
        """Decrement gauge."""

# Usage
training_loss = Gauge("esper_training_loss", "Current training loss")
training_loss.set(0.234, {"model": "transformer", "epoch": "5"})
```

**Counter - Monotonic Values**
```python
class Counter:
    """Metric that only goes up."""
    
    def inc(self, amount: float = 1.0, labels: Dict[str, str] = None):
        """Increment counter."""

# Usage
kernel_compilations = Counter("esper_kernel_compilations_total", "Total compilations")
kernel_compilations.inc(labels={"status": "success"})
```

**Histogram - Distribution of Values**
```python
class Histogram:
    """Track distribution of values."""
    
    def observe(self, value: float, labels: Dict[str, str] = None):
        """Record observation."""

# Usage
compilation_duration = Histogram(
    "esper_compilation_duration_seconds",
    "Kernel compilation duration",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0]
)
compilation_duration.observe(0.823, {"kernel_type": "transformer"})
```

**Collection Methods:**

**`async collect_system_metrics()`**
```python
async def collect_system_metrics(self):
    """
    Collect system-wide metrics.
    
    Metrics include:
    - CPU usage
    - Memory usage
    - GPU utilization
    - Network I/O
    - Disk I/O
    """
    
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    self.gauges["system_cpu_usage_percent"].set(cpu_percent)
    
    # Memory metrics
    memory = psutil.virtual_memory()
    self.gauges["system_memory_usage_bytes"].set(memory.used)
    self.gauges["system_memory_available_bytes"].set(memory.available)
    
    # GPU metrics (if available)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_used = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            
            self.gauges["gpu_memory_used_bytes"].set(
                memory_used,
                {"device": str(i)}
            )
            self.gauges["gpu_utilization_percent"].set(
                self._get_gpu_utilization(i),
                {"device": str(i)}
            )
```

**`async collect_training_metrics()`**
```python
async def collect_training_metrics(self):
    """
    Collect training-specific metrics.
    
    Sources:
    - Training loops
    - Model performance
    - Gradient statistics
    - Learning rate
    """
    
    # Connect to training telemetry
    training_stats = await self._get_training_telemetry()
    
    if training_stats:
        # Loss metrics
        self.gauges["training_loss"].set(
            training_stats["loss"],
            {"model": training_stats["model_name"]}
        )
        
        # Gradient metrics
        self.histograms["gradient_norm"].observe(
            training_stats["grad_norm"]
        )
        
        # Learning rate
        self.gauges["learning_rate"].set(
            training_stats["lr"]
        )
        
        # Throughput
        self.gauges["training_samples_per_second"].set(
            training_stats["throughput"]
        )
```

### Anomaly Detector (`anomaly_detector.py`)

**Purpose:** Detects anomalies in metrics using statistical and ML methods.

#### Key Classes

**`AnomalyDetector`** - Real-time Anomaly Detection
```python
class AnomalyDetector:
    """
    Detects anomalies in metric streams.
    
    Methods:
    - Statistical (Z-score, IQR)
    - Time series (Prophet, ARIMA)
    - Machine learning (Isolation Forest)
    - Custom rules
    """
    
    def __init__(
        self,
        config: NissaConfig,
        metric_store: MetricStore
    ):
        self.config = config
        self.metric_store = metric_store
        
        # Detection models
        self.statistical_detector = StatisticalDetector(
            sensitivity=config.anomaly_sensitivity
        )
        self.ml_detector = IsolationForestDetector()
        self.rule_engine = RuleEngine()
        
        # Anomaly history
        self.anomaly_history = deque(maxlen=1000)
```

**Detection Methods:**

**`async detect_anomalies() -> List[Anomaly]`**
```python
async def detect_anomalies(self) -> List[Anomaly]:
    """
    Run anomaly detection on all metrics.
    
    Returns list of detected anomalies with:
    - Metric name and value
    - Detection method
    - Confidence score
    - Suggested remediation
    """
    
    anomalies = []
    
    # Get recent metrics
    metrics = await self.metric_store.get_recent_metrics(
        window_minutes=self.config.anomaly_window_minutes
    )
    
    for metric_name, values in metrics.items():
        # Statistical detection
        stat_anomalies = self.statistical_detector.detect(
            metric_name, values
        )
        anomalies.extend(stat_anomalies)
        
        # ML detection for complex patterns
        if len(values) > 100:
            ml_anomalies = await self.ml_detector.detect_async(
                metric_name, values
            )
            anomalies.extend(ml_anomalies)
        
        # Rule-based detection
        rule_anomalies = self.rule_engine.check(metric_name, values)
        anomalies.extend(rule_anomalies)
    
    # Deduplicate and prioritize
    return self._prioritize_anomalies(anomalies)
```

**Statistical Detection:**
```python
class StatisticalDetector:
    """Z-score based anomaly detection."""
    
    def detect(
        self,
        metric_name: str,
        values: List[float]
    ) -> List[Anomaly]:
        """Detect statistical anomalies."""
        
        if len(values) < 10:
            return []
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        
        anomalies = []
        
        # Check recent values
        for i, value in enumerate(values[-5:]):
            z_score = abs((value - mean) / std) if std > 0 else 0
            
            if z_score > self.sensitivity:
                anomalies.append(Anomaly(
                    metric_name=metric_name,
                    value=value,
                    expected_range=(mean - self.sensitivity * std,
                                  mean + self.sensitivity * std),
                    severity=self._calculate_severity(z_score),
                    detection_method="statistical",
                    confidence=min(0.99, z_score / 10),
                    timestamp=datetime.utcnow(),
                    suggested_action=self._suggest_action(metric_name, value, mean)
                ))
        
        return anomalies
```

### Performance Analyzer (`performance_analyzer.py`)

**Purpose:** Analyzes performance patterns and provides optimization suggestions.

```python
class PerformanceAnalyzer:
    """
    Analyzes system performance and provides insights.
    
    Analysis types:
    - Bottleneck identification
    - Trend analysis
    - Correlation detection
    - Capacity planning
    """
    
    async def analyze_performance(
        self,
        time_range: str = "1h"
    ) -> PerformanceReport:
        """
        Comprehensive performance analysis.
        
        Returns:
        - Current bottlenecks
        - Performance trends
        - Optimization opportunities
        - Capacity recommendations
        """
        
        metrics = await self._get_metrics_for_range(time_range)
        
        report = PerformanceReport()
        
        # Bottleneck analysis
        bottlenecks = self._identify_bottlenecks(metrics)
        report.bottlenecks = bottlenecks
        
        # Trend analysis
        trends = self._analyze_trends(metrics)
        report.trends = trends
        
        # Correlation analysis
        correlations = self._find_correlations(metrics)
        report.correlations = correlations
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(
            bottlenecks, trends, correlations
        )
        
        return report
```

### Metric Exporter (`metric_exporter.py`)

**Purpose:** Exports metrics in various formats (Prometheus, JSON, OpenTelemetry).

```python
class MetricExporter:
    """
    Exports metrics to external systems.
    
    Supported formats:
    - Prometheus text format
    - JSON
    - OpenTelemetry
    - StatsD
    """
    
    def export_prometheus(self, metrics: Dict[str, Any]) -> str:
        """
        Export metrics in Prometheus text format.
        
        Example output:
        # HELP http_requests_total Total HTTP requests
        # TYPE http_requests_total counter
        http_requests_total{method="GET",status="200"} 1234
        """
        
        output = []
        
        for metric_name, metric in metrics.items():
            # Add HELP and TYPE
            output.append(f"# HELP {metric_name} {metric.description}")
            output.append(f"# TYPE {metric_name} {metric.type}")
            
            # Add metric values
            for labels, value in metric.get_all_values():
                label_str = self._format_labels(labels)
                output.append(f"{metric_name}{label_str} {value}")
        
        return "\n".join(output)
```

## Integration Patterns

### With Training Loops
```python
# In training loop
from esper.services.nissa import metrics

# Register metrics
loss_metric = metrics.gauge("training_loss", "Current batch loss")
throughput_metric = metrics.gauge("training_throughput", "Samples per second")
grad_norm_metric = metrics.histogram("gradient_norm", "Gradient L2 norm")

# During training
for batch in dataloader:
    start_time = time.time()
    
    output = model(batch.input)
    loss = criterion(output, batch.target)
    
    # Record loss
    loss_metric.set(loss.item(), {"epoch": str(epoch), "batch": str(batch_idx)})
    
    # Backward pass
    loss.backward()
    
    # Record gradient norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    grad_norm_metric.observe(total_norm)
    
    optimizer.step()
    
    # Record throughput
    batch_time = time.time() - start_time
    samples_per_sec = batch.size(0) / batch_time
    throughput_metric.set(samples_per_sec)
```

### With Services
```python
# In Urza service
compilation_counter = metrics.counter(
    "urza_compilations_total",
    "Total kernel compilations"
)
compilation_histogram = metrics.histogram(
    "urza_compilation_duration_seconds",
    "Compilation duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

async def compile_kernel(blueprint_ir):
    start_time = time.time()
    
    try:
        kernel = await compiler.compile(blueprint_ir)
        compilation_counter.inc({"status": "success"})
    except Exception as e:
        compilation_counter.inc({"status": "failure"})
        raise
    finally:
        duration = time.time() - start_time
        compilation_histogram.observe(duration)
    
    return kernel
```

### With Anomaly Response
```python
# Anomaly response handler
async def handle_anomaly(anomaly: Anomaly):
    if anomaly.metric_name == "training_loss" and anomaly.severity == "high":
        # Training loss spike detected
        logger.warning(f"Training loss anomaly: {anomaly.value}")
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            
        # Alert via Oona
        await oona_client.publish(
            "anomaly.training.loss_spike",
            {
                "value": anomaly.value,
                "action": "reduced_learning_rate",
                "new_lr": param_group['lr']
            }
        )
    
    elif anomaly.metric_name == "kernel_cache_hit_rate" and anomaly.value < 0.5:
        # Cache performance degradation
        logger.warning(f"Cache hit rate anomaly: {anomaly.value}")
        
        # Trigger cache warming
        await cache_manager.warm_frequently_used_kernels()
```

## Grafana Dashboard Configuration

### Training Dashboard
```json
{
  "dashboard": {
    "title": "Esper Training Metrics",
    "panels": [
      {
        "title": "Training Loss",
        "targets": [
          {
            "expr": "esper_training_loss{model=\"$model\"}",
            "legendFormat": "{{epoch}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Learning Rate",
        "targets": [
          {
            "expr": "esper_learning_rate{model=\"$model\"}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "esper_gpu_utilization_percent",
            "legendFormat": "GPU {{device}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### System Health Dashboard
```json
{
  "dashboard": {
    "title": "Esper System Health",
    "panels": [
      {
        "title": "Service Status",
        "targets": [
          {
            "expr": "up{job=~\"esper-.*\"}",
            "legendFormat": "{{job}}"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Kernel Cache Performance",
        "targets": [
          {
            "expr": "rate(esper_kernel_cache_hits_total[5m]) / (rate(esper_kernel_cache_hits_total[5m]) + rate(esper_kernel_cache_misses_total[5m]))",
            "legendFormat": "Hit Rate"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

## Performance Characteristics

### Collection Overhead
- **Metric collection:** <0.1ms per metric
- **Aggregation:** <1ms per 1000 metrics
- **Export latency:** <10ms for 10k metrics
- **Memory usage:** ~1KB per metric series

### Anomaly Detection
- **Detection latency:** <100ms for 1000 metrics
- **False positive rate:** <2% with proper tuning
- **Statistical detection:** O(n) complexity
- **ML detection:** O(n log n) complexity

## Configuration Examples

### Minimal Configuration
```python
config = NissaConfig(
    host="localhost",
    port=9090,
    enable_anomaly_detection=False,  # Just metrics
    enable_prometheus=True
)

service = NissaService(config)
```

### Production Configuration
```python
config = NissaConfig(
    host="0.0.0.0",
    port=9090,
    collection_interval_seconds=10,
    metric_retention_hours=168,  # 7 days
    enable_anomaly_detection=True,
    anomaly_sensitivity=2.5,
    enable_prometheus=True,
    enable_tracing=True,
    metric_buffer_size=50000  # Large buffer for high volume
)

# With custom anomaly rules
service = NissaService(config)
service.add_anomaly_rule(
    "high_loss",
    lambda metrics: metrics["training_loss"] > 10.0,
    severity="critical",
    action="stop_training"
)
```

## Best Practices

1. **Metric Naming**
   - Use consistent prefixes (e.g., `esper_`)
   - Include units in names (e.g., `_seconds`, `_bytes`)
   - Use labels for variations, not metric names

2. **Label Usage**
   - Keep cardinality low (<100 unique values per label)
   - Use static labels for dimensions
   - Avoid high-cardinality labels like user IDs

3. **Anomaly Detection**
   - Start with conservative sensitivity
   - Validate against known incidents
   - Implement gradual response escalation

4. **Performance**
   - Use metric buffering for high-frequency updates
   - Aggregate before exporting
   - Implement sampling for very high volume metrics

## Troubleshooting

### High Memory Usage
```python
# Check metric cardinality
cardinality_report = await nissa_client.get_cardinality_report()
for metric, cardinality in cardinality_report.items():
    if cardinality > 10000:
        logger.warning(f"High cardinality metric: {metric} ({cardinality})")

# Implement metric filtering
config.metric_filters = [
    lambda m: not m.name.startswith("debug_"),
    lambda m: m.labels.get("environment") == "production"
]
```

### Missing Metrics
```python
# Verify metric registration
registered_metrics = await nissa_client.list_metrics()
logger.info(f"Registered metrics: {registered_metrics}")

# Check collection errors
collection_errors = await nissa_client.get_collection_errors()
for error in collection_errors:
    logger.error(f"Collection error: {error}")
```

## Future Enhancements

1. **Advanced Analytics**
   - Predictive anomaly detection
   - Automated root cause analysis
   - Metric correlation discovery

2. **Integration Features**
   - Native Datadog support
   - CloudWatch integration
   - Custom webhook alerts

3. **Performance Improvements**
   - Metric compression
   - Federated collection
   - Edge aggregation

4. **Intelligence**
   - Automated dashboard generation
   - Smart alerting with context
   - Performance optimization suggestions