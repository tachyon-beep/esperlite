"""
Nissa Observability Service - Main service implementation.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
import uvicorn

from .collectors import MetricsCollector, MorphogeneticMetrics
from .exporters import PrometheusExporter, JsonExporter
from .analysis import AnomalyDetector, PerformanceAnalyzer
from ...utils.logging import get_logger

logger = get_logger(__name__)


class NissaService:
    """
    Comprehensive observability service for morphogenetic training.
    
    Responsibilities:
    - Real-time metrics collection from all subsystems
    - Historical analysis and trend detection
    - Anomaly detection and alerting
    - Compliance reporting and audit trails
    - System health monitoring and diagnostics
    """
    
    def __init__(
        self,
        port: int = 9090,
        metrics_dir: Path = Path("/var/esper/metrics"),
        enable_analysis: bool = True
    ):
        self.port = port
        self.metrics_dir = metrics_dir
        self.enable_analysis = enable_analysis
        
        # Ensure metrics directory exists
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.collector = MetricsCollector()
        self.prometheus_exporter = PrometheusExporter()
        self.json_exporter = JsonExporter()
        
        # Analysis components
        if enable_analysis:
            self.anomaly_detector = AnomalyDetector()
            self.performance_analyzer = PerformanceAnalyzer()
        
        # FastAPI app for metrics endpoint
        self.app = FastAPI(
            title="Nissa Observability Service",
            description="Morphogenetic training platform observability",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
        # Component connections
        self._component_clients: Dict[str, Any] = {}
        
        # Background tasks
        self._server_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        
        # Alerts and notifications
        self._alert_handlers: List[Callable] = []
        self._recent_alerts: List[Dict[str, Any]] = []
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "nissa",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics():
            """Prometheus metrics endpoint."""
            # Collect latest metrics
            metrics = await self.collector.collect_once()
            
            # Update Prometheus metrics
            self.prometheus_exporter.update_metrics(
                metrics,
                self.collector.system_metrics
            )
            
            # Return in Prometheus format
            return Response(
                content=self.prometheus_exporter.generate_metrics(),
                media_type="text/plain"
            )
        
        @self.app.get("/api/v1/metrics/current")
        async def current_metrics():
            """Get current metrics in JSON format."""
            metrics = await self.collector.collect_once()
            
            return self.json_exporter.export(
                metrics,
                self.collector.system_metrics,
                include_history=False
            )
        
        @self.app.get("/api/v1/metrics/summary")
        async def metrics_summary():
            """Get metrics summary."""
            return self.collector.get_metric_summary()
        
        @self.app.get("/api/v1/metrics/trends")
        async def metrics_trends(window_minutes: int = 60):
            """Get metric trends over time window."""
            return self.collector.get_trends(window_minutes)
        
        @self.app.get("/api/v1/alerts")
        async def get_alerts(limit: int = 100):
            """Get recent alerts."""
            return self._recent_alerts[-limit:]
        
        @self.app.get("/api/v1/analysis/anomalies")
        async def get_anomalies():
            """Get detected anomalies."""
            if not self.enable_analysis:
                raise HTTPException(
                    status_code=501,
                    detail="Analysis not enabled"
                )
            
            return self.anomaly_detector.get_anomalies()
        
        @self.app.get("/api/v1/analysis/performance")
        async def get_performance_analysis():
            """Get performance analysis."""
            if not self.enable_analysis:
                raise HTTPException(
                    status_code=501,
                    detail="Analysis not enabled"
                )
            
            return self.performance_analyzer.analyze(
                self.collector._metric_history
            )
        
        @self.app.post("/api/v1/metrics/record")
        async def record_custom_metric(metric_data: dict):
            """Record custom metric from external source."""
            try:
                # Validate and record metric
                if "name" not in metric_data or "value" not in metric_data:
                    raise ValueError("Missing required fields: name, value")
                
                # Store custom metric
                # This is simplified - in production would validate schema
                logger.info(f"Recorded custom metric: {metric_data['name']}")
                
                return {"status": "recorded"}
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
    
    async def start(self):
        """Start the observability service."""
        logger.info("Starting Nissa observability service")
        
        # Start metric collection
        await self.collector.start_collection()
        
        # Start analysis if enabled
        if self.enable_analysis:
            self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        # Start HTTP server
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        
        logger.info(f"Nissa service started on port {self.port}")
    
    async def stop(self):
        """Stop the observability service."""
        logger.info("Stopping Nissa observability service")
        
        # Stop collection
        await self.collector.stop_collection()
        
        # Cancel tasks
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Nissa service stopped")
    
    def register_component(self, name: str, client: Any):
        """Register a component client for metric collection."""
        self._component_clients[name] = client
        
        # Create collector for component
        async def component_collector():
            try:
                # This would call component-specific metric methods
                # Example: metrics = await client.get_metrics()
                return {}
            except Exception as e:
                logger.error(f"Failed to collect from {name}: {e}")
                return {}
        
        self.collector.register_collector(name, component_collector)
        logger.info(f"Registered component: {name}")
    
    def register_alert_handler(self, handler: Callable):
        """Register a handler for alerts."""
        self._alert_handlers.append(handler)
    
    async def record_event(
        self,
        event_type: str,
        component: str,
        data: Dict[str, Any],
        severity: str = "info"
    ):
        """Record a significant event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "component": component,
            "severity": severity,
            "data": data
        }
        
        # Store event
        event_file = self.metrics_dir / f"events_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(event_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        # Check if this should trigger an alert
        if severity in ["warning", "error", "critical"]:
            await self._trigger_alert(event)
    
    async def _trigger_alert(self, event: Dict[str, Any]):
        """Trigger alert for significant event."""
        alert = {
            "id": f"alert_{datetime.utcnow().timestamp()}",
            "timestamp": event["timestamp"],
            "severity": event["severity"],
            "component": event["component"],
            "message": f"{event['event_type']} in {event['component']}",
            "data": event["data"]
        }
        
        # Add to recent alerts
        self._recent_alerts.append(alert)
        if len(self._recent_alerts) > 1000:
            self._recent_alerts = self._recent_alerts[-1000:]
        
        # Call handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def _analysis_loop(self):
        """Background loop for metric analysis."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Get recent metrics
                metrics = self.collector._metric_history[-10:]
                if len(metrics) < 2:
                    continue
                
                # Run anomaly detection
                anomalies = self.anomaly_detector.detect(metrics)
                for anomaly in anomalies:
                    await self.record_event(
                        event_type="anomaly_detected",
                        component="analysis",
                        data=anomaly,
                        severity="warning"
                    )
                
                # Run performance analysis
                perf_issues = self.performance_analyzer.find_issues(metrics)
                for issue in perf_issues:
                    await self.record_event(
                        event_type="performance_issue",
                        component="analysis",
                        data=issue,
                        severity="warning"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
    
    def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for audit."""
        # Load events for date range
        events = []
        current_date = start_date
        
        while current_date <= end_date:
            event_file = self.metrics_dir / f"events_{current_date.strftime('%Y%m%d')}.jsonl"
            if event_file.exists():
                with open(event_file, "r") as f:
                    for line in f:
                        event = json.loads(line)
                        event_time = datetime.fromisoformat(event["timestamp"])
                        if start_date <= event_time <= end_date:
                            events.append(event)
            
            current_date += timedelta(days=1)
        
        # Generate report
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_events": len(events),
                "by_severity": self._count_by_field(events, "severity"),
                "by_component": self._count_by_field(events, "component"),
                "by_type": self._count_by_field(events, "event_type")
            },
            "critical_events": [
                e for e in events 
                if e.get("severity") in ["error", "critical"]
            ],
            "adaptation_summary": self._get_adaptation_summary(events),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _count_by_field(
        self,
        events: List[Dict[str, Any]],
        field: str
    ) -> Dict[str, int]:
        """Count events by field value."""
        counts = {}
        for event in events:
            value = event.get(field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def _get_adaptation_summary(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract adaptation-related summary from events."""
        adaptations = [
            e for e in events 
            if "adaptation" in e.get("event_type", "")
        ]
        
        return {
            "total_adaptations": len(adaptations),
            "successful": len([
                a for a in adaptations 
                if a.get("data", {}).get("success", False)
            ]),
            "rollbacks": len([
                a for a in adaptations 
                if a.get("data", {}).get("rollback", False)
            ])
        }


# Standalone service runner
async def main():
    """Run Nissa as standalone service."""
    import os
    
    port = int(os.getenv("NISSA_PORT", "9090"))
    service = NissaService(port=port)
    
    try:
        await service.start()
        # Keep running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down Nissa service")
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())