"""
Advanced Performance Monitoring and Metrics Collection
Provides comprehensive observability and performance optimization
"""
import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json
from enum import Enum
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PerformanceStats:
    """Performance statistics for operations"""
    operation_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, duration: float, success: bool = True):
        """Update statistics with new measurement"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.avg_duration = self.total_duration / self.total_calls
        self.last_updated = datetime.utcnow()
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.performance_stats: Dict[str, PerformanceStats] = {}
        self.duration_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.max_history = max_history
        self.collection_interval = 60  # seconds
        self.active = False
        self.collection_task: Optional[asyncio.Task] = None
        self.custom_collectors: List[Callable] = []
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "memory_available": 0.0,
            "disk_usage": 0.0,
            "network_sent": 0.0,
            "network_recv": 0.0
        }
    
    async def start_collection(self):
        """Start metrics collection"""
        if self.active:
            return
        
        self.active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.active = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.active:
            try:
                await self._collect_system_metrics()
                await self._collect_custom_metrics()
                await self._cleanup_old_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric("system_cpu_usage", cpu_percent, MetricType.GAUGE)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_usage", memory.percent, MetricType.GAUGE)
            self.record_metric("system_memory_available", memory.available / 1024 / 1024, MetricType.GAUGE)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system_disk_usage", disk_percent, MetricType.GAUGE)
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric("system_network_sent", network.bytes_sent, MetricType.COUNTER)
            self.record_metric("system_network_recv", network.bytes_recv, MetricType.COUNTER)
            
            # Process-specific metrics
            process = psutil.Process()
            self.record_metric("process_memory_rss", process.memory_info().rss / 1024 / 1024, MetricType.GAUGE)
            self.record_metric("process_cpu_percent", process.cpu_percent(), MetricType.GAUGE)
            self.record_metric("process_threads", process.num_threads(), MetricType.GAUGE)
            
            # Update system metrics cache
            self.system_metrics.update({
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available / 1024 / 1024,
                "disk_usage": disk_percent,
                "network_sent": network.bytes_sent,
                "network_recv": network.bytes_recv
            })
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_custom_metrics(self):
        """Collect custom metrics from registered collectors"""
        for collector in self.custom_collectors:
            try:
                await collector(self)
            except Exception as e:
                logger.error(f"Custom metrics collector failed: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for metric_name, metric_list in self.metrics.items():
            # Keep only recent metrics
            self.metrics[metric_name] = [
                m for m in metric_list
                if m.timestamp > cutoff_time
            ][-self.max_history:]
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     labels: Dict[str, str] = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        
        # Keep only recent metrics in memory
        if len(self.metrics[name]) > self.max_history:
            self.metrics[name] = self.metrics[name][-self.max_history:]
    
    def record_duration(self, operation: str, duration: float, success: bool = True):
        """Record operation duration and update performance stats"""
        # Update performance statistics
        if operation not in self.performance_stats:
            self.performance_stats[operation] = PerformanceStats(operation)
        
        self.performance_stats[operation].update(duration, success)
        
        # Store duration for percentile calculations
        self.duration_history[operation].append(duration)
        
        # Update percentiles
        self._update_percentiles(operation)
        
        # Record as metric
        self.record_metric(f"{operation}_duration", duration, MetricType.HISTOGRAM)
        self.record_metric(f"{operation}_success", 1 if success else 0, MetricType.COUNTER)
    
    def _update_percentiles(self, operation: str):
        """Update percentile calculations"""
        durations = sorted(self.duration_history[operation])
        if len(durations) < 10:  # Need minimum samples
            return
        
        p95_idx = int(len(durations) * 0.95)
        p99_idx = int(len(durations) * 0.99)
        
        stats = self.performance_stats[operation]
        stats.p95_duration = durations[p95_idx]
        stats.p99_duration = durations[p99_idx]
    
    def register_custom_collector(self, collector: Callable):
        """Register a custom metrics collector"""
        self.custom_collectors.append(collector)
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "system_metrics": self.system_metrics,
            "performance_stats": {},
            "metric_counts": {}
        }
        
        # Performance statistics
        for op_name, stats in self.performance_stats.items():
            summary["performance_stats"][op_name] = {
                "total_calls": stats.total_calls,
                "success_rate": (stats.successful_calls / stats.total_calls * 100) if stats.total_calls > 0 else 0,
                "error_rate": stats.error_rate,
                "avg_duration": stats.avg_duration,
                "p95_duration": stats.p95_duration,
                "p99_duration": stats.p99_duration,
                "min_duration": stats.min_duration if stats.min_duration != float('inf') else 0,
                "max_duration": stats.max_duration
            }
        
        # Metric counts
        for metric_name, metric_list in self.metrics.items():
            summary["metric_counts"][metric_name] = len(metric_list)
        
        return summary
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics"""
        return {
            "system_healthy": (
                self.system_metrics["cpu_usage"] < 80 and
                self.system_metrics["memory_usage"] < 85 and
                self.system_metrics["disk_usage"] < 90
            ),
            "performance_healthy": all(
                stats.error_rate < 5.0 and stats.avg_duration < 30.0
                for stats in self.performance_stats.values()
            ),
            "system_metrics": self.system_metrics,
            "high_error_operations": [
                op_name for op_name, stats in self.performance_stats.items()
                if stats.error_rate > 5.0
            ],
            "slow_operations": [
                op_name for op_name, stats in self.performance_stats.items()
                if stats.avg_duration > 10.0
            ]
        }

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, metrics_collector: MetricsCollector):
        self.operation_name = operation_name
        self.metrics_collector = metrics_collector
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            self.success = exc_type is None
            self.metrics_collector.record_duration(
                self.operation_name, duration, self.success
            )
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            self.success = exc_type is None
            self.metrics_collector.record_duration(
                self.operation_name, duration, self.success
            )

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_thresholds = {
            "error_rate": 5.0,  # percentage
            "response_time": 10.0,  # seconds
            "memory_usage": 85.0,  # percentage
            "cpu_usage": 80.0  # percentage
        }
        self.alert_callbacks: List[Callable] = []
        self.active = False
    
    async def start(self):
        """Start performance monitoring"""
        if self.active:
            return
        
        self.active = True
        await self.metrics_collector.start_collection()
        
        # Register default collectors
        self._register_default_collectors()
        
        # Start alert monitoring
        asyncio.create_task(self._alert_monitor())
        
        logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring"""
        self.active = False
        await self.metrics_collector.stop_collection()
        logger.info("Performance monitoring stopped")
    
    def _register_default_collectors(self):
        """Register default metric collectors"""
        async def collect_async_metrics(collector: MetricsCollector):
            # Collect asyncio task metrics
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            collector.record_metric("asyncio_active_tasks", len(tasks), MetricType.GAUGE)
            
            # Collect pending tasks
            pending_tasks = [task for task in tasks if not task.done()]
            collector.record_metric("asyncio_pending_tasks", len(pending_tasks), MetricType.GAUGE)
        
        self.metrics_collector.register_custom_collector(collect_async_metrics)
    
    async def _alert_monitor(self):
        """Monitor for alert conditions"""
        while self.active:
            try:
                health_metrics = self.metrics_collector.get_health_metrics()
                
                # Check for alert conditions
                alerts = []
                
                # System resource alerts
                if health_metrics["system_metrics"]["cpu_usage"] > self.alert_thresholds["cpu_usage"]:
                    alerts.append({
                        "type": "high_cpu",
                        "value": health_metrics["system_metrics"]["cpu_usage"],
                        "threshold": self.alert_thresholds["cpu_usage"]
                    })
                
                if health_metrics["system_metrics"]["memory_usage"] > self.alert_thresholds["memory_usage"]:
                    alerts.append({
                        "type": "high_memory",
                        "value": health_metrics["system_metrics"]["memory_usage"],
                        "threshold": self.alert_thresholds["memory_usage"]
                    })
                
                # Performance alerts
                for op_name in health_metrics["high_error_operations"]:
                    alerts.append({
                        "type": "high_error_rate",
                        "operation": op_name,
                        "threshold": self.alert_thresholds["error_rate"]
                    })
                
                for op_name in health_metrics["slow_operations"]:
                    alerts.append({
                        "type": "slow_response",
                        "operation": op_name,
                        "threshold": self.alert_thresholds["response_time"]
                    })
                
                # Trigger alert callbacks
                if alerts:
                    for callback in self.alert_callbacks:
                        try:
                            await callback(alerts)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(30)
    
    @asynccontextmanager
    async def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        timer = PerformanceTimer(operation_name, self.metrics_collector)
        async with timer:
            yield timer
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     labels: Dict[str, str] = None):
        """Record a custom metric"""
        self.metrics_collector.record_metric(name, value, metric_type, labels)
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        summary = self.metrics_collector.get_metric_summary()
        health = self.metrics_collector.get_health_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary,
            "health": health,
            "recommendations": self._generate_recommendations(summary, health)
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any], 
                                 health: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        memory_usage = summary["system_metrics"]["memory_usage"]
        if memory_usage > 80:
            recommendations.append(f"High memory usage ({memory_usage:.1f}%) - consider implementing memory cleanup")
        
        # CPU recommendations
        cpu_usage = summary["system_metrics"]["cpu_usage"]
        if cpu_usage > 70:
            recommendations.append(f"High CPU usage ({cpu_usage:.1f}%) - consider optimizing compute-intensive operations")
        
        # Performance recommendations
        for op_name, stats in summary["performance_stats"].items():
            if stats["error_rate"] > 1.0:
                recommendations.append(f"High error rate in {op_name} ({stats['error_rate']:.1f}%) - investigate failures")
            
            if stats["avg_duration"] > 5.0:
                recommendations.append(f"Slow operation {op_name} (avg: {stats['avg_duration']:.2f}s) - consider optimization")
        
        return recommendations

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

# Convenience decorators
def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                async with monitor.time_operation(operation_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                with PerformanceTimer(operation_name, monitor.metrics_collector):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator
