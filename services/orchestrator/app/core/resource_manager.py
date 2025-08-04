"""
Advanced Resource Management System
Handles memory, connections, and resource cleanup automatically
"""
import asyncio
import psutil
import gc
import weakref
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    MEMORY = "memory"
    CPU = "cpu"
    CONNECTION = "connection"
    FILE_HANDLE = "file_handle"
    ASYNC_TASK = "async_task"

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    memory_usage_mb: float = 0.0
    memory_percentage: float = 0.0
    cpu_percentage: float = 0.0
    active_connections: int = 0
    open_file_handles: int = 0
    active_tasks: int = 0
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    max_memory_mb: int = 2048
    max_memory_percentage: float = 85.0
    max_cpu_percentage: float = 80.0
    max_connections: int = 100
    max_file_handles: int = 1000
    max_concurrent_tasks: int = 50
    
    # Cleanup thresholds
    memory_cleanup_threshold: float = 75.0
    connection_idle_timeout: int = 300  # 5 minutes
    task_timeout: int = 1800  # 30 minutes

class ResourceTracker:
    """Tracks and manages system resources"""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.metrics = ResourceMetrics()
        self.start_time = datetime.utcnow()
        self.tracked_resources: Dict[str, Set[Any]] = {
            resource_type.value: set() for resource_type in ResourceType
        }
        self.resource_callbacks: Dict[str, List[Callable]] = {}
        self.cleanup_tasks: List[asyncio.Task] = []
        self.monitoring_active = False
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.cleanup_tasks = [
            asyncio.create_task(self._memory_monitor()),
            asyncio.create_task(self._connection_monitor()),
            asyncio.create_task(self._task_monitor()),
            asyncio.create_task(self._periodic_cleanup())
        ]
        
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        
        for task in self.cleanup_tasks:
            task.cancel()
        
        await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
        self.cleanup_tasks.clear()
        
        logger.info("Resource monitoring stopped")
    
    async def _memory_monitor(self):
        """Monitor memory usage"""
        while self.monitoring_active:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                self.metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
                self.metrics.memory_percentage = process.memory_percent()
                self.metrics.cpu_percentage = process.cpu_percent()
                self.metrics.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
                self.metrics.last_updated = datetime.utcnow()
                
                # Check for memory pressure
                if self.metrics.memory_percentage > self.limits.memory_cleanup_threshold:
                    await self._trigger_memory_cleanup()
                
                # Check for memory limit breach
                if (self.metrics.memory_usage_mb > self.limits.max_memory_mb or 
                    self.metrics.memory_percentage > self.limits.max_memory_percentage):
                    await self._handle_memory_limit_breach()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _connection_monitor(self):
        """Monitor connection usage"""
        while self.monitoring_active:
            try:
                connections = self.tracked_resources[ResourceType.CONNECTION.value]
                self.metrics.active_connections = len(connections)
                
                # Clean up idle connections
                current_time = datetime.utcnow()
                idle_connections = []
                
                for conn in list(connections):
                    if hasattr(conn, 'last_used'):
                        if (current_time - conn.last_used).total_seconds() > self.limits.connection_idle_timeout:
                            idle_connections.append(conn)
                
                for conn in idle_connections:
                    await self._cleanup_resource(conn, ResourceType.CONNECTION)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _task_monitor(self):
        """Monitor async task usage"""
        while self.monitoring_active:
            try:
                tasks = self.tracked_resources[ResourceType.ASYNC_TASK.value]
                active_tasks = [task for task in tasks if not task.done()]
                
                self.metrics.active_tasks = len(active_tasks)
                
                # Clean up completed tasks
                completed_tasks = [task for task in tasks if task.done()]
                for task in completed_tasks:
                    self.tracked_resources[ResourceType.ASYNC_TASK.value].discard(task)
                
                # Check for task timeout
                current_time = datetime.utcnow()
                for task in active_tasks:
                    if hasattr(task, 'created_at'):
                        if (current_time - task.created_at).total_seconds() > self.limits.task_timeout:
                            logger.warning(f"Cancelling long-running task: {task}")
                            task.cancel()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Task monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _periodic_cleanup(self):
        """Periodic resource cleanup"""
        while self.monitoring_active:
            try:
                # Force garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")
                
                # Clean up weak references
                for resource_type in self.tracked_resources.values():
                    dead_refs = [ref for ref in list(resource_type) if isinstance(ref, weakref.ref) and ref() is None]
                    for ref in dead_refs:
                        resource_type.discard(ref)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures"""
        logger.info("Triggering memory cleanup due to high usage")
        
        # Run callbacks for memory cleanup
        for callback in self.resource_callbacks.get(ResourceType.MEMORY.value, []):
            try:
                await callback()
            except Exception as e:
                logger.error(f"Memory cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear internal caches if available
        for resource_set in self.tracked_resources.values():
            for resource in list(resource_set):
                if hasattr(resource, 'clear_cache'):
                    try:
                        resource.clear_cache()
                    except Exception as e:
                        logger.error(f"Cache clear failed: {e}")
    
    async def _handle_memory_limit_breach(self):
        """Handle memory limit breach"""
        logger.warning(f"Memory limit breached: {self.metrics.memory_usage_mb}MB / {self.metrics.memory_percentage}%")
        
        # Emergency cleanup
        await self._trigger_memory_cleanup()
        
        # Cancel non-essential tasks
        tasks = self.tracked_resources[ResourceType.ASYNC_TASK.value]
        for task in list(tasks):
            if hasattr(task, 'priority') and task.priority == 'low':
                task.cancel()
                tasks.discard(task)
        
        # Close idle connections
        connections = self.tracked_resources[ResourceType.CONNECTION.value]
        for conn in list(connections):
            if hasattr(conn, 'is_idle') and conn.is_idle():
                await self._cleanup_resource(conn, ResourceType.CONNECTION)
    
    def track_resource(self, resource: Any, resource_type: ResourceType, metadata: Dict[str, Any] = None):
        """Track a resource"""
        self.tracked_resources[resource_type.value].add(resource)
        
        # Add metadata if provided
        if metadata:
            if not hasattr(resource, '_resource_metadata'):
                resource._resource_metadata = {}
            resource._resource_metadata.update(metadata)
        
        # Add creation timestamp
        if not hasattr(resource, 'created_at'):
            resource.created_at = datetime.utcnow()
        
        logger.debug(f"Tracking {resource_type.value}: {type(resource).__name__}")
    
    def untrack_resource(self, resource: Any, resource_type: ResourceType):
        """Stop tracking a resource"""
        self.tracked_resources[resource_type.value].discard(resource)
        logger.debug(f"Untracked {resource_type.value}: {type(resource).__name__}")
    
    async def _cleanup_resource(self, resource: Any, resource_type: ResourceType):
        """Cleanup a specific resource"""
        try:
            # Call resource-specific cleanup
            if hasattr(resource, 'close'):
                await resource.close() if asyncio.iscoroutinefunction(resource.close) else resource.close()
            elif hasattr(resource, 'cancel'):
                resource.cancel()
            elif hasattr(resource, 'disconnect'):
                await resource.disconnect() if asyncio.iscoroutinefunction(resource.disconnect) else resource.disconnect()
            
            self.untrack_resource(resource, resource_type)
            logger.debug(f"Cleaned up {resource_type.value}: {type(resource).__name__}")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
    
    def register_cleanup_callback(self, resource_type: ResourceType, callback: Callable):
        """Register a cleanup callback for a resource type"""
        if resource_type.value not in self.resource_callbacks:
            self.resource_callbacks[resource_type.value] = []
        self.resource_callbacks[resource_type.value].append(callback)
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        return self.metrics
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of tracked resources"""
        return {
            resource_type: len(resources)
            for resource_type, resources in self.tracked_resources.items()
        }

class ManagedResource:
    """Context manager for automatic resource cleanup"""
    
    def __init__(self, resource: Any, resource_type: ResourceType, tracker: ResourceTracker):
        self.resource = resource
        self.resource_type = resource_type
        self.tracker = tracker
    
    async def __aenter__(self):
        self.tracker.track_resource(self.resource, self.resource_type)
        return self.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.tracker._cleanup_resource(self.resource, self.resource_type)

class ResourceManager:
    """Main resource management system"""
    
    def __init__(self, limits: ResourceLimits = None):
        self.limits = limits or ResourceLimits()
        self.tracker = ResourceTracker(self.limits)
        self.connection_pools: Dict[str, Any] = {}
        self.active = False
    
    async def start(self):
        """Start resource management"""
        if self.active:
            return
        
        self.active = True
        await self.tracker.start_monitoring()
        logger.info("Resource manager started")
    
    async def stop(self):
        """Stop resource management"""
        if not self.active:
            return
        
        self.active = False
        
        # Cleanup all tracked resources
        for resource_type, resources in self.tracker.tracked_resources.items():
            for resource in list(resources):
                await self.tracker._cleanup_resource(resource, ResourceType(resource_type))
        
        await self.tracker.stop_monitoring()
        logger.info("Resource manager stopped")
    
    @asynccontextmanager
    async def managed_resource(self, resource: Any, resource_type: ResourceType):
        """Context manager for automatic resource management"""
        managed = ManagedResource(resource, resource_type, self.tracker)
        async with managed as r:
            yield r
    
    def create_connection_pool(self, name: str, factory_func: Callable, 
                             min_size: int = 5, max_size: int = 20) -> 'ConnectionPool':
        """Create a managed connection pool"""
        pool = ConnectionPool(name, factory_func, min_size, max_size, self.tracker)
        self.connection_pools[name] = pool
        return pool
    
    def get_connection_pool(self, name: str) -> Optional['ConnectionPool']:
        """Get a connection pool by name"""
        return self.connection_pools.get(name)
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        return self.tracker.get_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get resource health status"""
        metrics = self.get_metrics()
        limits = self.limits
        
        return {
            "healthy": (
                metrics.memory_percentage < limits.max_memory_percentage and
                metrics.cpu_percentage < limits.max_cpu_percentage and
                metrics.active_connections < limits.max_connections
            ),
            "metrics": {
                "memory_usage_mb": metrics.memory_usage_mb,
                "memory_percentage": metrics.memory_percentage,
                "cpu_percentage": metrics.cpu_percentage,
                "active_connections": metrics.active_connections,
                "active_tasks": metrics.active_tasks,
                "uptime_seconds": metrics.uptime_seconds
            },
            "limits": {
                "max_memory_mb": limits.max_memory_mb,
                "max_memory_percentage": limits.max_memory_percentage,
                "max_cpu_percentage": limits.max_cpu_percentage,
                "max_connections": limits.max_connections
            },
            "resource_summary": self.tracker.get_resource_summary()
        }

class ConnectionPool:
    """Managed connection pool with automatic cleanup"""
    
    def __init__(self, name: str, factory_func: Callable, 
                 min_size: int, max_size: int, tracker: ResourceTracker):
        self.name = name
        self.factory_func = factory_func
        self.min_size = min_size
        self.max_size = max_size
        self.tracker = tracker
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.active_connections: Set[Any] = set()
        self.creating_lock = asyncio.Lock()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the connection pool"""
        if self.initialized:
            return
        
        async with self.creating_lock:
            if self.initialized:
                return
            
            # Create minimum connections
            for _ in range(self.min_size):
                try:
                    conn = await self.factory_func()
                    conn.last_used = datetime.utcnow()
                    await self.pool.put(conn)
                    self.tracker.track_resource(conn, ResourceType.CONNECTION)
                except Exception as e:
                    logger.error(f"Failed to create connection for pool {self.name}: {e}")
            
            self.initialized = True
            logger.info(f"Connection pool {self.name} initialized with {self.pool.qsize()} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self.initialized:
            await self.initialize()
        
        connection = None
        try:
            # Try to get an existing connection
            try:
                connection = self.pool.get_nowait()
            except asyncio.QueueEmpty:
                # Create a new connection if under max limit
                if len(self.active_connections) < self.max_size:
                    connection = await self.factory_func()
                    self.tracker.track_resource(connection, ResourceType.CONNECTION)
                else:
                    # Wait for a connection to become available
                    connection = await asyncio.wait_for(self.pool.get(), timeout=30)
            
            # Mark as active
            connection.last_used = datetime.utcnow()
            self.active_connections.add(connection)
            
            yield connection
            
        finally:
            if connection:
                # Return to pool
                self.active_connections.discard(connection)
                connection.last_used = datetime.utcnow()
                
                # Check if connection is still healthy
                if self._is_connection_healthy(connection):
                    try:
                        self.pool.put_nowait(connection)
                    except asyncio.QueueFull:
                        # Pool is full, close the connection
                        await self._close_connection(connection)
                else:
                    await self._close_connection(connection)
    
    def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if a connection is healthy"""
        try:
            if hasattr(connection, 'ping'):
                return connection.ping()
            elif hasattr(connection, 'is_connected'):
                return connection.is_connected()
            return True
        except:
            return False
    
    async def _close_connection(self, connection: Any):
        """Close a connection"""
        try:
            if hasattr(connection, 'close'):
                await connection.close() if asyncio.iscoroutinefunction(connection.close) else connection.close()
            self.tracker.untrack_resource(connection, ResourceType.CONNECTION)
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")
    
    async def close_all(self):
        """Close all connections in the pool"""
        # Close active connections
        for conn in list(self.active_connections):
            await self._close_connection(conn)
        
        # Close pooled connections
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                await self._close_connection(conn)
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Connection pool {self.name} closed")

# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """Get the global resource manager"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager

async def initialize_resource_management():
    """Initialize global resource management"""
    manager = get_resource_manager()
    await manager.start()

async def cleanup_resource_management():
    """Cleanup global resource management"""
    manager = get_resource_manager()
    await manager.stop()
