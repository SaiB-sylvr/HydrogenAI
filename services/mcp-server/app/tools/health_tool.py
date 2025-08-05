"""
Health Check Tool - Demonstrates MCP server tool implementation
"""
import asyncio
from typing import Dict, Any
from datetime import datetime
import logging
import psutil
import platform

logger = logging.getLogger(__name__)

try:
    # Import the base Tool class from our tool registry
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from tool_registry import Tool
except ImportError:
    # Fallback Tool class if import fails
    class Tool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
            self.parameters = {}

class HealthCheckTool(Tool):
    """Tool for checking system health and status"""
    
    def __init__(self):
        super().__init__(
            name="health_check",
            description="Check system health, resource usage, and service status"
        )
        self.parameters = {
            "detailed": {
                "type": "boolean",
                "description": "Whether to include detailed system metrics",
                "default": False
            },
            "components": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific components to check (optional)",
                "default": ["cpu", "memory", "disk"]
            }
        }
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute health check"""
        try:
            detailed = params.get("detailed", False)
            components = params.get("components", ["cpu", "memory", "disk"])
            
            health_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy",
                "service": "mcp-server",
                "version": "1.0.0",
                "uptime": self._get_uptime(),
                "system": {
                    "platform": platform.system(),
                    "python_version": platform.python_version()
                }
            }
            
            # Add component-specific health checks
            if "cpu" in components:
                health_data["cpu"] = await self._check_cpu(detailed)
            
            if "memory" in components:
                health_data["memory"] = await self._check_memory(detailed)
            
            if "disk" in components:
                health_data["disk"] = await self._check_disk(detailed)
            
            # Determine overall health status
            health_data["status"] = self._determine_status(health_data)
            
            logger.info(f"Health check completed: {health_data['status']}")
            return health_data
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e),
                "service": "mcp-server"
            }
    
    async def _check_cpu(self, detailed: bool) -> Dict[str, Any]:
        """Check CPU metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_data = {
            "usage_percent": cpu_percent,
            "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical"
        }
        
        if detailed:
            cpu_data.update({
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            })
        
        return cpu_data
    
    async def _check_memory(self, detailed: bool) -> Dict[str, Any]:
        """Check memory metrics"""
        memory = psutil.virtual_memory()
        memory_data = {
            "usage_percent": memory.percent,
            "available_gb": round(memory.available / (1024**3), 2),
            "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical"
        }
        
        if detailed:
            memory_data.update({
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "free_gb": round(memory.free / (1024**3), 2)
            })
        
        return memory_data
    
    async def _check_disk(self, detailed: bool) -> Dict[str, Any]:
        """Check disk metrics"""
        disk = psutil.disk_usage('/')
        disk_data = {
            "usage_percent": round((disk.used / disk.total) * 100, 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "status": "healthy" if disk.used / disk.total < 0.8 else "warning" if disk.used / disk.total < 0.95 else "critical"
        }
        
        if detailed:
            disk_data.update({
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2)
            })
        
        return disk_data
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            return f"{days}d {hours}h {minutes}m"
        except:
            return "unknown"
    
    def _determine_status(self, health_data: Dict[str, Any]) -> str:
        """Determine overall health status"""
        statuses = []
        
        # Check component statuses
        for component in ["cpu", "memory", "disk"]:
            if component in health_data:
                statuses.append(health_data[component].get("status", "unknown"))
        
        # Determine overall status
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"

class SystemInfoTool(Tool):
    """Tool for getting detailed system information"""
    
    def __init__(self):
        super().__init__(
            name="system_info",
            description="Get detailed system and environment information"
        )
        self.parameters = {
            "include_env": {
                "type": "boolean",
                "description": "Whether to include environment variables (filtered)",
                "default": False
            }
        }
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute system info collection"""
        try:
            include_env = params.get("include_env", False)
            
            info = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "platform": platform.platform(),
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "python_build": platform.python_build(),
                    "python_compiler": platform.python_compiler()
                },
                "resources": {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                    "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
                }
            }
            
            if include_env:
                # Only include safe environment variables
                safe_env_vars = {}
                safe_keys = ["PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "TZ"]
                for key in safe_keys:
                    if key in os.environ:
                        safe_env_vars[key] = os.environ[key]
                info["environment"] = safe_env_vars
            
            return info
            
        except Exception as e:
            logger.error(f"System info collection failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

# Export tools for plugin registration
TOOLS = [HealthCheckTool(), SystemInfoTool()]
