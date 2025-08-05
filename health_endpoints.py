"""
Quick Health Check Endpoint for HydrogenAI
This file can be added to the orchestrator service to provide basic health monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio
import time
from datetime import datetime
from typing import Dict, Any
import psutil
import os

# Create router for health check endpoints
health_router = APIRouter(prefix="/health", tags=["health"])

class HealthChecker:
    """Health check utilities for the HydrogenAI system."""
    
    def __init__(self):
        self.start_time = time.time()
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        checks = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "status": "healthy",
            "checks": {}
        }
        
        # System resource checks
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            checks["checks"]["memory"] = {
                "status": "healthy" if memory.percent < 85 else "warning",
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            checks["checks"]["disk"] = {
                "status": "healthy" if disk.percent < 85 else "warning", 
                "usage_percent": disk.percent,
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            }
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            checks["checks"]["cpu"] = {
                "status": "healthy" if cpu_percent < 80 else "warning",
                "usage_percent": cpu_percent,
                "cores": psutil.cpu_count()
            }
            
        except Exception as e:
            checks["checks"]["system"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Environment checks
        try:
            required_env_vars = [
                "GROQ_API_KEY",
                "MONGO_URI", 
                "REDIS_URL",
                "QDRANT_HOST"
            ]
            
            env_status = {}
            for var in required_env_vars:
                env_status[var] = "present" if os.getenv(var) else "missing"
            
            checks["checks"]["environment"] = {
                "status": "healthy" if all(os.getenv(var) for var in required_env_vars) else "error",
                "variables": env_status
            }
            
        except Exception as e:
            checks["checks"]["environment"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Determine overall status
        all_checks = [check.get("status", "error") for check in checks["checks"].values()]
        if "error" in all_checks:
            checks["status"] = "unhealthy"
        elif "warning" in all_checks:
            checks["status"] = "warning"
        else:
            checks["status"] = "healthy"
            
        return checks

# Global health checker instance
health_checker = HealthChecker()

@health_router.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@health_router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics."""
    try:
        health_data = await health_checker.check_system_health()
        
        # Return appropriate HTTP status based on health
        if health_data["status"] == "unhealthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_data
            )
        elif health_data["status"] == "warning":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=health_data
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=health_data
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@health_router.get("/readiness")
async def readiness_check():
    """Kubernetes-style readiness probe."""
    # Check if all required services are ready
    try:
        # Add specific readiness checks here
        # For example: database connections, external APIs, etc.
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "database": "checking...",  # Implement actual check
                "redis": "checking...",     # Implement actual check
                "ai_providers": "checking..." # Implement actual check
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )

@health_router.get("/liveness")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    # Simple check to ensure the application is still running
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - health_checker.start_time
    }

# Usage instructions for adding to main FastAPI app:
"""
To add this to your main FastAPI application:

1. Add this to your main.py imports:
   from .health_endpoints import health_router

2. Include the router in your FastAPI app:
   app.include_router(health_router)

3. Install required dependencies:
   pip install psutil

4. Test the endpoints:
   - GET /health/ - Simple health check
   - GET /health/detailed - Detailed system metrics
   - GET /health/readiness - Readiness probe
   - GET /health/liveness - Liveness probe
"""
