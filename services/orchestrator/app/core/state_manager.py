# services/orchestrator/app/core/state_manager.py
import asyncio
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis
import logging
import hashlib

logger = logging.getLogger(__name__)

class StateManager:
    """State management with event sourcing and schema caching"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.state_ttl = 7200  # 2 hours (was 3600)
        self.schema_cache_ttl = 86400  # 24 hours
        self._schema_cache_key = "schema:catalog"
        self._schema_version_key = "schema:version"
        self.result_ttl = 3600  # 1 hour for results
    
    async def initialize(self):
        """Initialize Redis connection"""
        import os
        
        # Get Redis URL from environment or use default
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        
        try:
            self.redis_client = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.redis_client:
            await self.redis_client.close()
    
    # ===== Request State Management =====
    
    def create_request(self) -> str:
        """Create new request ID"""
        return str(uuid.uuid4())
    
    async def update_state(self, request_id: str, updates: Dict[str, Any]):
        """Update request state with event sourcing"""
        key = f"state:{request_id}"
        
        # Get current state
        current = await self.get_state(request_id) or {}
        
        # Apply updates
        current.update(updates)
        current["updated_at"] = datetime.utcnow().isoformat()
        
        # Store state
        await self.redis_client.setex(
            key,
            self.state_ttl,
            json.dumps(current)
        )
        
        # Store event for event sourcing
        event_key = f"events:{request_id}"
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "state_update",
            "data": updates
        }
        await self.redis_client.rpush(event_key, json.dumps(event))
        await self.redis_client.expire(event_key, self.state_ttl)
        
        # Publish state change event
        await self._publish_state_change(request_id, updates)
    
    async def get_state(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get current state"""
        key = f"state:{request_id}"
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None
    
    async def get_events(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all events for a request"""
        key = f"events:{request_id}"
        events = await self.redis_client.lrange(key, 0, -1)
        return [json.loads(event) for event in events]
    
    # ===== Schema Caching =====
    
    async def get_schema_cache(self) -> Optional[Dict[str, Any]]:
        """Get cached schema with version check"""
        try:
            # Check if cache exists
            schema_data = await self.redis_client.get(self._schema_cache_key)
            if not schema_data:
                logger.info("No schema cache found")
                return None
            
            # Check cache validity
            cache_time = await self.redis_client.get(f"{self._schema_cache_key}:timestamp")
            if cache_time:
                cached_at = datetime.fromisoformat(cache_time)
                if datetime.utcnow() - cached_at > timedelta(seconds=self.schema_cache_ttl):
                    logger.info("Schema cache expired")
                    return None
            
            schema = json.loads(schema_data)
            logger.info("Schema loaded from cache")
            return schema
            
        except Exception as e:
            logger.error(f"Error reading schema cache: {e}")
            return None
    
    async def set_schema_cache(self, schema: Dict[str, Any]):
        """Cache schema with metadata"""
        try:
            # Generate version hash
            schema_str = json.dumps(schema, sort_keys=True)
            version_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:8]
            
            # Store schema
            await self.redis_client.setex(
                self._schema_cache_key,
                self.schema_cache_ttl,
                schema_str
            )
            
            # Store metadata
            await self.redis_client.setex(
                f"{self._schema_cache_key}:timestamp",
                self.schema_cache_ttl,
                datetime.utcnow().isoformat()
            )
            
            await self.redis_client.setex(
                self._schema_version_key,
                self.schema_cache_ttl,
                version_hash
            )
            
            # Store collection info for quick access
            if "collections" in schema:
                for collection in schema["collections"]:
                    await self.redis_client.setex(
                        f"schema:collection:{collection['name']}",
                        self.schema_cache_ttl,
                        json.dumps(collection)
                    )
            
            logger.info(f"Schema cached successfully (version: {version_hash})")
            
        except Exception as e:
            logger.error(f"Error caching schema: {e}")
            raise
    
    async def invalidate_schema_cache(self):
        """Invalidate schema cache"""
        try:
            keys = await self.redis_client.keys("schema:*")
            if keys:
                await self.redis_client.delete(*keys)
            logger.info("Schema cache invalidated")
        except Exception as e:
            logger.error(f"Error invalidating schema cache: {e}")
    
    async def get_collection_schema(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for specific collection"""
        try:
            data = await self.redis_client.get(f"schema:collection:{collection_name}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting collection schema: {e}")
            return None
    
    # ===== Query Result Caching =====
    
    async def get_query_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        try:
            data = await self.redis_client.get(f"query:cache:{query_hash}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error reading query cache: {e}")
            return None
    
    async def set_query_cache(self, query_hash: str, result: Dict[str, Any], ttl: int = 300):
        """Cache query result (5 min default)"""
        try:
            await self.redis_client.setex(
                f"query:cache:{query_hash}",
                ttl,
                json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Error caching query result: {e}")
    
    # ===== Workflow State =====
    
    async def save_workflow_checkpoint(self, request_id: str, checkpoint: Dict[str, Any]):
        """Save workflow checkpoint for recovery"""
        key = f"workflow:checkpoint:{request_id}"
        await self.redis_client.setex(
            key,
            self.state_ttl,
            json.dumps(checkpoint)
        )
    
    async def get_workflow_checkpoint(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow checkpoint"""
        key = f"workflow:checkpoint:{request_id}"
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None
    
    # ===== Metrics and Monitoring =====
    
    async def increment_metric(self, metric_name: str, value: int = 1):
        """Increment a metric counter"""
        key = f"metrics:{metric_name}:{datetime.utcnow().strftime('%Y%m%d')}"
        await self.redis_client.incrby(key, value)
        await self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
    
    async def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency"""
        key = f"latency:{operation}:{datetime.utcnow().strftime('%Y%m%d%H')}"
        await self.redis_client.rpush(key, latency_ms)
        await self.redis_client.expire(key, 3600 * 24)  # Keep for 24 hours
    
    # ===== Private Methods =====
    
    async def _publish_state_change(self, request_id: str, changes: Dict[str, Any]):
        """Publish state change to Redis pub/sub"""
        channel = f"state:changes:{request_id}"
        message = json.dumps({
            "request_id": request_id,
            "changes": changes,
            "timestamp": datetime.utcnow().isoformat()
        })
        await self.redis_client.publish(channel, message)
    
    # ===== Health Check =====
    
    async def health_check(self) -> Dict[str, Any]:
        """Check state manager health"""
        try:
            # Ping Redis
            await self.redis_client.ping()
            
            # Get Redis info
            info = await self.redis_client.info()
            
            # Check schema cache status
            schema_cached = await self.redis_client.exists(self._schema_cache_key)
            schema_version = await self.redis_client.get(self._schema_version_key)
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "schema_cached": bool(schema_cached),
                "schema_version": schema_version or "none"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e)
            }