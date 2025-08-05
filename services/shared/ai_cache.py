"""
AI Response Caching System
"""
import hashlib
import json
import time
import logging
from typing import Any, Dict, Optional, Union
import redis
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AIResponseCache:
    """Cache AI responses to reduce API calls"""
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.default_ttl = default_ttl
        self.redis_client = None
        self._connect()
    
    async def initialize(self):
        """Initialize the AI cache"""
        if not self.redis_client:
            self._connect()
        logger.info("✅ AI Response Cache initialized")
    
    async def cleanup(self):
        """Cleanup cache resources"""
        if self.redis_client:
            self.redis_client.close()
        logger.info("✅ AI Response Cache cleanup completed")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        if not self.redis_client:
            return {"status": "disconnected", "hits": 0, "misses": 0}
        
        try:
            info = self.redis_client.info()
            return {
                "status": "connected",
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "memory_usage": info.get("used_memory", 0)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for AI response caching")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. AI caching disabled.")
            self.redis_client = None
    
    def _generate_cache_key(self, query: str, context: Dict = None, model: str = None) -> str:
        """Generate a cache key for the query"""
        # Create a deterministic hash of the query and context
        cache_data = {
            "query": query.strip().lower(),
            "context": context or {},
            "model": model or "default"
        }
        
        # Sort to ensure consistent hashing
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        
        return f"ai_response:{cache_hash}"
    
    async def get(self, cache_key: str = None, query: str = None, context: Dict = None, model: str = None) -> Optional[Dict]:
        """Get cached AI response"""
        if not self.redis_client:
            return None
        
        try:
            # Use provided cache_key or generate from query
            if cache_key:
                key = cache_key
            else:
                key = self._generate_cache_key(query, context, model)
                
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                response_data = json.loads(cached_data)
                logger.debug(f"Cache HIT for key: {key[:20]}...")
                return response_data
            
            logger.debug(f"Cache MISS for key: {key[:20]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached response: {e}")
            return None
    
    async def set(self, cache_key: str = None, response: Dict = None, query: str = None, context: Dict = None, model: str = None, ttl: int = None) -> bool:
        """Cache AI response"""
        if not self.redis_client:
            return False
        
        try:
            # Use provided cache_key or generate from query
            if cache_key:
                key = cache_key
            else:
                key = self._generate_cache_key(query, context, model)
                
            cache_ttl = ttl or self.default_ttl
            
            # Add metadata
            cache_data = {
                "response": response,
                "cached_at": datetime.now().isoformat(),
                "query": query,
                "model": model,
                "ttl": cache_ttl
            }
            
            success = self.redis_client.setex(
                key,
                cache_ttl,
                json.dumps(cache_data)
            )
            
            if success:
                logger.debug(f"Cached AI response for key: {key[:20]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cached responses matching a pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(f"ai_response:*{pattern}*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cached responses matching pattern: {pattern}")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache pattern {pattern}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disabled", "reason": "Redis not available"}
        
        try:
            # Get all AI response cache keys
            keys = self.redis_client.keys("ai_response:*")
            
            stats = {
                "status": "enabled",
                "total_cached_responses": len(keys),
                "redis_info": {
                    "used_memory": self.redis_client.info("memory").get("used_memory_human"),
                    "connected_clients": self.redis_client.info("clients").get("connected_clients"),
                    "keyspace_hits": self.redis_client.info("stats").get("keyspace_hits"),
                    "keyspace_misses": self.redis_client.info("stats").get("keyspace_misses")
                }
            }
            
            # Calculate hit rate if available
            hits = stats["redis_info"]["keyspace_hits"] or 0
            misses = stats["redis_info"]["keyspace_misses"] or 0
            total = hits + misses
            if total > 0:
                stats["cache_hit_rate"] = round((hits / total) * 100, 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_expired(self) -> int:
        """Clear expired cache entries (Redis handles this automatically, but useful for stats)"""
        if not self.redis_client:
            return 0
        
        try:
            # This is mainly for monitoring - Redis handles TTL automatically
            keys = self.redis_client.keys("ai_response:*")
            active_keys = 0
            
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl > 0:  # Key exists and has TTL
                    active_keys += 1
            
            expired = len(keys) - active_keys
            logger.info(f"Cache status: {active_keys} active, {expired} expired entries")
            return expired
            
        except Exception as e:
            logger.error(f"Error checking expired cache entries: {e}")
            return 0

# Global cache instance
ai_cache = AIResponseCache()

# Convenience functions for easy import
def cache_ai_response(query: str, response: Dict, context: Dict = None, model: str = None, ttl: int = None) -> bool:
    """Cache an AI response"""
    return ai_cache.set(query, response, context, model, ttl)

def get_cached_ai_response(query: str, context: Dict = None, model: str = None) -> Optional[Dict]:
    """Get a cached AI response"""
    return ai_cache.get(query, context, model)

def invalidate_ai_cache(pattern: str = None) -> int:
    """Invalidate AI cache entries"""
    if pattern:
        return ai_cache.invalidate_pattern(pattern)
    else:
        # Clear all AI cache
        return ai_cache.invalidate_pattern("")

def get_ai_cache_stats() -> Dict[str, Any]:
    """Get AI cache statistics"""
    return ai_cache.get_cache_stats()

# Decorator for automatic caching
def cache_ai_call(ttl: int = 3600, cache_key_func=None):
    """Decorator to automatically cache AI function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: use function name and args
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached = get_cached_ai_response(cache_key)
            if cached:
                return cached.get("response")
            
            # Call function and cache result
            result = func(*args, **kwargs)
            if result:
                cache_ai_response(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator
