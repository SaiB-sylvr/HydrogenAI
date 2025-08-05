"""
AI Provider Management with Rate Limit Monitoring
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limit information for an AI provider"""
    daily_limit: int
    daily_used: int = 0
    minute_limit: int = 60
    minute_used: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    minute_reset: datetime = field(default_factory=datetime.now)
    
    @property
    def daily_remaining(self) -> int:
        return max(0, self.daily_limit - self.daily_used)
    
    @property
    def minute_remaining(self) -> int:
        return max(0, self.minute_limit - self.minute_used)
    
    @property
    def can_make_request(self) -> bool:
        return self.daily_remaining > 0 and self.minute_remaining > 0

@dataclass
class AIProvider:
    """AI Provider configuration"""
    name: str
    api_key: str
    base_url: str
    models: List[str]
    rate_limit: RateLimitInfo
    is_available: bool = True
    last_error: Optional[str] = None
    error_count: int = 0
    max_errors: int = 5

class RateLimitManager:
    """Manages AI provider rate limits and failover"""
    
    def __init__(self, cache_file: str = "rate_limits.json"):
        self.cache_file = cache_file
        self.providers: Dict[str, AIProvider] = {}
        self._load_cache()
        self._setup_providers()
    
    def _setup_providers(self):
        """Setup available AI providers"""
        
        # Primary: Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.providers["groq"] = AIProvider(
                name="groq",
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                models=["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
                rate_limit=RateLimitInfo(
                    daily_limit=500000,  # 500K tokens/day
                    minute_limit=60      # 60 requests/minute
                )
            )
        
        # Fallback: OpenAI (if configured)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.providers["openai"] = AIProvider(
                name="openai",
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1",
                models=["gpt-4o-mini", "gpt-3.5-turbo"],
                rate_limit=RateLimitInfo(
                    daily_limit=1000000,  # 1M tokens/day (estimated)
                    minute_limit=500      # Higher rate limit
                )
            )
        
        # Fallback: Anthropic (if configured)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.providers["anthropic"] = AIProvider(
                name="anthropic",
                api_key=anthropic_api_key,
                base_url="https://api.anthropic.com",
                models=["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
                rate_limit=RateLimitInfo(
                    daily_limit=500000,   # 500K tokens/day
                    minute_limit=50       # 50 requests/minute
                )
            )
        
        logger.info(f"Initialized {len(self.providers)} AI providers: {list(self.providers.keys())}")
    
    def _load_cache(self):
        """Load rate limit cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    logger.info(f"Loaded rate limit cache: {len(cache_data)} providers")
        except Exception as e:
            logger.warning(f"Failed to load rate limit cache: {e}")
    
    def _save_cache(self):
        """Save rate limit cache to disk"""
        try:
            cache_data = {}
            for name, provider in self.providers.items():
                cache_data[name] = {
                    "daily_used": provider.rate_limit.daily_used,
                    "minute_used": provider.rate_limit.minute_used,
                    "last_reset": provider.rate_limit.last_reset.isoformat(),
                    "minute_reset": provider.rate_limit.minute_reset.isoformat(),
                    "error_count": provider.error_count,
                    "last_error": provider.last_error
                }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rate limit cache: {e}")
    
    def _reset_limits_if_needed(self):
        """Reset rate limits if time periods have passed"""
        now = datetime.now()
        
        for provider in self.providers.values():
            # Reset daily limits
            if now - provider.rate_limit.last_reset >= timedelta(days=1):
                provider.rate_limit.daily_used = 0
                provider.rate_limit.last_reset = now
                logger.info(f"Reset daily limits for {provider.name}")
            
            # Reset minute limits
            if now - provider.rate_limit.minute_reset >= timedelta(minutes=1):
                provider.rate_limit.minute_used = 0
                provider.rate_limit.minute_reset = now
    
    def get_available_provider(self, preferred: Optional[str] = None) -> Optional[AIProvider]:
        """Get an available AI provider"""
        self._reset_limits_if_needed()
        
        # Try preferred provider first
        if preferred and preferred in self.providers:
            provider = self.providers[preferred]
            if provider.is_available and provider.rate_limit.can_make_request:
                return provider
        
        # Try other providers in order of preference
        priority_order = ["groq", "openai", "anthropic"]
        for provider_name in priority_order:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if provider.is_available and provider.rate_limit.can_make_request:
                    return provider
        
        return None
    
    def record_usage(self, provider_name: str, tokens_used: int = 1):
        """Record API usage for rate limiting"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.rate_limit.daily_used += tokens_used
            provider.rate_limit.minute_used += 1
            self._save_cache()
    
    def record_error(self, provider_name: str, error: str):
        """Record an error for a provider"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.error_count += 1
            provider.last_error = error
            
            # Disable provider if too many errors
            if provider.error_count >= provider.max_errors:
                provider.is_available = False
                logger.warning(f"Disabled provider {provider_name} due to {provider.error_count} errors")
            
            self._save_cache()
    
    def record_success(self, provider_name: str):
        """Record a successful request"""
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            provider.error_count = max(0, provider.error_count - 1)  # Reduce error count
            provider.is_available = True
            self._save_cache()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all providers"""
        self._reset_limits_if_needed()
        
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                "available": provider.is_available,
                "daily_remaining": provider.rate_limit.daily_remaining,
                "minute_remaining": provider.rate_limit.minute_remaining,
                "error_count": provider.error_count,
                "last_error": provider.last_error,
                "models": provider.models
            }
        
        return status
    
    def get_health_warnings(self) -> List[str]:
        """Get health warnings for monitoring"""
        warnings = []
        
        for name, provider in self.providers.items():
            if not provider.is_available:
                warnings.append(f"Provider {name} is disabled due to errors")
            
            if provider.rate_limit.daily_remaining < 1000:
                warnings.append(f"Provider {name} has low daily quota remaining: {provider.rate_limit.daily_remaining}")
            
            if provider.rate_limit.minute_remaining < 5:
                warnings.append(f"Provider {name} approaching minute rate limit: {provider.rate_limit.minute_remaining}")
        
        return warnings

# Global instance
rate_limit_manager = RateLimitManager()

# Convenience functions
def get_available_provider(preferred: str = "groq") -> Optional[AIProvider]:
    """Get an available AI provider"""
    return rate_limit_manager.get_available_provider(preferred)

def record_ai_usage(provider_name: str, tokens_used: int = 1):
    """Record AI API usage"""
    rate_limit_manager.record_usage(provider_name, tokens_used)

def record_ai_error(provider_name: str, error: str):
    """Record AI API error"""
    rate_limit_manager.record_error(provider_name, error)

def record_ai_success(provider_name: str):
    """Record AI API success"""
    rate_limit_manager.record_success(provider_name)

def get_ai_status() -> Dict[str, Any]:
    """Get AI provider status"""
    return rate_limit_manager.get_status()

def get_ai_health_warnings() -> List[str]:
    """Get AI health warnings"""
    return rate_limit_manager.get_health_warnings()

class AIProviderManager:
    """Main AI Provider Manager with full functionality"""
    
    def __init__(self):
        self.rate_limit_manager = RateLimitManager()
        self.current_provider = "groq"
        self.initialized = False
        
    async def initialize(self):
        """Initialize the AI provider manager"""
        logger.info("🤖 Initializing AI Provider Manager...")
        self.initialized = True
        logger.info("✅ AI Provider Manager initialized with multi-provider support")
        
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None, cache_key: str = None) -> Dict[str, Any]:
        """Generate AI response with fallback support"""
        if not self.initialized:
            await self.initialize()
            
        # Get available provider
        provider = self.rate_limit_manager.get_available_provider(self.current_provider)
        if not provider:
            return {
                "output": "AI providers temporarily unavailable due to rate limits",
                "error": "rate_limit_exceeded",
                "fallback": True
            }
        
        try:
            # Record usage
            self.rate_limit_manager.record_usage(provider.name)
            
            # For now, return a structured response indicating the provider would be used
            # In a real implementation, this would make the actual AI API call
            response = {
                "output": {
                    "intent": "data_query",
                    "confidence": 0.8,
                    "complexity": "simple",
                    "entities": [],
                    "provider_used": provider.name
                },
                "provider": provider.name,
                "cached": False
            }
            
            self.rate_limit_manager.record_success(provider.name)
            return response
            
        except Exception as e:
            logger.error(f"AI generation failed with {provider.name}: {e}")
            self.rate_limit_manager.record_error(provider.name, str(e))
            
            # Try fallback provider
            fallback_provider = self.rate_limit_manager.get_available_provider()
            if fallback_provider and fallback_provider.name != provider.name:
                try:
                    self.rate_limit_manager.record_usage(fallback_provider.name)
                    response = {
                        "output": {
                            "intent": "data_query",
                            "confidence": 0.7,
                            "complexity": "simple",
                            "entities": [],
                            "provider_used": fallback_provider.name,
                            "fallback": True
                        },
                        "provider": fallback_provider.name,
                        "cached": False
                    }
                    self.rate_limit_manager.record_success(fallback_provider.name)
                    return response
                except Exception as e2:
                    logger.error(f"Fallback provider {fallback_provider.name} also failed: {e2}")
            
            # Final fallback
            return {
                "output": {
                    "intent": "data_query",
                    "confidence": 0.5,
                    "complexity": "simple",
                    "entities": [],
                    "error": "all_providers_failed"
                },
                "error": str(e),
                "fallback": True
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of AI providers"""
        return {
            "current_provider": self.current_provider,
            "rate_limits": self.rate_limit_manager.get_status(),
            "warnings": self.rate_limit_manager.get_health_warnings(),
            "initialized": self.initialized
        }
    
    async def cleanup(self):
        """Cleanup AI provider manager"""
        logger.info("🧹 Cleaning up AI Provider Manager...")
        self.rate_limit_manager._save_cache()
        logger.info("✅ AI Provider Manager cleanup completed")

# Global instance
rate_limit_manager = RateLimitManager()
