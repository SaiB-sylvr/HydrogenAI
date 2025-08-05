"""
Enhanced Configuration Management with Dynamic Timeouts and Caching
Reduces configuration complexity and provides intelligent defaults
"""
import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class TimeoutConfig:
    """Dynamic timeout configuration based on environment and operation type"""
    query_timeout: int = field(default=300)  # 5 minutes default
    health_check_timeout: int = field(default=30)
    connection_timeout: int = field(default=60)
    llm_timeout: int = field(default=120)
    aggregation_timeout: int = field(default=600)  # 10 minutes for complex operations
    
    def get_timeout(self, operation_type: str, complexity: str = "medium") -> int:
        """Get dynamic timeout based on operation and complexity"""
        base_timeouts = {
            "query": self.query_timeout,
            "health": self.health_check_timeout,
            "connection": self.connection_timeout,
            "llm": self.llm_timeout,
            "aggregation": self.aggregation_timeout
        }
        
        base = base_timeouts.get(operation_type, self.query_timeout)
        
        # Adjust based on complexity
        multipliers = {
            "simple": 0.5,
            "medium": 1.0,
            "complex": 2.0,
            "very_complex": 4.0
        }
        
        return int(base * multipliers.get(complexity, 1.0))

@dataclass
class CacheConfig:
    """Intelligent caching configuration"""
    schema_cache_ttl: int = field(default=86400)  # 24 hours
    query_cache_ttl: int = field(default=300)     # 5 minutes
    result_cache_ttl: int = field(default=3600)   # 1 hour
    state_ttl: int = field(default=7200)          # 2 hours
    
    # Cache sizes (in MB)
    schema_cache_size: int = field(default=100)
    query_cache_size: int = field(default=500)
    result_cache_size: int = field(default=1000)
    
    # Cache strategies
    enable_distributed_cache: bool = field(default=True)
    enable_local_cache: bool = field(default=True)
    cache_compression: bool = field(default=True)
    
    def get_ttl(self, cache_type: str, data_volatility: str = "medium") -> int:
        """Get dynamic TTL based on cache type and data volatility"""
        base_ttls = {
            "schema": self.schema_cache_ttl,
            "query": self.query_cache_ttl,
            "result": self.result_cache_ttl,
            "state": self.state_ttl
        }
        
        base = base_ttls.get(cache_type, self.result_cache_ttl)
        
        # Adjust based on data volatility
        volatility_multipliers = {
            "static": 10.0,    # Data rarely changes
            "low": 2.0,        # Data changes infrequently
            "medium": 1.0,     # Normal change rate
            "high": 0.5,       # Data changes frequently
            "realtime": 0.1    # Data changes constantly
        }
        
        return int(base * volatility_multipliers.get(data_volatility, 1.0))

@dataclass
class ResourceConfig:
    """Resource management configuration"""
    max_concurrent_queries: int = field(default=10)
    max_memory_usage_mb: int = field(default=2048)
    max_cpu_percentage: int = field(default=80)
    
    # Connection pools
    mongo_pool_size: int = field(default=20)
    redis_pool_size: int = field(default=10)
    
    # Request limits
    max_request_size_mb: int = field(default=10)
    max_response_size_mb: int = field(default=50)
    
    # Resource cleanup
    idle_connection_timeout: int = field(default=300)  # 5 minutes
    resource_cleanup_interval: int = field(default=600)  # 10 minutes

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = field(default=True)
    enable_tracing: bool = field(default=True)
    enable_profiling: bool = field(default=False)
    
    # Metrics collection
    metrics_interval: int = field(default=60)  # 1 minute
    health_check_interval: int = field(default=30)  # 30 seconds
    
    # Log levels by component
    log_levels: Dict[str, str] = field(default_factory=lambda: {
        "orchestrator": "INFO",
        "mcp_server": "INFO",
        "mongodb": "WARNING",
        "redis": "WARNING",
        "agents": "INFO"
    })
    
    # Alert thresholds
    error_rate_threshold: float = field(default=0.05)  # 5%
    response_time_threshold: float = field(default=10.0)  # 10 seconds
    memory_usage_threshold: float = field(default=0.85)  # 85%

class EnhancedConfigManager:
    """Enhanced configuration manager with environment-aware settings"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config_cache: Dict[str, Any] = {}
        self.watchers: List = []
        
        # Initialize configurations
        self.timeout_config = TimeoutConfig()
        self.cache_config = CacheConfig()
        self.resource_config = ResourceConfig()
        self.monitoring_config = MonitoringConfig()
        
        self._load_environment_config()
        self._apply_environment_optimizations()
    
    def _load_environment_config(self):
        """Load configuration from environment variables and files"""
        # Load from environment variables
        env_mappings = {
            "QUERY_TIMEOUT": ("timeout_config", "query_timeout"),
            "SCHEMA_CACHE_TTL": ("cache_config", "schema_cache_ttl"),
            "MAX_CONCURRENT_QUERIES": ("resource_config", "max_concurrent_queries"),
            "ENABLE_METRICS": ("monitoring_config", "enable_metrics")
        }
        
        for env_var, (config_section, config_key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, config_section)
                try:
                    # Convert to appropriate type
                    if hasattr(config_obj, config_key):
                        current_value = getattr(config_obj, config_key)
                        if isinstance(current_value, bool):
                            setattr(config_obj, config_key, value.lower() in ('true', '1', 'yes'))
                        elif isinstance(current_value, int):
                            setattr(config_obj, config_key, int(value))
                        elif isinstance(current_value, float):
                            setattr(config_obj, config_key, float(value))
                        else:
                            setattr(config_obj, config_key, value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to set {env_var}: {e}")
        
        # Load from config files
        config_paths = [
            f"config/environments/{self.environment.value}.yaml",
            "config/default.yaml",
            ".env.yaml"
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        file_config = yaml.safe_load(f)
                        self._merge_config(file_config)
                    logger.info(f"Loaded configuration from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")
    
    def _apply_environment_optimizations(self):
        """Apply environment-specific optimizations"""
        if self.environment == Environment.DEVELOPMENT:
            # Development optimizations
            self.timeout_config.query_timeout = min(self.timeout_config.query_timeout, 180)  # 3 minutes max
            self.cache_config.schema_cache_ttl = 3600  # 1 hour for faster development
            self.monitoring_config.enable_profiling = True
            self.monitoring_config.log_levels["agents"] = "DEBUG"
            
        elif self.environment == Environment.PRODUCTION:
            # Production optimizations
            self.cache_config.enable_distributed_cache = True
            self.cache_config.cache_compression = True
            self.resource_config.max_concurrent_queries = 50
            self.monitoring_config.enable_tracing = True
            self.monitoring_config.log_levels = {k: "WARNING" for k in self.monitoring_config.log_levels}
            
        elif self.environment == Environment.STAGING:
            # Staging optimizations (balance between dev and prod)
            self.resource_config.max_concurrent_queries = 20
            self.monitoring_config.enable_profiling = True
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge configuration from file"""
        for section, values in file_config.items():
            if hasattr(self, f"{section}_config"):
                config_obj = getattr(self, f"{section}_config")
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def get_timeout(self, operation_type: str, complexity: str = "medium") -> int:
        """Get dynamic timeout for operation"""
        return self.timeout_config.get_timeout(operation_type, complexity)
    
    def get_cache_ttl(self, cache_type: str, data_volatility: str = "medium") -> int:
        """Get dynamic cache TTL"""
        return self.cache_config.get_ttl(cache_type, data_volatility)
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return self.cache_config
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits"""
        return {
            "max_concurrent_queries": self.resource_config.max_concurrent_queries,
            "max_memory_mb": self.resource_config.max_memory_usage_mb,
            "max_cpu_percentage": self.resource_config.max_cpu_percentage,
            "max_request_size_mb": self.resource_config.max_request_size_mb,
            "max_response_size_mb": self.resource_config.max_response_size_mb
        }
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.monitoring_config
    
    def create_service_config(self, service_name: str) -> Dict[str, Any]:
        """Create service-specific configuration"""
        base_config = {
            "environment": self.environment.value,
            "timeouts": {
                "query": self.get_timeout("query"),
                "health": self.get_timeout("health"),
                "connection": self.get_timeout("connection")
            },
            "cache": {
                "ttl": self.get_cache_ttl("result"),
                "size_mb": self.cache_config.result_cache_size,
                "compression": self.cache_config.cache_compression
            },
            "resources": self.get_resource_limits(),
            "monitoring": {
                "log_level": self.monitoring_config.log_levels.get(service_name, "INFO"),
                "enable_metrics": self.monitoring_config.enable_metrics,
                "enable_tracing": self.monitoring_config.enable_tracing
            }
        }
        
        # Service-specific configurations
        if service_name == "orchestrator":
            base_config.update({
                "agent_timeout": self.get_timeout("llm"),
                "workflow_timeout": self.get_timeout("query", "complex"),
                "state_ttl": self.cache_config.state_ttl
            })
        
        elif service_name == "mcp_server":
            base_config.update({
                "tool_timeout": self.get_timeout("query"),
                "plugin_reload_interval": 300 if self.environment == Environment.DEVELOPMENT else 3600
            })
        
        elif service_name == "mongodb":
            base_config.update({
                "pool_size": self.resource_config.mongo_pool_size,
                "idle_timeout": self.resource_config.idle_connection_timeout,
                "aggregation_timeout": self.get_timeout("aggregation")
            })
        
        return base_config
    
    def watch_config_changes(self, callback):
        """Watch for configuration changes"""
        self.watchers.append(callback)
    
    def reload_config(self):
        """Reload configuration from sources"""
        old_config = {
            "timeout": self.timeout_config,
            "cache": self.cache_config,
            "resource": self.resource_config,
            "monitoring": self.monitoring_config
        }
        
        self._load_environment_config()
        self._apply_environment_optimizations()
        
        # Notify watchers
        for callback in self.watchers:
            try:
                callback(old_config, {
                    "timeout": self.timeout_config,
                    "cache": self.cache_config,
                    "resource": self.resource_config,
                    "monitoring": self.monitoring_config
                })
            except Exception as e:
                logger.error(f"Config change callback failed: {e}")
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            "environment": self.environment.value,
            "timeout_config": self.timeout_config.__dict__,
            "cache_config": self.cache_config.__dict__,
            "resource_config": self.resource_config.__dict__,
            "monitoring_config": self.monitoring_config.__dict__
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Validate timeouts
        if self.timeout_config.query_timeout < 10:
            issues.append("Query timeout too low (minimum 10 seconds)")
        
        if self.timeout_config.query_timeout > 1800:
            issues.append("Query timeout too high (maximum 30 minutes)")
        
        # Validate cache settings
        if self.cache_config.schema_cache_ttl < 300:
            issues.append("Schema cache TTL too low (minimum 5 minutes)")
        
        # Validate resources
        if self.resource_config.max_concurrent_queries < 1:
            issues.append("Max concurrent queries must be at least 1")
        
        if self.resource_config.max_memory_usage_mb < 512:
            issues.append("Max memory usage too low (minimum 512MB)")
        
        return issues

# Global configuration manager
def get_config_manager() -> EnhancedConfigManager:
    """Get the global configuration manager"""
    if not hasattr(get_config_manager, '_instance'):
        env = Environment(os.getenv('ENVIRONMENT', 'development'))
        get_config_manager._instance = EnhancedConfigManager(env)
    return get_config_manager._instance

# Convenience functions
def get_timeout(operation_type: str, complexity: str = "medium") -> int:
    """Get timeout for operation"""
    return get_config_manager().get_timeout(operation_type, complexity)

def get_cache_ttl(cache_type: str, data_volatility: str = "medium") -> int:
    """Get cache TTL"""
    return get_config_manager().get_cache_ttl(cache_type, data_volatility)
