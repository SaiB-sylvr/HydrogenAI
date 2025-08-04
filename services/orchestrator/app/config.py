"""
Enhanced configuration for AI-powered Orchestrator
"""
import os

class Settings:
    # Core settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SERVICE_NAME = "orchestrator"
    
    # External services
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
    REDIS_URL = os.getenv("REDIS_URL")
    
    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    
    # AI/LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
    GROQ_API_BASE = os.getenv("GROQ_API_BASE")
    
    # Performance
    SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL", "3600"))
    QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "300"))
    RESULT_CACHE_TTL = int(os.getenv("RESULT_CACHE_TTL", "3600"))  # 1 hour
    STATE_TTL = int(os.getenv("STATE_TTL", "7200"))  # 2 hours
    
    # Circuit breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30"))

settings = Settings()