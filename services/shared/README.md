# 📚 Shared Services - Common Functionality Library

## 📋 **Overview**
The Shared Services directory contains common functionality, utilities, and libraries used across all microservices in the HydrogenAI platform. This promotes code reuse, consistency, and maintainability across the distributed architecture.

## 🏗️ **Architecture Pattern**

### **Design Principles**
- **Single Responsibility**: Each module has a focused purpose
- **Dependency Injection**: Services are injected rather than instantiated
- **Interface Segregation**: Clear interfaces for different concerns
- **Open/Closed Principle**: Extensible without modification
- **DRY (Don't Repeat Yourself)**: Common code centralized here

### **Import Pattern**
```python
# All services import shared functionality
from services.shared.ai_provider_manager import AIProviderManager
from services.shared.ai_cache import AIResponseCache
from services.shared.models import QueryRequest, QueryResponse
from services.shared.config_validator import ConfigValidator
from services.shared.utils import sanitize_input, generate_id
```

## 📁 **Module Structure**

```
shared/
├── __init__.py                  # Package initialization and exports
├── models.py                    # 📊 Pydantic data models
├── utils.py                     # 🛠️ Common utilities
├── config_validator.py          # ✅ Configuration validation
├── ai_provider_manager.py       # 🤖 AI provider orchestration
└── ai_cache.py                  # 🗄️ Intelligent caching system
```

## 🧠 **Core Components Deep Dive**

### **1. AI Provider Manager (`ai_provider_manager.py`)**
**Purpose**: Multi-provider AI orchestration with intelligent failover

**Key Features**:
- **Multi-Provider Support**: Groq, OpenAI, Anthropic integration
- **Automatic Failover**: Seamless provider switching on failures
- **Rate Limit Management**: Real-time tracking and automatic throttling
- **Cost Optimization**: Provider selection based on cost and performance
- **Usage Analytics**: Detailed provider usage statistics

**Provider Hierarchy**:
```python
class AIProviderManager:
    def __init__(self):
        self.providers = {
            "groq": GroqProvider(),      # Primary (fast, cheap)
            "openai": OpenAIProvider(),  # Fallback (reliable)
            "anthropic": AnthropicProvider()  # Emergency (diverse)
        }
        self.fallback_order = ["groq", "openai", "anthropic"]
```

**Core Methods**:
- `generate_response()`: Main AI generation with failover
- `classify_query()`: Query classification across providers
- `check_rate_limits()`: Real-time rate limit monitoring
- `get_provider_stats()`: Usage and performance analytics
- `switch_provider()`: Manual provider switching

**Integration Example**:
```python
# Used in Orchestrator service
ai_manager = AIProviderManager()
response = await ai_manager.generate_response(
    prompt="Analyze this data query",
    model_preferences=["groq", "openai"],
    fallback_enabled=True
)
```

### **2. AI Cache System (`ai_cache.py`)**
**Purpose**: Sophisticated Redis-based caching with AI-powered optimization

**Key Features**:
- **Semantic Caching**: AI-powered cache key generation based on meaning
- **Conversation Awareness**: Context-sensitive caching for multi-turn conversations
- **Dynamic TTL**: Intelligent expiration based on content type and usage patterns
- **Cache Warming**: Predictive cache population for performance
- **Hit Rate Optimization**: Continuous cache strategy improvement

**Cache Types**:
```python
class AIResponseCache:
    def __init__(self):
        self.semantic_cache = SemanticCache()      # AI-powered similarity
        self.conversation_cache = ConversationCache()  # Context-aware
        self.result_cache = ResultCache()          # Query results
        self.schema_cache = SchemaCache()          # Database schemas
```

**Caching Strategy**:
- **L1 Cache**: In-memory for hot data (microsecond access)
- **L2 Cache**: Redis for shared data (millisecond access)
- **L3 Cache**: Database query results (second access)
- **Performance**: 3-5x improvement with intelligent caching

**Advanced Features**:
- **Semantic Similarity**: Uses embeddings to find similar cached queries
- **Conversation Context**: Maintains conversation threads with proper context
- **Cache Invalidation**: Smart invalidation based on data changes
- **Performance Monitoring**: Detailed cache hit/miss analytics

### **3. Data Models (`models.py`)**
**Purpose**: Pydantic models for data validation and serialization

**Key Model Categories**:

#### **Request/Response Models**
```python
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict] = None
    preferences: Optional[Dict] = None

class QueryResponse(BaseModel):
    result: Any
    query_type: str
    execution_time: float
    provider_used: str
    cached: bool
```

#### **AI Provider Models**
```python
class ProviderConfig(BaseModel):
    name: str
    api_key: str
    base_url: str
    model_name: str
    rate_limit: int
    timeout: int

class ProviderResponse(BaseModel):
    content: str
    provider: str
    model: str
    tokens_used: int
    cost: float
```

#### **Caching Models**
```python
class CacheEntry(BaseModel):
    key: str
    value: Any
    ttl: int
    created_at: datetime
    access_count: int
    last_accessed: datetime
```

#### **Configuration Models**
```python
class ServiceConfig(BaseModel):
    name: str
    version: str
    environment: str
    debug: bool
    log_level: str
```

### **4. Configuration Validator (`config_validator.py`)**
**Purpose**: Comprehensive configuration validation and management

**Validation Categories**:
- **Environment Variables**: Required/optional variable validation
- **API Keys**: Format and accessibility validation
- **Database Connections**: Connection string and accessibility validation
- **Service Dependencies**: Service availability and health validation
- **Performance Settings**: Resource limit and timeout validation

**Key Features**:
```python
class ConfigValidator:
    def validate_environment(self) -> ValidationResult:
        """Validate all environment variables"""
        
    def validate_services(self) -> ValidationResult:
        """Validate external service connectivity"""
        
    def validate_ai_providers(self) -> ValidationResult:
        """Validate AI provider configurations"""
        
    def validate_performance_settings(self) -> ValidationResult:
        """Validate performance and resource settings"""
```

**Validation Results**:
- **Critical Errors**: Missing required configurations
- **Warnings**: Suboptimal but functional settings
- **Recommendations**: Performance optimization suggestions
- **Health Status**: Overall system health assessment

### **5. Common Utilities (`utils.py`)**
**Purpose**: Shared utility functions across all services

**Utility Categories**:

#### **Data Processing**
```python
def sanitize_input(data: str) -> str:
    """Sanitize user input for security"""

def normalize_query(query: str) -> str:
    """Normalize queries for consistency"""

def extract_entities(text: str) -> List[str]:
    """Extract named entities from text"""
```

#### **ID Generation**
```python
def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracking"""

def generate_session_id() -> str:
    """Generate secure session identifiers"""

def generate_cache_key(data: Any) -> str:
    """Generate consistent cache keys"""
```

#### **Performance Utilities**
```python
def measure_execution_time(func):
    """Decorator for measuring function execution time"""

def retry_with_backoff(max_retries: int, backoff_factor: float):
    """Decorator for retry logic with exponential backoff"""

def rate_limit(requests_per_minute: int):
    """Decorator for rate limiting function calls"""
```

#### **Security Utilities**
```python
def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for logging"""

def validate_api_key(key: str) -> bool:
    """Validate API key format"""

def encrypt_config_value(value: str) -> str:
    """Encrypt configuration values"""
```

## 🔄 **Service Integration Patterns**

### **Dependency Injection Pattern**
```python
# In service initialization
class OrchestorService:
    def __init__(self):
        self.ai_manager = AIProviderManager()
        self.cache = AIResponseCache()
        self.config = ConfigValidator()
```

### **Factory Pattern**
```python
# Service factory for consistent initialization
class SharedServiceFactory:
    @staticmethod
    def create_ai_manager() -> AIProviderManager:
        return AIProviderManager()
    
    @staticmethod
    def create_cache() -> AIResponseCache:
        return AIResponseCache()
```

### **Observer Pattern**
```python
# Event-driven updates across services
class CacheInvalidationObserver:
    def on_data_change(self, event: DataChangeEvent):
        """Invalidate relevant cache entries"""
        
    def on_config_change(self, event: ConfigChangeEvent):
        """Update configurations across services"""
```

## ⚡ **Performance Optimizations**

### **Caching Strategy**
- **Multi-Level Caching**: In-memory → Redis → Database
- **Intelligent Prefetching**: Predictive cache warming
- **Cache Clustering**: Distributed cache for scalability
- **Performance Monitoring**: Real-time cache performance tracking

### **AI Provider Optimization**
- **Response Caching**: Cache AI responses for identical queries
- **Model Selection**: Optimal model selection based on query type
- **Batch Processing**: Batch multiple requests for efficiency
- **Connection Pooling**: Persistent connections to AI providers

### **Memory Management**
- **Object Pooling**: Reuse expensive objects
- **Lazy Loading**: Load resources only when needed
- **Garbage Collection**: Proactive memory cleanup
- **Resource Monitoring**: Track memory usage patterns

## 🔐 **Security Features**

### **Data Protection**
- **Input Sanitization**: Prevent injection attacks
- **Output Filtering**: Sanitize responses before caching
- **Encryption**: Encrypt sensitive cached data
- **Access Control**: Role-based access to shared services

### **API Security**
- **Key Validation**: Validate API keys before use
- **Rate Limiting**: Prevent abuse of shared services
- **Audit Logging**: Log all shared service usage
- **Error Handling**: Secure error messages without data leakage

## 📊 **Monitoring and Observability**

### **Performance Metrics**
- Cache hit/miss ratios
- AI provider response times
- Configuration validation results
- Utility function usage statistics

### **Health Monitoring**
```python
class SharedServiceHealth:
    def check_ai_providers(self) -> HealthStatus:
        """Check AI provider connectivity"""
        
    def check_cache_health(self) -> HealthStatus:
        """Check cache performance and connectivity"""
        
    def check_configuration(self) -> HealthStatus:
        """Validate current configuration"""
```

### **Logging Strategy**
- **Structured Logging**: JSON-formatted logs
- **Correlation Tracking**: Track requests across services
- **Performance Logging**: Log execution times and resource usage
- **Error Aggregation**: Centralized error collection and analysis

## 🧪 **Testing Strategy**

### **Unit Testing**
- Individual function testing
- Mock external dependencies
- Edge case validation
- Performance benchmarking

### **Integration Testing**
- Cross-service integration
- Cache consistency testing
- AI provider failover testing
- Configuration validation testing

### **Load Testing**
- Cache performance under load
- AI provider scaling
- Memory usage patterns
- Concurrent access testing

## 🤝 **Developer Guidelines**

### **Adding New Shared Services**
1. Define clear interface in base class
2. Implement comprehensive error handling
3. Add configuration validation
4. Include performance monitoring
5. Write comprehensive tests
6. Update service documentation

### **Extending Existing Services**
1. Maintain backward compatibility
2. Add deprecation warnings for old methods
3. Update all dependent services
4. Test integration thoroughly
5. Update documentation

### **Performance Optimization**
1. Profile shared service usage
2. Identify bottlenecks and hotspots
3. Implement caching where beneficial
4. Optimize database queries
5. Monitor resource usage

---

## 🎯 **Usage Examples**

### **AI Provider Integration**
```python
from services.shared.ai_provider_manager import AIProviderManager

ai_manager = AIProviderManager()
response = await ai_manager.generate_response(
    prompt="Analyze sales data trends",
    model_preferences=["groq"],
    max_tokens=1000
)
```

### **Caching Integration**
```python
from services.shared.ai_cache import AIResponseCache

cache = AIResponseCache()
cached_result = await cache.get_semantic_match(query)
if not cached_result:
    result = await process_query(query)
    await cache.store_with_context(query, result, context)
```

### **Configuration Validation**
```python
from services.shared.config_validator import ConfigValidator

validator = ConfigValidator()
validation_result = validator.validate_all()
if not validation_result.is_valid:
    raise ConfigurationError(validation_result.errors)
```

The Shared Services layer provides a robust, efficient, and secure foundation that enables all microservices to benefit from common functionality while maintaining their independence and scalability.
