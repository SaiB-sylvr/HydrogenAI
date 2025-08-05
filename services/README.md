# 🏗️ Services Directory - Microservices Architecture

## 📋 **Overview**
This directory contains all microservices that form the HydrogenAI platform. Each service is independently deployable, containerized, and communicates via well-defined APIs and event systems.

## 🏛️ **Architecture Pattern**
- **Microservices Architecture**: Each service has distinct responsibilities
- **Domain-Driven Design**: Services organized around business capabilities
- **Event-Driven Communication**: NATS/Redis event bus for loose coupling
- **API Gateway Pattern**: Kong gateway for unified API access
- **Shared Libraries**: Common functionality in `shared/` directory

## 📦 **Service Structure**

### **🎯 Orchestrator Service (`orchestrator/`)**
- **Purpose**: Main AI orchestration and query processing engine
- **Responsibilities**:
  - AI-powered query classification and routing
  - Multi-provider AI management (Groq, OpenAI, Anthropic)
  - Workflow execution and agent coordination
  - State management and caching
  - WebSocket connections for real-time communication
- **Key Components**:
  - `main.py` - Core FastAPI application (3,626 lines)
  - `agents/` - AI agent system with LangChain integration
  - `core/` - Infrastructure components (state, events, circuits)
- **Interactions**:
  - **→ MCP Server**: Tool execution via HTTP
  - **→ Redis**: Caching and state management
  - **→ NATS**: Event publishing/subscribing
  - **→ Kong**: Receives API requests
  - **→ Shared Services**: AI providers and caching

### **🛠️ MCP Server (`mcp-server/`)**
- **Purpose**: MongoDB operations and RAG (Retrieval-Augmented Generation) functionality
- **Responsibilities**:
  - MongoDB CRUD operations and aggregations
  - Document embedding and vector storage
  - RAG pipeline implementation
  - Plugin system for extensible tools
  - Database optimization and statistics
- **Key Components**:
  - `main.py` - FastAPI server with tool execution
  - `plugin_manager.py` - Dynamic plugin loading
  - `tool_registry.py` - Centralized tool management
- **Interactions**:
  - **← Orchestrator**: Receives tool execution requests
  - **→ MongoDB**: Database operations
  - **→ Qdrant**: Vector storage for RAG
  - **→ Plugin System**: Extensible tool ecosystem

### **📚 Shared Services (`shared/`)**
- **Purpose**: Common functionality shared across all services
- **Responsibilities**:
  - AI provider management with fallback logic
  - Redis-based intelligent caching system
  - Data models and validation
  - Configuration validation
  - Common utilities
- **Key Components**:
  - `ai_provider_manager.py` - Multi-provider AI orchestration
  - `ai_cache.py` - Semantic caching with conversation awareness
  - `models.py` - Pydantic data models
  - `config_validator.py` - Configuration validation
- **Interactions**:
  - **← All Services**: Import shared functionality
  - **→ External APIs**: Groq, OpenAI, Anthropic
  - **→ Redis**: Caching operations

### **🐳 Base Services (`base/`)**
- **Purpose**: Base Docker image and common dependencies
- **Responsibilities**:
  - Common Python dependencies
  - Base container configuration
  - Shared system packages
- **Usage**: Parent image for all service containers

### **🌐 Gateway Services (`gateway/`)**
- **Purpose**: Kong API gateway extensions
- **Responsibilities**:
  - Circuit breaker implementation in Lua
  - Custom Kong plugins
  - Advanced routing logic
- **Interactions**:
  - **← Kong**: Lua script execution
  - **→ All Services**: Request routing and protection

## 🔄 **Service Communication Patterns**

### **Synchronous Communication**
```
Client → Kong Gateway → Orchestrator → MCP Server
                     ↓
               Shared Services
```

### **Asynchronous Communication**
```
Orchestrator → NATS Event Bus → MCP Server
           ↓
    Redis Cache ← Shared Services
```

### **Data Flow**
```
1. API Request arrives at Kong Gateway
2. Kong routes to Orchestrator service
3. Orchestrator classifies query using AI
4. Orchestrator calls MCP Server for tools
5. MCP Server executes MongoDB/RAG operations
6. Results flow back through the chain
7. Shared Services handle caching and AI providers
```

## 🧠 **AI Intelligence Integration**

### **Multi-Provider AI System**
- **Primary**: Groq (fast, cost-effective)
- **Fallback**: OpenAI, Anthropic
- **Local**: Pattern-based classification
- **Features**: Rate limiting, automatic failover, cost optimization

### **Caching Strategy**
- **Semantic Caching**: AI-powered cache key generation
- **Conversation Cache**: Context-aware caching
- **TTL Management**: Dynamic expiration policies
- **Performance**: 3-5x query performance improvement

## 🔧 **Development Guidelines**

### **Adding New Services**
1. Create service directory following naming convention
2. Include `Dockerfile`, `requirements.txt`, `__init__.py`
3. Implement health check endpoints
4. Add service to `docker-compose.yml`
5. Configure Kong routing if needed
6. Update shared models if required

### **Service Dependencies**
- All services depend on `shared/` libraries
- Import shared components: `from services.shared import *`
- Use dependency injection for external services
- Implement graceful degradation for failures

### **Error Handling**
- Circuit breaker pattern for external calls
- Retry logic with exponential backoff
- Comprehensive logging with correlation IDs
- Health check endpoints for monitoring

## 📊 **Monitoring and Observability**

### **Health Checks**
- Each service exposes `/health` endpoint
- Dependency health verification
- Resource usage monitoring
- Service discovery support

### **Logging**
- Structured JSON logging
- Correlation ID tracking
- Error aggregation
- Performance metrics

### **Metrics**
- Prometheus metrics collection
- Service-specific KPIs
- Resource utilization tracking
- Business metrics

## 🚀 **Deployment Considerations**

### **Container Strategy**
- Multi-stage Docker builds
- Optimized layer caching
- Security-hardened base images
- Resource limits and requests

### **Scaling**
- Horizontal pod autoscaling
- Load balancer configuration
- Database connection pooling
- Cache distribution

### **Configuration**
- Environment-based configuration
- Secret management
- Feature flags
- Configuration validation

## 🔐 **Security**

### **Service-to-Service Communication**
- mTLS for production environments
- API key validation
- Request/response validation
- Rate limiting per service

### **Data Protection**
- Input sanitization
- SQL/NoSQL injection prevention
- Credential management
- Audit logging

## 📈 **Performance Optimization**

### **Caching Strategy**
- Multi-level caching (Redis, in-memory)
- Cache warming strategies
- Intelligent cache invalidation
- Cache hit ratio monitoring

### **Database Optimization**
- Connection pooling
- Query optimization
- Index management
- Read/write separation

### **Resource Management**
- Memory leak prevention
- Connection cleanup
- Graceful shutdown
- Resource pooling

---

## 🎯 **Quick Start for Developers**

1. **Understand the Architecture**: Review this README and service-specific documentation
2. **Set Up Environment**: Copy `.env.example` to `.env` and configure
3. **Start Services**: `docker-compose up -d`
4. **Verify Health**: Check all `/health` endpoints
5. **Run Tests**: Execute test suite to verify functionality
6. **Read Service Docs**: Review each service's README for specific details

## 🤝 **Contributing**

- Follow service-specific contribution guidelines
- Ensure all tests pass before submitting
- Update documentation for any architectural changes
- Maintain backward compatibility in shared services
- Use semantic versioning for service updates

This services architecture provides a robust, scalable, and maintainable foundation for the HydrogenAI platform's AI-powered data orchestration capabilities.
