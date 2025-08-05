# 🛠️ MCP Server - MongoDB Operations & RAG Engine

## 📋 **Service Overview**
The MCP (MongoDB Control Plane) Server is a specialized microservice responsible for all database operations, RAG (Retrieval-Augmented Generation) functionality, and extensible tool management. It serves as the data layer gateway for the HydrogenAI platform.

## 🏗️ **Architecture**

### **Core Responsibilities**
- **MongoDB Operations**: CRUD, aggregations, indexing, statistics
- **RAG Pipeline**: Document embedding, vector storage, semantic search
- **Tool Management**: Dynamic plugin system for extensible functionality
- **Data Processing**: Large dataset handling with streaming and pagination
- **Vector Operations**: Qdrant integration for semantic search

### **Technical Stack**
- **Framework**: FastAPI (async Python web framework)
- **Database**: MongoDB with Motor (async driver)
- **Vector DB**: Qdrant for embeddings and semantic search
- **ML/AI**: Sentence Transformers for embeddings
- **Plugin System**: Dynamic Python plugin loading
- **Containerization**: Docker with ML model optimization

## 📁 **Directory Structure**

```
mcp-server/
├── Dockerfile              # Container with ML dependencies
├── requirements.txt         # Python dependencies (ML heavy)
├── __init__.py             # Package initialization
└── app/                    # Main application
    ├── __init__.py         # App package init
    ├── main.py             # 🎯 CORE SERVER (MongoDB + RAG tools)
    ├── plugin_manager.py   # Dynamic plugin system
    ├── tool_registry.py    # Centralized tool management
    ├── tools/              # Built-in tools
    │   ├── __init__.py
    │   └── health_tool.py  # System health monitoring
    └── plugins/            # Plugin ecosystem
        └── system/         # System monitoring plugin
            ├── __init__.py
            └── plugin.yaml # Plugin configuration
```

## 🧠 **Core Components Deep Dive**

### **1. Main Server (`main.py`)**
**Purpose**: FastAPI server with comprehensive tool execution engine

**Key Classes**:
- `MongoDBTools`: Complete MongoDB operations suite
- `MemoryTools`: Fallback tools for resilience
- `ExecuteRequest`: Tool execution request model

**Critical Features**:
- **21 Total Tools**: 15 MCP tools + 6 RAG tools
- Streaming support for large datasets
- Timeout protection for long operations
- Plugin integration and management
- Health monitoring and diagnostics

**Tool Categories**:
```python
# MongoDB Tools (6 core operations)
mongodb_find          # Query documents with pagination
mongodb_count         # Count documents efficiently  
mongodb_aggregate     # Complex aggregation pipelines
mongodb_list_collections  # Collection discovery
mongodb_database_stats    # Database statistics
mongodb_collection_stats  # Collection analytics

# Memory Tools (fallback)
answer_question       # In-memory question answering

# Plugin Tools (dynamic loading)
# RAG Tools, System Tools, Custom Tools
```

### **2. Plugin Manager (`plugin_manager.py`)**
**Purpose**: Dynamic plugin loading and lifecycle management

**Key Features**:
- Automatic plugin discovery from `/app/plugins/`
- YAML-based plugin configuration
- Hot-reload capability for development
- Plugin dependency management
- Error isolation for plugin failures

**Plugin Structure**:
```yaml
# Example plugin.yaml
name: mongodb
version: 1.0.0
description: MongoDB operations plugin
class: MongoDBPlugin
author: HydrogenAI
category: database

tools:
  - name: create_index
    description: Create database index
    type: optimization
```

### **3. Tool Registry (`tool_registry.py`)**
**Purpose**: Centralized tool registration and discovery

**Features**:
- Tool schema validation
- Capability discovery
- Tool versioning
- Performance monitoring
- Usage analytics

## 🔄 **Service Interactions**

### **Inbound Communications**
```
Orchestrator → MCP Server
- Tool execution requests on port 8000
- Tool discovery queries
- Health checks
- Plugin management operations
```

### **Outbound Communications**
```
MCP Server → MongoDB
- Database queries and operations
- Index management
- Statistics collection

MCP Server → Qdrant
- Vector storage operations
- Semantic search queries
- Embedding management

MCP Server → Plugin System
- Dynamic tool loading
- Plugin lifecycle management
```

## 📊 **Tool Execution Flow**

### **1. Tool Request Processing**
```python
@app.post("/execute")
async def execute_tool(request: ExecuteRequest):
    # 1. Validate tool request
    tool_name = request.tool
    params = request.params
    
    # 2. Check tool availability
    if tool_name not in app_state["tools"]:
        # Check plugin tools
        if app_state.get("tool_registry"):
            plugin_tool = app_state["tool_registry"].get_tool(tool_name)
            if plugin_tool:
                return await plugin_tool.execute(params)
    
    # 3. Execute legacy tool
    tool_func = app_state["tools"][tool_name]
    result = await tool_func(params)
    
    return {"success": True, "result": result}
```

### **2. MongoDB Operations**
- **Connection Management**: Async MongoDB client with connection pooling
- **Query Optimization**: Efficient querying with proper indexing
- **Large Dataset Handling**: Streaming and pagination for memory efficiency
- **Error Handling**: Comprehensive error catching and reporting

### **3. RAG Pipeline**
```python
# RAG workflow (conceptual)
async def rag_query(query: str, collection: str):
    # 1. Generate query embedding
    embedding = await generate_embedding(query)
    
    # 2. Vector similarity search
    similar_docs = await qdrant_search(embedding, collection)
    
    # 3. Retrieve full documents
    documents = await mongodb_find(similar_docs.ids)
    
    # 4. Generate response using retrieved context
    response = await ai_generate_response(query, documents)
    
    return response
```

## 🧠 **MongoDB Tool Suite**

### **Core Database Operations**

#### **1. Document Queries (`mongodb_find`)**
```python
async def find(params: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced document querying with pagination"""
    # Features:
    # - Efficient pagination (skip/limit)
    # - Complex query filters
    # - Projection for specific fields
    # - Sort optimization
    # - Memory management for large results
```

#### **2. Aggregation Pipeline (`mongodb_aggregate`)**
```python
async def aggregate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Complex aggregation with allowDiskUse"""
    # Features:
    # - Multi-stage pipelines
    # - Cross-collection joins ($lookup)
    # - Advanced analytics
    # - Memory optimization (allowDiskUse)
    # - Performance monitoring
```

#### **3. Statistics and Analytics**
- **Database Stats**: Storage size, collection count, index usage
- **Collection Stats**: Document count, average size, index efficiency
- **Performance Metrics**: Query execution times, resource usage

#### **4. Index Management** (via plugins)
- Dynamic index creation
- Compound index optimization
- Index usage analytics
- Performance impact assessment

## 📚 **RAG (Retrieval-Augmented Generation) System**

### **Document Processing Pipeline**
1. **Document Ingestion**: Text extraction and preprocessing
2. **Embedding Generation**: Sentence Transformers for vector creation
3. **Vector Storage**: Qdrant for efficient similarity search
4. **Retrieval**: Semantic search based on query embeddings
5. **Generation**: AI-powered response using retrieved context

### **RAG Tools** (via plugins)
```
rag_add_document      # Add documents to vector store
rag_search_documents  # Semantic document search
rag_update_document   # Update existing documents
rag_delete_document   # Remove documents
rag_list_documents    # Browse document collection
rag_query_with_context # Full RAG query with generation
```

### **Vector Database Integration**
- **Qdrant Client**: Async vector database operations
- **Collection Management**: Automatic collection creation and management
- **Embedding Models**: Configurable embedding models (default: all-MiniLM-L6-v2)
- **Similarity Search**: Cosine similarity with configurable thresholds

## ⚡ **Performance Optimizations**

### **Database Optimizations**
- **Connection Pooling**: Efficient MongoDB connection management
- **Query Optimization**: Proper indexing and query planning
- **Pagination**: Memory-efficient large dataset handling
- **Async Operations**: Non-blocking database operations

### **Memory Management**
- **Streaming Results**: Large dataset streaming to prevent memory overflow
- **Connection Cleanup**: Automatic resource cleanup
- **Model Caching**: ML model caching for performance
- **Garbage Collection**: Proactive memory management

### **Caching Strategy**
- **Query Result Caching**: Redis-based result caching
- **Model Caching**: In-memory embedding model caching
- **Vector Caching**: Qdrant-based vector caching
- **Schema Caching**: Database schema caching

## 🔌 **Plugin System Architecture**

### **Plugin Discovery**
```python
class PluginManager:
    def load_plugins(self, plugin_dir: str):
        """Dynamically load plugins from directory"""
        # 1. Scan for plugin.yaml files
        # 2. Validate plugin configuration
        # 3. Load plugin classes
        # 4. Register tools with registry
        # 5. Handle plugin dependencies
```

### **Plugin Types**
- **Database Plugins**: MongoDB, PostgreSQL, Redis tools
- **AI/ML Plugins**: RAG tools, embedding tools, model tools
- **System Plugins**: Health monitoring, performance tools
- **Custom Plugins**: Domain-specific business tools

### **Plugin Configuration**
```yaml
# Plugin metadata
name: rag
version: 2.0.0
description: Retrieval-Augmented Generation tools
class: RAGPlugin
author: HydrogenAI
category: ai

# Tool definitions
tools:
  - name: rag_query
    description: Query documents with AI generation
    type: generation
    timeout: 60

# Plugin dependencies
requirements:
  - sentence-transformers
  - qdrant-client
```

## 🔧 **Configuration Management**

### **Environment Variables**
```bash
# Database configuration
MONGO_URI=mongodb://...
MONGO_DB_NAME=database_name

# Vector database
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hydrogen_documents

# ML configuration
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DISABLE_QDRANT=false

# Performance tuning
TOOL_TIMEOUT=30
PLUGIN_RELOAD_INTERVAL=300
```

### **Dynamic Configuration**
- Runtime configuration updates
- Plugin hot-reloading
- Model switching
- Performance auto-tuning

## 🚀 **Deployment Considerations**

### **Container Optimization**
```dockerfile
# ML-optimized container
FROM python:3.11-slim
# Install ML dependencies
RUN pip install torch sentence-transformers
# Optimize for production
ENV TRANSFORMERS_CACHE=/app/cache
```

### **Resource Requirements**
- **CPU**: 2+ cores for ML operations
- **Memory**: 4GB+ for embedding models
- **Storage**: SSD for vector database performance
- **Network**: Low latency to MongoDB and Qdrant

### **Scaling Strategy**
- **Horizontal Scaling**: Multiple MCP server instances
- **Load Balancing**: Round-robin with health checks
- **Database Sharding**: MongoDB cluster support
- **Vector Partitioning**: Qdrant collection partitioning

## 🔐 **Security Features**

### **Input Validation**
- **NoSQL Injection Prevention**: Parameter sanitization
- **Schema Validation**: Pydantic model validation
- **Rate Limiting**: Per-tool execution limits
- **Request Size Limits**: Prevent memory exhaustion

### **Access Control**
- **Tool Permissions**: Role-based tool access
- **Database Security**: Connection string encryption
- **Plugin Isolation**: Sandboxed plugin execution
- **Audit Logging**: Comprehensive operation logging

## 📊 **Monitoring and Health**

### **Health Checks**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mongodb": "connected" if mongo_client else "disconnected",
        "tools_loaded": len(app_state["tools"]),
        "plugins": plugin_manager.get_status(),
        "total_tools": total_tools
    }
```

### **Metrics Collection**
- Tool execution times
- Database operation performance
- Plugin usage statistics
- Error rates and patterns
- Resource utilization

## 🧪 **Testing and Development**

### **Test Coverage**
- Unit tests for individual tools
- Integration tests for database operations
- Plugin system tests
- Performance benchmarks
- Load testing for concurrent operations

### **Development Tools**
- Local MongoDB with Docker
- Qdrant local instance
- Plugin development framework
- Performance profiling tools

## 🤝 **Developer Guidelines**

### **Adding New Tools**
1. Create tool function in appropriate module
2. Register tool in `register_tools()` function
3. Add tool documentation and schema
4. Implement comprehensive error handling
5. Add unit and integration tests

### **Creating Plugins**
1. Create plugin directory with `plugin.yaml`
2. Implement plugin class with required methods
3. Define tool schemas and documentation
4. Test plugin loading and execution
5. Add plugin to plugin registry

### **Performance Optimization**
1. Profile tool execution times
2. Optimize database queries and indexes
3. Implement caching where beneficial
4. Monitor memory usage patterns
5. Use async/await for I/O operations

---

## 🎯 **Quick Start for Developers**

1. **Environment Setup**: Configure MongoDB and Qdrant connections
2. **Dependencies**: Install ML dependencies with `pip install -r requirements.txt`
3. **Development**: Use `docker-compose up mcp-server` for isolated testing
4. **Plugin Development**: Create plugins in `/app/plugins/` directory
5. **Testing**: Run comprehensive test suite including database operations

The MCP Server represents a sophisticated data layer that combines traditional database operations with modern AI capabilities, providing a robust foundation for the HydrogenAI platform's data processing needs.
