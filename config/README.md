# ⚙️ Configuration System - Centralized Settings Management

## 📋 **Overview**
The Configuration System provides centralized, hierarchical, and environment-aware configuration management for all HydrogenAI services. It enables flexible deployment across different environments while maintaining consistency and validation.

## 🏗️ **Architecture Pattern**

### **Design Principles**
- **Hierarchical Configuration**: Layered configuration with inheritance
- **Environment Awareness**: Different settings for dev/staging/production
- **Validation First**: All configurations validated before use
- **Hot Reload**: Runtime configuration updates without restart
- **Security**: Secure handling of sensitive configuration data

### **Configuration Hierarchy**
```
Environment Variables (Highest Priority)
         ↓
    Local Config Files
         ↓
    Default Agent/Tool/Workflow Configs
         ↓
    Built-in Defaults (Lowest Priority)
```

## 📁 **Directory Structure**

```
config/
├── agents/                      # AI agent configurations
│   ├── default.yaml            # Default general-purpose agent
│   ├── execution.yaml          # Tool execution specialist agent
│   ├── query_planning.yaml     # Query optimization agent
│   └── schema_discovery.yaml   # Database analysis agent
├── tools/                      # Tool configurations
│   ├── mongodb.yaml           # MongoDB tool settings
│   └── rag.yaml               # RAG tool configurations
└── workflows/                  # Workflow definitions
    ├── simple_query.yaml      # Basic query workflows
    ├── complex_aggregation.yaml # Advanced aggregation workflows
    ├── rag_query.yaml         # RAG-specific workflows
    └── schema_discovery.yaml  # Schema analysis workflows
```

## 🤖 **Agent Configurations (`agents/`)**

### **Default Agent (`default.yaml`)**
**Purpose**: General-purpose AI agent for standard query processing

```yaml
name: "default"
version: "1.0.0"
description: "Default general-purpose agent for query processing"

model_settings:
  temperature: 0.7
  max_tokens: 2000
  timeout: 30
  fallback_providers: ["groq", "openai", "anthropic"]

system_prompt: |
  You are an intelligent AI assistant specialized in data analysis and query processing.
  Your role is to understand user queries and provide accurate, helpful responses.
  
  Key capabilities:
  - Data query analysis and optimization
  - Natural language to database query translation
  - Result interpretation and explanation
  - Error handling and user guidance

tools:
  - mongodb_find
  - mongodb_count
  - mongodb_aggregate
  - mongodb_collection_stats
  - rag_search_documents
  - rag_query_with_context

conversation:
  memory_enabled: true
  max_history: 10
  context_window: 4000

performance:
  cache_responses: true
  cache_ttl: 3600
  max_concurrent_requests: 5
```

**Integration Points**:
- **Orchestrator**: Loads agent for general query processing
- **AI Provider Manager**: Uses model settings for provider selection
- **Tool Registry**: Binds specified tools to agent
- **Cache System**: Uses caching settings for performance optimization

### **Execution Agent (`execution.yaml`)**
**Purpose**: Specialized agent for tool execution and workflow orchestration

```yaml
name: "execution"
version: "1.0.0"
description: "Specialized agent for tool execution and workflow orchestration"

model_settings:
  temperature: 0.3  # Lower temperature for more deterministic execution
  max_tokens: 1500
  timeout: 45
  fallback_providers: ["groq", "openai"]

system_prompt: |
  You are a specialized execution agent responsible for tool orchestration and workflow management.
  Your role is to execute complex operations accurately and efficiently.
  
  Key responsibilities:
  - Tool selection and parameter preparation
  - Workflow step execution
  - Error handling and recovery
  - Result validation and formatting

tools:
  - mongodb_find
  - mongodb_count
  - mongodb_aggregate
  - mongodb_list_collections
  - mongodb_database_stats
  - mongodb_collection_stats
  - mongodb_create_index
  - rag_add_document
  - rag_search_documents
  - health_check
  - system_info

execution:
  retry_failed_tools: true
  max_retries: 3
  retry_delay: 2
  parallel_execution: true
  max_parallel_tools: 3

validation:
  validate_tool_params: true
  validate_results: true
  schema_validation: true
```

### **Query Planning Agent (`query_planning.yaml`)**
**Purpose**: Optimization and planning for complex queries

```yaml
name: "query_planning"
version: "1.0.0"
description: "Query optimization and planning specialist"

model_settings:
  temperature: 0.2  # Very low temperature for analytical tasks
  max_tokens: 2500
  timeout: 60
  fallback_providers: ["groq", "openai", "anthropic"]

system_prompt: |
  You are a query planning specialist focused on optimization and performance.
  Your role is to analyze queries and create optimal execution plans.
  
  Key responsibilities:
  - Query complexity analysis
  - Execution plan optimization
  - Index recommendation
  - Performance prediction
  - Resource usage estimation

tools:
  - mongodb_find
  - mongodb_count
  - mongodb_aggregate
  - mongodb_collection_stats
  - mongodb_database_stats
  - mongodb_create_index

optimization:
  analyze_query_complexity: true
  recommend_indexes: true
  estimate_performance: true
  suggest_alternatives: true

planning:
  max_execution_steps: 10
  complexity_threshold: 0.8
  performance_target: 1000  # ms
```

### **Schema Discovery Agent (`schema_discovery.yaml`)**
**Purpose**: Database schema analysis and discovery

```yaml
name: "schema_discovery"
version: "1.0.0"
description: "Database schema analysis and discovery specialist"

model_settings:
  temperature: 0.1  # Minimal temperature for factual analysis
  max_tokens: 3000
  timeout: 90
  fallback_providers: ["groq", "openai"]

system_prompt: |
  You are a database schema analysis specialist.
  Your role is to understand and explain database structures and relationships.
  
  Key responsibilities:
  - Schema discovery and analysis
  - Relationship mapping
  - Data type analysis
  - Index usage analysis
  - Performance recommendations

tools:
  - mongodb_list_collections
  - mongodb_collection_stats
  - mongodb_database_stats
  - mongodb_find  # For sample data analysis

analysis:
  sample_size: 100
  analyze_relationships: true
  identify_patterns: true
  performance_analysis: true

output:
  include_samples: true
  format_schema: true
  generate_documentation: true
```

## 🛠️ **Tool Configurations (`tools/`)**

### **MongoDB Tools (`mongodb.yaml`)**
**Purpose**: Configuration for MongoDB operations and optimizations

```yaml
name: "mongodb_tools"
version: "2.0.0"
description: "MongoDB operations and optimization tools"

connection:
  pool_size: 10
  idle_timeout: 300
  socket_timeout: 30
  server_selection_timeout: 5

operations:
  find:
    default_limit: 100
    max_limit: 10000
    timeout: 30
    allow_disk_use: false
  
  aggregate:
    default_timeout: 60
    allow_disk_use: true
    max_memory_usage: 100  # MB
    cursor_batch_size: 1000
  
  count:
    timeout: 15
    use_estimated_count: false
  
  stats:
    include_indexes: true
    include_size_details: true
    cache_ttl: 300

indexing:
  auto_create_recommendations: true
  analyze_query_patterns: true
  performance_threshold: 100  # ms
  
optimization:
  enable_query_analysis: true
  suggest_indexes: true
  monitor_performance: true

security:
  validate_queries: true
  prevent_injection: true
  sanitize_inputs: true
  audit_operations: true
```

### **RAG Tools (`rag.yaml`)**
**Purpose**: Configuration for Retrieval-Augmented Generation tools

```yaml
name: "rag_tools"
version: "2.0.0"
description: "Retrieval-Augmented Generation tools configuration"

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimensions: 384
  normalize_embeddings: true
  batch_size: 32
  device: "cpu"  # or "cuda" for GPU

vector_store:
  collection_name: "hydrogen_documents"
  distance_metric: "cosine"
  index_type: "hnsw"
  ef_construct: 200
  m: 16

search:
  default_limit: 5
  max_limit: 50
  similarity_threshold: 0.7
  rerank_results: true
  include_metadata: true

generation:
  max_context_length: 4000
  context_overlap: 200
  temperature: 0.7
  max_tokens: 1000
  include_sources: true

documents:
  max_size: 10485760  # 10MB
  supported_formats: ["txt", "pdf", "docx", "md"]
  chunk_size: 1000
  chunk_overlap: 200
  auto_extract_metadata: true

performance:
  cache_embeddings: true
  embedding_cache_ttl: 86400  # 24 hours
  precompute_similar: false
  parallel_processing: true
```

## 🔄 **Workflow Configurations (`workflows/`)**

### **Simple Query Workflow (`simple_query.yaml`)**
**Purpose**: Basic query processing workflow

```yaml
name: "simple_query"
version: "1.0.0"
description: "Basic query processing workflow"

trigger_conditions:
  query_types: ["simple", "basic"]
  complexity_score: "< 0.3"
  estimated_time: "< 5 seconds"

steps:
  - name: "validate_input"
    type: "validation"
    agent: "default"
    tools: []
    timeout: 5
    
  - name: "classify_query"
    type: "classification"
    agent: "default"
    tools: []
    timeout: 10
    
  - name: "execute_query"
    type: "execution"
    agent: "execution"
    tools: ["mongodb_find", "mongodb_count"]
    timeout: 30
    retry: true
    
  - name: "format_response"
    type: "formatting"
    agent: "default"
    tools: []
    timeout: 5

performance:
  cache_results: true
  cache_ttl: 1800  # 30 minutes
  max_execution_time: 60
  
error_handling:
  retry_on_failure: true
  max_retries: 2
  fallback_workflow: "basic_fallback"
```

### **Complex Aggregation Workflow (`complex_aggregation.yaml`)**
**Purpose**: Advanced multi-collection aggregation workflow

```yaml
name: "complex_aggregation"
version: "1.0.0"
description: "Advanced aggregation workflow for complex queries"

trigger_conditions:
  query_types: ["aggregation", "analytics", "complex"]
  complexity_score: "> 0.6"
  estimated_time: "> 10 seconds"

steps:
  - name: "analyze_requirements"
    type: "analysis"
    agent: "query_planning"
    tools: ["mongodb_collection_stats", "mongodb_database_stats"]
    timeout: 30
    
  - name: "optimize_query"
    type: "optimization"
    agent: "query_planning"
    tools: ["mongodb_create_index"]
    timeout: 60
    optional: true
    
  - name: "execute_aggregation"
    type: "execution"
    agent: "execution"
    tools: ["mongodb_aggregate"]
    timeout: 300  # 5 minutes for complex aggregations
    retry: true
    
  - name: "validate_results"
    type: "validation"
    agent: "execution"
    tools: ["mongodb_count"]
    timeout: 30
    
  - name: "format_response"
    type: "formatting"
    agent: "default"
    tools: []
    timeout: 15

optimization:
  pre_execution_analysis: true
  index_recommendations: true
  performance_monitoring: true
  
performance:
  cache_results: true
  cache_ttl: 3600  # 1 hour
  max_execution_time: 600  # 10 minutes
  
resource_limits:
  max_memory_usage: 500  # MB
  max_cpu_usage: 80  # percentage
```

### **RAG Query Workflow (`rag_query.yaml`)**
**Purpose**: Retrieval-Augmented Generation workflow

```yaml
name: "rag_query"
version: "1.0.0"
description: "Retrieval-Augmented Generation workflow"

trigger_conditions:
  query_types: ["rag", "document_search", "knowledge"]
  contains_keywords: ["find documents", "search for", "what do you know about"]

steps:
  - name: "understand_query"
    type: "analysis"
    agent: "default"
    tools: []
    timeout: 10
    
  - name: "search_documents"
    type: "retrieval"
    agent: "execution"
    tools: ["rag_search_documents"]
    timeout: 30
    
  - name: "generate_response"
    type: "generation"
    agent: "default"
    tools: ["rag_query_with_context"]
    timeout: 60
    
  - name: "validate_sources"
    type: "validation"
    agent: "execution"
    tools: []
    timeout: 10

rag_settings:
  retrieval_count: 5
  context_length: 3000
  include_sources: true
  rerank_results: true
  
generation:
  temperature: 0.7
  max_tokens: 1500
  include_citations: true
  
performance:
  cache_embeddings: true
  cache_results: true
  cache_ttl: 1800  # 30 minutes
```

### **Schema Discovery Workflow (`schema_discovery.yaml`)**
**Purpose**: Database schema analysis and documentation workflow

```yaml
name: "schema_discovery"
version: "1.0.0"
description: "Database schema analysis workflow"

trigger_conditions:
  query_types: ["schema", "structure", "discovery"]
  contains_keywords: ["schema", "structure", "collections", "tables"]

steps:
  - name: "discover_collections"
    type: "discovery"
    agent: "schema_discovery"
    tools: ["mongodb_list_collections"]
    timeout: 30
    
  - name: "analyze_structure"
    type: "analysis"
    agent: "schema_discovery"
    tools: ["mongodb_collection_stats", "mongodb_find"]
    timeout: 120
    parallel: true
    
  - name: "generate_documentation"
    type: "documentation"
    agent: "schema_discovery"
    tools: []
    timeout: 60
    
  - name: "recommend_optimizations"
    type: "optimization"
    agent: "query_planning"
    tools: ["mongodb_create_index"]
    timeout: 30
    optional: true

analysis_settings:
  sample_size: 100
  max_collections: 20
  analyze_relationships: true
  identify_patterns: true
  
output:
  format: "markdown"
  include_samples: true
  include_statistics: true
  include_recommendations: true
  
performance:
  cache_results: true
  cache_ttl: 7200  # 2 hours
  parallel_analysis: true
```

## ⚙️ **Configuration Management System**

### **Environment-Aware Loading**
```python
class ConfigManager:
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config_cache = {}
        
    def load_agent_config(self, agent_name: str) -> AgentConfig:
        """Load agent configuration with environment overrides"""
        # 1. Load base configuration from YAML
        # 2. Apply environment-specific overrides
        # 3. Validate configuration schema
        # 4. Cache for performance
        
    def load_workflow_config(self, workflow_name: str) -> WorkflowConfig:
        """Load workflow configuration with validation"""
        # 1. Load workflow definition
        # 2. Validate step dependencies
        # 3. Check tool availability
        # 4. Apply performance settings
```

### **Hot Reload Capability**
```python
class ConfigReloader:
    async def reload_configurations(self):
        """Hot reload configurations without service restart"""
        # 1. Detect configuration file changes
        # 2. Validate new configurations
        # 3. Update active configurations
        # 4. Notify services of changes
        # 5. Update cache and registries
```

### **Validation System**
```python
class ConfigValidator:
    def validate_agent_config(self, config: Dict) -> ValidationResult:
        """Validate agent configuration"""
        # - Check required fields
        # - Validate tool availability
        # - Check model settings
        # - Validate timeout values
        
    def validate_workflow_config(self, config: Dict) -> ValidationResult:
        """Validate workflow configuration"""
        # - Check step dependencies
        # - Validate agent references
        # - Check timeout values
        # - Validate trigger conditions
```

## 🔐 **Security and Access Control**

### **Configuration Security**
- **Sensitive Data**: Encrypted storage of API keys and credentials
- **Access Control**: Role-based access to configuration files
- **Audit Logging**: Track all configuration changes
- **Validation**: Schema validation for all configurations

### **Environment Separation**
```yaml
# Development environment overrides
development:
  model_settings:
    temperature: 0.9  # Higher creativity for testing
    timeout: 120      # Longer timeouts for debugging
  
  performance:
    cache_ttl: 60     # Shorter cache for rapid testing
    
# Production environment overrides
production:
  model_settings:
    temperature: 0.3  # Lower temperature for consistency
    timeout: 30       # Strict timeouts for performance
  
  performance:
    cache_ttl: 3600   # Longer cache for performance
```

## 📊 **Configuration Monitoring**

### **Usage Analytics**
- Configuration usage patterns
- Performance impact of settings
- Error rates by configuration
- Resource utilization trends

### **Health Monitoring**
```python
class ConfigHealth:
    def check_configuration_health(self) -> HealthReport:
        """Monitor configuration system health"""
        # - Validate active configurations
        # - Check file accessibility
        # - Monitor performance impact
        # - Detect configuration drift
```

## 🤝 **Developer Guidelines**

### **Adding New Configurations**
1. **Create Configuration File**: Follow YAML schema standards
2. **Add Validation**: Implement schema validation
3. **Document Settings**: Provide comprehensive documentation
4. **Test Thoroughly**: Test all configuration scenarios
5. **Environment Variants**: Create environment-specific overrides

### **Configuration Best Practices**
1. **Use Descriptive Names**: Clear, self-documenting configuration keys
2. **Provide Defaults**: Sensible default values for all settings
3. **Validate Early**: Validate configurations at startup
4. **Document Dependencies**: Clearly document configuration dependencies
5. **Version Control**: Track configuration changes with versioning

---

## 🎯 **Quick Start for Configuration Management**

1. **Environment Setup**: Set environment variables for your deployment
2. **Configuration Review**: Review default configurations for your use case
3. **Customization**: Override defaults with environment-specific settings
4. **Validation**: Run configuration validation before deployment
5. **Monitoring**: Set up monitoring for configuration health and performance

The Configuration System provides a robust, flexible, and secure foundation for managing all aspects of the HydrogenAI platform's behavior across different environments and deployment scenarios.
