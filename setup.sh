#!/bin/bash

echo "Setting up HydrogenAI v2..."

# Create directory structure
echo "Creating directory structure..."

# Service directories
mkdir -p services/orchestrator/app/{agents,core}
mkdir -p services/mcp-server/app/tools
mkdir -p services/shared

# Plugin directories
mkdir -p plugins/mongodb
mkdir -p plugins/rag

# Config directories
mkdir -p config/{agents,workflows,tools}
mkdir -p kong

# Create __init__.py files
echo "Creating __init__.py files..."

# Orchestrator
touch services/orchestrator/__init__.py
touch services/orchestrator/app/__init__.py
touch services/orchestrator/app/agents/__init__.py
touch services/orchestrator/app/core/__init__.py

# MCP Server
touch services/mcp-server/__init__.py
touch services/mcp-server/app/__init__.py
touch services/mcp-server/app/tools/__init__.py

# Shared
touch services/shared/__init__.py

# Plugins
touch plugins/__init__.py
touch plugins/mongodb/__init__.py
touch plugins/rag/__init__.py

# Move Dockerfile if needed
if [ -f "services/orchestrator/app/Dockerfile" ]; then
    echo "Moving orchestrator Dockerfile to correct location..."
    mv services/orchestrator/app/Dockerfile services/orchestrator/Dockerfile
fi

# Copy .env template
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your configuration"
fi

# Create default config files
echo "Creating default configuration files..."

# Agent config
cat > config/agents/default.yaml << 'EOF'
name: default_agents
version: 1.0.0
description: Default agent configuration

agents:
  - name: schema_profiler
    role: Schema Profiler
    goal: Discover and analyze database schemas
    backstory: Expert in database structures and relationships
    
  - name: query_planner
    role: Query Planning Manager
    goal: Create optimal query execution plans
    backstory: Strategic thinker for data retrieval
    
  - name: executor
    role: Execution Manager
    goal: Execute queries efficiently
    backstory: Operations expert for query execution
    
  - name: insight_generator
    role: Insight Generator
    goal: Generate meaningful insights from data
    backstory: Data analyst who tells stories with data
EOF

# Workflow config
cat > config/workflows/schema_discovery.yaml << 'EOF'
name: schema_discovery
version: 1.0.0
description: Workflow for discovering database schema

steps:
  - name: discover_collections
    type: tool
    tool: mongodb_list_collections
    
  - name: analyze_schema
    type: agent
    agent: schema_profiler
    task: Analyze the schema of discovered collections
    
  - name: cache_schema
    type: custom
    action: cache_schema_results
EOF

# MongoDB plugin config update
cat > plugins/mongodb/plugin.yaml << 'EOF'
name: mongodb
version: 1.0.0
class: MongoDBPlugin
description: MongoDB tools for database operations
author: Hydrogen AI Team

configuration:
  default_database: ${MONGO_DB_NAME}
  connection_string: ${MONGO_URI}
  
  # Optional: Multi-cluster configuration
  # clusters:
  #   - name: default
  #     env_var: MONGO_URI
  #   - name: analytics
  #     env_var: MONGO_URI_ANALYTICS
  
tools:
  - name: mongodb_find
    description: Find documents in MongoDB
  - name: mongodb_count
    description: Count documents in MongoDB
  - name: mongodb_aggregate
    description: Run aggregation pipeline
  - name: mongodb_list_collections
    description: List all collections
  - name: mongodb_sample_documents
    description: Sample documents for schema discovery
  - name: mongodb_get_indexes
    description: Get collection indexes
EOF

# Set permissions
chmod +x setup.sh

echo "Setup complete! Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: docker-compose up -d"
echo "3. Check health: curl http://localhost:8080/api/health"
echo "4. View logs: docker-compose logs -f"