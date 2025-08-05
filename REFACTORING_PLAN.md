# HydrogenAI Refactoring Plan

## Critical Issues Found

### 1. Monolithic File Structure
- `services/orchestrator/app/main.py`: 3,628 lines (CRITICAL)
- Single file contains: FastAPI app, AI providers, agents, workflows, database

### 2. Recommended File Structure

```
services/orchestrator/app/
├── main.py                 # FastAPI app only (~100 lines)
├── config/
│   ├── __init__.py
│   └── settings.py         # Environment configuration
├── providers/
│   ├── __init__.py
│   ├── base_provider.py    # Abstract base class
│   ├── groq_provider.py
│   ├── openai_provider.py
│   └── anthropic_provider.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── query_agent.py
│   ├── execution_agent.py
│   └── schema_agent.py
├── workflows/
│   ├── __init__.py
│   ├── workflow_engine.py
│   └── workflow_types.py
├── database/
│   ├── __init__.py
│   ├── mongodb.py
│   └── models.py
├── cache/
│   ├── __init__.py
│   └── redis_client.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   └── exceptions.py
└── api/
    ├── __init__.py
    ├── dependencies.py
    └── routes/
        ├── __init__.py
        ├── health.py
        ├── query.py
        └── agents.py
```

### 3. Implementation Priority

1. **High Priority:**
   - Split main.py into modules
   - Fix circular imports
   - Add proper error handling

2. **Medium Priority:**
   - Implement connection pooling
   - Add comprehensive logging
   - Optimize Docker images

3. **Low Priority:**
   - Add metrics collection
   - Implement caching strategy
   - Security hardening

### 4. Quick Wins

1. Add health check endpoints
2. Implement proper logging configuration
3. Add connection pooling for MongoDB
4. Configure Redis caching properly

## Next Steps

1. Create modular structure
2. Extract provider classes
3. Implement dependency injection
4. Add comprehensive tests
