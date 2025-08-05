# Hydrogen AI Orchestrator Container Fix Plan

## Problem Analysis

The hydrogen-orchestrator container is failing during startup due to missing Python dependencies, causing the health check to fail and preventing Kong from starting.

## Root Cause Identified

### Critical Missing Dependencies:
1. **langgraph** - Required by `workflow_engine.py` (lines 13-14)
2. **nats-py** - Required by `event_bus.py` (line 4)
3. **Version conflicts** in LangChain packages

### Import Chain Analysis:
```
main.py → workflow_engine.py → langgraph (MISSING)
main.py → event_bus.py → nats.aio.client (MISSING)
main.py → agent_system.py → langchain (version conflicts)
```

## Required Fixes

### 1. Update requirements.txt
Add missing dependencies to `services/orchestrator/requirements.txt`:

```txt
# Add these missing dependencies:
langgraph>=0.0.40
nats-py>=2.6.0

# Pin LangChain versions to avoid conflicts:
langchain==0.1.0
langchain-openai==0.0.5
langchain-community==0.0.10

# Ensure async MongoDB support:
motor>=3.3.0
```

### 2. Fix Circular Import Issues
In `services/orchestrator/app/main.py` line 86:
- Currently passes `event_bus=None` to WorkflowEngine
- Need to properly initialize EventBus or make it optional

### 3. Handle Optional Dependencies
Add try/catch blocks for optional imports in:
- `workflow_engine.py` for langgraph
- `event_bus.py` for nats
- `agent_system.py` for langchain (already has fallback)

### 4. Docker Configuration Updates
Consider increasing health check start period from 90s to 120s in `docker-compose.yml` to allow for dependency initialization.

## Implementation Steps

1. **Switch to Code Mode** to make the necessary file changes
2. **Update requirements.txt** with missing dependencies
3. **Add import fallbacks** for graceful degradation
4. **Test container startup** locally
5. **Verify health check** endpoint functionality
6. **Update Docker configuration** if needed

## Expected Outcome

After implementing these fixes:
- Container will start successfully
- Health check will pass
- Kong will be able to start and depend on orchestrator
- System will be fully operational

## Prevention Measures

1. Add dependency validation in CI/CD
2. Use dependency scanning tools
3. Implement proper error handling for missing dependencies
4. Add integration tests for container startup