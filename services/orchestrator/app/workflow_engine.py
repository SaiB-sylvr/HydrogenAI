"""
Enhanced Workflow Engine with LangGraph Integration
"""
import asyncio
import yaml
from typing import Dict, Any, Optional, List, Set, Callable
from pathlib import Path
import logging
from enum import Enum
from datetime import datetime
import hashlib

# Import LangGraph with fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("LangGraph not available, using fallback implementation")
    LANGGRAPH_AVAILABLE = False
    
    # Mock classes for fallback
    class StateGraph:
        def __init__(self, state_class):
            self.state_class = state_class
            self.nodes = {}
            self.edges = []
            self.entry_point = None
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            self.edges.append((from_node, to_node))
        
        def set_entry_point(self, node):
            self.entry_point = node
        
        def compile(self, checkpointer=None):
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph):
            self.graph = graph
        
        async def astream(self, initial_state, config):
            # Simple sequential execution
            current_state = initial_state
            for node_name in self.graph.nodes:
                if node_name in self.graph.nodes:
                    current_state = await self.graph.nodes[node_name](current_state)
                    yield current_state
    
    class MemorySaver:
        def __init__(self):
            pass
    
    END = "END"

from app.core.event_bus import EventBus
from app.core.state_manager import StateManager
from app.core.circuit_breaker import CircuitBreaker
from app.agents.agent_system import AgentRuntime

logger = logging.getLogger(__name__)

class WorkflowState(dict):
    """Workflow state with typed access"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["messages"] = []
        self["errors"] = []
        self["metrics"] = {}
    
    def add_message(self, role: str, content: str):
        """Add a message to the workflow"""
        self["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_error(self, error: str, context: Dict[str, Any] = None):
        """Add an error to the workflow"""
        self["errors"].append({
            "error": error,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def record_metric(self, metric_name: str, value: Any):
        """Record a metric"""
        self["metrics"][metric_name] = value

class WorkflowEngine:
    """Enhanced workflow engine with dynamic graph generation"""
    
    def __init__(
        self,
        event_bus: EventBus,
        state_manager: StateManager,
        circuit_breaker: CircuitBreaker
    ):
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.circuit_breaker = circuit_breaker
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_saver = MemorySaver()
        self.agent_runtime: Optional[AgentRuntime] = None
        
    def set_agent_runtime(self, agent_runtime: AgentRuntime):
        """Set the agent runtime"""
        self.agent_runtime = agent_runtime
    
    def load_workflows(self, workflow_dir: str):
        """Load workflow definitions"""
        workflow_path = Path(workflow_dir)
        if not workflow_path.exists():
            logger.warning(f"Workflow directory {workflow_path} does not exist")
            self._create_default_workflows()
            return
        
        for workflow_file in workflow_path.glob("*.yaml"):
            try:
                with open(workflow_file, "r") as f:
                    workflow = yaml.safe_load(f)
                
                workflow_name = workflow.get("name", workflow_file.stem)
                self.workflows[workflow_name] = workflow
                logger.info(f"Loaded workflow: {workflow_name}")
                
            except Exception as e:
                logger.error(f"Failed to load workflow {workflow_file}: {e}")
    
    def _create_default_workflows(self):
        """Create default workflows if none exist"""
        self.workflows = {
            "simple_query": {
                "name": "simple_query",
                "description": "Direct query execution",
                "steps": [
                    {"type": "execute_tool", "name": "execute_query"},
                    {"type": "format_result", "name": "format_response"}
                ]
            },
            "complex_aggregation": {
                "name": "complex_aggregation",
                "description": "Complex aggregation workflow",
                "steps": [
                    {"type": "agent", "name": "analyze_query", "agent": "query_planner"},
                    {"type": "agent", "name": "discover_schema", "agent": "schema_profiler"},
                    {"type": "agent", "name": "execute_query", "agent": "executor"},
                    {"type": "agent", "name": "generate_insights", "agent": "insight_generator"}
                ]
            },
            "rag_query": {
                "name": "rag_query",
                "description": "RAG-based query workflow",
                "steps": [
                    {"type": "tool", "name": "search_documents", "tool": "rag_search"},
                    {"type": "tool", "name": "rerank_results", "tool": "rerank_documents"},
                    {"type": "agent", "name": "synthesize_answer", "agent": "insight_generator"}
                ]
            }
        }
    
    async def execute_workflow(self, workflow_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        request_id = context.get("request_id", "unknown")
        
        try:
            # Create workflow graph
            graph = await self._create_workflow_graph(workflow_name, context)
            
            # Initialize state
            initial_state = WorkflowState(context)
            
            # Execute with checkpointing
            config = {"configurable": {"thread_id": request_id}}
            
            # Track execution time
            start_time = datetime.utcnow()
            
            # Run the graph
            final_state = None
            async for state in graph.astream(initial_state, config):
                # Broadcast progress
                await self.event_bus.publish("workflow.progress", {
                    "request_id": request_id,
                    "state": self._serialize_state(state),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Save checkpoint
                await self.state_manager.save_workflow_checkpoint(
                    request_id,
                    {"state": self._serialize_state(state), "step": state.get("current_step")}
                )
                
                final_state = state
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Handle completion
            result = await self._handle_workflow_completion(
                workflow_name,
                request_id,
                final_state,
                execution_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Handle failure
            await self._handle_workflow_failure(workflow_name, request_id, str(e))
            
            raise
    
    async def _create_workflow_graph(self, workflow_name: str, context: Dict[str, Any]) -> StateGraph:
        """Create a workflow graph dynamically"""
        workflow_def = self.workflows[workflow_name]
        graph = StateGraph(WorkflowState)
        
        # Add nodes for each step
        steps = workflow_def.get("steps", [])
        node_names = []
        
        for step in steps:
            node_name = step["name"]
            node_names.append(node_name)
            
            if step["type"] == "agent":
                graph.add_node(node_name, self._create_agent_node(step))
            elif step["type"] == "tool":
                graph.add_node(node_name, self._create_tool_node(step))
            elif step["type"] == "execute_tool":
                graph.add_node(node_name, self._create_direct_tool_node(step))
            elif step["type"] == "format_result":
                graph.add_node(node_name, self._create_format_node(step))
            else:
                graph.add_node(node_name, self._create_custom_node(step))
        
        # Set entry point
        if node_names:
            graph.set_entry_point(node_names[0])
        
        # Add edges sequentially
        for i in range(len(node_names) - 1):
            graph.add_edge(node_names[i], node_names[i + 1])
        
        # Add final edge to END
        if node_names:
            graph.add_edge(node_names[-1], END)
        
        # Compile the graph
        return graph.compile(checkpointer=self.checkpoint_saver)
    
    def _create_agent_node(self, step: Dict[str, Any]) -> Callable:
        """Create a node that executes an agent"""
        agent_name = step.get("agent")
        
        async def agent_node(state: WorkflowState) -> WorkflowState:
            state["current_step"] = step["name"]
            state.add_message("system", f"Executing agent: {agent_name}")
            
            try:
                # Get task from state or step config
                task = step.get("task", state.get("query", ""))
                
                # Execute agent through circuit breaker
                result = await self.circuit_breaker.call(
                    f"agent_{agent_name}",
                    self.agent_runtime.execute_agent,
                    agent_name,
                    task,
                    state
                )
                
                # Store result
                state[f"{step['name']}_result"] = result
                
                if result.get("success"):
                    state.add_message(agent_name, result.get("output", ""))
                else:
                    state.add_error(f"Agent {agent_name} failed: {result.get('error')}")
                
            except Exception as e:
                logger.error(f"Agent node {agent_name} failed: {e}")
                state.add_error(f"Agent execution failed: {str(e)}")
            
            return state
        
        return agent_node
    
    def _create_tool_node(self, step: Dict[str, Any]) -> Callable:
        """Create a node that executes a tool"""
        tool_name = step.get("tool")
        
        async def tool_node(state: WorkflowState) -> WorkflowState:
            state["current_step"] = step["name"]
            state.add_message("system", f"Executing tool: {tool_name}")
            
            try:
                # Get parameters from state or step config
                params = step.get("params", {})
                
                # Resolve parameter references
                resolved_params = self._resolve_params(params, state)
                
                # Execute tool through MCP server
                result = await self._execute_mcp_tool(tool_name, resolved_params)
                
                # Store result
                state[f"{step['name']}_result"] = result
                
            except Exception as e:
                logger.error(f"Tool node {tool_name} failed: {e}")
                state.add_error(f"Tool execution failed: {str(e)}")
            
            return state
        
        return tool_node
    
    def _create_direct_tool_node(self, step: Dict[str, Any]) -> Callable:
        """Create a node for direct tool execution (simple queries)"""
        async def direct_tool_node(state: WorkflowState) -> WorkflowState:
            state["current_step"] = step["name"]
            
            try:
                # Get tool and params from classification
                classification = state.get("classification", {})
                tool_name = classification.get("tool")
                params = classification.get("params", {})
                
                if not tool_name:
                    raise ValueError("No tool specified in classification")
                
                # Execute tool
                result = await self._execute_mcp_tool(tool_name, params)
                
                state["query_result"] = result
                
            except Exception as e:
                logger.error(f"Direct tool execution failed: {e}")
                state.add_error(f"Query execution failed: {str(e)}")
            
            return state
        
        return direct_tool_node
    
    def _create_format_node(self, step: Dict[str, Any]) -> Callable:
        """Create a node that formats results"""
        async def format_node(state: WorkflowState) -> WorkflowState:
            state["current_step"] = step["name"]
            
            try:
                # Get result to format
                result = state.get("query_result") or state.get(f"{step.get('input_from', 'execute_query')}_result")
                
                if not result:
                    state["final_result"] = {
                        "success": False,
                        "error": "No result to format"
                    }
                else:
                    # Format based on query type
                    formatted = self._format_result(result, state.get("query"))
                    state["final_result"] = formatted
                
            except Exception as e:
                logger.error(f"Format node failed: {e}")
                state.add_error(f"Result formatting failed: {str(e)}")
                state["final_result"] = {
                    "success": False,
                    "error": str(e)
                }
            
            return state
        
        return format_node
    
    def _create_custom_node(self, step: Dict[str, Any]) -> Callable:
        """Create a custom node"""
        async def custom_node(state: WorkflowState) -> WorkflowState:
            state["current_step"] = step["name"]
            # Implement custom logic based on step type
            return state
        
        return custom_node
    
    async def _execute_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool through MCP server"""
        import httpx
        import os
        
        # Get MCP server URL from environment or use default
        mcp_url = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{mcp_url}/execute",
                json={"tool": tool_name, "params": params},
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            return result.get("result")
    
    def _resolve_params(self, params: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Resolve parameter references from state"""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to state value
                path = value[1:].split(".")
                resolved_value = state
                
                for part in path:
                    if isinstance(resolved_value, dict):
                        resolved_value = resolved_value.get(part)
                    else:
                        resolved_value = None
                        break
                
                resolved[key] = resolved_value
            else:
                resolved[key] = value
        
        return resolved
    
    def _format_result(self, result: Any, query: str) -> Dict[str, Any]:
        """Format result for response"""
        if isinstance(result, dict) and "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }
        
        # Basic formatting - enhance as needed
        return {
            "success": True,
            "query": query,
            "result": result,
            "formatted_at": datetime.utcnow().isoformat()
        }
    
    def _serialize_state(self, state: WorkflowState) -> Dict[str, Any]:
        """Serialize state for storage/transmission"""
        # Remove non-serializable objects
        serialized = {}
        
        for key, value in state.items():
            try:
                # Test if serializable
                import json
                json.dumps(value)
                serialized[key] = value
            except:
                # Store string representation
                serialized[key] = str(value)
        
        return serialized
    
    async def _handle_workflow_completion(
        self,
        workflow_name: str,
        request_id: str,
        final_state: WorkflowState,
        execution_time: float
    ) -> Dict[str, Any]:
        """Handle workflow completion"""
        # Extract final result
        final_result = final_state.get("final_result")
        
        if not final_result:
            # Try to construct result from state
            if final_state.get("errors"):
                final_result = {
                    "success": False,
                    "errors": final_state["errors"]
                }
            else:
                # Collect all results
                results = {}
                for key, value in final_state.items():
                    if key.endswith("_result") and value:
                        results[key.replace("_result", "")] = value
                
                final_result = {
                    "success": True,
                    "results": results,
                    "messages": final_state.get("messages", [])
                }
        
        # Add metrics
        final_result["metrics"] = {
            **final_state.get("metrics", {}),
            "execution_time": execution_time,
            "workflow": workflow_name,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # Update state manager
        await self.state_manager.update_state(request_id, {
            "status": "completed",
            "result": final_result,
            "execution_time": execution_time
        })
        
        # Record metrics
        await self.state_manager.increment_metric(f"workflow_completions:{workflow_name}")
        await self.state_manager.record_latency(f"workflow:{workflow_name}", execution_time * 1000)
        
        # Publish completion event
        await self.event_bus.publish("workflow.completed", {
            "request_id": request_id,
            "workflow": workflow_name,
            "success": final_result.get("success", False),
            "execution_time": execution_time,
            "result": final_result
        })
        
        logger.info(f"Workflow {workflow_name} completed in {execution_time:.2f}s")
        
        return final_result
    
    async def _handle_workflow_failure(self, workflow_name: str, request_id: str, error: str):
        """Handle workflow failure"""
        # Update state
        await self.state_manager.update_state(request_id, {
            "status": "failed",
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        })
        
        # Record metrics
        await self.state_manager.increment_metric(f"workflow_failures:{workflow_name}")
        
        # Publish failure event
        await self.event_bus.publish("workflow.failed", {
            "request_id": request_id,
            "workflow": workflow_name,
            "error": error
        })
        
        logger.error(f"Workflow {workflow_name} failed: {error}")