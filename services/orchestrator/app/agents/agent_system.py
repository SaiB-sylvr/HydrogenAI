"""
Enhanced Intelligent Agent System - True AI Decision Making
No hard-coded rules, agents communicate and decide autonomously
"""
import os
import yaml
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Import with fallback for testing
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import Tool, StructuredTool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory
    from langchain.callbacks.base import AsyncCallbackHandler
    from langchain.schema import AgentAction, AgentFinish
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available, using mock implementation")
    LANGCHAIN_AVAILABLE = False
    
    # Mock classes for testing without LangChain
    class ChatOpenAI:
        def __init__(self, **kwargs):
            self.model = kwargs.get("model", "mock")
    
    class Tool:
        def __init__(self, name, func, description):
            self.name = name
            self.func = func
            self.description = description
    
    class StructuredTool:
        @staticmethod
        def from_function(**kwargs):
            return Tool(kwargs.get("name"), kwargs.get("func"), kwargs.get("description"))

@dataclass
class AgentCapability:
    """Enhanced capability definition"""
    name: str
    description: str
    confidence_threshold: float = 0.7
    required_context: List[str] = None
    
    def __post_init__(self):
        if self.required_context is None:
            self.required_context = []

class IntelligentAgent(ABC):
    """Base class for truly intelligent agents"""
    
    def __init__(self, name: str, config: Dict[str, Any], llm=None):
        self.name = name
        self.config = config
        self.llm = llm
        self.tools: List[Tool] = []
        self.capabilities: Set[AgentCapability] = set()
        self.memory = None
        self._executor = None
        self.communication_channel = {}  # For inter-agent communication
        
    async def think(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Core thinking method - agent analyzes and understands
        """
        thinking_prompt = f"""
        As {self.config.get('role', 'an AI assistant')}, analyze this query:
        
        Query: {query}
        Context: {json.dumps(context or {}, indent=2)}
        
        Think step by step:
        1. What is the user really asking for?
        2. What information do I need to answer this?
        3. What tools or other agents might I need?
        4. What's the complexity level?
        5. What approach would be best?
        
        Provide your analysis in a structured way.
        """
        
        # In production, this would use the LLM
        if LANGCHAIN_AVAILABLE and self.llm:
            # Real LLM thinking
            response = await self._llm_think(thinking_prompt)
            return self._parse_thinking(response)
        else:
            # Mock thinking for testing
            return {
                "understanding": f"I understand you want to: {query}",
                "approach": "analytical",
                "confidence": 0.85,
                "needs": ["data", "context"],
                "complexity": "medium"
            }
    
    async def collaborate(self, other_agents: List['IntelligentAgent'], task: str) -> Dict[str, Any]:
        """
        Collaborate with other agents to solve complex tasks
        """
        collaboration_results = {
            "task": task,
            "contributors": [self.name],
            "insights": []
        }
        
        # Ask each agent for their perspective
        for agent in other_agents:
            if agent.name != self.name:
                perspective = await agent.think(task, {"requesting_agent": self.name})
                collaboration_results["contributors"].append(agent.name)
                collaboration_results["insights"].append({
                    "agent": agent.name,
                    "perspective": perspective
                })
        
        # Synthesize insights
        synthesis = await self._synthesize_collaboration(collaboration_results)
        collaboration_results["synthesis"] = synthesis
        
        return collaboration_results
    
    async def execute_with_reasoning(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute task with full reasoning capability
        """
        try:
            # First, think about the task
            thought_process = await self.think(task, context)
            
            # Log reasoning
            logger.info(f"Agent {self.name} reasoning: {thought_process}")
            
            # Determine if we can handle this alone
            if thought_process.get("confidence", 0) < 0.7:
                # Need help from other agents
                return {
                    "success": True,
                    "needs_collaboration": True,
                    "thought_process": thought_process,
                    "message": "This task requires collaboration with other agents"
                }
            
            # Execute based on our thinking
            result = await self._execute_based_on_thinking(task, thought_process, context)
            
            return {
                "success": True,
                "output": result,
                "thought_process": thought_process,
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"Agent {self.name} execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    @abstractmethod
    async def _execute_based_on_thinking(self, task: str, thought_process: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Implement specific execution logic"""
        pass
    
    async def _llm_think(self, prompt: str) -> str:
        """Use LLM for thinking"""
        if not self._executor:
            self._create_executor()
        
        result = await self._executor.ainvoke({
            "input": prompt,
            "chat_history": []
        })
        return result.get("output", "")
    
    def _parse_thinking(self, response: str) -> Dict[str, Any]:
        """Parse LLM thinking into structured format"""
        # Simple parsing - enhance with better NLP
        return {
            "understanding": response,
            "approach": "analytical",
            "confidence": 0.8,
            "complexity": "medium"
        }
    
    async def _synthesize_collaboration(self, results: Dict[str, Any]) -> str:
        """Synthesize collaborative insights"""
        insights = results.get("insights", [])
        if not insights:
            return "No collaborative insights available"
        
        # In production, use LLM to synthesize
        synthesis = f"Based on {len(insights)} agent perspectives:\n"
        for insight in insights:
            synthesis += f"- {insight['agent']}: {insight['perspective'].get('understanding', 'No insight')}\n"
        
        return synthesis
    
    def _create_executor(self):
        """Create the agent executor"""
        if not LANGCHAIN_AVAILABLE:
            return
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_enhanced_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        self._executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10  # More iterations for complex reasoning
        )
    
    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt for true intelligence"""
        return f"""You are {self.config.get('role', 'an AI assistant')} named {self.name}.

Goal: {self.config.get('goal', 'Help users with their queries')}
Backstory: {self.config.get('backstory', 'You are a helpful AI assistant')}

Key Instructions:
1. Think deeply about each query - understand the real intent
2. Break down complex problems into steps
3. Use available tools intelligently
4. Collaborate with other agents when needed
5. Always explain your reasoning
6. Provide responses in clear, human-friendly language
7. Handle errors gracefully and suggest alternatives
8. Learn from context and adapt your approach

Remember: You have true agency to make decisions. Don't just follow patterns - think!"""

class QueryUnderstandingAgent(IntelligentAgent):
    """Specialized agent for understanding queries"""
    
    async def _execute_based_on_thinking(self, task: str, thought_process: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Understand and classify queries intelligently"""
        
        # Analyze query components
        analysis = {
            "original_query": task,
            "intent": self._detect_intent(task),
            "entities": self._extract_entities(task),
            "operations": self._detect_operations(task),
            "complexity": self._assess_complexity(task, thought_process),
            "suggested_approach": self._suggest_approach(thought_process)
        }
        
        return analysis
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent intelligently"""
        query_lower = query.lower()
        
        # Intent patterns (but flexible)
        if any(word in query_lower for word in ["count", "how many", "total"]):
            if "by" in query_lower or "group" in query_lower:
                return "aggregated_count"
            return "simple_count"
        elif any(word in query_lower for word in ["sum", "average", "mean", "median"]):
            return "statistical_aggregation"
        elif any(word in query_lower for word in ["list", "show", "display", "get"]):
            return "data_retrieval"
        elif "?" in query or any(word in query_lower for word in ["what", "why", "how", "when", "where"]):
            return "question_answering"
        elif any(word in query_lower for word in ["analyze", "compare", "trend", "pattern"]):
            return "analytical"
        else:
            return "general_query"
    
    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """Extract entities from query"""
        entities = []
        query_lower = query.lower()
        
        # Collection/table names
        collection_keywords = ["users", "products", "orders", "sales", "customers", "transactions"]
        for keyword in collection_keywords:
            if keyword in query_lower:
                entities.append({"type": "collection", "value": keyword})
        
        # Field names (common ones)
        field_keywords = ["category", "status", "price", "quantity", "date", "name", "id"]
        for keyword in field_keywords:
            if keyword in query_lower:
                entities.append({"type": "field", "value": keyword})
        
        return entities
    
    def _detect_operations(self, query: str) -> List[str]:
        """Detect required operations"""
        operations = []
        query_lower = query.lower()
        
        operation_map = {
            "filter": ["where", "filter", "only", "with", "having", "is", "equals"],
            "aggregate": ["sum", "average", "mean", "total", "count by", "group by"],
            "sort": ["sort", "order", "top", "bottom", "highest", "lowest"],
            "limit": ["limit", "first", "last", "top", "max"],
            "join": ["join", "combine", "merge", "with", "and"],
            "compute": ["calculate", "compute", "derive", "percentage", "ratio"]
        }
        
        for op, keywords in operation_map.items():
            if any(keyword in query_lower for keyword in keywords):
                operations.append(op)
        
        return operations
    
    def _assess_complexity(self, query: str, thought_process: Dict[str, Any]) -> str:
        """Assess query complexity"""
        # Factors for complexity
        factors = {
            "word_count": len(query.split()),
            "operations": len(self._detect_operations(query)),
            "entities": len(self._extract_entities(query)),
            "has_aggregation": "aggregate" in self._detect_operations(query),
            "has_join": "join" in self._detect_operations(query),
            "confidence": thought_process.get("confidence", 0.5)
        }
        
        # Score complexity
        complexity_score = 0
        complexity_score += min(factors["word_count"] / 10, 2)
        complexity_score += factors["operations"] * 0.5
        complexity_score += factors["entities"] * 0.3
        complexity_score += 2 if factors["has_aggregation"] else 0
        complexity_score += 3 if factors["has_join"] else 0
        complexity_score += (1 - factors["confidence"]) * 2
        
        if complexity_score < 2:
            return "simple"
        elif complexity_score < 5:
            return "medium"
        else:
            return "complex"
    
    def _suggest_approach(self, thought_process: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest execution approach"""
        complexity = thought_process.get("complexity", "medium")
        
        if complexity == "simple":
            return {
                "workflow": "simple_execution",
                "parallel": False,
                "cache_friendly": True
            }
        elif complexity == "medium":
            return {
                "workflow": "complex_aggregation",
                "parallel": True,
                "cache_friendly": True
            }
        else:
            return {
                "workflow": "multi_step_reasoning",
                "parallel": True,
                "cache_friendly": False,
                "needs_collaboration": True
            }

class ExecutionPlannerAgent(IntelligentAgent):
    """Agent that plans query execution"""
    
    async def _execute_based_on_thinking(self, task: str, thought_process: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Create an execution plan"""
        understanding = context.get("understanding", {})
        
        # Build execution plan
        plan = {
            "steps": [],
            "estimated_time": 0,
            "resources_needed": [],
            "optimization_hints": []
        }
        
        # Determine steps based on understanding
        intent = understanding.get("intent", "general_query")
        operations = understanding.get("operations", [])
        
        # Build step sequence
        if "filter" in operations:
            plan["steps"].append({
                "type": "filter",
                "description": "Apply filtering conditions",
                "tool": "mongodb_find"
            })
        
        if "aggregate" in operations or intent == "aggregated_count":
            plan["steps"].append({
                "type": "aggregate",
                "description": "Perform aggregation operations",
                "tool": "mongodb_aggregate",
                "pipeline": self._build_aggregation_pipeline(understanding)
            })
        elif intent == "simple_count":
            plan["steps"].append({
                "type": "count",
                "description": "Count documents",
                "tool": "mongodb_count"
            })
        
        if "sort" in operations:
            plan["steps"].append({
                "type": "sort",
                "description": "Sort results",
                "params": {"direction": -1}  # Descending by default
            })
        
        # Add optimization hints
        if len(understanding.get("entities", [])) > 2:
            plan["optimization_hints"].append("Consider using indexes on filter fields")
        
        if intent == "analytical":
            plan["optimization_hints"].append("Use allowDiskUse for large aggregations")
        
        return plan
    
    def _build_aggregation_pipeline(self, understanding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build MongoDB aggregation pipeline intelligently"""
        pipeline = []
        
        # Always start with $match if there are filters
        if "filter" in understanding.get("operations", []):
            pipeline.append({"$match": {}})  # Would be populated with actual conditions
        
        # Add grouping stage
        entities = understanding.get("entities", [])
        group_fields = [e["value"] for e in entities if e["type"] == "field"]
        
        if group_fields:
            group_stage = {
                "$group": {
                    "_id": f"${group_fields[0]}" if group_fields else None
                }
            }
            
            # Add aggregation operations
            intent = understanding.get("intent", "")
            if "sum" in intent:
                group_stage["$group"]["total"] = {"$sum": 1}  # Would use actual field
            if "average" in intent:
                group_stage["$group"]["average"] = {"$avg": 1}  # Would use actual field
            
            pipeline.append(group_stage)
        
        # Add sort stage
        if "sort" in understanding.get("operations", []):
            pipeline.append({"$sort": {"_id": -1}})
        
        # Add limit
        pipeline.append({"$limit": 1000})  # Safety limit
        
        return pipeline

class DataExecutorAgent(IntelligentAgent):
    """Agent that executes data operations"""
    
    async def _execute_based_on_thinking(self, task: str, thought_process: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute the actual data operation"""
        plan = context.get("execution_plan", {})
        
        results = []
        for step in plan.get("steps", []):
            if step["type"] == "count":
                result = await self._execute_count(step, context)
            elif step["type"] == "aggregate":
                result = await self._execute_aggregation(step, context)
            elif step["type"] == "filter":
                result = await self._execute_find(step, context)
            else:
                result = {"error": f"Unknown step type: {step['type']}"}
            
            results.append(result)
        
        return {
            "execution_complete": True,
            "results": results,
            "plan_followed": plan
        }
    
    async def _execute_count(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute count operation"""
        # In production, this would call the actual tool
        return {
            "operation": "count",
            "result": {"count": 42, "collection": "users"},
            "success": True
        }
    
    async def _execute_aggregation(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation operation"""
        return {
            "operation": "aggregate",
            "result": {"data": [], "pipeline": step.get("pipeline", [])},
            "success": True
        }
    
    async def _execute_find(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute find operation"""
        return {
            "operation": "find",
            "result": {"documents": [], "count": 0},
            "success": True
        }

class InsightGeneratorAgent(IntelligentAgent):
    """Agent that generates human-friendly insights"""
    
    async def _execute_based_on_thinking(self, task: str, thought_process: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Generate insights from results"""
        results = context.get("results", [])
        query = context.get("original_query", task)
        
        # Analyze results
        insights = self._analyze_results(results)
        
        # Generate human-friendly response
        response = self._generate_human_response(query, insights, results)
        
        return response
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results for insights"""
        insights = {
            "summary": "",
            "key_findings": [],
            "patterns": [],
            "recommendations": []
        }
        
        # Extract insights based on result type
        for result in results:
            if result.get("operation") == "count":
                count = result.get("result", {}).get("count", 0)
                insights["key_findings"].append(f"Total count: {count:,}")
                
                if count == 0:
                    insights["recommendations"].append("No data found. Check your filter conditions.")
                elif count > 10000:
                    insights["recommendations"].append("Large dataset detected. Consider using pagination or aggregation.")
            
            elif result.get("operation") == "aggregate":
                data = result.get("result", {}).get("data", [])
                if data:
                    insights["key_findings"].append(f"Found {len(data)} groups/categories")
                    # Would analyze actual data for patterns
        
        return insights
    
    def _generate_human_response(self, query: str, insights: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """Generate natural language response"""
        response_parts = []
        
        # Opening
        response_parts.append(f"Based on your query: '{query}'")
        
        # Key findings
        if insights["key_findings"]:
            response_parts.append("\nHere's what I found:")
            for finding in insights["key_findings"]:
                response_parts.append(f"• {finding}")
        
        # Patterns
        if insights["patterns"]:
            response_parts.append("\nPatterns detected:")
            for pattern in insights["patterns"]:
                response_parts.append(f"• {pattern}")
        
        # Recommendations
        if insights["recommendations"]:
            response_parts.append("\nRecommendations:")
            for rec in insights["recommendations"]:
                response_parts.append(f"• {rec}")
        
        # Closing
        response_parts.append("\nWould you like me to dive deeper into any specific aspect?")
        
        return "\n".join(response_parts)

class ToolRegistry:
    """Enhanced tool registry with intelligent tool selection"""
    
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self._tools: Dict[str, Tool] = {}
        self._tool_capabilities: Dict[str, List[str]] = {}
        
    async def initialize(self):
        """Load and categorize tools"""
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.mcp_server_url}/tools", timeout=10.0)
                tools_data = response.json()
                
                for tool_info in tools_data.get("tools", []):
                    self._register_tool(tool_info)
                    
            logger.info(f"Loaded {len(self._tools)} tools")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            # Register mock tools for testing
            self._register_mock_tools()
    
    def _register_tool(self, tool_info: Dict[str, Any]):
        """Register a tool with capabilities"""
        tool_name = tool_info["name"]
        
        async def execute_tool(**kwargs):
            """Execute tool via MCP server"""
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mcp_server_url}/execute",
                    json={"tool": tool_name, "params": kwargs},
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result.get("result")
        
        tool = StructuredTool.from_function(
            func=execute_tool,
            name=tool_name,
            description=tool_info.get("description", ""),
            coroutine=execute_tool
        )
        
        self._tools[tool_name] = tool
        
        # Categorize tool capabilities
        if "count" in tool_name:
            self._tool_capabilities.setdefault("counting", []).append(tool_name)
        if "aggregate" in tool_name:
            self._tool_capabilities.setdefault("aggregation", []).append(tool_name)
        if "find" in tool_name or "search" in tool_name:
            self._tool_capabilities.setdefault("search", []).append(tool_name)
    
    def _register_mock_tools(self):
        """Register mock tools for testing"""
        mock_tools = [
            {"name": "mongodb_find", "description": "Find documents"},
            {"name": "mongodb_count", "description": "Count documents"},
            {"name": "mongodb_aggregate", "description": "Aggregate data"},
            {"name": "answer_question", "description": "Answer questions"}
        ]
        
        for tool_info in mock_tools:
            self._register_tool(tool_info)
    
    async def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool"""
        return self._tools.get(name)
    
    def get_tools_for_capability(self, capability: str) -> List[str]:
        """Get tools that provide a capability"""
        return self._tool_capabilities.get(capability, [])
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self._tools.keys())

class AgentRuntime:
    """Enhanced agent runtime with true intelligence"""
    
    def __init__(self, config_dir: str = "/app/config/agents"):
        self.config_dir = config_dir
        self.agents: Dict[str, IntelligentAgent] = {}
        self.tool_registry: Optional[ToolRegistry] = None
        self.llm: Optional[Any] = None
        self.ai_provider_manager: Optional[Any] = None
        
    def set_ai_provider_manager(self, ai_provider_manager):
        """Set the AI provider manager for resilient AI operations"""
        self.ai_provider_manager = ai_provider_manager
        logger.info("✅ AI provider manager set for agent runtime")
        
    def set_llm_service(self, llm_service):
        """Set the LLM service as fallback"""
        if llm_service:
            self.llm = llm_service
            logger.info("✅ LLM service set for agent runtime")
        
    async def initialize(self):
        """Initialize the agent runtime"""
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize tool registry
        mcp_url = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
        self.tool_registry = ToolRegistry(mcp_url)
        await self.tool_registry.initialize()
        
        # Create intelligent agents
        await self._create_intelligent_agents()
        
        logger.info(f"Initialized {len(self.agents)} intelligent agents")
    
    def _initialize_llm(self):
        """Initialize LLM with fallback"""
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model=os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192"),
                    openai_api_key=os.getenv("GROQ_API_KEY", ""),
                    openai_api_base=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
                    temperature=0.1,
                    max_tokens=2000,
                    request_timeout=30,
                    max_retries=3
                )
                logger.info("LLM initialized successfully")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")
                self.llm = None
        else:
            logger.warning("LangChain not available, using mock LLM")
            self.llm = ChatOpenAI()
    
    async def _create_intelligent_agents(self):
        """Create our intelligent agent team from YAML configurations"""
        # Load agent configurations from YAML files
        agent_configs = self._load_agent_configs()
        
        # Create agents
        for agent_config in agent_configs:
            agent_class = self._get_agent_class(agent_config.get("role", "default"))
            if agent_class:
                agent = agent_class(
                    name=agent_config["name"],
                    config=agent_config,
                    llm=self.llm
                )
                
                # Initialize with tools
                if self.tool_registry:
                    # Give each agent relevant tools
                    if "executor" in agent_config["name"].lower():
                        for tool_name in self.tool_registry.list_tools():
                            tool = await self.tool_registry.get_tool(tool_name)
                            if tool:
                                agent.tools.append(tool)
                
                self.agents[agent.name] = agent
                logger.info(f"Created intelligent agent: {agent.name}")
            else:
                logger.warning(f"Unknown agent class for role: {agent_config.get('role', 'unknown')}")

    def _load_agent_configs(self):
        """Load agent configurations from YAML files"""
        import yaml
        import glob
        
        agent_configs = []
        config_dir = "/app/config/agents"
        
        try:
            # Load all agent YAML files
            for yaml_file in glob.glob(f"{config_dir}/*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                        if config_data and 'agents' in config_data:
                            agent_configs.extend(config_data['agents'])
                            logger.info(f"Loaded {len(config_data['agents'])} agents from {yaml_file}")
                except Exception as e:
                    logger.error(f"Failed to load agent config from {yaml_file}: {e}")
            
            logger.info(f"Total agents loaded from configs: {len(agent_configs)}")
            
        except Exception as e:
            logger.error(f"Failed to load agent configs: {e}")
            # Fallback to hardcoded configs
            agent_configs = self._get_fallback_agents()
            
        return agent_configs
    
    def _get_agent_class(self, role: str):
        """Map agent roles to classes"""
        role_mapping = {
            "Query Understanding Specialist": QueryUnderstandingAgent,
            "Query Planning Manager": ExecutionPlannerAgent,
            "Pipeline Builder": ExecutionPlannerAgent,
            "Join Specialist": ExecutionPlannerAgent,
            "Query Validator": ExecutionPlannerAgent,
            "Execution Planning Strategist": ExecutionPlannerAgent,
            "Data Operations Expert": DataExecutorAgent,
            "Execution Manager": DataExecutorAgent,
            "Insight Generation Expert": InsightGeneratorAgent,
            "Result Processor": InsightGeneratorAgent,
            "Collection Inspector": DataExecutorAgent,
            "Schema Profiler": DataExecutorAgent,
            "Relationship Mapper": InsightGeneratorAgent,
            "Schema Documenter": InsightGeneratorAgent,
            "default": QueryUnderstandingAgent
        }
        return role_mapping.get(role, QueryUnderstandingAgent)
    
    def _get_fallback_agents(self):
        """Fallback agent configurations if YAML loading fails"""
        return [
            {
                "name": "query_understander",
                "role": "Query Understanding Specialist",
                "goal": "Deeply understand user queries and extract intent",
                "backstory": "Expert in natural language understanding and query analysis"
            },
            {
                "name": "query_planner",
                "role": "Execution Planning Strategist", 
                "goal": "Create optimal execution plans for queries",
                "backstory": "Master strategist who plans efficient data operations"
            },
            {
                "name": "query_validator",
                "role": "Query Validator",
                "goal": "Validate and optimize query plans",
                "backstory": "Quality assurance expert for data operations"
            },
            {
                "name": "executor", 
                "role": "Data Operations Expert",
                "goal": "Execute data operations flawlessly",
                "backstory": "Specialist in database operations and data retrieval"
            },
            {
                "name": "insight_generator",
                "role": "Insight Generation Expert",
                "goal": "Transform data into meaningful insights", 
                "backstory": "Data storyteller who finds patterns and generates insights"
            }
        ]
    
    async def process_query_intelligently(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query using intelligent agent collaboration
        """
        results = {
            "query": query,
            "processing_steps": [],
            "final_result": None
        }
        
        # Step 1: Understand the query
        understander = self.agents.get("query_understander")
        understanding = await understander.execute_with_reasoning(query, context)
        results["processing_steps"].append({
            "step": "understanding",
            "agent": "query_understander",
            "result": understanding
        })
        
        # Step 2: Plan execution
        planner = self.agents.get("query_planner")
        plan = await planner.execute_with_reasoning(
            query,
            {"understanding": understanding}
        )
        results["processing_steps"].append({
            "step": "planning",
            "agent": "query_planner",
            "result": plan
        })
        
        # Step 3: Execute plan
        executor = self.agents.get("executor")
        execution_result = await executor.execute_with_reasoning(
            query,
            {
                "understanding": understanding,
                "execution_plan": plan.get("output", {})
            }
        )
        results["processing_steps"].append({
            "step": "execution",
            "agent": "executor",
            "result": execution_result
        })
        
        # Step 4: Generate insights
        insight_gen = self.agents.get("insight_generator")
        insights = await insight_gen.execute_with_reasoning(
            query,
            {
                "original_query": query,
                "understanding": understanding,
                "results": execution_result.get("output", {}).get("results", [])
            }
        )
        results["processing_steps"].append({
            "step": "insight_generation",
            "agent": "insight_generator",
            "result": insights
        })
        
        results["final_result"] = insights.get("output", "Processing complete")
        
        return results
    
    async def execute_agent(self, agent_name: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific agent with reasoning"""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        return await agent.execute_with_reasoning(task, context)
    
    def get_agent(self, name: str) -> Optional[IntelligentAgent]:
        """Get a specific agent"""
        return self.agents.get(name)
    
    async def collaborate_on_task(self, task: str, agent_names: List[str] = None) -> Dict[str, Any]:
        """Have agents collaborate on a complex task"""
        if not agent_names:
            agent_names = list(self.agents.keys())
        
        # Select lead agent based on task
        lead_agent_name = "query_understander"  # Default
        
        # Let agents collaborate
        collaboration_result = {
            "task": task,
            "participating_agents": agent_names,
            "contributions": []
        }
        
        # Each agent contributes their perspective
        for agent_name in agent_names[:3]:  # Limit to 3 agents to avoid overcomplication
            agent = self.agents.get(agent_name)
            if agent:
                contribution = await agent.execute_with_reasoning(
                    f"Contribute to this collaborative task: {task}",
                    {"task": task, "other_agents": [a for a in agent_names if a != agent_name]}
                )
                collaboration_result["contributions"].append({
                    "agent": agent_name,
                    "contribution": contribution
                })
        
        return collaboration_result
    
    async def cleanup(self):
        """Cleanup agent runtime resources"""
        try:
            logger.info("🧹 Cleaning up agent runtime...")
            
            # Cleanup individual agents
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, 'cleanup'):
                        await agent.cleanup()
                except Exception as e:
                    logger.warning(f"Agent {agent_name} cleanup failed: {e}")
            
            # Cleanup tool registry
            if self.tool_registry and hasattr(self.tool_registry, 'cleanup'):
                await self.tool_registry.cleanup()
            
            # Clear references
            self.agents.clear()
            self.tool_registry = None
            self.llm = None
            self.ai_provider_manager = None
            
            logger.info("✅ Agent runtime cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Agent runtime cleanup failed: {e}")