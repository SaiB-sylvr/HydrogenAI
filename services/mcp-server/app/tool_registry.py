import asyncio
from typing import Dict, Any, Optional
import logging
import inspect

logger = logging.getLogger(__name__)

class Tool:
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        """Execute the tool"""
        # Default implementation provides a basic response
        # Concrete tool classes should override this method
        logger.info(f"Executing base tool '{self.name}' with parameters: {params}")
        return {
            "tool": self.name,
            "description": self.description,
            "params": params,
            "result": "Tool executed successfully (base implementation)",
            "timestamp": "2025-08-03T00:00:00Z"
        }
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        if not isinstance(params, dict):
            return False
        
        # Check required parameters if defined
        required_params = self.get_schema().get("required", [])
        for param in required_params:
            if param not in params:
                logger.error(f"Missing required parameter '{param}' for tool '{self.name}'")
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema"""
        return {
            "type": "object",
            "properties": self.parameters,
            "required": [],
            "description": self.description
        }

class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.tools: Dict[str, Tool] = {}
    
    async def initialize(self):
        """Initialize tool registry"""
        self.tools.clear()
        
        # Load tools from plugins
        for plugin_name, plugin in self.plugin_manager.plugins.items():
            try:
                tools = plugin.get_tools()
                for tool in tools:
                    self.register_tool(tool)
                    logger.info(f"Registered tool '{tool.name}' from plugin '{plugin_name}'")
            except Exception as e:
                logger.error(f"Failed to load tools from plugin '{plugin_name}': {e}")
    
    def register_tool(self, tool: Tool):
        """Register a tool"""
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        
        self.tools[tool.name] = tool
    
    async def execute(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool"""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Validate parameters
        if not tool.validate_params(params):
            raise ValueError(f"Invalid parameters for tool '{tool_name}'")
        
        # Execute tool
        try:
            if inspect.iscoroutinefunction(tool.execute):
                result = await tool.execute(params)
            else:
                # Run sync functions in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.execute, params)
            
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            raise