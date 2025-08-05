"""
System Plugin for MCP Server
Provides system health and information tools
"""
import sys
import os
from typing import Dict, Any, List
import logging

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    # Try to import from the actual service location first
    from services.mcp_server.app.plugin_manager import Plugin
except ImportError:
    try:
        # Fallback to the mounted location
        sys.path.insert(0, '/app/app')
        from plugin_manager import Plugin
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Import error in system plugin: {e}")
        # Final fallback Plugin class
        class Plugin:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.name = config.get("name", "system")
                self.version = config.get("version", "1.0.0")
                self.tools = []

logger = logging.getLogger(__name__)

class SystemPlugin(Plugin):
    """System monitoring and information plugin"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tools = []
        logger.info(f"Initializing SystemPlugin with config: {config}")
    
    def get_tools(self) -> List:
        """Get tools provided by this plugin"""
        if not self.tools:
            try:
                # Import tools dynamically to avoid circular imports
                from tools.health_tool import TOOLS as HEALTH_TOOLS
                self.tools = HEALTH_TOOLS
                logger.info(f"SystemPlugin loaded {len(self.tools)} tools")
            except ImportError as e:
                logger.error(f"Failed to import health tools: {e}")
                # Create simple tools directly
                self.tools = self._create_basic_tools()
        
        return self.tools
    
    def _create_basic_tools(self):
        """Create basic tools when health_tool import fails"""
        try:
            import psutil
            from datetime import datetime
            
            class BasicHealthTool:
                def __init__(self):
                    self.name = "health_check"
                    self.description = "Basic system health check"
                    self.parameters = {
                        "detailed": {"type": "boolean", "default": False}
                    }
                
                async def execute(self, params: Dict[str, Any]) -> Any:
                    try:
                        return {
                            "timestamp": datetime.utcnow().isoformat(),
                            "status": "healthy",
                            "service": "mcp-server",
                            "cpu_percent": psutil.cpu_percent(interval=1),
                            "memory_percent": psutil.virtual_memory().percent,
                            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
                        }
                    except Exception as e:
                        return {
                            "timestamp": datetime.utcnow().isoformat(),
                            "status": "error",
                            "error": str(e)
                        }
                
                def validate_params(self, params: Dict[str, Any]) -> bool:
                    return isinstance(params, dict)
                
                def get_schema(self) -> Dict[str, Any]:
                    return {
                        "type": "object",
                        "properties": self.parameters,
                        "description": self.description
                    }
            
            return [BasicHealthTool()]
        except ImportError:
            logger.error("psutil not available, creating minimal tool")
            return []
    
    def initialize(self):
        """Initialize the plugin"""
        logger.info(f"Initializing SystemPlugin v{self.version}")
        
        # Load tools
        tools = self.get_tools()
        
        logger.info(f"SystemPlugin initialization complete with {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
    
    def cleanup(self):
        """Cleanup plugin resources"""
        logger.info("Cleaning up SystemPlugin")
        self.tools.clear()

# Export the plugin class for the plugin manager
Plugin = SystemPlugin
