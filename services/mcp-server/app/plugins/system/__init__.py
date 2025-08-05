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
    from plugin_manager import Plugin
    from tools.health_tool import TOOLS
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error in system plugin: {e}")
    # Fallback Plugin class
    class Plugin:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.name = config.get("name", "system")
            self.version = config.get("version", "1.0.0")
            self.tools = []
    TOOLS = []

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
                self.tools = []
        
        return self.tools
    
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
