"""
MongoDB Plugin for MCP Server
"""
from .mongodb_tool import MongoDBPlugin

# Export the plugin class for the plugin manager
Plugin = MongoDBPlugin

__all__ = ["MongoDBPlugin", "Plugin"]