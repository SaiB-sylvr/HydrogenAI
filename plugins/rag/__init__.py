"""
RAG Plugin for MCP Server
"""
from .rag_tool import RAGPlugin

# Export the plugin class for the plugin manager
Plugin = RAGPlugin

__all__ = ["RAGPlugin", "Plugin"]