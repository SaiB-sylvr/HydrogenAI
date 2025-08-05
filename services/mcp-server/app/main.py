"""
Minimal MCP Server - Guaranteed to Work
Handles MongoDB operations with streaming for large datasets
"""
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
from typing import Dict, Any, List, Optional
import os
import logging
import json
import time
from datetime import datetime
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Request models
class ExecuteRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = {}

# Global state
app_state = {
    "mongo_client": None,
    "ready": False,
    "tools": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    try:
        logger.info("Starting MCP Server...")
        
        # Try to connect to MongoDB
        try:
            from pymongo import MongoClient
            app_state["mongo_client"] = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            app_state["mongo_client"].admin.command('ping')
            logger.info("MongoDB connected")
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}")
            app_state["mongo_client"] = None
        
        # Load plugins from mounted directory
        try:
            from app.plugin_manager import PluginManager
            from app.tool_registry import ToolRegistry
            
            plugin_manager = PluginManager()
            plugin_manager.load_plugins("/app/plugins")
            app_state["plugin_manager"] = plugin_manager
            
            # Initialize tool registry
            tool_registry = ToolRegistry(plugin_manager)
            await tool_registry.initialize()
            app_state["tool_registry"] = tool_registry
            
            logger.info(f"Plugins loaded: {len(plugin_manager.plugins)} plugins")
            logger.info(f"Tools registered: {len(tool_registry.tools)} tools")
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            app_state["plugin_manager"] = None
            app_state["tool_registry"] = None
        
        # Register tools
        register_tools()
        
        app_state["ready"] = True
        logger.info(f"MCP Server ready with {len(app_state['tools'])} tools")
        
        yield
        
    finally:
        if app_state["mongo_client"]:
            app_state["mongo_client"].close()

app = FastAPI(
    title="MCP Server",
    version="2.0.0",
    lifespan=lifespan
)

# Tool implementations
class MongoDBTools:
    """MongoDB tools with pagination for large datasets"""
    
    @staticmethod
    async def find(params: Dict[str, Any]) -> Dict[str, Any]:
        """Find documents with improved pagination and resource management"""
        collection = params.get("collection", "users")
        filter_query = params.get("filter", {})
        limit = min(params.get("limit", 10), 5000)  # Cap at 5000 for real database analysis
        skip = params.get("skip", 0)
        projection = params.get("projection")
        
        if not app_state["mongo_client"]:
            return {
                "success": False,
                "error": "Database not connected",
                "documents": [],
                "count": 0,
                "has_more": False
            }
        
        cursor = None
        try:
            db = app_state["mongo_client"][MONGO_DB_NAME]
            
            # Validate collection exists
            if collection not in db.list_collection_names():
                return {
                    "success": False,
                    "error": f"Collection '{collection}' does not exist",
                    "documents": [],
                    "count": 0,
                    "has_more": False
                }
            
            # Build query with timeout and proper resource management
            cursor = db[collection].find(
                filter_query, 
                projection=projection,
                no_cursor_timeout=False,  # Ensure cursor times out
                max_time_ms=30000  # 30 second timeout
            ).skip(skip).limit(limit + 1)  # Get one extra to check if more exist
            
            # Convert to list with proper ObjectId handling
            documents = []
            doc_count = 0
            
            # Convert ObjectId and datetime to JSON-serializable formats
            def convert_doc(obj):
                from bson import ObjectId
                if isinstance(obj, ObjectId):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_doc(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_doc(item) for item in obj]
                elif hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                return obj
            
            for doc in cursor:
                if doc_count >= limit:
                    break
                
                documents.append(convert_doc(doc))
                doc_count += 1
            
            # Check if there are more documents
            has_more = False
            try:
                next_doc = next(cursor, None)
                has_more = next_doc is not None
            except StopIteration:
                has_more = False
            
            # Get total count for better pagination info
            total_count = db[collection].count_documents(filter_query)
            
            return {
                "success": True,
                "documents": documents,
                "count": len(documents),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "next_skip": skip + limit if has_more else None,
                    "total_pages": (total_count + limit - 1) // limit,
                    "current_page": (skip // limit) + 1
                }
            }
            
        except Exception as e:
            logger.error(f"Find error: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "count": 0,
                "has_more": False
            }
        finally:
            # Ensure cursor is properly closed
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
    
    @staticmethod
    async def count(params: Dict[str, Any]) -> Dict[str, Any]:
        """Count documents efficiently"""
        collection = params.get("collection", "users")
        filter_query = params.get("filter", {})
        use_estimate = params.get("use_estimate", True)
        
        if not app_state["mongo_client"]:
            return {
                "success": False,
                "error": "Database not connected",
                "count": 0
            }
        
        try:
            db = app_state["mongo_client"][MONGO_DB_NAME]
            
            if use_estimate and not filter_query:
                # Use fast estimated count for large collections
                count = db[collection].estimated_document_count()
                is_estimate = True
            else:
                # Use accurate count with filter
                count = db[collection].count_documents(filter_query)
                is_estimate = False
            
            return {
                "success": True,
                "count": count,
                "collection": collection,
                "is_estimate": is_estimate,
                "filter": filter_query
            }
            
        except Exception as e:
            logger.error(f"Count error: {e}")
            return {
                "success": False,
                "error": str(e),
                "count": 0
            }
    
    @staticmethod
    async def aggregate(params: Dict[str, Any]) -> Dict[str, Any]:
        """Run aggregation pipeline with allowDiskUse for large datasets"""
        collection = params.get("collection", "users")
        pipeline = params.get("pipeline", [])
        options = params.get("options", {})
        
        if not app_state["mongo_client"]:
            return {
                "success": False,
                "error": "Database not connected",
                "results": []
            }
        
        try:
            db = app_state["mongo_client"][MONGO_DB_NAME]
            
            # Enable disk use for large aggregations
            if "allowDiskUse" not in options:
                options["allowDiskUse"] = True
            
            # Add default batch size
            if "batchSize" not in options:
                options["batchSize"] = 1000
            
            # Execute pipeline
            cursor = db[collection].aggregate(pipeline, **options)
            
            # Collect results with limit
            results = []
            max_results = 10000  # Prevent memory issues
            
            for doc in cursor:
                if "_id" in doc and hasattr(doc["_id"], "__str__"):
                    doc["_id"] = str(doc["_id"])
                results.append(doc)
                
                if len(results) >= max_results:
                    break
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "truncated": len(results) == max_results,
                "pipeline_stages": len(pipeline)
            }
            
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    @staticmethod
    async def list_collections(params: Dict[str, Any]) -> Dict[str, Any]:
        """List all collections with stats"""
        include_stats = params.get("include_stats", False)
        
        if not app_state["mongo_client"]:
            return {
                "success": False,
                "error": "Database not connected",
                "collections": []
            }
        
        try:
            db = app_state["mongo_client"][MONGO_DB_NAME]
            collection_names = db.list_collection_names()
            
            collections = []
            for name in collection_names:
                collection_info = {"name": name}
                
                if include_stats:
                    try:
                        stats = db.command("collStats", name)
                        collection_info.update({
                            "count": stats.get("count", 0),
                            "size": stats.get("size", 0),
                            "avgObjSize": stats.get("avgObjSize", 0),
                            "indexCount": len(stats.get("indexSizes", {}))
                        })
                    except:
                        pass
                
                collections.append(collection_info)
            
            return {
                "success": True,
                "collections": collections,
                "database": MONGO_DB_NAME,
                "total": len(collections)
            }
            
        except Exception as e:
            logger.error(f"List collections error: {e}")
            return {
                "success": False,
                "error": str(e),
                "collections": []
            }

    @staticmethod
    async def database_stats(params: Dict[str, Any]) -> Dict[str, Any]:
        """Get database statistics"""
        if not app_state["mongo_client"]:
            return {
                "success": False,
                "error": "Database not connected"
            }
        
        try:
            db = app_state["mongo_client"][MONGO_DB_NAME]
            stats = db.command("dbStats")
            
            return {
                "success": True,
                "database": MONGO_DB_NAME,
                "stats": {
                    "collections": stats.get("collections", 0),
                    "objects": stats.get("objects", 0),
                    "dataSize": stats.get("dataSize", 0),
                    "storageSize": stats.get("storageSize", 0),
                    "indexes": stats.get("indexes", 0),
                    "indexSize": stats.get("indexSize", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def collection_stats(params: Dict[str, Any]) -> Dict[str, Any]:
        """Get collection statistics"""
        collection = params.get("collection")
        if not collection:
            return {
                "success": False,
                "error": "Collection name required"
            }
            
        if not app_state["mongo_client"]:
            return {
                "success": False,
                "error": "Database not connected"
            }
        
        try:
            db = app_state["mongo_client"][MONGO_DB_NAME]
            stats = db.command("collStats", collection)
            
            return {
                "success": True,
                "collection": collection,
                "stats": {
                    "count": stats.get("count", 0),
                    "size": stats.get("size", 0),
                    "avgObjSize": stats.get("avgObjSize", 0),
                    "storageSize": stats.get("storageSize", 0),
                    "indexCount": len(stats.get("indexSizes", {})),
                    "totalIndexSize": stats.get("totalIndexSize", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Collection stats error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Simple in-memory tool implementations
class MemoryTools:
    """Fallback tools that work without external dependencies"""
    
    @staticmethod
    async def answer_question(params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple question answering without LLM"""
        query = params.get("query", "")
        
        # Simple pattern-based responses
        responses = {
            "hello": "Hello! I'm Hydrogen AI. I can help you query and analyze data.",
            "help": "I can help you with: counting records, finding data, running aggregations, and answering questions about your database.",
            "what": "I'm an AI assistant that helps you interact with your data through natural language queries.",
            "how": "Just ask me questions about your data in plain English, and I'll help you get the answers you need."
        }
        
        # Find matching response
        query_lower = query.lower()
        for key, response in responses.items():
            if key in query_lower:
                return {
                    "success": True,
                    "result": {
                        "answer": response,
                        "confidence": 0.8
                    }
                }
        
        # Default response
        return {
            "success": True,
            "result": {
                "answer": "I understand you're asking about: " + query + ". Please be more specific about what data you'd like to see.",
                "confidence": 0.5
            }
        }

def register_tools():
    """Register all available tools"""
    # MongoDB tools
    app_state["tools"]["mongodb_find"] = MongoDBTools.find
    app_state["tools"]["mongodb_count"] = MongoDBTools.count
    app_state["tools"]["mongodb_aggregate"] = MongoDBTools.aggregate
    app_state["tools"]["mongodb_list_collections"] = MongoDBTools.list_collections
    app_state["tools"]["mongodb_database_stats"] = MongoDBTools.database_stats
    app_state["tools"]["mongodb_collection_stats"] = MongoDBTools.collection_stats
    
    # Memory tools (always available)
    app_state["tools"]["answer_question"] = MemoryTools.answer_question
    
    logger.info(f"Registered {len(app_state['tools'])} tools")

# API Endpoints
@app.post("/execute")
async def execute_tool(request: Request):
    """Execute a tool with timeout protection"""
    try:
        body = await request.json()
        logger.info(f"Received request: {body}")
        tool_name = body.get("tool")
        params = body.get("params", {})
    except Exception as e:
        logger.error(f"Failed to parse request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    if not tool_name:
        raise HTTPException(status_code=400, detail="Tool name required")
    
    # Try plugin system first
    if app_state.get("tool_registry"):
        try:
            result = await asyncio.wait_for(
                app_state["tool_registry"].execute(tool_name, params),
                timeout=30.0
            )
            return {"success": True, "result": result}
        except ValueError as e:
            # Tool not found in plugin system, try legacy tools
            logger.info(f"Tool '{tool_name}' not in plugin system, trying legacy: {e}")
        except Exception as e:
            logger.error(f"Plugin tool execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Fallback to legacy tool system
    if tool_name not in app_state["tools"]:
        # Get available tools from both systems
        available = list(app_state["tools"].keys())
        if app_state.get("tool_registry"):
            available.extend(list(app_state["tool_registry"].tools.keys()))
        
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available: {available}"
        )
    
    try:
        # Execute legacy tool with timeout
        tool_func = app_state["tools"][tool_name]
        result = await asyncio.wait_for(
            tool_func(params),
            timeout=30.0
        )
        
        return {
            "success": True,
            "tool": tool_name,
            "result": result
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Tool execution timed out")
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e)
        }

@app.get("/tools")
async def list_tools():
    """List available tools"""
    tools = []
    
    # Add legacy tools
    for name in app_state["tools"]:
        tools.append({
            "name": name,
            "type": "legacy",
            "available": True,
            "description": f"Execute {name} operations"
        })
    
    # Add plugin tools
    if app_state.get("tool_registry"):
        for name, tool in app_state["tool_registry"].tools.items():
            # Safely get schema, fallback to basic schema if method doesn't exist
            try:
                schema = tool.get_schema() if hasattr(tool, 'get_schema') else {}
            except Exception as e:
                logger.warning(f"Failed to get schema for plugin tool {name}: {e}")
                schema = {"type": "object", "properties": {}}
                
            tools.append({
                "name": name,
                "type": "plugin",
                "available": True,
                "description": tool.description,
                "schema": schema
            })
    
    return {
        "tools": tools,
        "total": len(tools)
    }

@app.get("/health")
async def health_check():
    """Health check"""
    plugin_status = {}
    tool_counts = {"legacy": len(app_state["tools"]), "plugin": 0}
    
    if app_state.get("plugin_manager"):
        plugin_status = app_state["plugin_manager"].get_status()
    
    if app_state.get("tool_registry"):
        tool_counts["plugin"] = len(app_state["tool_registry"].tools)
    
    total_tools = sum(tool_counts.values())
    
    return {
        "status": "healthy" if app_state["ready"] else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "mongodb": "connected" if app_state["mongo_client"] else "not_connected",
        "tools": tool_counts,
        "plugins": plugin_status,
        "tools_loaded": total_tools,  # For compatibility with test
        "total_tools": total_tools
    }

@app.get("/")
async def root():
    return {
        "service": "MCP Server",
        "version": "2.0.0",
        "status": "running"
    }