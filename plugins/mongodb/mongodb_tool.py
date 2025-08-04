"""
MongoDB Plugin with Security Fixes and Enhanced Features
"""
import os
import re
from typing import Dict, Any, List, Optional, Union
from pymongo import MongoClient
from bson import ObjectId
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json
from pymongo.errors import OperationFailure, ConfigurationError

logger = logging.getLogger(__name__)

class MongoDBSecurityValidator:
    """Validates and sanitizes MongoDB queries"""
    
    @staticmethod
    def validate_collection_name(name: str) -> bool:
        """Validate collection name against injection"""
        if not name or not isinstance(name, str):
            return False
        # Only allow alphanumeric, underscore, and hyphen
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))
    
    @staticmethod
    def sanitize_filter(filter_query: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize filter to prevent NoSQL injection"""
        if not isinstance(filter_query, dict):
            return {}
        
        sanitized = {}
        dangerous_operators = ['$where', '$function', '$accumulator', '$javascript']
        
        def clean_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
            cleaned = {}
            for key, value in obj.items():
                # Check for dangerous operators
                if isinstance(key, str) and key in dangerous_operators:
                    logger.warning(f"Blocked dangerous operator: {key}")
                    continue
                
                # Recursively clean nested objects
                if isinstance(value, dict):
                    cleaned[key] = clean_dict(value)
                elif isinstance(value, list):
                    cleaned[key] = [clean_dict(item) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned[key] = value
            
            return cleaned
        
        return clean_dict(filter_query)
    
    @staticmethod
    def validate_pipeline(pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate aggregation pipeline"""
        if not isinstance(pipeline, list):
            return []
        
        validated = []
        dangerous_stages = ['$function', '$accumulator']
        
        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
            
            # Check stage operators
            stage_op = next(iter(stage.keys())) if stage else None
            if stage_op in dangerous_stages:
                logger.warning(f"Blocked dangerous pipeline stage: {stage_op}")
                continue
            
            # Sanitize stage content
            validated.append(MongoDBSecurityValidator.sanitize_filter(stage))
        
        return validated

class MongoDBBaseTool:
    """Base class for MongoDB tools with connection pooling and security"""
    
    def __init__(self, name: str, description: str, clients: Dict[str, MongoClient], default_database: Optional[str] = None):
        self.name = name
        self.description = description
        self.clients = clients
        self.default_client = list(clients.keys())[0] if clients else None
        self.default_database = default_database
        self.validator = MongoDBSecurityValidator()
    
    def _get_client(self, cluster: Optional[str] = None) -> MongoClient:
        """Get MongoDB client for specified cluster"""
        cluster = cluster or self.default_client
        if cluster not in self.clients:
            raise ValueError(f"Unknown cluster: {cluster}")
        return self.clients[cluster]
    
    def _convert_objectid(self, data: Any) -> Any:
        """Convert ObjectId to string recursively"""
        if isinstance(data, ObjectId):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {k: self._convert_objectid(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_objectid(item) for item in data]
        return data
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema for this tool"""
        return {}
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        # Override in subclasses
        return True

class MongoDBFindTool(MongoDBBaseTool):
    """Tool for finding documents with advanced features and security"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        # Validate parameters
        if not self.validate_params(params):
            raise ValueError("Invalid parameters")
        
        cluster = params.get("cluster", self.default_client)
        database = params.get("database")
        collection = params.get("collection")
        
        # Validate collection name
        if not self.validator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        # Sanitize filter
        filter_query = self.validator.sanitize_filter(params.get("filter", {}))
        projection = params.get("projection")
        sort = params.get("sort")
        limit = min(params.get("limit", 10), 1000)  # Cap at 1000
        skip = max(params.get("skip", 0), 0)
        
        try:
            client = self._get_client(cluster)
            # Use provided database, or default database, or fail gracefully
            if database:
                db = client[database]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            coll = db[collection]
            
            # Build query with timeout
            cursor = coll.find(filter_query, projection, max_time_ms=30000)  # 30 second timeout
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            documents = list(cursor)
            
            return {
                "documents": self._convert_objectid(documents),
                "count": len(documents),
                "cluster": cluster,
                "database": db.name,
                "collection": collection
            }
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {e}")
            raise ValueError(f"Query execution failed: {e}")
        except Exception as e:
            logger.error(f"MongoDB find error: {e}")
            raise
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate find parameters"""
        required = ["collection"]
        for field in required:
            if field not in params:
                return False
        
        # Validate types
        if not isinstance(params.get("filter", {}), dict):
            return False
        if not isinstance(params.get("limit", 10), int):
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cluster": {"type": "string", "description": "MongoDB cluster name"},
                "database": {"type": "string", "description": "Database name"},
                "collection": {"type": "string", "description": "Collection name", "pattern": "^[a-zA-Z0-9_-]+$"},
                "filter": {"type": "object", "description": "Query filter"},
                "projection": {"type": "object", "description": "Field projection"},
                "sort": {"type": "object", "description": "Sort specification"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 10},
                "skip": {"type": "integer", "minimum": 0, "default": 0}
            },
            "required": ["collection"]
        }

class MongoDBAggregateTool(MongoDBBaseTool):
    """Tool for aggregation pipelines with optimization and security"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        if not self.validate_params(params):
            raise ValueError("Invalid parameters")
        
        cluster = params.get("cluster", self.default_client)
        database = params.get("database")
        collection = params.get("collection")
        
        # Validate collection
        if not self.validator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        # Validate and sanitize pipeline
        pipeline = self.validator.validate_pipeline(params.get("pipeline", []))
        if not pipeline:
            raise ValueError("Empty or invalid pipeline")
        
        options = params.get("options", {})
        
        try:
            client = self._get_client(cluster)
            # Use provided database, or default database, or fail gracefully
            if database:
                db = client[database]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            coll = db[collection]
            
            # Add performance hints
            if "allowDiskUse" not in options:
                options["allowDiskUse"] = True
            
            # Add timeout
            options["maxTimeMS"] = options.get("maxTimeMS", 30000)  # 30 seconds
            
            # Execute pipeline
            cursor = coll.aggregate(pipeline, **options)
            results = list(cursor)
            
            return {
                "results": self._convert_objectid(results),
                "count": len(results),
                "cluster": cluster,
                "database": db.name,
                "collection": collection,
                "pipeline_stages": len(pipeline)
            }
        except OperationFailure as e:
            logger.error(f"MongoDB aggregation failed: {e}")
            raise ValueError(f"Aggregation failed: {e}")
        except Exception as e:
            logger.error(f"MongoDB aggregate error: {e}")
            raise
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate aggregation parameters"""
        if "collection" not in params:
            return False
        if not isinstance(params.get("pipeline", []), list):
            return False
        return True

class MongoDBListCollectionsTool(MongoDBBaseTool):
    """Tool for listing collections"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        cluster = params.get("cluster", self.default_client)
        database = params.get("database")
        
        try:
            client = self._get_client(cluster)
            # Use provided database, or default database, or fail gracefully
            if database:
                db = client[database]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            
            collections = db.list_collection_names()
            
            # Get basic info for each collection
            collection_info = []
            for coll_name in collections:
                try:
                    stats = db.command("collStats", coll_name, indexDetails=False)
                    collection_info.append({
                        "name": coll_name,
                        "count": stats.get("count", 0),
                        "size": stats.get("size", 0),
                        "avgObjSize": stats.get("avgObjSize", 0)
                    })
                except:
                    collection_info.append({"name": coll_name})
            
            return {
                "collections": collection_info,
                "total": len(collections),
                "database": db.name,
                "cluster": cluster
            }
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

class MongoDBSampleDocumentsTool(MongoDBBaseTool):
    """Tool for sampling documents to understand schema"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        collection = params.get("collection")
        if not self.validator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        sample_size = min(params.get("sample_size", 10), 100)
        
        try:
            client = self._get_client()
            # Use provided database, or default database, or fail gracefully
            if params.get("database"):
                db = client[params.get("database")]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            coll = db[collection]
            
            # Use aggregation for random sampling
            pipeline = [{"$sample": {"size": sample_size}}]
            samples = list(coll.aggregate(pipeline))
            
            # Analyze schema
            schema = self._analyze_schema(samples)
            
            return {
                "samples": self._convert_objectid(samples),
                "schema": schema,
                "sample_size": len(samples),
                "collection": collection
            }
        except Exception as e:
            logger.error(f"Failed to sample documents: {e}")
            raise
    
    def _analyze_schema(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze document schema"""
        if not documents:
            return {}
        
        field_types = {}
        field_counts = {}
        
        for doc in documents:
            for field, value in doc.items():
                # Track field frequency
                field_counts[field] = field_counts.get(field, 0) + 1
                
                # Track field types
                type_name = type(value).__name__
                if field not in field_types:
                    field_types[field] = set()
                field_types[field].add(type_name)
        
        # Build schema summary
        schema = {}
        total_docs = len(documents)
        
        for field, count in field_counts.items():
            schema[field] = {
                "types": list(field_types.get(field, [])),
                "frequency": count / total_docs,
                "required": count == total_docs
            }
        
        return schema

class MongoDBGetIndexesTool(MongoDBBaseTool):
    """Tool for getting collection indexes"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        collection = params.get("collection")
        if not self.validator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        try:
            client = self._get_client()
            # Use provided database, or default database, or fail gracefully
            if params.get("database"):
                db = client[params.get("database")]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            coll = db[collection]
            
            indexes = []
            for index in coll.list_indexes():
                indexes.append({
                    "name": index.get("name"),
                    "keys": index.get("key"),
                    "unique": index.get("unique", False),
                    "sparse": index.get("sparse", False)
                })
            
            return {
                "indexes": indexes,
                "total": len(indexes),
                "collection": collection
            }
        except Exception as e:
            logger.error(f"Failed to get indexes: {e}")
            raise

class MongoDBCountTool(MongoDBBaseTool):
    """Tool for counting documents"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        collection = params.get("collection")
        if not self.validator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        filter_query = self.validator.sanitize_filter(params.get("filter", {}))
        
        try:
            client = self._get_client()
            # Use provided database, or default database, or fail gracefully
            if params.get("database"):
                db = client[params.get("database")]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            coll = db[collection]
            
            count = coll.count_documents(filter_query, maxTimeMS=10000)
            
            return {
                "count": count,
                "collection": collection,
                "filter": filter_query
            }
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            raise


class MongoDBCreateIndexTool(MongoDBBaseTool):
    """Tool for creating indexes on collections"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        collection = params.get("collection")
        if not MongoDBSecurityValidator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        index_spec = params.get("index", {})
        background = params.get("background", True)
        index_name = params.get("name")
        
        try:
            client = self._get_client()
            # Use provided database, or default database, or fail gracefully
            if params.get("database"):
                db = client[params.get("database")]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            coll = db[collection]
            
            # Create index options
            index_options = {"background": background}
            if index_name:
                index_options["name"] = index_name
            
            # Create the index
            result = coll.create_index(
                list(index_spec.items()),
                **index_options
            )
            
            return {
                "index_name": result,
                "collection": collection,
                "success": True
            }
        except Exception as e:
            logger.warning(f"Index creation may have failed (possibly already exists): {e}")
            return {
                "error": str(e),
                "success": False,
                "message": "Index may already exist"
            }


class MongoDBCollectionStatsTool(MongoDBBaseTool):
    """Tool for getting collection statistics"""
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        collection = params.get("collection")
        if not MongoDBSecurityValidator.validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")
        
        try:
            client = self._get_client()
            # Use provided database, or default database, or fail gracefully
            if params.get("database"):
                db = client[params.get("database")]
            elif self.default_database:
                db = client[self.default_database]
            else:
                return {
                    "error": "No default database defined",
                    "success": False
                }
            
            # Get collection stats
            stats = db.command("collStats", collection)
            
            return {
                "success": True,
                "result": {
                    "count": stats.get("count", 0),
                    "size": stats.get("size", 0),
                    "avgObjSize": stats.get("avgObjSize", 0),
                    "storageSize": stats.get("storageSize", 0),
                    "nindexes": stats.get("nindexes", 0),
                    "totalIndexSize": stats.get("totalIndexSize", 0)
                },
                "collection": collection
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "error": str(e),
                "success": False
            }

class MongoDBPlugin:
    """Enhanced MongoDB plugin with security and multi-cluster support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "mongodb")
        self.version = config.get("version", "1.0.0")
        self.clients: Dict[str, MongoClient] = {}
        
    def initialize(self):
        """Initialize MongoDB connections"""
        # Check for multi-cluster configuration
        if "clusters" in self.config.get("configuration", {}):
            # Multi-cluster setup
            for cluster_config in self.config["configuration"]["clusters"]:
                name = cluster_config["name"]
                connection_string = os.getenv(
                    cluster_config.get("env_var", f"MONGO_URI_{name.upper()}"),
                    cluster_config.get("connection_string")
                )
                
                try:
                    client = MongoClient(
                        connection_string,
                        serverSelectionTimeoutMS=5000,
                        maxPoolSize=50,
                        minPoolSize=10
                    )
                    client.admin.command('ping')
                    self.clients[name] = client
                    logger.info(f"Connected to MongoDB cluster: {name}")
                except Exception as e:
                    logger.error(f"Failed to connect to cluster {name}: {e}")
        else:
            # Single cluster setup (backward compatible)
            connection_string = os.getenv(
                "MONGO_URI",
                self.config["configuration"].get("connection_string")
            )
            
            try:
                client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000,
                    maxPoolSize=50,
                    minPoolSize=10
                )
                client.admin.command('ping')
                self.clients["default"] = client
                logger.info("Connected to default MongoDB cluster")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
    
    def get_tools(self) -> List[Any]:
        """Get tools provided by this plugin"""
        if not self.clients:
            logger.warning("No MongoDB connections available")
            return []
        
        # Get default database from configuration
        default_database = self.config.get("configuration", {}).get("default_database")
        
        tools = [
            MongoDBFindTool("mongodb_find", "Find documents in MongoDB with security", self.clients, default_database),
            MongoDBAggregateTool("mongodb_aggregate", "Run aggregation pipeline securely", self.clients, default_database),
            MongoDBListCollectionsTool("mongodb_list_collections", "List all collections", self.clients, default_database),
            MongoDBSampleDocumentsTool("mongodb_sample_documents", "Sample documents for schema", self.clients, default_database),
            MongoDBGetIndexesTool("mongodb_get_indexes", "Get collection indexes", self.clients, default_database),
            MongoDBCountTool("mongodb_count", "Count documents in collection", self.clients, default_database),
            MongoDBCreateIndexTool("mongodb_create_index", "Create index on collection", self.clients, default_database),
            MongoDBCollectionStatsTool("mongodb_collection_stats", "Get collection statistics", self.clients, default_database)
        ]
        
        return tools
    
    def cleanup(self):
        """Cleanup MongoDB connections"""
        for name, client in self.clients.items():
            try:
                client.close()
                logger.info(f"Closed connection to cluster: {name}")
            except Exception as e:
                logger.error(f"Error closing connection to {name}: {e}")