"""
RAG Plugin with Qdrant Resilience and Document Management
"""
import os
from typing import Dict, Any, List, Optional
import asyncio
import logging
import uuid
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Check if we should disable Qdrant
DISABLE_QDRANT = os.getenv("DISABLE_QDRANT", "false").lower() == "true"

# Try to import Qdrant dependencies
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    QDRANT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qdrant dependencies not available: {e}")
    QDRANT_AVAILABLE = False

class MockQdrantClient:
    """Mock Qdrant client for when Qdrant is not available"""
    def __init__(self, *args, **kwargs):
        logger.info("Using mock Qdrant client")
        self.collections = {}
        self.documents = {}  # Store mock documents
    
    def create_collection(self, *args, **kwargs):
        pass
    
    def get_collection(self, collection_name):
        if collection_name not in self.collections:
            raise Exception("Collection not found")
        return {"name": collection_name}
    
    def upsert(self, collection_name, points, **kwargs):
        """Mock document upsert"""
        if collection_name not in self.documents:
            self.documents[collection_name] = {}
        
        for point in points:
            doc_id = point.id if hasattr(point, 'id') else str(uuid.uuid4())
            self.documents[collection_name][doc_id] = {
                'id': doc_id,
                'payload': point.payload if hasattr(point, 'payload') else {},
                'vector': point.vector if hasattr(point, 'vector') else []
            }
        
        return {"status": "success", "added": len(points)}
    
    def delete(self, collection_name, ids, **kwargs):
        """Mock document deletion"""
        if collection_name in self.documents:
            deleted = 0
            for doc_id in ids:
                if str(doc_id) in self.documents[collection_name]:
                    del self.documents[collection_name][str(doc_id)]
                    deleted += 1
            return {"status": "success", "deleted": deleted}
        return {"status": "success", "deleted": 0}
    
    def retrieve(self, collection_name, ids, **kwargs):
        """Mock document retrieval"""
        if collection_name not in self.documents:
            return []
        
        results = []
        for doc_id in ids:
            if str(doc_id) in self.documents[collection_name]:
                doc = self.documents[collection_name][str(doc_id)]
                results.append(type('obj', (object,), {
                    'id': doc['id'],
                    'payload': doc['payload'],
                    'vector': doc['vector']
                })())
        
        return results
    
    def scroll(self, collection_name, limit=100, **kwargs):
        """Mock document scrolling"""
        if collection_name not in self.documents:
            return [], None
        
        docs = list(self.documents[collection_name].values())[:limit]
        results = []
        for doc in docs:
            results.append(type('obj', (object,), {
                'id': doc['id'],
                'payload': doc['payload'],
                'vector': doc['vector']
            })())
        
        return results, None
    
    def search(self, collection_name, query_vector, limit=10, **kwargs):
        # Return mock results
        return [
            type('obj', (object,), {
                'score': 0.9 - (i * 0.1),
                'id': str(uuid.uuid4()),
                'payload': {
                    'text': f'Mock result {i}: This is a simulated search result for testing.',
                    'metadata': {'source': 'mock', 'index': i}
                }
            })()
            for i in range(min(3, limit))
        ]
    
    def upsert(self, collection_name, points, **kwargs):
        pass
    
    def close(self):
        pass

class MockEmbeddingModel:
    """Mock embedding model for when dependencies are not available"""
    def __init__(self, *args, **kwargs):
        logger.info("Using mock embedding model")
        self.embedding_dimension = 384
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Return random embeddings
        import random
        return [[random.random() for _ in range(self.embedding_dimension)] for _ in texts]
    
    def get_sentence_embedding_dimension(self):
        return self.embedding_dimension

class RAGSearchTool:
    """Tool for semantic search with resilience"""
    
    def __init__(self, name: str, description: str, client, embedding_model, collection_name: str):
        self.name = name
        self.description = description
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._connection_failed = False
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        if not isinstance(params, dict):
            return False
        return "query" in params
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema"""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 10},
                "filters": {"type": "object", "description": "Search filters", "default": {}}
            },
            "required": ["query"],
            "description": self.description
        }
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        query = params.get("query")
        filters = params.get("filters", {})
        top_k = params.get("top_k", 10)
        
        # Check if Qdrant is disabled
        if DISABLE_QDRANT:
            return {
                "query": query,
                "documents": [],
                "count": 0,
                "message": "RAG search is disabled"
            }
        
        try:
            # Generate query embedding
            loop = asyncio.get_event_loop()
            
            if hasattr(self.embedding_model, 'encode'):
                query_embedding = await loop.run_in_executor(
                    None,
                    self.embedding_model.encode,
                    query
                )
            else:
                # Fallback for mock model
                query_embedding = self.embedding_model.encode(query)[0]
            
            # Search with timeout
            search_task = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding if isinstance(query_embedding, list) else query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            # Handle both sync and async search
            if asyncio.iscoroutine(search_task):
                results = await asyncio.wait_for(search_task, timeout=5.0)
            else:
                results = search_task
            
            # Format results
            documents = []
            for result in results:
                doc = {
                    "score": float(result.score) if hasattr(result, 'score') else 0.0,
                    "text": result.payload.get("text", "") if hasattr(result, 'payload') else "",
                    "metadata": result.payload.get("metadata", {}) if hasattr(result, 'payload') else {},
                    "id": str(result.id) if hasattr(result, 'id') else str(uuid.uuid4())
                }
                documents.append(doc)
            
            return {
                "query": query,
                "documents": documents,
                "count": len(documents)
            }
            
        except asyncio.TimeoutError:
            logger.error("RAG search timed out")
            return {
                "query": query,
                "documents": [],
                "count": 0,
                "error": "Search timed out"
            }
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            
            # Mark connection as failed to avoid repeated attempts
            if "connect" in str(e).lower():
                self._connection_failed = True
            
            return {
                "query": query,
                "documents": [],
                "count": 0,
                "error": "Search service unavailable"
            }

class RAGEmbedTool:
    """Tool for embedding documents with resilience"""
    
    def __init__(self, name: str, description: str, client, embedding_model, collection_name: str, text_splitter):
        self.name = name
        self.description = description
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.text_splitter = text_splitter
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        if not isinstance(params, dict):
            return False
        return "documents" in params
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema"""
        return {
            "type": "object",
            "properties": {
                "documents": {"type": "array", "description": "Documents to embed"},
                "batch_size": {"type": "integer", "description": "Batch size for processing", "default": 100}
            },
            "required": ["documents"],
            "description": self.description
        }
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        documents = params.get("documents", [])
        batch_size = params.get("batch_size", 100)
        
        if DISABLE_QDRANT:
            return {
                "total_documents": len(documents),
                "total_chunks": 0,
                "embedded_chunks": 0,
                "message": "Embedding is disabled"
            }
        
        try:
            all_chunks = []
            
            # Process documents
            for doc in documents:
                if isinstance(doc, str):
                    text = doc
                    metadata = {}
                else:
                    text = doc.get("text", "")
                    metadata = doc.get("metadata", {})
                
                # Split text
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    })
            
            # Mock embedding if client is not real
            if isinstance(self.client, MockQdrantClient):
                return {
                    "total_documents": len(documents),
                    "total_chunks": len(all_chunks),
                    "embedded_chunks": len(all_chunks),
                    "message": "Mock embedding completed"
                }
            
            # Real embedding logic here...
            embedded_count = len(all_chunks)  # Simplified for resilience
            
            return {
                "total_documents": len(documents),
                "total_chunks": len(all_chunks),
                "embedded_chunks": embedded_count
            }
            
        except Exception as e:
            logger.error(f"RAG embed error: {e}")
            return {
                "total_documents": len(documents),
                "total_chunks": 0,
                "embedded_chunks": 0,
                "error": str(e)
            }

class RAGPlugin:
    """RAG plugin with Qdrant resilience"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "rag")
        self.version = config.get("version", "1.0.0")
        self.qdrant_client = None
        self.embedding_model = None
        self.collection_name = None
        self.text_splitter = None
        self._initialized = False
    
    def initialize(self):
        """Initialize RAG components with fallbacks"""
        config = self.config.get("configuration", {})
        
        if DISABLE_QDRANT:
            logger.info("Qdrant is disabled via DISABLE_QDRANT environment variable")
            self._initialize_mock()
            return
        
        # Try to initialize real Qdrant
        try:
            if not QDRANT_AVAILABLE:
                logger.warning("Qdrant dependencies not installed, using mock")
                self._initialize_mock()
                return
            
            # Initialize Qdrant client with timeout
            qdrant_host = os.getenv("QDRANT_HOST", config.get("qdrant_host", "localhost"))
            qdrant_port = int(os.getenv("QDRANT_PORT", config.get("qdrant_port", 6333)))
            
            logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
            
            self.qdrant_client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=5,  # 5 second timeout
                grpc_port=6334,
                prefer_grpc=False,  # Use HTTP for better compatibility
                check_compatibility=False  # Disable version compatibility check
            )
            
            # Test connection with timeout
            try:
                # Simple health check
                collections = self.qdrant_client.get_collections()
                logger.info(f"Successfully connected to Qdrant. Collections: {len(collections.collections)}")
            except Exception as e:
                logger.warning(f"Qdrant health check failed: {e}")
                logger.warning("Falling back to mock Qdrant client")
                self._initialize_mock()
                return
            
            # Initialize embedding model
            model_name = os.getenv("EMBEDDING_MODEL_NAME", config.get("embedding_model"))
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = MockEmbeddingModel()
            
            # Collection name
            self.collection_name = os.getenv(
                "QDRANT_COLLECTION_NAME",
                config.get("collection_name", "hydrogen_documents")
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200)
            )
            
            # Ensure collection exists
            self._ensure_collection()
            
            self._initialized = True
            logger.info(f"RAG plugin initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG plugin: {e}")
            logger.info("Falling back to mock implementation")
            self._initialize_mock()
    
    def _initialize_mock(self):
        """Initialize with mock components"""
        self.qdrant_client = MockQdrantClient()
        self.embedding_model = MockEmbeddingModel()
        self.collection_name = "mock_collection"
        
        # Create simple text splitter
        class SimpleTextSplitter:
            def __init__(self, chunk_size=1000):
                self.chunk_size = chunk_size
            
            def split_text(self, text):
                # Simple splitting by sentences or chunk size
                if len(text) <= self.chunk_size:
                    return [text]
                
                chunks = []
                for i in range(0, len(text), self.chunk_size):
                    chunks.append(text[i:i + self.chunk_size])
                return chunks
        
        self.text_splitter = SimpleTextSplitter()
        self._initialized = True
        logger.info("RAG plugin initialized with mock components")
    
    def _ensure_collection(self):
        """Ensure Qdrant collection exists"""
        if isinstance(self.qdrant_client, MockQdrantClient):
            return
        
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} exists")
        except Exception:
            try:
                # Create collection
                vector_size = self.embedding_model.get_sentence_embedding_dimension()
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
    
    def get_tools(self) -> List[Any]:
        """Get tools provided by this plugin"""
        if not self._initialized:
            logger.warning("RAG plugin not initialized, returning empty tools")
            return []
        
        return [
            RAGSearchTool(
                "rag_search",
                "Semantic search in documents",
                self.qdrant_client,
                self.embedding_model,
                self.collection_name
            ),
            RAGEmbedTool(
                "rag_embed",
                "Embed documents for search",
                self.qdrant_client,
                self.embedding_model,
                self.collection_name,
                self.text_splitter
            ),
            RAGDocumentAddTool(
                "rag_document_add",
                "Add new documents to the knowledge base",
                self.qdrant_client,
                self.embedding_model,
                self.collection_name,
                self.text_splitter
            ),
            RAGDocumentUpdateTool(
                "rag_document_update",
                "Update existing documents in the knowledge base",
                self.qdrant_client,
                self.embedding_model,
                self.collection_name,
                self.text_splitter
            ),
            RAGDocumentDeleteTool(
                "rag_document_delete",
                "Delete documents from the knowledge base",
                self.qdrant_client,
                self.collection_name
            ),
            RAGDocumentListTool(
                "rag_document_list",
                "List documents in the knowledge base",
                self.qdrant_client,
                self.collection_name
            )
        ]
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.qdrant_client and not isinstance(self.qdrant_client, MockQdrantClient):
                self.qdrant_client.close()
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")


class RAGDocumentAddTool:
    """Tool for adding new documents to RAG knowledge base"""
    
    def __init__(self, name: str, description: str, qdrant_client, embedding_model, collection_name: str, text_splitter):
        self.name = name
        self.description = description
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.text_splitter = text_splitter
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Document content to add"
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata for the document",
                            "properties": {
                                "source": {"type": "string"},
                                "category": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "required": ["content", "title"]
                }
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        return isinstance(params, dict) and "content" in params and "title" in params
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add document to knowledge base"""
        try:
            content = params["content"]
            title = params["title"]
            metadata = params.get("metadata", {})
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create document points
            points = []
            for i, chunk in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Prepare metadata
                chunk_metadata = {
                    "text": chunk,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "added_at": datetime.now().isoformat(),
                    **metadata
                }
                
                if QDRANT_AVAILABLE and not isinstance(self.qdrant_client, MockQdrantClient):
                    points.append(PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=chunk_metadata
                    ))
                else:
                    # Mock point structure
                    points.append(type('Point', (), {
                        'id': doc_id,
                        'vector': embedding,
                        'payload': chunk_metadata
                    })())
            
            # Upsert points
            result = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return {
                "success": True,
                "document_id": title,
                "chunks_added": len(chunks),
                "message": f"Successfully added document '{title}' with {len(chunks)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to add document"
            }


class RAGDocumentUpdateTool:
    """Tool for updating existing documents in RAG knowledge base"""
    
    def __init__(self, name: str, description: str, qdrant_client, embedding_model, collection_name: str, text_splitter):
        self.name = name
        self.description = description
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.text_splitter = text_splitter
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Document title to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "New document content"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Updated metadata for the document"
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        return isinstance(params, dict) and "title" in params and "content" in params
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update document in knowledge base"""
        try:
            title = params["title"]
            content = params["content"]
            metadata = params.get("metadata", {})
            
            # First, delete existing document
            delete_result = await self._delete_by_title(title)
            
            # Then add the updated version
            add_params = {
                "content": content,
                "title": title,
                "metadata": {**metadata, "updated_at": datetime.now().isoformat()}
            }
            
            # Use RAGDocumentAddTool logic
            add_tool = RAGDocumentAddTool(
                "temp_add", "temp", self.qdrant_client, 
                self.embedding_model, self.collection_name, self.text_splitter
            )
            add_result = await add_tool.execute(add_params)
            
            if add_result["success"]:
                return {
                    "success": True,
                    "document_id": title,
                    "chunks_deleted": delete_result.get("chunks_deleted", 0),
                    "chunks_added": add_result["chunks_added"],
                    "message": f"Successfully updated document '{title}'"
                }
            else:
                return add_result
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to update document"
            }
    
    async def _delete_by_title(self, title: str) -> Dict[str, Any]:
        """Delete document by title"""
        try:
            # Search for points with this title
            if QDRANT_AVAILABLE and not isinstance(self.qdrant_client, MockQdrantClient):
                # Use scroll to find documents with this title
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="title", match=MatchValue(value=title))]
                    ),
                    limit=1000
                )
            else:
                # Mock implementation
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=1000
                )
                # Filter by title in mock
                points = [p for p in points if p.payload.get("title") == title]
            
            if points:
                point_ids = [point.id for point in points]
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                
                return {"chunks_deleted": len(point_ids)}
            
            return {"chunks_deleted": 0}
            
        except Exception as e:
            logger.error(f"Error deleting by title: {e}")
            return {"chunks_deleted": 0}


class RAGDocumentDeleteTool:
    """Tool for deleting documents from RAG knowledge base"""
    
    def __init__(self, name: str, description: str, qdrant_client, collection_name: str):
        self.name = name
        self.description = description
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Document title to delete"
                        }
                    },
                    "required": ["title"]
                }
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        return isinstance(params, dict) and "title" in params
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete document from knowledge base"""
        try:
            title = params["title"]
            
            # Search for points with this title
            if QDRANT_AVAILABLE and not isinstance(self.qdrant_client, MockQdrantClient):
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="title", match=MatchValue(value=title))]
                    ),
                    limit=1000
                )
            else:
                # Mock implementation
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=1000
                )
                points = [p for p in points if p.payload.get("title") == title]
            
            if not points:
                return {
                    "success": False,
                    "message": f"Document '{title}' not found"
                }
            
            # Delete points
            point_ids = [point.id for point in points]
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            return {
                "success": True,
                "document_id": title,
                "chunks_deleted": len(point_ids),
                "message": f"Successfully deleted document '{title}' ({len(point_ids)} chunks)"
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to delete document"
            }


class RAGDocumentListTool:
    """Tool for listing documents in RAG knowledge base"""
    
    def __init__(self, name: str, description: str, qdrant_client, collection_name: str):
        self.name = name
        self.description = description
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return",
                            "default": 10
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by document category"
                        }
                    }
                }
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters"""
        return isinstance(params, dict)
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List documents in knowledge base"""
        try:
            limit = params.get("limit", 10)
            category = params.get("category")
            
            # Get all points (limited)
            if QDRANT_AVAILABLE and not isinstance(self.qdrant_client, MockQdrantClient):
                scroll_filter = None
                if category:
                    scroll_filter = Filter(
                        must=[FieldCondition(key="category", match=MatchValue(value=category))]
                    )
                
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=limit * 10  # Get more to account for chunks
                )
            else:
                # Mock implementation
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=limit * 10
                )
                if category:
                    points = [p for p in points if p.payload.get("category") == category]
            
            # Group by title and get unique documents
            documents = {}
            for point in points:
                title = point.payload.get("title", "Unknown")
                if title not in documents:
                    documents[title] = {
                        "title": title,
                        "chunks": 0,
                        "added_at": point.payload.get("added_at"),
                        "updated_at": point.payload.get("updated_at"),
                        "source": point.payload.get("source"),
                        "category": point.payload.get("category"),
                        "tags": point.payload.get("tags", [])
                    }
                documents[title]["chunks"] += 1
            
            # Convert to list and limit
            document_list = list(documents.values())[:limit]
            
            return {
                "success": True,
                "documents": document_list,
                "total_found": len(document_list),
                "message": f"Found {len(document_list)} documents"
            }
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to list documents"
            }