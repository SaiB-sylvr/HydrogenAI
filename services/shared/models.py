from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class QueryType(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    RAG = "rag"
    AGGREGATION = "aggregation"

class ToolParameter(BaseModel):
    name: str
    type: str
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    category: Optional[str] = None
    version: str = "1.0.0"

class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = {}
    context: Optional[Dict[str, Any]] = {}
    options: Optional[Dict[str, Any]] = {}

class QueryResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    execution_time: Optional[float] = None

class WorkflowStep(BaseModel):
    name: str
    type: str
    params: Dict[str, Any] = {}
    dependencies: List[str] = []
    critical: bool = True
    timeout: Optional[int] = None

class WorkflowDefinition(BaseModel):
    name: str
    version: str
    description: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = {}

class AgentMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

class EventMessage(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

class HealthStatus(BaseModel):
    service: str
    status: str
    details: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PluginInfo(BaseModel):
    name: str
    version: str
    description: str
    author: Optional[str] = None
    tools: List[str] = []
    status: str = "loaded"