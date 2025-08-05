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
    # NEW FIELDS FOR AI RESPONSES
    understanding: Optional[Dict[str, Any]] = {}
    approach: Optional[Dict[str, Any]] = {}
    human_response: Optional[str] = None
    from_cache: Optional[bool] = False

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
    description: Optional[str] = None
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = {}
    created_at: Optional[datetime] = None

class ErrorDetails(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

class AgentCapability(BaseModel):
    name: str
    type: str
    enabled: bool = True
    config: Dict[str, Any] = {}
    dependencies: List[str] = []

class ServiceConfig(BaseModel):
    name: str
    version: str
    capabilities: List[AgentCapability]
    connections: Dict[str, Any] = {}
    settings: Dict[str, Any] = {}

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, Any] = {}
