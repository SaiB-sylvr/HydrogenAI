"""
Intelligent Query Classification with NLP and Pattern Matching
Enhanced with Rate Limit Management and Fallback Logic
"""
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import yaml
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

# Import our new AI management modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

try:
    from ai_provider_manager import get_available_provider, record_ai_usage, record_ai_error, record_ai_success
    from ai_cache import get_cached_ai_response, cache_ai_response
    AI_MANAGEMENT_AVAILABLE = True
except ImportError:
    logger.warning("AI management modules not available, using basic classification")
    AI_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    AGGREGATION = "aggregation"
    RAG = "rag"
    SCHEMA = "schema"
    ANALYTICAL = "analytical"

@dataclass
class QueryPattern:
    """Query pattern definition"""
    name: str
    regex: str
    type: QueryType
    workflow: str
    tool: Optional[str] = None
    confidence: float = 1.0
    param_extractors: Optional[Dict[str, str]] = None

class QueryClassifier:
    """Intelligent query classification for optimal routing"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self._compile_patterns()
        
        # Enhanced keyword sets
        self.simple_keywords = {
            "count", "total", "number", "how many",
            "list", "show", "get", "find", "fetch",
            "exists", "check", "verify"
        }
        
        self.complex_keywords = {
            "analyze", "compare", "trend", "pattern",
            "relationship", "correlation", "insight",
            "summarize", "explain", "predict", "forecast"
        }
        
        # Fallback classification patterns for when AI is unavailable
        self.fallback_patterns = {
            QueryType.SIMPLE: [
                r"\b(count|total|number|how many)\b.*\b(users?|orders?|products?)\b",
                r"\b(list|show|get|find)\b.*\b(all|latest|recent)\b",
                r"\b(what is|who is|when was|where is)\b",
                r"\b(exists|check|verify)\b"
            ],
            QueryType.AGGREGATION: [
                r"\b(sum|total|average|mean|max|min)\b",
                r"\b(group by|breakdown|distribution)\b",
                r"\b(statistics|stats|summary)\b",
                r"\b(top \d+|bottom \d+|highest|lowest)\b"
            ],
            QueryType.RAG: [
                r"\b(explain|describe|tell me about|what does)\b",
                r"\b(search for|find information|look up)\b",
                r"\b(documentation|help|guide|manual)\b"
            ],
            QueryType.SCHEMA: [
                r"\b(schema|structure|fields|columns)\b",
                r"\b(collections?|tables?|databases?)\b",
                r"\b(describe|show structure)\b"
            ],
            QueryType.ANALYTICAL: [
                r"\b(analyze|analysis|insights?)\b",
                r"\b(trends?|patterns?|correlations?)\b",
                r"\b(predict|forecast|projection)\b",
                r"\b(compare|comparison|vs|versus)\b"
            ]
        }
        
        # Compile fallback patterns
        self.compiled_fallback = {}
        for query_type, patterns in self.fallback_patterns.items():
            self.compiled_fallback[query_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        self.aggregation_keywords = {
            "sum", "average", "mean", "median", "max", "min",
            "group by", "aggregate", "total by", "breakdown",
            "distribution", "statistics"
        }
        
        self.schema_keywords = {
            "schema", "structure", "fields", "columns",
            "indexes", "collections", "tables", "describe",
            "data model", "relationships"
        }
        
        # Collection name patterns
        self.collection_patterns = [
            r'\b(users?|customers?|clients?)\b',
            r'\b(orders?|purchases?|transactions?)\b',
            r'\b(products?|items?|inventory)\b',
            r'\b(employees?|staff|personnel)\b',
            r'\b(payments?|invoices?|bills?)\b',
            r'\b(logs?|events?|activities)\b'
        ]
    
    def _load_patterns(self) -> List[QueryPattern]:
        """Load classification patterns from config"""
        default_patterns = [
            # Schema queries
            QueryPattern(
                name="schema_discovery",
                regex=r"(show|list|what).*(collections?|tables?|schemas?|databases?)",
                type=QueryType.SCHEMA,
                workflow="schema_discovery"
            ),
            QueryPattern(
                name="schema_details",
                regex=r"(describe|explain|show).*(structure|schema|fields|columns).*\b(\w+)",
                type=QueryType.SCHEMA,
                workflow="schema_discovery",
                param_extractors={"collection": r"of\s+(\w+)|for\s+(\w+)|in\s+(\w+)"}
            ),
            
            # Count queries
            QueryPattern(
                name="count_simple",
                regex=r"(count|how many|total number of)\s+(\w+)",
                type=QueryType.SIMPLE,
                workflow="simple_query",
                tool="mongodb_count",
                param_extractors={"collection": r"(count|how many|total number of)\s+(\w+)"}
            ),
            
            # Aggregation queries
            QueryPattern(
                name="sum_aggregation",
                regex=r"(sum|total|add up).*(by|per|group)",
                type=QueryType.AGGREGATION,
                workflow="complex_aggregation"
            ),
            QueryPattern(
                name="average_aggregation",
                regex=r"(average|mean|avg).*(by|per|group)",
                type=QueryType.AGGREGATION,
                workflow="complex_aggregation"
            ),
            
            # Analytical queries
            QueryPattern(
                name="trend_analysis",
                regex=r"(trend|pattern|change|growth|decline).*(over time|period|monthly|yearly)",
                type=QueryType.ANALYTICAL,
                workflow="complex_aggregation"
            ),
            
            # RAG queries
            QueryPattern(
                name="explanation",
                regex=r"(explain|what is|how does|why|tell me about)",
                type=QueryType.RAG,
                workflow="rag_query",
                confidence=0.8
            ),
            QueryPattern(
                name="question",
                regex=r".*\?$",
                type=QueryType.RAG,
                workflow="rag_query",
                confidence=0.6
            )
        ]
        
        # Try to load from config file
        try:
            with open("/app/config/query_patterns.yaml", "r") as f:
                config = yaml.safe_load(f)
                patterns = []
                
                for pattern_config in config.get("patterns", []):
                    patterns.append(QueryPattern(
                        name=pattern_config["name"],
                        regex=pattern_config["regex"],
                        type=QueryType(pattern_config["type"]),
                        workflow=pattern_config.get("workflow"),
                        tool=pattern_config.get("tool"),
                        confidence=pattern_config.get("confidence", 1.0),
                        param_extractors=pattern_config.get("param_extractors")
                    ))
                
                return patterns
        except Exception as e:
            logger.warning(f"Failed to load patterns from config: {e}")
            return default_patterns
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        for pattern in self.patterns:
            pattern.compiled = re.compile(pattern.regex, re.IGNORECASE)
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """Classify query for optimal routing with confidence scoring and fallback"""
        query_lower = query.lower().strip()
        
        # Check cache first if AI management is available
        cached_result = None
        if AI_MANAGEMENT_AVAILABLE:
            try:
                cached_result = get_cached_ai_response(query, model="classification")
                if cached_result and cached_result.get("response"):
                    logger.debug("Using cached classification result")
                    return cached_result["response"]
            except Exception as e:
                logger.debug(f"Cache lookup failed: {e}")
        
        # Try pattern matching first (fastest)
        pattern_result = self._pattern_classification(query_lower)
        if pattern_result["confidence"] >= 0.8:
            # High confidence pattern match
            if AI_MANAGEMENT_AVAILABLE:
                try:
                    cache_ai_response(query, pattern_result, model="classification", ttl=3600)
                except Exception:
                    pass
            return pattern_result
        
        # Try AI classification if available and not rate limited
        if AI_MANAGEMENT_AVAILABLE:
            ai_result = await self._ai_classification(query)
            if ai_result:
                return ai_result
        
        # Fallback to enhanced pattern matching
        fallback_result = self._fallback_classification(query_lower)
        
        # Cache the result
        if AI_MANAGEMENT_AVAILABLE:
            try:
                cache_ai_response(query, fallback_result, model="classification", ttl=1800)
            except Exception:
                pass
        
        return fallback_result
    
    def _pattern_classification(self, query_lower: str) -> Dict[str, Any]:
        """Pattern-based classification (fast path)"""
        best_match = None
        best_confidence = 0.0
        
        for pattern in self.patterns:
            if pattern.compiled.search(query_lower):
                confidence = pattern.confidence
                
                # Boost confidence for exact keyword matches
                if pattern.type == QueryType.SIMPLE and any(kw in query_lower for kw in self.simple_keywords):
                    confidence *= 1.2
                elif pattern.type == QueryType.COMPLEX and any(kw in query_lower for kw in self.complex_keywords):
                    confidence *= 1.2
                
                if confidence > best_confidence:
                    best_match = pattern
                    best_confidence = confidence
        
        if best_match:
            classification = {
                "type": best_match.type.value,
                "workflow": best_match.workflow,
                "confidence": best_confidence,
                "pattern_match": best_match.name,
                "method": "pattern_matching"
            }
            
            if best_match.tool:
                classification["tool"] = best_match.tool
            
            if best_match.param_extractors:
                params = self._extract_params(query_lower, best_match)
                if params:
                    classification["params"] = params
            
            return classification
        
        return {"confidence": 0.0}
    
    def _fallback_classification(self, query_lower: str) -> Dict[str, Any]:
        """Enhanced fallback classification using compiled patterns"""
        best_type = QueryType.SIMPLE
        best_score = 0
        
        # Test against fallback patterns
        for query_type, patterns in self.compiled_fallback.items():
            score = 0
            for pattern in patterns:
                if pattern.search(query_lower):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_type = query_type
        
        # If no pattern matches, use keyword scoring
        if best_score == 0:
            return self._keyword_classification(query_lower)
        
        # Determine workflow based on type
        workflow_map = {
            QueryType.SIMPLE: "simple_query",
            QueryType.AGGREGATION: "complex_aggregation", 
            QueryType.RAG: "rag_query",
            QueryType.SCHEMA: "schema_discovery",
            QueryType.ANALYTICAL: "complex_aggregation"
        }
        
        confidence = min(0.9, 0.5 + (best_score * 0.1))  # Scale confidence
        
        classification = {
            "type": best_type.value,
            "workflow": workflow_map.get(best_type, "simple_query"),
            "confidence": confidence,
            "method": "fallback_patterns",
            "fallback_score": best_score
        }
        
        return classification
    
    async def _ai_classification(self, query: str) -> Optional[Dict[str, Any]]:
        """AI-powered classification with provider management"""
        try:
            provider = get_available_provider("groq")
            if not provider:
                logger.warning("No AI provider available for classification")
                return None
            
            # Simple classification prompt that uses fewer tokens
            prompt = f"""Classify this query into one of these types:
- simple: Basic queries (count, list, find, get)
- aggregation: Statistical queries (sum, average, group by)
- rag: Questions needing document search/explanation
- schema: Database structure queries
- analytical: Complex analysis, trends, insights

Query: "{query}"

Response format: {{"type": "...", "confidence": 0.0-1.0}}"""
            
            # Make AI request (implement your AI client call here)
            # This is a placeholder - you'd implement the actual AI call
            # based on your current AI client setup
            
            # For now, return None to use fallback
            return None
            
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            if AI_MANAGEMENT_AVAILABLE:
                record_ai_error("groq", str(e))
            return None
            classification = {
                "type": best_match.type.value,
                "workflow": best_match.workflow,
                "confidence": best_confidence,
                "pattern_match": best_match.name
            }
            
            # Add tool if specified
            if best_match.tool:
                classification["tool"] = best_match.tool
            
            # Extract parameters
            if best_match.param_extractors:
                params = self._extract_params(query, best_match)
                if params:
                    classification["params"] = params
            
            # Enhance with additional analysis
            self._enhance_classification(query, classification)
            
            return classification
        
        # Fallback to keyword-based classification
        return self._keyword_classification(query)
    
    def _keyword_classification(self, query: str) -> Dict[str, Any]:
        """Fallback keyword-based classification"""
        query_lower = query.lower()
        
        # Count keyword matches
        simple_score = sum(1 for kw in self.simple_keywords if kw in query_lower)
        complex_score = sum(1 for kw in self.complex_keywords if kw in query_lower)
        aggregation_score = sum(1 for kw in self.aggregation_keywords if kw in query_lower)
        schema_score = sum(1 for kw in self.schema_keywords if kw in query_lower)
        
        # Determine type based on scores
        scores = {
            QueryType.SIMPLE: simple_score,
            QueryType.COMPLEX: complex_score,
            QueryType.AGGREGATION: aggregation_score,
            QueryType.SCHEMA: schema_score
        }
        
        # Get type with highest score
        query_type = max(scores, key=scores.get)
        
        # If no clear winner, analyze further
        if scores[query_type] == 0:
            # Check query characteristics
            if "?" in query or len(query.split()) > 15:
                query_type = QueryType.RAG
            elif any(op in query_lower for op in ["and", "or", "join", "combine"]):
                query_type = QueryType.COMPLEX
            else:
                query_type = QueryType.SIMPLE
        
        classification = {
            "type": query_type.value,
            "confidence": min(scores[query_type] / 3.0, 1.0),  # Normalize confidence
            "keyword_scores": {k.value: v for k, v in scores.items() if v > 0}
        }
        
        # Set workflow
        workflow_map = {
            QueryType.SIMPLE: "simple_query",
            QueryType.COMPLEX: "complex_aggregation",
            QueryType.AGGREGATION: "complex_aggregation",
            QueryType.SCHEMA: "schema_discovery",
            QueryType.RAG: "rag_query",
            QueryType.ANALYTICAL: "complex_aggregation"
        }
        classification["workflow"] = workflow_map.get(query_type, "complex_aggregation")
        
        # Try to determine tool for simple queries
        if query_type == QueryType.SIMPLE:
            tool_info = self._determine_simple_tool(query)
            if tool_info:
                classification.update(tool_info)
        
        # Enhance classification
        self._enhance_classification(query, classification)
        
        return classification
    
    def _determine_simple_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Determine tool and params for simple queries"""
        query_lower = query.lower()
        
        # Count queries
        if any(kw in query_lower for kw in ["count", "how many", "total number"]):
            collection = self._extract_collection(query)
            if collection != "unknown":
                return {
                    "tool": "mongodb_count",
                    "params": {
                        "collection": collection,
                        "filter": self._extract_filter(query)
                    }
                }
        
        # Find/List queries
        elif any(kw in query_lower for kw in ["list", "show", "get", "find"]):
            collection = self._extract_collection(query)
            if collection != "unknown":
                return {
                    "tool": "mongodb_find",
                    "params": {
                        "collection": collection,
                        "filter": self._extract_filter(query),
                        "limit": self._extract_limit(query)
                    }
                }
        
        return None
    
    def _extract_params(self, query: str, pattern: QueryPattern) -> Dict[str, Any]:
        """Extract parameters based on pattern extractors"""
        params = {}
        
        if not pattern.param_extractors:
            return params
        
        for param, extractor in pattern.param_extractors.items():
            match = re.search(extractor, query.lower())
            if match:
                # Get the first non-None group
                value = next((g for g in match.groups() if g), None)
                if value:
                    params[param] = value
        
        return params
    
    def _extract_collection(self, query: str) -> str:
        """Extract collection name from query"""
        query_lower = query.lower()
        
        # Check against known patterns
        for pattern in self.collection_patterns:
            match = re.search(pattern, query_lower)
            if match:
                collection = match.group(1)
                # Singularize if needed
                if collection.endswith('s') and not collection.endswith('ss'):
                    return collection[:-1]
                return collection
        
        # Look for "collection_name" pattern
        match = re.search(r'(?:from|in|of)\s+["\'`]?(\w+)["\'`]?', query_lower)
        if match:
            return match.group(1)
        
        return "unknown"
    
    def _extract_filter(self, query: str) -> Dict[str, Any]:
        """Extract filter conditions from query"""
        filters = {}
        query_lower = query.lower()
        
        # Extract equality conditions
        eq_patterns = [
            r'where\s+(\w+)\s*=\s*["\']?([^"\']+)["\']?',
            r'(\w+)\s+is\s+["\']?([^"\']+)["\']?',
            r'(\w+)\s+equals?\s+["\']?([^"\']+)["\']?'
        ]
        
        for pattern in eq_patterns:
            matches = re.findall(pattern, query_lower)
            for field, value in matches:
                # Try to parse value type
                if value.isdigit():
                    filters[field] = int(value)
                elif value.replace('.', '').isdigit():
                    filters[field] = float(value)
                elif value in ['true', 'false']:
                    filters[field] = value == 'true'
                else:
                    filters[field] = value
        
        # Extract comparison conditions
        comp_patterns = [
            (r'(\w+)\s*>\s*(\d+)', '$gt'),
            (r'(\w+)\s*<\s*(\d+)', '$lt'),
            (r'(\w+)\s*>=\s*(\d+)', '$gte'),
            (r'(\w+)\s*<=\s*(\d+)', '$lte')
        ]
        
        for pattern, op in comp_patterns:
            matches = re.findall(pattern, query_lower)
            for field, value in matches:
                if field not in filters:
                    filters[field] = {}
                filters[field][op] = int(value)
        
        return filters
    
    def _extract_limit(self, query: str) -> int:
        """Extract limit from query"""
        # Look for explicit limit
        limit_match = re.search(r'limit\s+(\d+)', query.lower())
        if limit_match:
            return min(int(limit_match.group(1)), 1000)
        
        # Look for top/first N
        top_match = re.search(r'(?:top|first)\s+(\d+)', query.lower())
        if top_match:
            return min(int(top_match.group(1)), 1000)
        
        # Look for any number
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers and len(numbers) == 1:
            return min(int(numbers[0]), 100)
        
        return 10
    
    def _enhance_classification(self, query: str, classification: Dict[str, Any]):
        """Enhance classification with additional metadata"""
        # Add query complexity
        words = query.split()
        classification["complexity_score"] = len(words) / 10.0
        
        # Add entity extraction
        entities = self._extract_entities(query)
        if entities:
            classification["entities"] = entities
        
        # Add time-based indicators
        time_indicators = self._extract_time_indicators(query)
        if time_indicators:
            classification["time_indicators"] = time_indicators
        
        # Add operation hints
        operations = self._extract_operations(query)
        if operations:
            classification["operations"] = operations
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query"""
        entities = []
        
        # Extract dates
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|month|year)\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                entities.append({
                    "type": "date",
                    "value": match,
                    "normalized": self._normalize_date(match)
                })
        
        # Extract numbers with context
        number_pattern = r'(\d+(?:\.\d+)?)\s*(\w+)'
        matches = re.findall(number_pattern, query)
        for number, unit in matches:
            entities.append({
                "type": "measure",
                "value": float(number) if '.' in number else int(number),
                "unit": unit
            })
        
        return entities
    
    def _extract_time_indicators(self, query: str) -> List[str]:
        """Extract time-related indicators"""
        indicators = []
        query_lower = query.lower()
        
        time_keywords = [
            "recent", "latest", "current", "now", "today",
            "yesterday", "last week", "last month", "last year",
            "historical", "past", "previous", "trend", "over time"
        ]
        
        for keyword in time_keywords:
            if keyword in query_lower:
                indicators.append(keyword)
        
        return indicators
    
    def _extract_operations(self, query: str) -> List[str]:
        """Extract operation hints from query"""
        operations = []
        query_lower = query.lower()
        
        operation_map = {
            "sum": ["sum", "total", "add up"],
            "average": ["average", "avg", "mean"],
            "count": ["count", "number", "how many"],
            "group": ["group by", "by", "per", "breakdown"],
            "sort": ["sort", "order by", "ranked", "top", "bottom"],
            "filter": ["where", "filter", "only", "with", "having"],
            "join": ["join", "combine", "merge", "with"]
        }
        
        for op, keywords in operation_map.items():
            if any(kw in query_lower for kw in keywords):
                operations.append(op)
        
        return operations
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date strings to ISO format"""
        # Simple normalization - in production, use dateparser or similar
        if date_str == "today":
            from datetime import date
            return date.today().isoformat()
        elif date_str == "yesterday":
            from datetime import date, timedelta
            return (date.today() - timedelta(days=1)).isoformat()
        
        return date_str