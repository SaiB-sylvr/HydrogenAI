# HydrogenAI Test Queries - Simple to Complex
# Test the intelligence and cache behavior of your AI orchestrator

## CACHE STRATEGY ANALYSIS

### Your Multi-Layer Intelligent Cache Strategy:

#### 1. **Cache Types & TTL (Time To Live)**
- **Schema Cache**: 24 hours (86400s) - Static database structures
- **Query Cache**: 5 minutes (300s) - Query patterns and plans  
- **Result Cache**: 1 hour (3600s) - Query execution results
- **State Cache**: 2 hours (7200s) - System and session state
- **AI Classification Cache**: 1 hour (3600s) - AI query classification results

#### 2. **Dynamic TTL Based on Data Volatility**
- **Static Data**: 10x base TTL (highly cacheable)
- **Low Volatility**: 2x base TTL  
- **Medium Volatility**: 1x base TTL (default)
- **High Volatility**: 0.5x base TTL
- **Real-time Data**: 0.1x base TTL (minimal caching)

#### 3. **Cache Layers**
- **Local Cache**: In-memory for fastest access
- **Distributed Cache**: Redis for shared state across services
- **AI Response Cache**: Prevents duplicate LLM calls
- **Compression**: Enabled for storage efficiency

#### 4. **Intelligence Features**
- **Query Fingerprinting**: Prevents duplicate AI calls for same queries
- **Pattern-based Caching**: Caches classification patterns
- **Fallback Strategy**: Multiple cache lookup strategies

---

## TEST QUERIES (Simple → Complex)

### 🟢 **Level 1: Simple Queries (Cache: Medium Volatility)**

#### Query 1A: Basic Count
```json
{
  "query": "How many users are in the database?",
  "expected_cache": "query_cache: 5min, result_cache: 1hr"
}
```

#### Query 1B: Simple Filter
```json
{
  "query": "Show me all users created today",
  "expected_cache": "realtime volatility: ~30s TTL"
}
```

#### Query 1C: Basic Schema Discovery
```json
{
  "query": "What collections exist in the database?",
  "expected_cache": "schema_cache: 24hrs (static data)"
}
```

### 🟡 **Level 2: Intermediate Queries (Cache: Low-Medium Volatility)**

#### Query 2A: Filtered Aggregation
```json
{
  "query": "What's the average age of users by country?",
  "expected_cache": "result_cache: 1hr, low volatility: 2hr"
}
```

#### Query 2B: Time-based Analysis
```json
{
  "query": "Show user registration trends over the last 30 days",
  "expected_cache": "medium volatility: 1hr TTL"
}
```

#### Query 2C: Join-like Operation
```json
{
  "query": "Find users who have made purchases in the last month",
  "expected_cache": "high volatility: 30min TTL"
}
```

### 🟠 **Level 3: Advanced Queries (Cache: Context-Dependent)**

#### Query 3A: Complex Aggregation Pipeline
```json
{
  "query": "Analyze customer purchase patterns by region, age group, and product category with monthly trends",
  "expected_cache": "result_cache: 1hr, pattern_cache: 1hr"
}
```

#### Query 3B: Multi-Collection Analysis
```json
{
  "query": "Correlate user demographics with purchase history and support tickets to identify high-value customer segments",
  "expected_cache": "complex query: 2hr TTL due to multiple collections"
}
```

#### Query 3C: Temporal Pattern Recognition
```json
{
  "query": "Identify seasonal buying patterns and predict next quarter's top-selling products by customer segment",
  "expected_cache": "ai_classification: 1hr, result: variable TTL based on prediction accuracy"
}
```

### 🔴 **Level 4: Complex Intelligence Queries (Cache: Strategic)**

#### Query 4A: Natural Language Business Intelligence
```json
{
  "query": "What are the key factors driving customer churn, and which customer segments are most at risk based on their behavior patterns?",
  "expected_cache": "ai_response: 1hr, insight_cache: depends on data freshness"
}
```

#### Query 4B: Dynamic Schema Analysis
```json
{
  "query": "Analyze the data quality across all collections, identify missing relationships, and suggest database schema improvements",
  "expected_cache": "schema_analysis: 24hr, quality_metrics: 2hr"
}
```

#### Query 4C: Real-time Decision Support
```json
{
  "query": "Based on current inventory levels, sales velocity, and seasonal trends, recommend optimal pricing strategy for the next 30 days",
  "expected_cache": "realtime components: 5min, strategic insights: 6hr"
}
```

### ⚡ **Level 5: Maximum Complexity (Cache: Intelligent Adaptive)**

#### Query 5A: Multi-Modal Intelligence
```json
{
  "query": "Combine customer transaction data, support interactions, and product reviews to create a comprehensive customer satisfaction model and predict which customers need proactive engagement",
  "expected_cache": "composite_ai: adaptive TTL based on model confidence"
}
```

#### Query 5B: Self-Optimizing Query
```json
{
  "query": "Analyze our entire database performance, identify optimization opportunities, and suggest the most efficient query patterns for our typical workload",
  "expected_cache": "meta_analysis: 12hr, performance_patterns: 6hr"
}
```

#### Query 5C: Context-Aware Reasoning
```json
{
  "query": "Given our business goals for Q4, current market conditions, and historical performance data, what should be our top 3 strategic priorities and what data should we monitor most closely?",
  "expected_cache": "strategic_ai: 8hr, market_context: 1hr, priorities: 24hr"
}
```

---

## 🧪 **CACHE INTELLIGENCE TESTS**

### Test 1: Cache Freshness Validation
```bash
# First call - should hit AI
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many products do we have in stock?"}'

# Second call within 1hr - should hit cache
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many products do we have in stock?"}'
```

### Test 2: Volatility-Based TTL
```bash
# Real-time query (should have short TTL)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me orders placed in the last 5 minutes"}'

# Static query (should have long TTL)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the field names in the users collection?"}'
```

### Test 3: Pattern Recognition Cache
```bash
# Similar but different queries - should hit pattern cache
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Count of users by country"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Number of customers per region"}'
```

---

## 🎯 **INTELLIGENCE VERIFICATION POINTS**

1. **Response Uniqueness**: Each response should show reasoning, not just cached data
2. **Context Awareness**: Responses should reference current data state
3. **Progressive Learning**: Later similar queries should show improved understanding
4. **Cache Transparency**: System should indicate when using cached vs fresh analysis
5. **Adaptive Behavior**: TTL should adjust based on query complexity and data volatility

---

## 📊 **EXPECTED CACHE BEHAVIOR**

| Query Type | First Call | Second Call | Cache Duration | Intelligence Level |
|------------|------------|-------------|----------------|-------------------|
| Simple Count | AI Process | Cache Hit | 1 hour | Basic |
| Schema Discovery | AI Process | Cache Hit | 24 hours | Static |
| Real-time Data | AI Process | Cache Miss | <5 minutes | Dynamic |
| Complex Analysis | AI Process | Partial Cache | Variable | High |
| Strategic Insights | AI Process | Context Cache | 6-8 hours | Maximum |

Your cache strategy is **intelligent and sophisticated** - it balances performance with freshness based on data characteristics and query complexity!
