#!/usr/bin/env python3
"""
HydrogenAI Test Suite - Progressive Query Testing
Tests intelligence, cache behavior, and response uniqueness
"""

import asyncio
import httpx
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any

class HydrogenAITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        
    async def test_query(self, query: str, test_name: str, expected_cache_behavior: str = None) -> Dict[str, Any]:
        """Test a single query and analyze response intelligence"""
        print(f"\n🧪 Testing: {test_name}")
        print(f"📝 Query: {query}")
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query",
                    json={"query": query},
                    headers={"Content-Type": "application/json"}
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Analyze response for intelligence indicators
                    intelligence_score = self._analyze_intelligence(result, query)
                    
                    test_result = {
                        "test_name": test_name,
                        "query": query,
                        "status": "SUCCESS",
                        "response_time": response_time,
                        "intelligence_score": intelligence_score,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                        "expected_cache": expected_cache_behavior
                    }
                    
                    print(f"✅ SUCCESS - Response time: {response_time:.2f}s")
                    print(f"🧠 Intelligence Score: {intelligence_score}/10")
                    
                    if "cached" in str(result).lower():
                        print("💾 Cache indicator detected in response")
                    
                    return test_result
                    
                else:
                    print(f"❌ FAILED - Status: {response.status_code}")
                    return {
                        "test_name": test_name,
                        "query": query,
                        "status": "FAILED",
                        "error": f"HTTP {response.status_code}",
                        "response_time": response_time,
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                print(f"💥 ERROR: {str(e)}")
                return {
                    "test_name": test_name,
                    "query": query, 
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    def _analyze_intelligence(self, result: Dict[str, Any], query: str) -> int:
        """Analyze response for intelligence indicators (0-10 scale)"""
        score = 0
        
        result_str = json.dumps(result, default=str).lower()
        
        # Check for reasoning indicators (+2 points)
        reasoning_indicators = ["because", "analysis", "based on", "considering", "pattern", "trend"]
        if any(indicator in result_str for indicator in reasoning_indicators):
            score += 2
            
        # Check for data-specific insights (+2 points)
        insight_indicators = ["insight", "correlation", "relationship", "significant", "notable"]
        if any(indicator in result_str for indicator in insight_indicators):
            score += 2
            
        # Check for context awareness (+2 points)
        context_indicators = ["current", "recent", "latest", "updated", "as of"]
        if any(indicator in result_str for indicator in context_indicators):
            score += 2
            
        # Check for structured analysis (+2 points)
        if "metadata" in result or "workflow" in result_str or "agents" in result_str:
            score += 2
            
        # Check for personalized response (+1 point)
        if len(result_str) > 100:  # Detailed response
            score += 1
            
        # Avoid generic responses (+1 point)
        generic_phrases = ["error", "could not", "unable to", "failed"]
        if not any(phrase in result_str for phrase in generic_phrases):
            score += 1
            
        return min(score, 10)  # Cap at 10
    
    async def test_cache_behavior(self, query: str, test_name: str, delay_seconds: int = 2):
        """Test cache behavior with repeated queries"""
        print(f"\n🔄 Cache Test: {test_name}")
        
        # First call
        print("📞 First call (should process with AI)...")
        first_result = await self.test_query(query, f"{test_name} - First Call")
        
        # Wait for a moment
        await asyncio.sleep(delay_seconds)
        
        # Second call
        print("📞 Second call (may hit cache)...")
        second_result = await self.test_query(query, f"{test_name} - Second Call")
        
        # Compare response times
        if (first_result.get("status") == "SUCCESS" and 
            second_result.get("status") == "SUCCESS"):
            
            time_diff = first_result["response_time"] - second_result["response_time"]
            
            if second_result["response_time"] < first_result["response_time"] * 0.5:
                print(f"⚡ Potential cache hit detected! {time_diff:.2f}s faster")
            else:
                print(f"🔄 Fresh processing detected (similar response times)")
                
        return {"first": first_result, "second": second_result}

# Test Suite Definitions
SIMPLE_QUERIES = [
    {
        "query": "How many collections exist in the database?",
        "name": "Simple Count Query",
        "cache": "schema_cache: 24hrs"
    },
    {
        "query": "Show me the structure of the users collection",
        "name": "Schema Discovery", 
        "cache": "schema_cache: 24hrs"
    },
    {
        "query": "What fields are available in the products collection?",
        "name": "Field Discovery",
        "cache": "schema_cache: 24hrs"
    }
]

INTERMEDIATE_QUERIES = [
    {
        "query": "Count users grouped by registration date",
        "name": "Temporal Grouping",
        "cache": "result_cache: 1hr"
    },
    {
        "query": "Find the average price of products by category",
        "name": "Aggregation Query",
        "cache": "result_cache: 1hr"
    },
    {
        "query": "Show me purchase trends over the last 30 days",
        "name": "Trend Analysis",
        "cache": "medium volatility: 1hr"
    }
]

COMPLEX_QUERIES = [
    {
        "query": "Analyze customer behavior patterns and identify high-value segments based on purchase history and engagement metrics",
        "name": "Customer Segmentation Analysis",
        "cache": "ai_response: 1hr, complex analysis"
    },
    {
        "query": "What are the correlations between product categories, customer demographics, and seasonal buying patterns?",
        "name": "Multi-dimensional Correlation",
        "cache": "complex result: 2hr TTL"
    },
    {
        "query": "Identify data quality issues across all collections and recommend schema improvements",
        "name": "Data Quality Assessment",
        "cache": "schema_analysis: 24hr"
    }
]

INTELLIGENCE_QUERIES = [
    {
        "query": "Based on current data patterns, what insights can you provide about our business performance and what should we monitor?",
        "name": "Business Intelligence Insights",
        "cache": "strategic_ai: adaptive TTL"
    },
    {
        "query": "Analyze our database structure and suggest the most efficient queries for typical business operations",
        "name": "Query Optimization Recommendations", 
        "cache": "meta_analysis: 12hr"
    },
    {
        "query": "What are the most important relationships in our data and how can we leverage them for better decision making?",
        "name": "Relationship Intelligence",
        "cache": "insight_cache: depends on complexity"
    }
]

async def main():
    """Run the complete test suite"""
    tester = HydrogenAITester()
    
    print("🚀 Starting HydrogenAI Intelligence and Cache Testing")
    print("=" * 60)
    
    # Test 1: Simple Queries
    print("\n📊 Level 1: Simple Queries")
    for test in SIMPLE_QUERIES:
        result = await tester.test_query(test["query"], test["name"], test["cache"])
        tester.test_results.append(result)
        await asyncio.sleep(1)
    
    # Test 2: Intermediate Queries  
    print("\n📈 Level 2: Intermediate Queries")
    for test in INTERMEDIATE_QUERIES:
        result = await tester.test_query(test["query"], test["name"], test["cache"])
        tester.test_results.append(result)
        await asyncio.sleep(1)
    
    # Test 3: Complex Queries
    print("\n🧠 Level 3: Complex Queries")
    for test in COMPLEX_QUERIES:
        result = await tester.test_query(test["query"], test["name"], test["cache"])
        tester.test_results.append(result)
        await asyncio.sleep(1)
        
    # Test 4: Intelligence Queries
    print("\n⚡ Level 4: Maximum Intelligence")
    for test in INTELLIGENCE_QUERIES:
        result = await tester.test_query(test["query"], test["name"], test["cache"])
        tester.test_results.append(result)
        await asyncio.sleep(1)
    
    # Test 5: Cache Behavior Tests
    print("\n💾 Cache Behavior Tests")
    cache_test_query = "What collections are in the database and what do they contain?"
    await tester.test_cache_behavior(cache_test_query, "Schema Cache Test", 3)
    
    # Generate Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for r in tester.test_results if r.get("status") == "SUCCESS"]
    failed_tests = [r for r in tester.test_results if r.get("status") != "SUCCESS"]
    
    print(f"✅ Successful Tests: {len(successful_tests)}")
    print(f"❌ Failed Tests: {len(failed_tests)}")
    
    if successful_tests:
        avg_intelligence = sum(r.get("intelligence_score", 0) for r in successful_tests) / len(successful_tests)
        avg_response_time = sum(r.get("response_time", 0) for r in successful_tests) / len(successful_tests)
        
        print(f"🧠 Average Intelligence Score: {avg_intelligence:.1f}/10")
        print(f"⏱️ Average Response Time: {avg_response_time:.2f}s")
        
        # Intelligence Analysis
        high_intelligence = [r for r in successful_tests if r.get("intelligence_score", 0) >= 7]
        print(f"🎯 High Intelligence Responses: {len(high_intelligence)}/{len(successful_tests)}")
    
    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(tester.test_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to test_results.json")
    print("🎉 Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())
