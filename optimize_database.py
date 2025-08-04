#!/usr/bin/env python3
"""
Database Performance Optimization for HydrogenAI
Creates indexes, optimizes queries, and improves database performance
"""

import asyncio
import httpx
import json
from datetime import datetime

MCP_SERVER_URL = "http://localhost:8001/execute"

class DatabaseOptimizer:
    
    async def create_performance_indexes(self):
        """Create strategic indexes for better query performance"""
        print("🔄 Creating performance indexes...")
        
        indexes = [
            # User Activity Indexes
            {
                "collection": "user_activity",
                "indexes": [
                    {"user_id": 1, "timestamp": -1},  # User timeline queries
                    {"event_type": 1, "timestamp": -1},  # Event type analysis
                    {"timestamp": -1},  # Time-based queries
                    {"event_data.content_id": 1, "user_id": 1},  # Content engagement
                    {"device_info.device_type": 1, "timestamp": -1},  # Device analysis
                    {"location.city": 1, "timestamp": -1},  # Geographic analysis
                    {"session_id": 1},  # Session tracking
                    {"conversion_flag": 1, "timestamp": -1}  # Conversion analysis
                ]
            },
            
            # Users Indexes
            {
                "collection": "users",
                "indexes": [
                    {"email": 1},  # Unique email lookups
                    {"status": 1, "created_at": -1},  # Status filtering
                    {"preferences.categories": 1},  # Category preferences
                    {"location.city": 1, "location.country": 1},  # Geographic queries
                    {"demographics.income_bracket": 1},  # Demographic analysis
                    {"lifetime_value": -1},  # Value-based sorting
                    {"last_login": -1}  # Activity analysis
                ]
            },
            
            # Products Indexes
            {
                "collection": "products",
                "indexes": [
                    {"category": 1, "subcategory": 1},  # Category browsing
                    {"sku": 1},  # Unique SKU lookups
                    {"price": 1, "rating": -1},  # Price/rating sorting
                    {"brand": 1, "category": 1},  # Brand filtering
                    {"status": 1, "created_at": -1},  # Status filtering
                    {"inventory.stock": 1},  # Stock level queries
                    {"rating": -1, "reviews_count": -1}  # Popular products
                ]
            },
            
            # Orders Indexes
            {
                "collection": "orders",
                "indexes": [
                    {"user_id": 1, "timestamps.created_at": -1},  # User order history
                    {"status": 1, "timestamps.created_at": -1},  # Status filtering
                    {"order_number": 1},  # Unique order lookups
                    {"payment.status": 1, "timestamps.created_at": -1},  # Payment analysis
                    {"pricing.total": -1, "timestamps.created_at": -1},  # Value analysis
                    {"shipping.address.city": 1},  # Geographic shipping
                    {"items.product_id": 1}  # Product order tracking
                ]
            },
            
            # Marketing Campaigns Indexes
            {
                "collection": "marketing_campaigns",
                "indexes": [
                    {"campaign_type": 1, "created_at": -1},  # Campaign type analysis
                    {"status": 1, "budget": -1},  # Active campaign sorting
                    {"target_audience": 1},  # Audience targeting
                    {"performance.roi": -1}  # ROI analysis
                ]
            },
            
            # Support Tickets Indexes
            {
                "collection": "support_tickets",
                "indexes": [
                    {"user_id": 1, "timestamps.created_at": -1},  # User ticket history
                    {"status": 1, "priority": 1},  # Status/priority filtering
                    {"ticket_number": 1},  # Unique ticket lookups
                    {"category": 1, "timestamps.created_at": -1},  # Category analysis
                    {"assigned_to": 1, "status": 1},  # Agent workload
                    {"customer_satisfaction": -1},  # Satisfaction analysis
                    {"resolution_time_hours": 1}  # Performance metrics
                ]
            }
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for collection_config in indexes:
                collection = collection_config["collection"]
                print(f"  📊 Creating indexes for {collection}...")
                
                for index_spec in collection_config["indexes"]:
                    try:
                        response = await client.post(
                            MCP_SERVER_URL,
                            json={
                                "tool": "mongodb_create_index",
                                "params": {
                                    "collection": collection,
                                    "index": index_spec,
                                    "background": True  # Non-blocking index creation
                                }
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                index_name = "_".join([f"{k}_{v}" for k, v in index_spec.items()])
                                print(f"    ✅ Created index: {index_name}")
                            else:
                                print(f"    ⚠️  Index may already exist: {index_spec}")
                        else:
                            print(f"    ❌ HTTP error {response.status_code} creating index: {index_spec}")
                            
                    except Exception as e:
                        print(f"    ❌ Exception creating index {index_spec}: {e}")
                
                # Small delay between collections
                await asyncio.sleep(0.2)

    async def create_compound_indexes(self):
        """Create compound indexes for complex queries"""
        print("🔄 Creating compound indexes for complex analytics...")
        
        compound_indexes = [
            # User behavior analysis
            {
                "collection": "user_activity",
                "index": {
                    "user_id": 1,
                    "event_type": 1,
                    "timestamp": -1,
                    "conversion_flag": 1
                },
                "name": "user_behavior_analysis"
            },
            
            # Content performance tracking
            {
                "collection": "user_activity",
                "index": {
                    "event_data.content_id": 1,
                    "event_type": 1,
                    "timestamp": -1,
                    "session_duration": -1
                },
                "name": "content_performance"
            },
            
            # Geographic user activity
            {
                "collection": "user_activity",
                "index": {
                    "location.country": 1,
                    "location.city": 1,
                    "timestamp": -1,
                    "event_type": 1
                },
                "name": "geographic_activity"
            },
            
            # Product catalog optimization
            {
                "collection": "products",
                "index": {
                    "category": 1,
                    "status": 1,
                    "price": 1,
                    "rating": -1
                },
                "name": "catalog_optimization"
            },
            
            # Order fulfillment tracking
            {
                "collection": "orders",
                "index": {
                    "status": 1,
                    "shipping.method": 1,
                    "timestamps.created_at": -1,
                    "pricing.total": -1
                },
                "name": "fulfillment_tracking"
            }
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for index_config in compound_indexes:
                try:
                    response = await client.post(
                        MCP_SERVER_URL,
                        json={
                            "tool": "mongodb_create_index",
                            "params": {
                                "collection": index_config["collection"],
                                "index": index_config["index"],
                                "name": index_config["name"],
                                "background": True
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            print(f"  ✅ Created compound index: {index_config['name']}")
                        else:
                            print(f"  ⚠️  Compound index may already exist: {index_config['name']}")
                    else:
                        print(f"  ❌ HTTP error creating compound index: {index_config['name']}")
                        
                except Exception as e:
                    print(f"  ❌ Exception creating compound index {index_config['name']}: {e}")

    async def analyze_query_performance(self):
        """Analyze and explain query performance"""
        print("🔍 Analyzing query performance...")
        
        test_queries = [
            {
                "name": "User Activity by Type",
                "collection": "user_activity",
                "pipeline": [
                    {"$match": {"event_type": "product_view", "timestamp": {"$gte": "2024-07-01"}}},
                    {"$group": {"_id": "$user_id", "views": {"$sum": 1}}},
                    {"$sort": {"views": -1}},
                    {"$limit": 10}
                ]
            },
            {
                "name": "Product Performance",
                "collection": "products",
                "pipeline": [
                    {"$match": {"status": "active", "category": "electronics"}},
                    {"$sort": {"rating": -1, "reviews_count": -1}},
                    {"$limit": 20}
                ]
            },
            {
                "name": "Order Analysis",
                "collection": "orders",
                "pipeline": [
                    {"$match": {"status": "delivered", "pricing.total": {"$gte": 100}}},
                    {"$group": {
                        "_id": {"$dateToString": {"format": "%Y-%m", "date": {"$dateFromString": {"dateString": "$timestamps.created_at"}}}},
                        "total_revenue": {"$sum": "$pricing.total"},
                        "order_count": {"$sum": 1}
                    }},
                    {"$sort": {"_id": -1}}
                ]
            }
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for query in test_queries:
                try:
                    start_time = datetime.now()
                    
                    response = await client.post(
                        MCP_SERVER_URL,
                        json={
                            "tool": "mongodb_aggregate",
                            "params": {
                                "collection": query["collection"],
                                "pipeline": query["pipeline"],
                                "explain": True  # Get query plan
                            }
                        }
                    )
                    
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            print(f"  ✅ {query['name']}: {execution_time:.3f}s")
                            
                            # Check if indexes were used
                            explain_data = result.get("result", {})
                            if "executionStats" in str(explain_data):
                                print(f"      📊 Query optimization details available")
                        else:
                            print(f"  ❌ Query failed: {query['name']}")
                    else:
                        print(f"  ❌ HTTP error for query: {query['name']}")
                        
                except Exception as e:
                    print(f"  ❌ Exception testing query {query['name']}: {e}")

    async def get_collection_stats(self):
        """Get detailed collection statistics"""
        print("📊 Getting collection statistics...")
        
        collections = ["users", "products", "orders", "user_activity", "marketing_campaigns", "support_tickets"]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for collection in collections:
                try:
                    response = await client.post(
                        MCP_SERVER_URL,
                        json={
                            "tool": "mongodb_collection_stats",
                            "params": {"collection": collection}
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            stats = result.get("result", {})
                            count = stats.get("count", 0)
                            size = stats.get("size", 0)
                            avg_obj_size = stats.get("avgObjSize", 0)
                            index_count = stats.get("nindexes", 0)
                            
                            print(f"  📊 {collection}:")
                            print(f"      Documents: {count:,}")
                            print(f"      Size: {size/1024/1024:.2f} MB")
                            print(f"      Avg Object Size: {avg_obj_size:,} bytes")
                            print(f"      Indexes: {index_count}")
                        else:
                            print(f"  ❌ Could not get stats for {collection}")
                    else:
                        print(f"  ❌ HTTP error getting stats for {collection}")
                        
                except Exception as e:
                    print(f"  ❌ Exception getting stats for {collection}: {e}")

async def main():
    """Run database performance optimization"""
    print("🚀 Starting database performance optimization...")
    
    optimizer = DatabaseOptimizer()
    
    try:
        # Get baseline stats
        await optimizer.get_collection_stats()
        
        # Create performance indexes
        await optimizer.create_performance_indexes()
        
        # Create compound indexes for complex queries
        await optimizer.create_compound_indexes()
        
        # Analyze query performance
        await optimizer.analyze_query_performance()
        
        # Get final stats
        print("\n📈 Final database statistics:")
        await optimizer.get_collection_stats()
        
        print("\n✅ Database performance optimization completed!")
        print("📊 Your MongoDB collections now have optimized indexes for better query performance")
        
    except Exception as e:
        print(f"❌ Error during database optimization: {e}")

if __name__ == "__main__":
    asyncio.run(main())
