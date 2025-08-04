#!/usr/bin/env python3
"""
Realistic Data Generation for HydrogenAI System
Generates comprehensive, realistic data to replace mock dependencies
"""

import asyncio
import httpx
import json
import random
from datetime import datetime, timedelta
from faker import Faker
import uuid

fake = Faker()

# MongoDB connection via MCP server
MCP_SERVER_URL = "http://localhost:8001/execute"

class RealisticDataGenerator:
    def __init__(self):
        self.users = []
        self.products = []
        self.orders = []
        self.campaigns = []
        
    async def generate_users(self, count=1000):
        """Generate realistic user profiles"""
        print(f"🔄 Generating {count} realistic users...")
        users = []
        
        for i in range(count):
            user = {
                "_id": str(uuid.uuid4()),
                "email": fake.email(),
                "name": fake.name(),
                "age": random.randint(18, 75),
                "location": {
                    "city": fake.city(),
                    "country": fake.country(),
                    "coordinates": [fake.longitude(), fake.latitude()]
                },
                "preferences": {
                    "categories": random.sample([
                        "electronics", "clothing", "books", "movies", "music", 
                        "sports", "travel", "food", "health", "technology"
                    ], random.randint(2, 5)),
                    "price_range": random.choice(["budget", "mid-range", "premium"]),
                    "communication": random.choice(["email", "sms", "push"])
                },
                "demographics": {
                    "income_bracket": random.choice(["low", "middle", "high"]),
                    "education": random.choice(["high_school", "bachelor", "master", "phd"]),
                    "occupation": fake.job()
                },
                "created_at": fake.date_time_between(start_date="-2y", end_date="now").isoformat(),
                "status": random.choice(["active", "inactive", "premium"]),
                "lifetime_value": round(random.uniform(100, 5000), 2),
                "last_login": fake.date_time_between(start_date="-30d", end_date="now").isoformat()
            }
            users.append(user)
        
        self.users = users
        await self._insert_batch("users", users)
        print(f"✅ Created {len(users)} realistic users")

    async def generate_products(self, count=500):
        """Generate realistic product catalog"""
        print(f"🔄 Generating {count} realistic products...")
        products = []
        
        categories = [
            {"name": "Electronics", "subcats": ["smartphones", "laptops", "tablets", "accessories"]},
            {"name": "Clothing", "subcats": ["shirts", "pants", "shoes", "accessories"]},
            {"name": "Books", "subcats": ["fiction", "non-fiction", "textbooks", "comics"]},
            {"name": "Home", "subcats": ["furniture", "kitchen", "decor", "garden"]},
            {"name": "Sports", "subcats": ["fitness", "outdoor", "team-sports", "individual"]}
        ]
        
        for i in range(count):
            category = random.choice(categories)
            subcategory = random.choice(category["subcats"])
            
            product = {
                "_id": str(uuid.uuid4()),
                "name": fake.catch_phrase(),
                "description": fake.text(max_nb_chars=200),
                "category": category["name"].lower(),
                "subcategory": subcategory,
                "price": round(random.uniform(10, 2000), 2),
                "cost": round(random.uniform(5, 1000), 2),
                "sku": fake.bothify(text="??###??").upper(),
                "brand": fake.company(),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "reviews_count": random.randint(0, 500),
                "inventory": {
                    "stock": random.randint(0, 1000),
                    "warehouse": fake.city(),
                    "reorder_level": random.randint(10, 50)
                },
                "attributes": {
                    "weight": round(random.uniform(0.1, 50), 2),
                    "dimensions": f"{random.randint(10, 100)}x{random.randint(10, 100)}x{random.randint(5, 50)}",
                    "color": fake.color_name(),
                    "material": random.choice(["plastic", "metal", "wood", "fabric", "glass"])
                },
                "seo": {
                    "keywords": fake.words(nb=5),
                    "meta_description": fake.sentence()
                },
                "created_at": fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
                "status": random.choice(["active", "discontinued", "out_of_stock"]),
                "supplier": {
                    "name": fake.company(),
                    "contact": fake.email()
                }
            }
            products.append(product)
        
        self.products = products
        await self._insert_batch("products", products)
        print(f"✅ Created {len(products)} realistic products")

    async def generate_orders(self, count=2000):
        """Generate realistic order history"""
        print(f"🔄 Generating {count} realistic orders...")
        
        if not self.users or not self.products:
            print("❌ Users and products must be generated first")
            return
            
        orders = []
        
        for i in range(count):
            user = random.choice(self.users)
            order_products = random.sample(self.products, random.randint(1, 5))
            
            subtotal = sum(p["price"] * random.randint(1, 3) for p in order_products)
            tax = round(subtotal * 0.08, 2)
            shipping = round(random.uniform(5, 25), 2) if subtotal < 100 else 0
            total = subtotal + tax + shipping
            
            order = {
                "_id": str(uuid.uuid4()),
                "user_id": user["_id"],
                "order_number": f"ORD-{fake.bothify(text='###???###').upper()}",
                "status": random.choice(["pending", "processing", "shipped", "delivered", "cancelled"]),
                "items": [
                    {
                        "product_id": p["_id"],
                        "product_name": p["name"],
                        "quantity": random.randint(1, 3),
                        "unit_price": p["price"],
                        "total_price": p["price"] * random.randint(1, 3)
                    }
                    for p in order_products
                ],
                "pricing": {
                    "subtotal": subtotal,
                    "tax": tax,
                    "shipping": shipping,
                    "total": total,
                    "currency": "USD"
                },
                "shipping": {
                    "address": {
                        "street": fake.street_address(),
                        "city": fake.city(),
                        "state": fake.state(),
                        "zip": fake.zipcode(),
                        "country": fake.country()
                    },
                    "method": random.choice(["standard", "express", "overnight"]),
                    "tracking": fake.bothify(text="??###??###").upper()
                },
                "payment": {
                    "method": random.choice(["credit_card", "debit_card", "paypal", "bank_transfer"]),
                    "status": random.choice(["paid", "pending", "failed"]),
                    "transaction_id": fake.uuid4()
                },
                "timestamps": {
                    "created_at": fake.date_time_between(start_date="-6m", end_date="now").isoformat(),
                    "updated_at": fake.date_time_between(start_date="-6m", end_date="now").isoformat()
                },
                "notes": fake.sentence() if random.random() < 0.3 else ""
            }
            orders.append(order)
        
        self.orders = orders
        await self._insert_batch("orders", orders)
        print(f"✅ Created {len(orders)} realistic orders")

    async def generate_support_tickets(self, count=300):
        """Generate realistic support tickets"""
        print(f"🔄 Generating {count} realistic support tickets...")
        
        if not self.users:
            print("❌ Users must be generated first")
            return
            
        tickets = []
        issues = [
            "Product not working as expected",
            "Delivery delayed",
            "Wrong item received",
            "Refund request",
            "Account access issues",
            "Payment problems",
            "Product quality concerns",
            "Feature request",
            "Technical support needed",
            "Billing inquiry"
        ]
        
        for i in range(count):
            user = random.choice(self.users)
            created_date = fake.date_time_between(start_date="-3m", end_date="now")
            
            ticket = {
                "_id": str(uuid.uuid4()),
                "ticket_number": f"TKT-{fake.bothify(text='####').upper()}",
                "user_id": user["_id"],
                "user_email": user["email"],
                "subject": random.choice(issues),
                "description": fake.text(max_nb_chars=500),
                "priority": random.choice(["low", "medium", "high", "urgent"]),
                "status": random.choice(["open", "in_progress", "resolved", "closed"]),
                "category": random.choice(["technical", "billing", "product", "shipping", "general"]),
                "assigned_to": fake.name() if random.random() < 0.7 else None,
                "tags": random.sample(["bug", "feature", "urgent", "vip", "refund", "exchange"], random.randint(0, 3)),
                "timestamps": {
                    "created_at": created_date.isoformat(),
                    "updated_at": fake.date_time_between(start_date=created_date, end_date="now").isoformat(),
                    "resolved_at": fake.date_time_between(start_date=created_date, end_date="now").isoformat() if random.random() < 0.6 else None
                },
                "customer_satisfaction": random.randint(1, 5) if random.random() < 0.5 else None,
                "resolution_time_hours": random.randint(1, 72) if random.random() < 0.6 else None
            }
            tickets.append(ticket)
        
        await self._insert_batch("support_tickets", tickets)
        print(f"✅ Created {len(tickets)} realistic support tickets")

    async def enhance_user_activity(self, count=5000):
        """Generate enhanced user activity data"""
        print(f"🔄 Generating {count} enhanced user activity records...")
        
        if not self.users or not self.products:
            print("❌ Users and products must be generated first")
            return
            
        activities = []
        
        events = [
            "page_view", "product_view", "add_to_cart", "remove_from_cart",
            "purchase", "search", "login", "logout", "profile_update",
            "review_write", "wishlist_add", "share_product", "contact_support"
        ]
        
        for i in range(count):
            user = random.choice(self.users)
            product = random.choice(self.products) if random.random() < 0.7 else None
            
            activity_time = fake.date_time_between(start_date="-30d", end_date="now")
            
            activity = {
                "_id": str(uuid.uuid4()),
                "user_id": user["_id"],
                "session_id": fake.uuid4(),
                "event_type": random.choice(events),
                "timestamp": activity_time.isoformat(),
                "device_info": {
                    "device_type": random.choice(["desktop", "mobile", "tablet"]),
                    "browser": random.choice(["chrome", "firefox", "safari", "edge"]),
                    "os": random.choice(["windows", "macos", "ios", "android", "linux"]),
                    "screen_size": random.choice(["1920x1080", "1366x768", "375x667", "768x1024"])
                },
                "location": {
                    "ip": fake.ipv4(),
                    "city": fake.city(),
                    "country": fake.country_code(),
                    "timezone": fake.timezone()
                },
                "event_data": {
                    "page_url": fake.url() if random.random() < 0.8 else None,
                    "referrer": fake.url() if random.random() < 0.5 else None,
                    "product_id": product["_id"] if product else None,
                    "search_query": fake.words(nb=random.randint(1, 4)) if random.random() < 0.3 else None,
                    "cart_value": round(random.uniform(10, 500), 2) if random.random() < 0.2 else None
                },
                "session_duration": random.randint(30, 1800),  # 30 seconds to 30 minutes
                "conversion_flag": random.random() < 0.15,  # 15% conversion rate
                "ab_test_group": random.choice(["A", "B", "control"]) if random.random() < 0.3 else None
            }
            activities.append(activity)
        
        await self._insert_batch("user_activity", activities)
        print(f"✅ Created {len(activities)} enhanced user activity records")

    async def _insert_batch(self, collection: str, documents: list, batch_size=100):
        """Insert documents in batches to avoid timeout"""
        total = len(documents)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, total, batch_size):
                batch = documents[i:i+batch_size]
                try:
                    response = await client.post(
                        MCP_SERVER_URL,
                        json={
                            "tool": "mongodb_insert_many",
                            "params": {
                                "collection": collection,
                                "documents": batch
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            print(f"  ✅ Inserted batch {i//batch_size + 1}/{(total-1)//batch_size + 1} into {collection}")
                        else:
                            print(f"  ❌ Error in batch {i//batch_size + 1}: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"  ❌ HTTP error {response.status_code} for batch {i//batch_size + 1}")
                        
                except Exception as e:
                    print(f"  ❌ Exception inserting batch {i//batch_size + 1}: {e}")
                
                # Small delay between batches
                await asyncio.sleep(0.1)

    async def verify_data_quality(self):
        """Verify the quality and completeness of generated data"""
        print("\n🔍 Verifying data quality...")
        
        collections = ["users", "products", "orders", "user_activity", "support_tickets"]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for collection in collections:
                try:
                    # Get count
                    response = await client.post(
                        MCP_SERVER_URL,
                        json={
                            "tool": "mongodb_count",
                            "params": {"collection": collection}
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            count = result["result"]["count"]
                            print(f"  📊 {collection}: {count:,} documents")
                        else:
                            print(f"  ❌ Error counting {collection}: {result.get('error')}")
                    else:
                        print(f"  ❌ HTTP error {response.status_code} for {collection}")
                        
                except Exception as e:
                    print(f"  ❌ Exception counting {collection}: {e}")

async def main():
    """Generate comprehensive realistic data"""
    print("🚀 Starting realistic data generation for HydrogenAI...")
    
    generator = RealisticDataGenerator()
    
    try:
        # Generate data in dependency order
        await generator.generate_users(1000)
        await generator.generate_products(500)
        await generator.generate_orders(2000)
        await generator.generate_support_tickets(300)
        await generator.enhance_user_activity(5000)
        
        # Verify the results
        await generator.verify_data_quality()
        
        print("\n✅ Realistic data generation completed successfully!")
        print("📈 Your HydrogenAI system now has comprehensive, realistic data for testing")
        
    except Exception as e:
        print(f"❌ Error during data generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
