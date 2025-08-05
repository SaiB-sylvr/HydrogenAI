#!/usr/bin/env python3
"""
Seed MongoDB with realistic user activity data for testing deep behavioral analysis
"""

import pymongo
import random
import datetime
from datetime import timedelta
import uuid

# MongoDB connection
MONGO_URI = "MONGO_URI"

def seed_user_activity_data():
    """Seed realistic user activity data"""
    
    client = pymongo.MongoClient(MONGO_URI)
    db = client.HydrogenAI
    collection = db.user_activity
    
    print("🌱 Seeding user activity data for deep behavioral analysis...")
    
    # Clear existing data
    collection.delete_many({})
    
    # Generate realistic user activity data
    user_profiles = [
        {"type": "high_engagement", "event_count": (200, 500), "devices": 3, "locations": 4},
        {"type": "conversion_champion", "event_count": (100, 300), "devices": 2, "locations": 2, "purchases": (5, 15)},
        {"type": "at_risk", "event_count": (10, 50), "devices": 1, "locations": 1, "last_active_days_ago": (15, 30)},
        {"type": "new_user", "event_count": (5, 20), "devices": 1, "locations": 1, "days_old": (1, 7)},
        {"type": "loyal_customer", "event_count": (300, 800), "devices": 4, "locations": 3, "purchases": (10, 25), "days_old": (60, 200)},
        {"type": "multi_device", "event_count": (150, 400), "devices": 5, "locations": 6}
    ]
    
    event_types = [
        "page_view", "click", "search", "add_to_cart", "purchase", 
        "sign_up", "login", "logout", "share", "comment", "like", 
        "video_watch", "download", "profile_update", "message_send"
    ]
    
    devices = ["desktop", "mobile", "tablet", "smart_tv", "laptop"]
    locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
    
    activities = []
    
    # Generate 500 users with diverse behavioral patterns
    for user_num in range(500):
        profile = random.choice(user_profiles)
        user_id = f"user_{user_num:03d}"
        
        # Base user characteristics
        event_count = random.randint(*profile["event_count"])
        user_devices = random.sample(devices, min(profile["devices"], len(devices)))
        user_locations = random.sample(locations, min(profile["locations"], len(locations)))
        
        # Time characteristics
        if "days_old" in profile:
            start_date = datetime.datetime.now() - timedelta(days=random.randint(*profile["days_old"]))
        else:
            start_date = datetime.datetime.now() - timedelta(days=random.randint(1, 90))
            
        if "last_active_days_ago" in profile:
            end_date = datetime.datetime.now() - timedelta(days=random.randint(*profile["last_active_days_ago"]))
        else:
            end_date = datetime.datetime.now() - timedelta(days=random.randint(0, 3))
        
        # Generate events for this user
        for _ in range(event_count):
            timestamp = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            event = {
                "user_id": user_id,
                "event_type": random.choice(event_types),
                "timestamp": timestamp,
                "device": random.choice(user_devices),
                "location": random.choice(user_locations),
                "session_id": str(uuid.uuid4())[:8],
                "value": random.uniform(1, 100) if random.random() > 0.7 else 0,
                "properties": {
                    "page": f"/page_{random.randint(1, 50)}",
                    "referrer": random.choice(["google", "facebook", "direct", "email", "ads"]),
                    "duration": random.randint(5, 300)
                }
            }
            
            # Add purchase-specific data for purchases
            if event["event_type"] == "purchase":
                event["revenue"] = random.uniform(10, 500)
                event["product_id"] = f"prod_{random.randint(1, 100)}"
                event["quantity"] = random.randint(1, 5)
            
            activities.append(event)
    
    # Insert in batches
    batch_size = 1000
    for i in range(0, len(activities), batch_size):
        batch = activities[i:i + batch_size]
        collection.insert_many(batch)
        print(f"Inserted batch {i//batch_size + 1}/{(len(activities)-1)//batch_size + 1}")
    
    print(f"✅ Seeded {len(activities)} user activity records for {500} users")
    
    # Verify data
    total_count = collection.count_documents({})
    unique_users = len(collection.distinct("user_id"))
    unique_events = len(collection.distinct("event_type"))
    
    print(f"📊 Verification:")
    print(f"   Total events: {total_count}")
    print(f"   Unique users: {unique_users}")
    print(f"   Event types: {unique_events}")
    
    client.close()

def seed_content_data():
    """Seed content/digital asset data"""
    
    client = pymongo.MongoClient(MONGO_URI)
    db = client.HydrogenAI
    collection = db.digital_content
    
    print("🎬 Seeding digital content data...")
    
    # Clear existing data
    collection.delete_many({})
    
    content_types = ["video", "article", "podcast", "image", "course", "webinar"]
    categories = ["technology", "business", "entertainment", "education", "health", "lifestyle"]
    
    contents = []
    
    # Generate 500 content items
    for i in range(500):
        content = {
            "content_id": f"content_{i:03d}",
            "title": f"Content Title {i}",
            "type": random.choice(content_types),
            "category": random.choice(categories),
            "views": random.randint(100, 10000),
            "likes": random.randint(10, 1000),
            "shares": random.randint(1, 200),
            "duration": random.randint(60, 3600),  # seconds
            "created_at": datetime.datetime.now() - timedelta(days=random.randint(1, 365)),
            "rating": round(random.uniform(1, 5), 1),
            "engagement_rate": round(random.uniform(0.1, 0.8), 2)
        }
        contents.append(content)
    
    collection.insert_many(contents)
    print(f"✅ Seeded {len(contents)} content records")
    
    client.close()

def seed_marketing_campaigns():
    """Seed marketing campaign data"""
    
    client = pymongo.MongoClient(MONGO_URI)
    db = client.HydrogenAI
    collection = db.marketing_campaigns
    
    print("📢 Seeding marketing campaign data...")
    
    # Clear existing data
    collection.delete_many({})
    
    campaign_types = ["email", "social", "search", "display", "video", "influencer"]
    statuses = ["active", "paused", "completed", "draft"]
    
    campaigns = []
    
    # Generate 50 campaigns
    for i in range(50):
        campaign = {
            "campaign_id": f"camp_{i:03d}",
            "name": f"Campaign {i}: {random.choice(['Summer Sale', 'Product Launch', 'Brand Awareness', 'Retargeting', 'Lead Gen'])}",
            "type": random.choice(campaign_types),
            "status": random.choice(statuses),
            "budget": random.randint(1000, 50000),
            "spent": random.randint(500, 45000),
            "impressions": random.randint(10000, 500000),
            "clicks": random.randint(100, 25000),
            "conversions": random.randint(10, 2000),
            "start_date": datetime.datetime.now() - timedelta(days=random.randint(1, 90)),
            "end_date": datetime.datetime.now() + timedelta(days=random.randint(1, 60)),
            "target_audience": {
                "age_range": f"{random.randint(18, 35)}-{random.randint(36, 65)}",
                "interests": random.sample(["technology", "business", "lifestyle", "health", "travel"], 2)
            }
        }
        campaigns.append(campaign)
    
    collection.insert_many(campaigns)
    print(f"✅ Seeded {len(campaigns)} marketing campaign records")
    
    client.close()

if __name__ == "__main__":
    print("🚀 Starting MongoDB seeding for deep behavioral analysis testing...")
    seed_user_activity_data()
    seed_content_data()
    seed_marketing_campaigns()
    print("🎉 Seeding complete! Ready for deep behavioral analysis testing.")

