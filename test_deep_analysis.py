#!/usr/bin/env python3
"""
Enhanced Deep User Analysis Testing Script
"""

import requests
import json
import time
from typing import Dict, Any

def test_enhanced_analysis():
    """Test the enhanced deep user analysis capabilities"""
    
    print("=" * 50)
    print("🚀 TESTING ENHANCED DEEP USER ANALYSIS")
    print("=" * 50)
    print()
    
    # Test 1: Deep User Behavioral Analysis
    print("📊 Test 1: Deep User Behavioral Analysis")
    print("-" * 40)
    
    test_query1 = {
        "query": "Analyze user behavior patterns and provide detailed insights about individual users, their engagement levels, purchase patterns, and strategic recommendations for each user segment",
        "context": {
            "analysis_type": "deep_user_behavioral",
            "include_individual_profiles": True,
            "require_strategic_insights": True
        }
    }
    
    try:
        response1 = requests.post(
            "http://localhost:8000/api/query",
            json=test_query1,
            timeout=60
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            content = data1.get('human_response', data1.get('response', str(data1)))
            
            print("✅ ENHANCED DEEP ANALYSIS SUCCESS!")
            print(f"Response Length: {len(content)} characters")
            print()
            
            # Check for enhanced intelligence markers
            intelligence_markers = [
                "engagement_metrics",
                "behavioral_analysis", 
                "customer_journey",
                "technology_profile",
                "geographic_profile",
                "temporal_patterns",
                "business_value",
                "individual user",
                "comprehensive analysis",
                "multi-dimensional user segmentation"
            ]
            
            found_markers = 0
            print("Intelligence Markers Found:")
            for marker in intelligence_markers:
                if marker.lower() in content.lower():
                    print(f"  ✓ {marker}")
                    found_markers += 1
                else:
                    print(f"  ✗ {marker}")
            
            print(f"\nIntelligence Score: {found_markers}/{len(intelligence_markers)}")
            
            if found_markers > 5:
                print("🎯 ENHANCED ANALYSIS CONFIRMED!")
            else:
                print("⚠️  Basic analysis detected - enhancement needed")
            
            print("\nSample Response Preview:")
            print("-" * 30)
            print(content[:800] + ("..." if len(content) > 800 else ""))
            
        else:
            print(f"❌ Request failed with status: {response1.status_code}")
            print(f"Error: {response1.text}")
            
    except Exception as e:
        print(f"❌ Test 1 Failed: {str(e)}")
    
    print("\n" + "=" * 50)
    
    # Test 2: Individual User Deep Profile Analysis
    print("👤 Test 2: Individual User Deep Profile Analysis")
    print("-" * 40)
    
    test_query2 = {
        "query": "Provide detailed analysis of individual users including their complete behavioral profile, device usage patterns, geographic insights, and personalized recommendations",
        "context": {
            "analysis_depth": "maximum",
            "include_user_profiles": True,
            "include_recommendations": True
        }
    }
    
    try:
        response2 = requests.post(
            "http://localhost:8000/api/query",
            json=test_query2,
            timeout=60
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            content2 = data2.get('human_response', data2.get('response', str(data2)))
            
            print("✅ INDIVIDUAL USER ANALYSIS SUCCESS!")
            print(f"Response Length: {len(content2)} characters")
            
            # Check for user profile markers
            user_profile_markers = [
                "engagement_level",
                "device_diversity",
                "location_patterns", 
                "revenue_generated",
                "customer_lifetime",
                "behavioral_segment",
                "recommendations",
                "strategic insights"
            ]
            
            found_profile_markers = 0
            print("\nUser Profile Analysis Markers:")
            for marker in user_profile_markers:
                if marker.lower() in content2.lower():
                    print(f"  ✓ {marker}")
                    found_profile_markers += 1
                else:
                    print(f"  ✗ {marker}")
            
            print(f"\nUser Profile Score: {found_profile_markers}/{len(user_profile_markers)}")
            
            if found_profile_markers > 4:
                print("🎯 DEEP USER PROFILING CONFIRMED!")
            else:
                print("⚠️  Surface-level analysis detected")
                
        else:
            print(f"❌ Request failed with status: {response2.status_code}")
            print(f"Error: {response2.text}")
            
    except Exception as e:
        print(f"❌ Test 2 Failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 ENHANCED ANALYSIS TESTING COMPLETE")
    print("=" * 50)
    
    # System Health Check
    print("\nSystem Status Check:")
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Orchestrator Health: OK")
        else:
            print("❌ Orchestrator Health: FAILED")
    except:
        print("❌ Orchestrator Health: UNREACHABLE")

if __name__ == "__main__":
    test_enhanced_analysis()
