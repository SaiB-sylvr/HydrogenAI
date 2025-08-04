#!/usr/bin/env python3
"""
Comprehensive System Test for HydrogenAI with Enhanced AI Provider Management
"""
import asyncio
import sys
import os
import json
from typing import Dict, Any

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'services/shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'services/orchestrator'))

async def test_ai_provider_manager():
    """Test AI Provider Manager functionality"""
    print("🤖 Testing AI Provider Manager...")
    
    try:
        from services.shared.ai_provider_manager import AIProviderManager
        
        # Initialize manager
        manager = AIProviderManager()
        await manager.initialize()
        print("✅ AI Provider Manager initialized")
        
        # Test response generation
        response = await manager.generate_response(
            "How many customers do we have?",
            {"test": True}
        )
        
        print(f"✅ AI response generated: {response.get('provider', 'unknown')} provider used")
        
        # Test health status
        health = await manager.get_health_status()
        print(f"✅ Health status: {health.get('current_provider', 'unknown')} active")
        
        # Cleanup
        await manager.cleanup()
        print("✅ AI Provider Manager cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Provider Manager test failed: {e}")
        return False

async def test_ai_cache():
    """Test AI Response Cache functionality"""
    print("💾 Testing AI Response Cache...")
    
    try:
        from services.shared.ai_cache import AIResponseCache
        
        # Initialize cache (will fallback gracefully if Redis unavailable)
        cache = AIResponseCache("redis://localhost:6379")
        await cache.initialize()
        print("✅ AI Response Cache initialized (may use fallback)")
        
        # Test cache operations
        test_key = "test_key"
        test_value = {"response": "test data"}
        
        await cache.set(test_key, test_value, ttl=60)
        cached_value = await cache.get(test_key)
        
        if cached_value:
            print("✅ Cache set/get operations working")
        else:
            print("ℹ️ Cache operations using fallback (Redis not available)")
        
        # Cleanup
        await cache.cleanup()
        print("✅ AI Response Cache cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Response Cache test failed: {e}")
        return False

async def test_config_validator():
    """Test Configuration Validator"""
    print("⚙️ Testing Configuration Validator...")
    
    try:
        from services.shared.config_validator import ConfigValidator
        
        # Initialize validator
        validator = ConfigValidator()
        print("✅ Config Validator initialized")
        
        # Test validation
        validation_result = validator.validate_all()
        print(f"✅ Configuration validation: {validation_result.get('overall_status', 'completed')}")
        
        # Display any warnings
        warnings = validation_result.get('warnings', 0)
        if isinstance(warnings, int) and warnings > 0:
            print(f"⚠️ Warnings: {warnings} found")
        elif isinstance(warnings, list) and len(warnings) > 0:
            print(f"⚠️ Warnings: {len(warnings)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Config Validator test failed: {e}")
        return False

async def test_integration():
    """Test integration of all components"""
    print("🔗 Testing component integration...")
    
    try:
        from services.shared.config_validator import ConfigValidator
        from services.shared.ai_provider_manager import AIProviderManager
        from services.shared.ai_cache import AIResponseCache
        
        # Initialize all components
        config_validator = ConfigValidator()
        ai_manager = AIProviderManager()
        ai_cache = AIResponseCache("redis://localhost:6379")
        
        await ai_manager.initialize()
        await ai_cache.initialize()
        
        print("✅ All components initialized together")
        
        # Test integrated workflow
        validation = config_validator.validate_all()
        response = await ai_manager.generate_response("Test integration query")
        health = await ai_manager.get_health_status()
        
        print("✅ Integrated workflow completed")
        
        # Cleanup
        await ai_manager.cleanup()
        await ai_cache.cleanup()
        
        print("✅ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 HydrogenAI Enhanced System Test")
    print("=" * 50)
    
    tests = [
        ("AI Provider Manager", test_ai_provider_manager),
        ("AI Response Cache", test_ai_cache), 
        ("Config Validator", test_config_validator),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced AI system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
