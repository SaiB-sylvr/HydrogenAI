#!/usr/bin/env python3
"""
Test script to verify that main.py imports work correctly
"""
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'services/shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'services/orchestrator'))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        print("✅ Testing shared services imports...")
        from services.shared.config_validator import ConfigValidator
        from services.shared.ai_provider_manager import AIProviderManager
        from services.shared.ai_cache import AIResponseCache
        print("✅ Shared services imports successful")
        
        # Test config validator
        config_validator = ConfigValidator()
        print("✅ Config validator instantiated")
        
        # Test validation (should work even without real config)
        validation_result = config_validator.validate_all()
        print(f"✅ Config validation completed: {validation_result.get('overall_status', 'unknown')}")
        
        print("🎉 All critical imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
