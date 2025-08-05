"""
Configuration Validation System
"""
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)

# Load .env file if available
def load_env_file(env_path: str = ".env"):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        logger.info(f"Loaded environment variables from {env_path}")

# Load .env on import
load_env_file()

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    level: ValidationLevel
    component: str
    message: str
    suggestion: Optional[str] = None

class ConfigValidator:
    """Validates system configuration at startup"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all configuration components"""
        self.results = []
        
        # Core validations
        self._validate_environment_variables()
        self._validate_database_config()
        self._validate_ai_providers()
        self._validate_redis_config()
        self._validate_security_config()
        self._validate_service_config()
        
        # Categorize results
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        info = [r for r in self.results if r.level == ValidationLevel.INFO]
        
        validation_summary = {
            "overall_status": "FAILED" if errors else "PASSED" if warnings else "OK",
            "status": "FAILED" if errors else "PASSED" if warnings else "OK", # Keep both for compatibility
            "total_issues": len(self.results),
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(info),
            "details": {
                "errors": [{"component": r.component, "message": r.message, "suggestion": r.suggestion} for r in errors],
                "warnings": [{"component": r.component, "message": r.message, "suggestion": r.suggestion} for r in warnings],
                "info": [{"component": r.component, "message": r.message, "suggestion": r.suggestion} for r in info]
            }
        }
        
        return validation_summary
    
    def _validate_environment_variables(self):
        """Validate required environment variables"""
        required_vars = {
            "MONGO_URI": "MongoDB connection string",
            "MONGO_DB_NAME": "MongoDB database name",
            "REDIS_URL": "Redis connection URL",
            "EVENT_BUS_URL": "NATS event bus URL"
        }
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    component="Environment",
                    message=f"Missing required environment variable: {var}",
                    suggestion=f"Set {var} to your {description}"
                ))
            elif var == "MONGO_URI" and not self._validate_mongo_uri(value):
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    component="Environment", 
                    message=f"Invalid MongoDB URI format: {var}",
                    suggestion="Ensure MONGO_URI follows mongodb:// or mongodb+srv:// format"
                ))
        
        # Check for hardcoded secrets
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                content = f.read()
                if "mongodb+srv://sai:" in content:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        component="Security",
                        message="Hardcoded MongoDB credentials detected in .env file",
                        suggestion="Use environment-specific credential injection"
                    ))
    
    def _validate_database_config(self):
        """Validate database configuration"""
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            # Check if using Atlas (recommended for production)
            if "mongodb+srv://" in mongo_uri:
                self.results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    component="Database",
                    message="Using MongoDB Atlas (recommended)",
                    suggestion=None
                ))
            elif "localhost" in mongo_uri or "127.0.0.1" in mongo_uri:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    component="Database",
                    message="Using local MongoDB instance",
                    suggestion="Consider MongoDB Atlas for production deployments"
                ))
            
            # Validate connection parameters
            if "retryWrites=true" not in mongo_uri:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    component="Database",
                    message="MongoDB URI missing retryWrites=true parameter",
                    suggestion="Add retryWrites=true for better reliability"
                ))
        
        # Check database name
        db_name = os.getenv("MONGO_DB_NAME")
        if db_name and not re.match(r"^[a-zA-Z0-9_-]+$", db_name):
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                component="Database",
                message="Database name contains special characters",
                suggestion="Use alphanumeric characters, underscores, and hyphens only"
            ))
    
    def _validate_ai_providers(self):
        """Validate AI provider configuration"""
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        provider_count = sum(1 for key in [groq_key, openai_key, anthropic_key] if key)
        
        if provider_count == 0:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                component="AI Providers",
                message="No AI provider API keys configured",
                suggestion="Set at least GROQ_API_KEY for basic functionality"
            ))
        elif provider_count == 1:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                component="AI Providers",
                message="Only one AI provider configured",
                suggestion="Add backup providers (OPENAI_API_KEY, ANTHROPIC_API_KEY) for resilience"
            ))
        else:
            self.results.append(ValidationResult(
                level=ValidationLevel.INFO,
                component="AI Providers",
                message=f"Multiple AI providers configured ({provider_count})",
                suggestion=None
            ))
        
        # Validate API key format
        if groq_key and not groq_key.startswith("gsk_"):
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                component="AI Providers",
                message="GROQ_API_KEY format appears incorrect",
                suggestion="Groq API keys should start with 'gsk_'"
            ))
        
        if openai_key and not openai_key.startswith("sk-"):
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                component="AI Providers",
                message="OPENAI_API_KEY format appears incorrect",
                suggestion="OpenAI API keys should start with 'sk-'"
            ))
    
    def _validate_redis_config(self):
        """Validate Redis configuration"""
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            if "localhost" in redis_url or "127.0.0.1" in redis_url:
                self.results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    component="Cache",
                    message="Using local Redis instance",
                    suggestion="Consider Redis Cloud for production deployments"
                ))
            
            # Check for authentication
            if "@" not in redis_url and "redis://redis:" not in redis_url:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    component="Cache",
                    message="Redis URL appears to have no authentication",
                    suggestion="Use password-protected Redis in production"
                ))
    
    def _validate_security_config(self):
        """Validate security configuration"""
        # Check for development settings in production
        environment = os.getenv("ENVIRONMENT", "development")
        log_level = os.getenv("LOG_LEVEL", "INFO")
        
        if environment == "production":
            if log_level == "DEBUG":
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    component="Security",
                    message="DEBUG logging enabled in production",
                    suggestion="Set LOG_LEVEL=INFO or higher for production"
                ))
            
            # Check for secrets in environment
            sensitive_patterns = [
                ("password", "passwords in environment variables"),
                ("secret", "secrets in environment variables"),
                ("key", "API keys visible in process list")
            ]
            
            for pattern, description in sensitive_patterns:
                env_vars = [k for k in os.environ.keys() if pattern.lower() in k.lower()]
                if env_vars and environment == "production":
                    self.results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        component="Security",
                        message=f"Potential {description}",
                        suggestion="Use secrets management system in production"
                    ))
    
    def _validate_service_config(self):
        """Validate service-specific configuration"""
        # Check for required service configurations
        service_configs = {
            "SCHEMA_CACHE_TTL": "Schema cache TTL",
            "QUERY_CACHE_TTL": "Query cache TTL"
        }
        
        for config, description in service_configs.items():
            value = os.getenv(config)
            if value and not value.isdigit():
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    component="Service Config",
                    message=f"Invalid {description} value: {value}",
                    suggestion=f"Set {config} to a numeric value (seconds)"
                ))
        
        # Check Qdrant configuration
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        
        if qdrant_host == "localhost" or qdrant_host == "127.0.0.1":
            self.results.append(ValidationResult(
                level=ValidationLevel.INFO,
                component="Vector DB",
                message="Using local Qdrant instance",
                suggestion="Consider Qdrant Cloud for production deployments"
            ))
        
        if not qdrant_port.isdigit():
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                component="Vector DB",
                message=f"Invalid Qdrant port: {qdrant_port}",
                suggestion="Set QDRANT_PORT to a valid port number"
            ))
    
    def _validate_mongo_uri(self, uri: str) -> bool:
        """Validate MongoDB URI format"""
        mongodb_patterns = [
            r"^mongodb://.*",
            r"^mongodb\+srv://.*"
        ]
        return any(re.match(pattern, uri) for pattern in mongodb_patterns)
    
    def print_validation_report(self):
        """Print a formatted validation report"""
        results = self.validate_all()
        
        print("🔍 HydrogenAI Configuration Validation Report")
        print("=" * 50)
        print(f"Status: {results['status']}")
        print(f"Total Issues: {results['total_issues']}")
        print(f"Errors: {results['errors']}, Warnings: {results['warnings']}, Info: {results['info']}")
        
        if results['details']['errors']:
            print("\n❌ ERRORS (Must Fix):")
            for error in results['details']['errors']:
                print(f"  • {error['component']}: {error['message']}")
                if error['suggestion']:
                    print(f"    💡 {error['suggestion']}")
        
        if results['details']['warnings']:
            print("\n⚠️ WARNINGS (Should Fix):")
            for warning in results['details']['warnings']:
                print(f"  • {warning['component']}: {warning['message']}")
                if warning['suggestion']:
                    print(f"    💡 {warning['suggestion']}")
        
        if results['details']['info']:
            print("\nℹ️ INFO:")
            for info in results['details']['info']:
                print(f"  • {info['component']}: {info['message']}")
        
        if results['status'] == "OK":
            print("\n✅ Configuration validation passed!")
        elif results['status'] == "PASSED":
            print("\n⚠️ Configuration validation passed with warnings")
        else:
            print("\n❌ Configuration validation failed - fix errors before deployment")
        
        return results

# Global validator instance
config_validator = ConfigValidator()

# Convenience function
def validate_config() -> Dict[str, Any]:
    """Validate system configuration"""
    return config_validator.validate_all()

def print_config_report():
    """Print configuration validation report"""
    return config_validator.print_validation_report()
