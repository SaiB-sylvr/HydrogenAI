import asyncio
from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.breakers: Dict[str, Dict[str, Any]] = {}
    
    def _get_breaker(self, name: str) -> Dict[str, Any]:
        """Get or create circuit breaker for service"""
        if name not in self.breakers:
            self.breakers[name] = {
                "state": CircuitState.CLOSED,
                "failures": 0,
                "last_failure": None,
                "success_count": 0
            }
        return self.breakers[name]
    
    async def call(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection"""
        breaker = self._get_breaker(service_name)
        
        # Check if circuit is open
        if breaker["state"] == CircuitState.OPEN:
            if self._should_attempt_reset(breaker):
                breaker["state"] = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker for {service_name} is HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker for {service_name} is OPEN")
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Success - update state
            self._on_success(service_name, breaker)
            return result
            
        except self.expected_exception as e:
            # Failure - update state
            self._on_failure(service_name, breaker)
            raise
    
    def _should_attempt_reset(self, breaker: Dict[str, Any]) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (
            breaker["last_failure"] and
            datetime.utcnow() - breaker["last_failure"] > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self, service_name: str, breaker: Dict[str, Any]):
        """Handle successful call"""
        breaker["failures"] = 0
        
        if breaker["state"] == CircuitState.HALF_OPEN:
            breaker["success_count"] += 1
            if breaker["success_count"] >= 2:
                breaker["state"] = CircuitState.CLOSED
                breaker["success_count"] = 0
                logger.info(f"Circuit breaker for {service_name} is CLOSED")
    
    def _on_failure(self, service_name: str, breaker: Dict[str, Any]):
        """Handle failed call"""
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.utcnow()
        breaker["success_count"] = 0
        
        if breaker["failures"] >= self.failure_threshold:
            breaker["state"] = CircuitState.OPEN
            logger.warning(f"Circuit breaker for {service_name} is OPEN")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        status = {}
        for name, breaker in self.breakers.items():
            status[name] = {
                "state": breaker["state"].value,
                "failures": breaker["failures"],
                "last_failure": breaker["last_failure"].isoformat() if breaker["last_failure"] else None
            }
        return status