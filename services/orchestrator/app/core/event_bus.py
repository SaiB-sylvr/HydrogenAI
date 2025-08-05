import asyncio
import json
from typing import Dict, Any, Callable, Optional
import logging

# Import NATS with fallback
try:
    from nats.aio.client import Client as NATS
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    
    # Mock NATS client for fallback
    class NATS:
        def __init__(self):
            self.connected = False
        
        async def connect(self, **kwargs):
            self.connected = True
        
        async def close(self):
            self.connected = False
        
        def jetstream(self):
            return MockJetStream()
    
    class MockJetStream:
        async def add_stream(self, **kwargs):
            pass
        
        async def publish(self, subject, payload):
            return MockAck()
        
        async def subscribe(self, subject, cb, durable):
            return MockSubscription()
    
    class MockAck:
        seq = 1
    
    class MockSubscription:
        async def unsubscribe(self):
            pass

logger = logging.getLogger(__name__)

class EventBus:
    """Event-driven communication using NATS JetStream"""
    
    def __init__(self):
        self.nc = NATS()
        self.js = None
        self.subscriptions: Dict[str, list] = {}
        self._connected = False
    
    async def connect(self):
        """Connect to NATS"""
        import os
        
        if not NATS_AVAILABLE:
            logger.warning("NATS not available, using mock event bus")
            self._connected = True
            return
        
        # Get NATS URL from environment
        nats_url = os.getenv("EVENT_BUS_URL", "nats://nats:4222")
        
        try:
            await self.nc.connect(
                servers=[nats_url],
                reconnect_time_wait=2,
                max_reconnect_attempts=60,
                connect_timeout=5
            )
            self.js = self.nc.jetstream()
            
            # Create streams with error handling
            try:
                await self.js.add_stream(
                    name="EVENTS",
                    subjects=["events.>"],
                    retention="limits",
                    max_msgs=100000,
                    max_age=86400  # 1 day
                )
            except Exception as stream_error:
                logger.warning(f"Stream creation failed (may already exist): {stream_error}")
            
            self._connected = True
            logger.info(f"Connected to NATS JetStream at {nats_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to NATS at {nats_url}: {e}")
            # Don't raise - allow graceful degradation
            logger.warning("Falling back to mock event bus")
            self._connected = True
    
    async def disconnect(self):
        """Disconnect from NATS"""
        if self._connected:
            await self.nc.close()
            self._connected = False
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event"""
        if not self._connected:
            raise RuntimeError("EventBus not connected")
        
        if not NATS_AVAILABLE:
            logger.debug(f"Mock publish event {event_type}: {data}")
            return
        
        subject = f"events.{event_type}"
        payload = json.dumps(data).encode()
        
        try:
            ack = await self.js.publish(subject, payload)
            logger.debug(f"Published event {event_type}: {ack.seq}")
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
            # Don't raise - allow graceful degradation
            logger.debug(f"Fallback: Mock publish event {event_type}")
    
    async def subscribe(
        self,
        event_pattern: str,
        handler: Callable,
        connection_id: Optional[str] = None
    ):
        """Subscribe to events"""
        if not self._connected:
            raise RuntimeError("EventBus not connected")
        
        subject = f"events.{event_pattern}"
        
        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                await handler(data)
                await msg.ack()
            except Exception as e:
                logger.error(f"Error handling message: {e}")
        
        sub = await self.js.subscribe(
            subject,
            cb=message_handler,
            durable=f"handler-{event_pattern}-{connection_id or 'default'}"
        )
        
        if event_pattern not in self.subscriptions:
            self.subscriptions[event_pattern] = []
        self.subscriptions[event_pattern].append((connection_id, sub))
    
    async def unsubscribe(self, event_pattern: str, connection_id: Optional[str] = None):
        """Unsubscribe from events"""
        if event_pattern in self.subscriptions:
            for conn_id, sub in self.subscriptions[event_pattern]:
                if conn_id == connection_id:
                    await sub.unsubscribe()
                    self.subscriptions[event_pattern].remove((conn_id, sub))
    
    async def health_check(self) -> Dict[str, Any]:
        """Check event bus health"""
        if not self._connected:
            return {"status": "unhealthy", "reason": "Not connected"}
        
        try:
            info = await self.js.stream_info("EVENTS")
            return {
                "status": "healthy",
                "messages": info.state.messages,
                "consumers": info.state.consumer_count
            }
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}