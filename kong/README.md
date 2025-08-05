# 🌐 Kong Gateway Configuration - API Gateway & Service Mesh

## 📋 **Overview**
The Kong directory contains the complete API gateway configuration for the HydrogenAI platform. Kong serves as the single entry point for all client requests, providing routing, security, rate limiting, and monitoring capabilities.

## 🏗️ **Architecture Pattern**

### **API Gateway Pattern**
- **Single Entry Point**: All external traffic flows through Kong
- **Service Discovery**: Dynamic routing to backend services
- **Cross-Cutting Concerns**: Security, monitoring, rate limiting centralized
- **Protocol Translation**: HTTP/HTTPS to internal service protocols
- **Load Balancing**: Intelligent request distribution

### **Traffic Flow**
```
Client Requests → Kong Gateway → Service Routing → Backend Services
                      ↓
              Rate Limiting, Security, Monitoring
                      ↓
              Request/Response Transformation
```

## 📁 **Directory Structure**

```
kong/
└── kong.yml                    # 🌐 Complete Kong configuration
```

## 🌐 **Kong Configuration (`kong.yml`)**

### **Configuration Overview**
**Purpose**: Complete API gateway setup with routing, security, and monitoring

**Key Features**:
- **Service Discovery**: Automatic backend service routing
- **Rate Limiting**: Per-service and global rate limits
- **Security**: CORS, authentication, and request validation
- **Monitoring**: Prometheus metrics and health checks
- **Load Balancing**: Intelligent request distribution

### **Service Definitions**

#### **1. Main API Service (Orchestrator)**
```yaml
services:
  - name: hydrogen-api
    url: http://orchestrator:8000
    connect_timeout: 60000
    write_timeout: 300000  # 5 minutes for long-running queries
    read_timeout: 300000
    retries: 3
    routes:
      - name: api-query
        paths: [/api/query]
        strip_path: false
        methods: [POST]
      - name: api-status
        paths: [/api/status]
        strip_path: false
        methods: [GET]
      - name: api-health
        paths: [/api/health]
        strip_path: false
        methods: [GET]
```

**Features**:
- **Long Timeouts**: 5-minute timeouts for complex AI queries
- **Retry Logic**: 3 retries for resilience
- **Path Preservation**: Maintains original API paths
- **Method Restrictions**: Specific HTTP methods per route

**Security Plugins**:
```yaml
plugins:
  - name: rate-limiting
    config:
      minute: 60
      hour: 1000
      policy: local
  - name: request-size-limiting
    config:
      allowed_payload_size: 10  # MB
  - name: correlation-id
    config:
      header_name: X-Request-ID
      generator: uuid
      echo_downstream: true
```

#### **2. WebSocket Service**
```yaml
  - name: hydrogen-websocket
    url: http://orchestrator:8000
    routes:
      - name: websocket
        paths: [/ws]
        strip_path: false
        protocols: [ws, wss]
```

**Features**:
- **Real-Time Communication**: WebSocket support for live updates
- **Protocol Support**: Both WS and WSS protocols
- **Direct Routing**: Minimal latency for real-time features

#### **3. Tool Service (MCP Server)**
```yaml
  - name: hydrogen-tools
    url: http://mcp-server:8000
    connect_timeout: 30000
    write_timeout: 60000
    read_timeout: 60000
    routes:
      - name: tool-execute
        paths: [/tools/execute]
        strip_path: true
        methods: [POST]
      - name: tool-list
        paths: [/tools/list]
        strip_path: true
        methods: [GET]
```

**Features**:
- **Tool Execution**: Dedicated routes for tool operations
- **Path Stripping**: Simplified backend routing
- **Shorter Timeouts**: Optimized for tool operations

**Security Configuration**:
```yaml
plugins:
  - name: rate-limiting
    config:
      minute: 100
      policy: local
```

### **Global Plugins**

#### **1. CORS Configuration**
```yaml
plugins:
  - name: cors
    config:
      origins: ["*"]
      methods: [GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH]
      headers:
        - Accept
        - Accept-Version
        - Content-Length
        - Content-MD5
        - Content-Type
        - Date
        - X-Auth-Token
        - Authorization
        - X-Request-ID
      exposed_headers:
        - X-Auth-Token
        - Content-Length
        - Content-Type
        - X-Request-ID
      credentials: true
      max_age: 3600
      preflight_continue: false
```

**Features**:
- **Cross-Origin Support**: Enable web client access
- **Header Management**: Comprehensive header handling
- **Security Headers**: Proper security header exposure
- **Preflight Handling**: Efficient OPTIONS request handling

#### **2. Prometheus Monitoring**
```yaml
  - name: prometheus
    config:
      per_consumer: false
```

**Features**:
- **Metrics Collection**: Automatic metrics generation
- **Performance Monitoring**: Request/response time tracking
- **Error Tracking**: HTTP status code monitoring
- **Throughput Analysis**: Request volume and patterns

#### **3. Response Transformation**
```yaml
  - name: response-transformer
    config:
      add:
        headers:
          - X-Kong-Proxy:true
```

**Features**:
- **Header Injection**: Add Kong identification headers
- **Response Modification**: Transform responses as needed
- **Debugging Support**: Easy identification of Kong-proxied requests

### **Consumer Groups and Authentication**

#### **Consumer Group Configuration**
```yaml
consumer_groups:
  - name: premium-users
    plugins:
      - name: rate-limiting
        config:
          minute: 200
          hour: 5000

consumers:
  - username: default-user
    groups:
      - name: premium-users
```

**Features**:
- **User Classification**: Different user tiers with varying limits
- **Tiered Rate Limiting**: Higher limits for premium users
- **Group Management**: Simplified user group administration
- **Scalable Authentication**: Ready for future auth implementation

### **Advanced Routing Configuration**

#### **Path-Based Routing**
```yaml
routes:
  - name: api-query
    paths: [/api/query]
    strip_path: false
    methods: [POST]
    
  - name: tool-execute
    paths: [/tools/execute]
    strip_path: true
    methods: [POST]
    
  - name: websocket
    paths: [/ws]
    strip_path: false
    protocols: [ws, wss]
```

**Routing Features**:
- **Exact Path Matching**: Precise route matching
- **Method-Specific Routing**: Different routes per HTTP method
- **Protocol-Aware Routing**: HTTP and WebSocket support
- **Path Transformation**: Flexible path handling

#### **Service Discovery Integration**
```yaml
services:
  - name: hydrogen-api
    url: http://orchestrator:8000  # Docker Compose service name
    
  - name: hydrogen-tools
    url: http://mcp-server:8000    # Docker Compose service name
```

**Features**:
- **Docker Integration**: Seamless Docker Compose integration
- **Service Names**: Use Docker service names for discovery
- **Network Isolation**: Internal container network communication
- **Health Check Integration**: Automatic unhealthy service removal

## 🔐 **Security Features**

### **Rate Limiting Strategy**
```yaml
# API service rate limits
rate-limiting:
  minute: 60      # 60 requests per minute
  hour: 1000      # 1000 requests per hour
  policy: local   # Local rate limiting

# Tool service rate limits  
rate-limiting:
  minute: 100     # 100 tool executions per minute
  policy: local
```

**Rate Limiting Benefits**:
- **DoS Protection**: Prevent denial of service attacks
- **Resource Protection**: Protect backend services from overload
- **Fair Usage**: Ensure fair resource distribution
- **Cost Control**: Control API usage costs

### **Request Validation**
```yaml
request-size-limiting:
  allowed_payload_size: 10  # 10MB limit
  
correlation-id:
  header_name: X-Request-ID
  generator: uuid
  echo_downstream: true
```

**Validation Features**:
- **Size Limits**: Prevent oversized requests
- **Request Tracking**: Unique ID for each request
- **Audit Trail**: Complete request traceability
- **Debug Support**: Easy request correlation

### **Authentication Ready**
```yaml
# Ready for future implementation
basic-auth:
  hide_credentials: true
  
# JWT plugin configuration (ready)
jwt:
  uri_param_names: [jwt]
  header_names: [Authorization]
```

**Authentication Features**:
- **Multiple Auth Methods**: Basic auth, JWT, API keys
- **Credential Security**: Hide credentials from backend
- **Flexible Configuration**: Multiple authentication sources
- **Future-Ready**: Easy authentication addition

## ⚡ **Performance Optimization**

### **Connection Management**
```yaml
services:
  - name: hydrogen-api
    connect_timeout: 60000
    write_timeout: 300000
    read_timeout: 300000
    retries: 3
```

**Performance Features**:
- **Optimized Timeouts**: Service-specific timeout configuration
- **Retry Logic**: Automatic retry for failed requests
- **Connection Pooling**: Efficient backend connections
- **Load Balancing**: Intelligent request distribution

### **Caching Strategy**
```yaml
# Future caching plugin configuration
proxy-cache:
  response_code: [200, 301, 404]
  request_method: [GET, HEAD]
  content_type: [text/plain, application/json]
  cache_ttl: 300
  cache_control: false
```

**Caching Benefits**:
- **Response Caching**: Cache frequent responses
- **Reduced Backend Load**: Lower backend service load
- **Improved Latency**: Faster response times
- **Bandwidth Savings**: Reduced network usage

## 📊 **Monitoring and Observability**

### **Prometheus Metrics**
```yaml
prometheus:
  per_consumer: false
```

**Available Metrics**:
- **Request Count**: Total requests per service
- **Response Times**: Request/response latency
- **Error Rates**: HTTP status code distribution
- **Throughput**: Requests per second
- **Backend Health**: Service availability metrics

### **Health Checks**
```yaml
healthcheck:
  test: ["CMD", "kong", "health"]
  interval: 30s
  timeout: 10s
  retries: 10
  start_period: 60s
```

**Health Monitoring**:
- **Gateway Health**: Kong service health
- **Backend Health**: Service dependency health
- **Route Availability**: Individual route health
- **Plugin Status**: Plugin operational status

### **Logging Configuration**
```yaml
# Enhanced logging (via environment variables)
KONG_PROXY_ACCESS_LOG: /dev/stdout
KONG_ADMIN_ACCESS_LOG: /dev/stdout
KONG_PROXY_ERROR_LOG: /dev/stderr
KONG_ADMIN_ERROR_LOG: /dev/stderr
KONG_LOG_LEVEL: warn
```

**Logging Features**:
- **Structured Logging**: JSON-formatted logs
- **Access Logs**: Complete request/response logging
- **Error Logs**: Detailed error information
- **Performance Logs**: Request timing and metrics

## 🔧 **Configuration Management**

### **Environment Variables**
```bash
# Kong configuration
KONG_DATABASE=off
KONG_DECLARATIVE_CONFIG=/kong/kong.yml
KONG_ADMIN_LISTEN=0.0.0.0:8001
KONG_PROXY_LISTEN=0.0.0.0:8000
KONG_LOG_LEVEL=warn

# Network configuration
KONG_PROXY_ACCESS_LOG=/dev/stdout
KONG_ADMIN_ACCESS_LOG=/dev/stdout
KONG_PROXY_ERROR_LOG=/dev/stderr
KONG_ADMIN_ERROR_LOG=/dev/stderr
```

### **Port Configuration**
```yaml
ports:
  - "8080:8000"  # Main proxy port
  - "8443:8443"  # HTTPS proxy port  
  - "8002:8001"  # Admin API port
  - "8444:8444"  # HTTPS admin port
```

**Port Mapping**:
- **8080**: Main HTTP proxy (client access)
- **8443**: HTTPS proxy (secure client access)
- **8002**: Admin API (configuration management)
- **8444**: HTTPS admin API (secure admin access)

## 🚀 **Deployment and Operations**

### **Docker Integration**
```yaml
kong:
  image: kong:3.4
  container_name: hydrogen-kong
  restart: unless-stopped
  environment:
    KONG_DATABASE: "off"
    KONG_DECLARATIVE_CONFIG: /kong/kong.yml
  volumes:
    - ./kong/kong.yml:/kong/kong.yml:ro
  depends_on:
    orchestrator:
      condition: service_healthy
    mcp-server:
      condition: service_healthy
```

**Deployment Features**:
- **Database-Less**: Declarative configuration without database
- **Health Dependencies**: Wait for backend services
- **Configuration Mounting**: External configuration file
- **Restart Policy**: Automatic restart on failure

### **Administration**
```bash
# Check Kong status
curl http://localhost:8002/status

# List configured services
curl http://localhost:8002/services

# List configured routes
curl http://localhost:8002/routes

# View metrics
curl http://localhost:8002/metrics

# Test service routing
curl http://localhost:8080/api/health
```

### **Debugging**
```bash
# Check Kong logs
docker logs hydrogen-kong

# Test specific routes
curl -v http://localhost:8080/api/query -d '{"query":"test"}'

# Check backend connectivity
curl http://localhost:8002/services/hydrogen-api/health

# Monitor real-time traffic
tail -f /var/log/kong/access.log
```

## 🎯 **Best Practices**

### **Configuration Management**
- **Version Control**: Keep kong.yml in version control
- **Environment Separation**: Different configs per environment
- **Validation**: Validate configuration before deployment
- **Backup**: Regular configuration backups

### **Security**
- **Rate Limiting**: Implement appropriate rate limits
- **Authentication**: Add authentication for production
- **HTTPS**: Use HTTPS in production environments
- **IP Filtering**: Implement IP-based access control

### **Performance**
- **Timeout Tuning**: Optimize timeouts per service
- **Caching**: Implement response caching where appropriate
- **Load Balancing**: Use multiple backend instances
- **Monitoring**: Continuous performance monitoring

### **Operations**
- **Health Checks**: Implement comprehensive health checks
- **Logging**: Structured logging for troubleshooting
- **Metrics**: Detailed metrics collection
- **Alerting**: Set up alerts for critical issues

---

## 🚀 **Quick Start Guide**

1. **Configuration Review**: Review kong.yml for your environment
2. **Service Dependencies**: Ensure backend services are running
3. **Kong Startup**: Start Kong with Docker Compose
4. **Health Verification**: Check Kong and route health
5. **Traffic Testing**: Test API routing and functionality

The Kong Gateway configuration provides a robust, secure, and performant API gateway solution that serves as the cornerstone of the HydrogenAI platform's external interface.
