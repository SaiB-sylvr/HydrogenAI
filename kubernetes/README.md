# ☸️ Kubernetes Deployment - Production Container Orchestration

## 📋 **Overview**
The Kubernetes directory contains production-ready deployment configurations for the HydrogenAI platform. It provides a complete container orchestration setup with base configurations and environment-specific overlays using Kustomize.

## 🏗️ **Architecture Pattern**

### **Kustomize Structure**
- **Base**: Common configurations shared across environments
- **Overlays**: Environment-specific customizations (development, production)
- **Patches**: Targeted modifications for specific environments
- **Resources**: Kubernetes resources (deployments, services, ingress)

### **Deployment Strategy**
```
Base Configuration (common resources)
         ↓
Environment Overlays (dev/prod customizations)
         ↓
Patches (specific modifications)
         ↓
Final Kubernetes Manifests
```

## 📁 **Directory Structure**

```
kubernetes/
├── base/                           # Base configurations
│   ├── namespace.yaml             # Namespace definition
│   ├── configmaps.yaml            # Configuration maps
│   └── kustomization.yaml         # Base kustomization
└── overlays/                      # Environment overlays
    ├── development/               # Development environment
    │   ├── ingress-dev.yaml      # Development ingress
    │   └── patches/              # Development patches
    │       └── deployment-patches.yaml
    └── production/               # Production environment
        ├── kustomization.yaml   # Production kustomization
        └── patches/             # Production patches
            └── hpa.yaml         # Horizontal Pod Autoscaler
```

## 🏛️ **Base Configuration (`base/`)**

### **Namespace Definition (`namespace.yaml`)**
**Purpose**: Isolate HydrogenAI resources in dedicated namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hydrogen
  labels:
    name: hydrogen
    environment: shared
    app: hydrogen-ai
```

**Features**:
- **Resource Isolation**: Separate namespace for all HydrogenAI components
- **Access Control**: Enable RBAC policies per namespace
- **Resource Quotas**: Apply resource limits and quotas
- **Network Policies**: Implement network segmentation

### **ConfigMaps (`configmaps.yaml`)**
**Purpose**: Centralized configuration management for all services

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hydrogen-config
  namespace: hydrogen
data:
  # Core service configuration
  LOG_LEVEL: "INFO"
  SERVICE_NAME: "hydrogen-ai"
  
  # Performance settings
  SCHEMA_CACHE_TTL: "86400"
  QUERY_CACHE_TTL: "300"
  RESULT_CACHE_TTL: "3600"
  
  # Service URLs (internal cluster)
  MCP_SERVER_URL: "http://mcp-server-service:8000"
  REDIS_URL: "redis://redis-service:6379"
  EVENT_BUS_URL: "nats://nats-service:4222"
  
  # AI configuration
  GROQ_MODEL_NAME: "llama-3.1-8b-instant"
  GROQ_API_BASE: "https://api.groq.com/openai/v1"
  
  # Vector database
  QDRANT_HOST: "qdrant-service"
  QDRANT_PORT: "6333"
  QDRANT_COLLECTION_NAME: "hydrogen_documents"
  EMBEDDING_MODEL_NAME: "sentence-transformers/all-MiniLM-L6-v2"

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: hydrogen-secrets-config
  namespace: hydrogen
data:
  # Non-sensitive configuration that references secrets
  MONGO_URI_KEY: "mongodb-uri"
  GROQ_API_KEY_KEY: "groq-api-key"
  OPENAI_API_KEY_KEY: "openai-api-key"
```

**Configuration Categories**:
- **Service Discovery**: Internal cluster service URLs
- **Performance Tuning**: Cache TTLs and timeouts
- **AI Integration**: Model names and API endpoints
- **Database Configuration**: Connection parameters
- **Security References**: Secret key references

### **Base Kustomization (`kustomization.yaml`)**
**Purpose**: Define base resources and common configurations

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Namespace for all resources
namespace: hydrogen

# Base resources
resources:
  - namespace.yaml
  - configmaps.yaml

# Common labels applied to all resources
commonLabels:
  app: hydrogen-ai
  version: v3.0.0
  component: ai-orchestration

# Common annotations
commonAnnotations:
  hydrogen.ai/managed-by: kustomize
  hydrogen.ai/version: "3.0.0"

# Image tag replacement
images:
  - name: hydrogen-orchestrator
    newTag: v3.0.0
  - name: hydrogen-mcp-server
    newTag: v3.0.0

# ConfigMap generation
configMapGenerator:
  - name: hydrogen-environment
    literals:
      - ENVIRONMENT=kubernetes
      - CLUSTER_NAME=hydrogen-cluster
      - DEPLOYMENT_TYPE=kustomize
```

## 🚀 **Development Overlay (`overlays/development/`)**

### **Development Ingress (`ingress-dev.yaml`)**
**Purpose**: Development environment ingress configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hydrogen-ingress-dev
  namespace: hydrogen
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/websocket-services: "orchestrator-service"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
spec:
  ingressClassName: nginx
  rules:
  - host: hydrogen-dev.local
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: kong-service
            port:
              number: 8000
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: orchestrator-service
            port:
              number: 8000
      - path: /tools
        pathType: Prefix
        backend:
          service:
            name: mcp-server-service
            port:
              number: 8000
  tls:
  - hosts:
    - hydrogen-dev.local
    secretName: hydrogen-dev-tls
```

**Development Features**:
- **Local Development**: `.local` domain for development
- **CORS Enabled**: Allow cross-origin requests for testing
- **WebSocket Support**: Enable real-time communication
- **Direct Service Access**: Direct access to individual services
- **Flexible Routing**: Multiple path-based routing options

### **Development Patches (`patches/deployment-patches.yaml`)**
**Purpose**: Development-specific deployment modifications

```yaml
# Orchestrator development patches
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestrator-deployment
spec:
  replicas: 1  # Single replica for development
  template:
    spec:
      containers:
      - name: orchestrator
        env:
        - name: LOG_LEVEL
          value: "DEBUG"  # Verbose logging for development
        - name: ENVIRONMENT
          value: "development"
        - name: HOT_RELOAD
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
# MCP Server development patches  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server-deployment
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mcp-server
        env:
        - name: LOG_LEVEL
          value: "DEBUG"
        - name: PLUGIN_RELOAD_INTERVAL
          value: "60"  # Fast plugin reloading
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 🏭 **Production Overlay (`overlays/production/`)**

### **Production Kustomization (`kustomization.yaml`)**
**Purpose**: Production environment configuration with HPA and production patches

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# Base configuration
resources:
  - ../../base

# Production-specific resources
patchesStrategicMerge:
  - patches/hpa.yaml

# Production-specific patches
patches:
  - target:
      kind: Deployment
      name: orchestrator-deployment
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 3
      - op: replace
        path: /spec/template/spec/containers/0/env/0/value
        value: "WARN"  # Reduced logging for production

  - target:
      kind: Deployment
      name: mcp-server-deployment
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 2
      - op: replace
        path: /spec/template/spec/containers/0/resources/requests/memory
        value: "2Gi"
      - op: replace
        path: /spec/template/spec/containers/0/resources/limits/memory
        value: "4Gi"

# Production images
images:
  - name: hydrogen-orchestrator
    newTag: v3.0.0-prod
  - name: hydrogen-mcp-server
    newTag: v3.0.0-prod

# Production labels
commonLabels:
  environment: production
  tier: production

# Production annotations
commonAnnotations:
  hydrogen.ai/environment: production
  hydrogen.ai/monitoring: enabled
  hydrogen.ai/backup: enabled
```

### **Horizontal Pod Autoscaler (`patches/hpa.yaml`)**
**Purpose**: Automatic scaling based on resource utilization

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orchestrator-hpa
  namespace: hydrogen
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: orchestrator-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
  namespace: hydrogen
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server-deployment
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 85
```

**Scaling Features**:
- **CPU-Based Scaling**: Scale based on CPU utilization
- **Memory-Based Scaling**: Scale based on memory usage
- **Stabilization Windows**: Prevent rapid scaling oscillations
- **Service-Specific Scaling**: Different scaling policies per service
- **Resource Protection**: Prevent resource exhaustion

## 🔧 **Deployment Commands**

### **Development Deployment**
```bash
# Deploy to development environment
kubectl apply -k overlays/development/

# Verify deployment
kubectl get pods -n hydrogen
kubectl get services -n hydrogen
kubectl get ingress -n hydrogen

# Check logs
kubectl logs -n hydrogen -l app=hydrogen-ai -f

# Port forwarding for local access
kubectl port-forward -n hydrogen svc/kong-service 8080:8000
```

### **Production Deployment**
```bash
# Deploy to production environment
kubectl apply -k overlays/production/

# Verify deployment and scaling
kubectl get pods -n hydrogen
kubectl get hpa -n hydrogen
kubectl top pods -n hydrogen

# Monitor scaling events
kubectl describe hpa -n hydrogen

# Check service status
kubectl get services -n hydrogen -o wide
```

### **Configuration Management**
```bash
# Update ConfigMaps
kubectl create configmap hydrogen-config --from-env-file=.env -n hydrogen --dry-run=client -o yaml | kubectl apply -f -

# Update secrets
kubectl create secret generic hydrogen-secrets \
  --from-literal=mongodb-uri="$MONGO_URI" \
  --from-literal=groq-api-key="$GROQ_API_KEY" \
  -n hydrogen

# Restart deployments to pick up new config
kubectl rollout restart deployment/orchestrator-deployment -n hydrogen
kubectl rollout restart deployment/mcp-server-deployment -n hydrogen
```

## 📊 **Monitoring and Observability**

### **Health Checks**
```bash
# Check pod health
kubectl get pods -n hydrogen -o wide

# Check service endpoints
kubectl get endpoints -n hydrogen

# View pod logs
kubectl logs -n hydrogen -l app=hydrogen-ai --tail=100

# Check resource usage
kubectl top pods -n hydrogen
kubectl top nodes
```

### **Scaling Monitoring**
```bash
# Monitor HPA status
kubectl get hpa -n hydrogen -w

# View scaling events
kubectl describe hpa orchestrator-hpa -n hydrogen

# Check resource metrics
kubectl top pods -n hydrogen --containers
```

## 🔐 **Security Configuration**

### **RBAC Setup**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: hydrogen
  name: hydrogen-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "update"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: hydrogen-rolebinding
  namespace: hydrogen
subjects:
- kind: ServiceAccount
  name: hydrogen-serviceaccount
  namespace: hydrogen
roleRef:
  kind: Role
  name: hydrogen-role
  apiGroup: rbac.authorization.k8s.io
```

### **Network Policies**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hydrogen-network-policy
  namespace: hydrogen
spec:
  podSelector:
    matchLabels:
      app: hydrogen-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
```

## 🎯 **Best Practices**

### **Resource Management**
- **Resource Requests**: Always specify resource requests
- **Resource Limits**: Set appropriate resource limits
- **Quality of Service**: Use guaranteed QoS for critical services
- **Node Affinity**: Use node affinity for optimal placement

### **Configuration Management**
- **ConfigMaps**: Use ConfigMaps for non-sensitive configuration
- **Secrets**: Use Secrets for sensitive data
- **Environment Variables**: Minimize environment variables
- **Volume Mounts**: Use volume mounts for large configurations

### **Deployment Strategy**
- **Rolling Updates**: Use rolling updates for zero-downtime deployments
- **Health Checks**: Implement proper liveness and readiness probes
- **Graceful Shutdown**: Handle SIGTERM signals properly
- **Resource Monitoring**: Monitor resource usage continuously

---

## 🚀 **Quick Start Guide**

1. **Prerequisites**: Ensure Kubernetes cluster and kubectl access
2. **Namespace Setup**: Apply base configuration for namespace and ConfigMaps
3. **Development Deploy**: Use development overlay for testing
4. **Production Deploy**: Use production overlay with HPA for production
5. **Monitoring**: Set up monitoring and alerting for production use

The Kubernetes deployment system provides a robust, scalable, and production-ready container orchestration solution for the HydrogenAI platform.
