#!/bin/bash

# HydrogenAI Quick Health Check Script
# This script performs quick checks on the system health

echo "=== HydrogenAI System Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check Docker Compose services
echo "🐳 Docker Services Status:"
docker-compose ps 2>/dev/null || echo "Docker Compose not running or not available"
echo ""

# Check running containers
echo "📦 Running Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Docker not available"
echo ""

# Check system resources
echo "💻 System Resources:"
echo "Memory Usage:"
free -h 2>/dev/null || echo "free command not available"
echo ""
echo "Disk Usage:"
df -h . 2>/dev/null || echo "df command not available"
echo ""

# Check network connectivity
echo "🌐 Network Tests:"
echo "Testing internet connectivity..."
curl -s --max-time 5 http://google.com > /dev/null && echo "✅ Internet: OK" || echo "❌ Internet: Failed"

# Check if MongoDB Atlas is reachable
echo "Testing MongoDB Atlas connectivity..."
if command -v mongo &> /dev/null; then
    mongo --eval "print('MongoDB connection test')" --quiet 2>/dev/null && echo "✅ MongoDB: OK" || echo "❌ MongoDB: Failed"
else
    echo "⚠️  MongoDB client not installed"
fi

echo ""
echo "=== Health Check Complete ==="
