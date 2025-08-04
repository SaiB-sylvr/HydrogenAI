# PowerShell Query Examples for HydrogenAI

## Simple Queries (Copy and paste these into PowerShell)

### 1. Customer Count Query
```powershell
$query = @{
  query = "How many customers do we have?"
  context = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $query -ContentType "application/json"
```

### 2. Product Count Query
```powershell
$query = @{
  query = "How many products are in our catalog?"
  context = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $query -ContentType "application/json"
```

### 3. Order Analysis Query
```powershell
$query = @{
  query = "Show me our order statistics"
  context = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $query -ContentType "application/json"
```

### 4. Business Performance Query
```powershell
$query = @{
  query = "How is our business performing?"
  context = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $query -ContentType "application/json"
```

### 5. Database Overview Query
```powershell
$query = @{
  query = "Give me an overview of our database"
  context = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $query -ContentType "application/json"
```

### 6. Growth Analysis Query
```powershell
$query = @{
  query = "What growth opportunities do you see in our data?"
  context = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $query -ContentType "application/json"
```

## Alternative: Using curl (if you have it installed)
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How many customers do we have?", "context": {}}'
```

## System Health Check
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
```

## API Status Check
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method GET
```

## System Statistics
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/stats" -Method GET
```

## Tips:
1. The system responds with intelligent, contextual answers
2. Each query gets a unique request_id for tracking
3. The system uses AI to understand your intent and provide relevant data
4. Response includes both human-readable answers and structured data
5. The enhanced system now has fallback AI providers, so queries should always work!
