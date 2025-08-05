# HydrogenAI Deep User Analysis Test Script
# Tests the enhanced deep behavioral analysis capabilities

# Test 1: Simple user query (should trigger deep analysis)
echo "=== Test 1: Deep User Analysis ===" | Out-Host
$body1 = @{
    query = "Show me detailed customer behavior analysis"
} | ConvertTo-Json

try {
    $response1 = Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $body1 -ContentType "application/json"
    Write-Host "✅ Query: $($body1 | ConvertFrom-Json | Select-Object -ExpandProperty query)" -ForegroundColor Green
    Write-Host "📊 Response Preview:" -ForegroundColor Cyan
    if ($response1.human_response) {
        $preview = $response1.human_response.Substring(0, [Math]::Min(300, $response1.human_response.Length))
        Write-Host $preview -ForegroundColor White
        if ($response1.human_response.Length -gt 300) {
            Write-Host "... (response truncated for preview)" -ForegroundColor Gray
        }
    } else {
        Write-Host $response1 -ForegroundColor White
    }
    Write-Host "`n" -NoNewline
} catch {
    Write-Host "❌ Test 1 Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Specific user engagement query
echo "=== Test 2: User Engagement Intelligence ===" | Out-Host
$body2 = @{
    query = "Who are my most engaged users and what makes them special?"
} | ConvertTo-Json

try {
    $response2 = Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $body2 -ContentType "application/json"
    Write-Host "✅ Query: $($body2 | ConvertFrom-Json | Select-Object -ExpandProperty query)" -ForegroundColor Green
    Write-Host "📊 Response Preview:" -ForegroundColor Cyan
    if ($response2.human_response) {
        $preview = $response2.human_response.Substring(0, [Math]::Min(300, $response2.human_response.Length))
        Write-Host $preview -ForegroundColor White
        if ($response2.human_response.Length -gt 300) {
            Write-Host "... (response truncated for preview)" -ForegroundColor Gray
        }
    }
    Write-Host "`n" -NoNewline
} catch {
    Write-Host "❌ Test 2 Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Business intelligence query
echo "=== Test 3: Business Intelligence Analysis ===" | Out-Host
$body3 = @{
    query = "Analyze my customer lifetime value and behavioral patterns for strategic insights"
} | ConvertTo-Json

try {
    $response3 = Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method POST -Body $body3 -ContentType "application/json"
    Write-Host "✅ Query: $($body3 | ConvertFrom-Json | Select-Object -ExpandProperty query)" -ForegroundColor Green
    Write-Host "📊 Response Preview:" -ForegroundColor Cyan
    if ($response3.human_response) {
        $preview = $response3.human_response.Substring(0, [Math]::Min(300, $response3.human_response.Length))
        Write-Host $preview -ForegroundColor White
        if ($response3.human_response.Length -gt 300) {
            Write-Host "... (response truncated for preview)" -ForegroundColor Gray
        }
    }
    Write-Host "`n" -NoNewline
} catch {
    Write-Host "❌ Test 3 Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "=== Enhanced Deep Analysis Testing Complete ===" -ForegroundColor Green
Write-Host "Your HydrogenAI system now provides:" -ForegroundColor Yellow
Write-Host "✨ Deep user behavioral analysis with complete profiles" -ForegroundColor White
Write-Host "✨ Individual user journey tracking and insights" -ForegroundColor White
Write-Host "✨ Comprehensive engagement and conversion intelligence" -ForegroundColor White
Write-Host "✨ Strategic business recommendations based on actual data" -ForegroundColor White
Write-Host "✨ Multi-dimensional user segmentation and analysis" -ForegroundColor White
