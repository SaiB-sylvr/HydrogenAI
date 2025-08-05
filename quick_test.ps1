# HydrogenAI Quick Test Script
# PowerShell script for rapid testing of queries and cache behavior

$BaseUrl = "http://localhost:8000"
$Headers = @{"Content-Type" = "application/json"}

function Test-HydrogenQuery {
    param(
        [string]$Query,
        [string]$TestName,
        [string]$ExpectedCache = ""
    )
    
    Write-Host "`n🧪 Testing: $TestName" -ForegroundColor Cyan
    Write-Host "📝 Query: $Query" -ForegroundColor Gray
    
    $Body = @{
        query = $Query
    } | ConvertTo-Json
    
    $StartTime = Get-Date
    
    try {
        $Response = Invoke-RestMethod -Uri "$BaseUrl/query" -Method POST -Body $Body -Headers $Headers -TimeoutSec 300
        $EndTime = Get-Date
        $ResponseTime = ($EndTime - $StartTime).TotalSeconds
        
        Write-Host "✅ SUCCESS - Response time: $($ResponseTime.ToString("F2"))s" -ForegroundColor Green
        
        # Analyze response for intelligence
        $ResponseText = $Response | ConvertTo-Json -Depth 10
        $IntelligenceScore = 0
        
        # Check for reasoning indicators
        if ($ResponseText -match "because|analysis|based on|considering|pattern|trend") { $IntelligenceScore += 2 }
        if ($ResponseText -match "insight|correlation|relationship|significant|notable") { $IntelligenceScore += 2 }
        if ($ResponseText -match "current|recent|latest|updated|as of") { $IntelligenceScore += 2 }
        if ($ResponseText.Length -gt 500) { $IntelligenceScore += 2 }
        if ($ResponseText -match "metadata|workflow|agents") { $IntelligenceScore += 1 }
        if ($ResponseText -notmatch "error|could not|unable to|failed") { $IntelligenceScore += 1 }
        
        Write-Host "🧠 Intelligence Score: $IntelligenceScore/10" -ForegroundColor Yellow
        
        if ($ResponseText -match "cached|cache") {
            Write-Host "💾 Cache indicator detected" -ForegroundColor Magenta
        }
        
        if ($ExpectedCache) {
            Write-Host "🎯 Expected Cache: $ExpectedCache" -ForegroundColor Blue
        }
        
        return @{
            Success = $true
            ResponseTime = $ResponseTime
            IntelligenceScore = $IntelligenceScore
            Response = $Response
        }
    }
    catch {
        Write-Host "❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
        return @{
            Success = $false
            Error = $_.Exception.Message
        }
    }
}

function Test-CacheBehavior {
    param(
        [string]$Query,
        [string]$TestName
    )
    
    Write-Host "`n🔄 Cache Test: $TestName" -ForegroundColor Yellow
    
    # First call
    Write-Host "📞 First call (should process with AI)..." -ForegroundColor Gray
    $FirstResult = Test-HydrogenQuery -Query $Query -TestName "$TestName - First Call"
    
    Start-Sleep -Seconds 2
    
    # Second call
    Write-Host "📞 Second call (may hit cache)..." -ForegroundColor Gray
    $SecondResult = Test-HydrogenQuery -Query $Query -TestName "$TestName - Second Call"
    
    if ($FirstResult.Success -and $SecondResult.Success) {
        $TimeDiff = $FirstResult.ResponseTime - $SecondResult.ResponseTime
        
        if ($SecondResult.ResponseTime -lt ($FirstResult.ResponseTime * 0.5)) {
            Write-Host "⚡ Potential cache hit detected! $($TimeDiff.ToString("F2"))s faster" -ForegroundColor Green
        } else {
            Write-Host "🔄 Fresh processing detected (similar response times)" -ForegroundColor Blue
        }
    }
}

# Main Test Execution
Write-Host "🚀 HydrogenAI Quick Intelligence & Cache Test Suite" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor White

# Level 1: Simple Queries
Write-Host "`n📊 Level 1: Simple Queries" -ForegroundColor Green
Test-HydrogenQuery -Query "How many collections exist in the database?" -TestName "Collection Count" -ExpectedCache "schema_cache: 24hrs"
Test-HydrogenQuery -Query "What fields are in the users collection?" -TestName "Schema Discovery" -ExpectedCache "schema_cache: 24hrs"
Test-HydrogenQuery -Query "Show me the available collections and their purposes" -TestName "Collection Overview" -ExpectedCache "schema_cache: 24hrs"

# Level 2: Intermediate Queries  
Write-Host "`n📈 Level 2: Intermediate Queries" -ForegroundColor Yellow
Test-HydrogenQuery -Query "Count records by creation date in the last 30 days" -TestName "Temporal Analysis" -ExpectedCache "result_cache: 1hr"
Test-HydrogenQuery -Query "What are the most common values in each field?" -TestName "Data Distribution" -ExpectedCache "result_cache: 1hr"
Test-HydrogenQuery -Query "Find relationships between different collections" -TestName "Relationship Discovery" -ExpectedCache "medium volatility"

# Level 3: Complex Intelligence
Write-Host "`n🧠 Level 3: Complex Intelligence" -ForegroundColor Magenta
Test-HydrogenQuery -Query "Analyze the data quality and structure across all collections" -TestName "Data Quality Analysis" -ExpectedCache "schema_analysis: 24hr"
Test-HydrogenQuery -Query "What insights can you provide about our data patterns and what should we focus on?" -TestName "Business Intelligence" -ExpectedCache "strategic_ai: adaptive"
Test-HydrogenQuery -Query "Based on the current data structure, suggest optimal query patterns for typical operations" -TestName "Query Optimization" -ExpectedCache "meta_analysis: 12hr"

# Cache Behavior Tests
Write-Host "`n💾 Cache Behavior Tests" -ForegroundColor Cyan
Test-CacheBehavior -Query "What collections exist and what data do they contain?" -TestName "Schema Cache Test"

# Real-time vs Static Cache Test
Write-Host "`n⏱️ Volatility-Based Cache Tests" -ForegroundColor Blue
Test-HydrogenQuery -Query "Show me data created in the last 5 minutes" -TestName "Real-time Query" -ExpectedCache "realtime: ~30s TTL"
Test-HydrogenQuery -Query "What are the field types in the users collection?" -TestName "Static Schema Query" -ExpectedCache "static: 10x TTL = 10 days"

Write-Host "`n🎉 Quick Test Suite Complete!" -ForegroundColor Green
Write-Host "💡 For detailed analysis, run: python test_intelligence_suite.py" -ForegroundColor White
