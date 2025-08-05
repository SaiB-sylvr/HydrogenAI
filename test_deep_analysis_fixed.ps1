#!/usr/bin/env pwsh

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "🚀 TESTING ENHANCED DEEP USER ANALYSIS" -ForegroundColor Cyan  
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Deep User Behavioral Analysis Query
Write-Host "📊 Test 1: Deep User Behavioral Analysis" -ForegroundColor Yellow
Write-Host ""

$testQuery1 = @{
    "query" = "Analyze user behavior patterns and provide detailed insights about individual users, their engagement levels, purchase patterns, and strategic recommendations for each user segment"
    "context" = @{
        "analysis_type" = "deep_user_behavioral"
        "include_individual_profiles" = $true
        "require_strategic_insights" = $true
    }
} | ConvertTo-Json -Depth 10

try {
    $response1 = Invoke-RestMethod -Uri "http://localhost:8001/query" -Method POST -Body $testQuery1 -ContentType "application/json" -TimeoutSec 60
    
    if ($response1) {
        Write-Host "✅ ENHANCED DEEP ANALYSIS SUCCESS!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Response Length: $($response1.response.Length) characters" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Testing for enhanced intelligence markers..." -ForegroundColor White
        
        $content = $response1.response
        $intelligenceMarkers = @(
            "engagement_metrics",
            "behavioral_analysis", 
            "customer_journey",
            "technology_profile",
            "geographic_profile",
            "temporal_patterns",
            "business_value",
            "individual user",
            "comprehensive analysis",
            "multi-dimensional user segmentation"
        )
        
        $foundMarkers = 0
        foreach ($marker in $intelligenceMarkers) {
            if ($content -match $marker) {
                Write-Host "  ✓ Found: $marker" -ForegroundColor Green
                $foundMarkers++
            } else {
                Write-Host "  ✗ Missing: $marker" -ForegroundColor Red
            }
        }
        
        Write-Host ""
        Write-Host "Intelligence Score: $foundMarkers/$($intelligenceMarkers.Count)" -ForegroundColor Cyan
        
        if ($foundMarkers -gt 5) {
            Write-Host "🎯 ENHANCED ANALYSIS CONFIRMED!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Basic analysis detected - enhancement needed" -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "Sample Response Preview:" -ForegroundColor White
        Write-Host "------------------------" -ForegroundColor Gray
        Write-Host $content.Substring(0, [Math]::Min(800, $content.Length)) -ForegroundColor White
        if ($content.Length -gt 800) {
            Write-Host "... [truncated]" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "❌ Test 1 Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan

# Test 2: Individual User Deep Dive
Write-Host "👤 Test 2: Individual User Deep Profile Analysis" -ForegroundColor Yellow
Write-Host ""

$testQuery2 = @{
    "query" = "Provide detailed analysis of individual users including their complete behavioral profile, device usage patterns, geographic insights, and personalized recommendations"
    "context" = @{
        "analysis_depth" = "maximum"
        "include_user_profiles" = $true
        "include_recommendations" = $true
    }
} | ConvertTo-Json -Depth 10

try {
    $response2 = Invoke-RestMethod -Uri "http://localhost:8001/query" -Method POST -Body $testQuery2 -ContentType "application/json" -TimeoutSec 60
    
    if ($response2) {
        Write-Host "✅ INDIVIDUAL USER ANALYSIS SUCCESS!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Response Length: $($response2.response.Length) characters" -ForegroundColor Cyan
        
        $content2 = $response2.response
        $userProfileMarkers = @(
            "engagement_level",
            "device_diversity",
            "location_patterns", 
            "revenue_generated",
            "customer_lifetime",
            "behavioral_segment",
            "recommendations",
            "strategic insights"
        )
        
        $foundProfileMarkers = 0
        Write-Host ""
        Write-Host "User Profile Analysis Markers:" -ForegroundColor White
        foreach ($marker in $userProfileMarkers) {
            if ($content2 -match $marker) {
                Write-Host "  ✓ Found: $marker" -ForegroundColor Green
                $foundProfileMarkers++
            } else {
                Write-Host "  ✗ Missing: $marker" -ForegroundColor Red
            }
        }
        
        Write-Host ""
        Write-Host "User Profile Score: $foundProfileMarkers/$($userProfileMarkers.Count)" -ForegroundColor Cyan
        
        if ($foundProfileMarkers -gt 4) {
            Write-Host "🎯 DEEP USER PROFILING CONFIRMED!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Surface-level analysis detected" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Test 2 Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "🎉 ENHANCED ANALYSIS TESTING COMPLETE" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "System Status Check:" -ForegroundColor White
try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method GET -TimeoutSec 10
    Write-Host "✅ Orchestrator Health: OK" -ForegroundColor Green
} catch {
    Write-Host "❌ Orchestrator Health: FAILED" -ForegroundColor Red
}
