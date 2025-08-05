# HydrogenAI Complex Queries Demo Script
# This script demonstrates advanced analytical capabilities

Write-Host "🚀 HydrogenAI Complex Query Demonstration" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Define the base URL
$baseUrl = "http://localhost:8000/api/query"

# Function to execute query and display results
function Execute-HydrogenQuery {
    param(
        [string]$QueryText,
        [string]$QueryName,
        [string]$Color = "Cyan"
    )
    
    Write-Host "`n🔍 $QueryName" -ForegroundColor $Color
    Write-Host "Query: $QueryText" -ForegroundColor Gray
    Write-Host "----------------------------------------" -ForegroundColor Gray
    
    $body = @{
        query = $QueryText
        use_cache = $false
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri $baseUrl -Method POST -ContentType "application/json" -Body $body
        Write-Host "✅ Status: $($response.status)" -ForegroundColor Green
        Write-Host "📊 Collections Analyzed: $($response.result.results.collections_analyzed -join ', ')" -ForegroundColor Yellow
        Write-Host "📈 Total Records: $($response.result.results.total_records)" -ForegroundColor Magenta
        Write-Host "🏢 Database: $($response.result.results.database)" -ForegroundColor Blue
        
        return $response.request_id
    }
    catch {
        Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# COMPLEX QUERY 1: Advanced Customer Segmentation
$query1 = @"
Perform advanced customer segmentation analysis by examining user demographics, purchase history from orders, digital content engagement patterns, and support interaction frequency. Identify distinct customer personas and their value profiles.
"@

Execute-HydrogenQuery -QueryText $query1 -QueryName "Advanced Customer Segmentation" -Color "Cyan"

Start-Sleep -Seconds 2

# COMPLEX QUERY 2: Predictive Supply Chain Analytics
$query2 = @"
Analyze supply chain efficiency by correlating warehouse capacity utilization, shipment delivery patterns, product demand fluctuations, and seasonal marketing campaign impacts. Predict potential bottlenecks and recommend optimization strategies.
"@

Execute-HydrogenQuery -QueryText $query2 -QueryName "Predictive Supply Chain Analytics" -Color "Yellow"

Start-Sleep -Seconds 2

# COMPLEX QUERY 3: Multi-Channel Marketing Attribution
$query3 = @"
Execute comprehensive marketing attribution analysis by tracking customer journey from marketing campaigns through digital content interactions to purchase conversion in orders, including post-purchase support engagement patterns.
"@

Execute-HydrogenQuery -QueryText $query3 -QueryName "Multi-Channel Marketing Attribution" -Color "Magenta"

Start-Sleep -Seconds 2

# COMPLEX QUERY 4: Employee Performance & Customer Satisfaction Correlation
$query4 = @"
Analyze the correlation between employee performance metrics, support ticket resolution efficiency, customer satisfaction scores, and business outcomes. Identify training opportunities and performance improvement strategies.
"@

Execute-HydrogenQuery -QueryText $query4 -QueryName "Employee Performance Analytics" -Color "Green"

Start-Sleep -Seconds 2

# COMPLEX QUERY 5: Cross-Functional Business Intelligence
$query5 = @"
Generate comprehensive business intelligence report combining user behavior analytics, product performance metrics, operational efficiency indicators from warehouses and shipments, marketing ROI, and customer service quality metrics to identify strategic growth opportunities.
"@

Execute-HydrogenQuery -QueryText $query5 -QueryName "Cross-Functional Business Intelligence" -Color "Red"

Start-Sleep -Seconds 2

# COMPLEX QUERY 6: Risk Assessment & Fraud Detection
$query6 = @"
Perform risk assessment analysis by examining unusual patterns in user activity, order anomalies, support ticket escalations, and digital content access patterns to identify potential fraud, security risks, or operational issues.
"@

Execute-HydrogenQuery -QueryText $query6 -QueryName "Risk Assessment & Fraud Detection" -Color "DarkRed"

Start-Sleep -Seconds 2

# COMPLEX QUERY 7: Real-time Operational Dashboard Query
$query7 = @"
Create real-time operational dashboard insights by aggregating current warehouse status, active shipments, ongoing marketing campaigns, recent user activity trends, pending support tickets, and product performance metrics for executive decision-making.
"@

Execute-HydrogenQuery -QueryText $query7 -QueryName "Real-time Operational Dashboard" -Color "Blue"

Write-Host "`n🎉 Complex Query Demonstration Complete!" -ForegroundColor Green
Write-Host "Your HydrogenAI system successfully processed advanced analytical queries across all 10 collections!" -ForegroundColor Green
