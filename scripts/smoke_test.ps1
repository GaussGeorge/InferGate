param(
  [int]$Requests = 20,
  [int]$Concurrency = 4,
  [string]$Policy = "infergate_admission",
  [string]$Workload = "mixed_short_long",
  [switch]$UseRealVllm
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$results = Join-Path $repo "results\smoke"
New-Item -ItemType Directory -Force -Path $results | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $results "client_results.jsonl")
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $results "infergate_trace.jsonl")

$env:INFERGATE_POLICY = $Policy
$env:INFERGATE_RESULTS_DIR = $results
$env:VLLM_BASE_URL = if ($env:VLLM_BASE_URL) { $env:VLLM_BASE_URL } else { "http://127.0.0.1:8000" }
$env:VLLM_METRICS_URL = if ($env:VLLM_METRICS_URL) { $env:VLLM_METRICS_URL } else { "http://127.0.0.1:8000/metrics" }
$env:CACHE_BACKEND = if ($env:CACHE_BACKEND) { $env:CACHE_BACKEND } else { "vllm_apc" }

$mockProcess = $null
$gateProcess = $null

function Stop-StartedProcess {
  param(
    [System.Diagnostics.Process]$Process,
    [string]$Name
  )
  if ($null -eq $Process) {
    return
  }
  try {
    $Process.Refresh()
    if (-not $Process.HasExited) {
      Stop-Process -Id $Process.Id -Force -ErrorAction Stop
      $Process.WaitForExit(5000) | Out-Null
    }
  }
  catch {
    Write-Warning "Failed to stop $Name process $($Process.Id): $($_.Exception.Message)"
  }
}

try {
  if (-not $UseRealVllm) {
    $mockProcess = Start-Process -FilePath "python" `
      -ArgumentList "-m","uvicorn","experiments.mock_vllm_server:app","--host","127.0.0.1","--port","8000" `
      -WorkingDirectory $repo -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 2
  }

  $gateProcess = Start-Process -FilePath "python" `
    -ArgumentList "-m","uvicorn","infergate.app:app","--host","127.0.0.1","--port","8080" `
    -WorkingDirectory $repo -PassThru -WindowStyle Hidden
  Start-Sleep -Seconds 2

  $experimentArgs = @(
    "-m", "experiments.run_experiment",
    "--target-url", "http://127.0.0.1:8080/v1/chat/completions",
    "--workload", $Workload,
    "--requests", $Requests,
    "--concurrency", $Concurrency,
    "--policy", $Policy,
    "--output", (Join-Path $results "client_results.jsonl")
  )
  if ($env:MODEL_ID) {
    $experimentArgs += @("--model", $env:MODEL_ID)
  }
  python @experimentArgs
  if ($LASTEXITCODE -ne 0) {
    throw "experiments.run_experiment failed with exit code $LASTEXITCODE"
  }

  Write-Host "Smoke results written to $results"
}
finally {
  Stop-StartedProcess -Process $gateProcess -Name "InferGate"
  Stop-StartedProcess -Process $mockProcess -Name "mock vLLM"
}
