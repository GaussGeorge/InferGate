param(
  [string]$ModelId = $env:MODEL_ID,
  [int]$Port = 8000
)

if (-not $ModelId) {
  Write-Error "Set MODEL_ID or pass -ModelId. This script does not download a model."
  exit 1
}

python -m vllm.entrypoints.openai.api_server `
  --model $ModelId `
  --host 127.0.0.1 `
  --port $Port `
  --enable-prefix-caching `
  --gpu-memory-utilization 0.88 `
  --max-model-len 8192

