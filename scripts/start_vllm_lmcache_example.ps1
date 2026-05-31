param(
  [string]$ModelId = $env:MODEL_ID,
  [int]$Port = 8000
)

if (-not $ModelId) {
  Write-Error "Set MODEL_ID or pass -ModelId. This script does not download a model."
  exit 1
}

$env:VLLM_USE_V1 = "1"
$env:LMCACHE_USE_EXPERIMENTAL = "True"
$env:LMCACHE_CHUNK_SIZE = "256"

python -m vllm.entrypoints.openai.api_server `
  --model $ModelId `
  --host 127.0.0.1 `
  --port $Port `
  --enable-prefix-caching `
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' `
  --gpu-memory-utilization 0.88 `
  --max-model-len 8192

