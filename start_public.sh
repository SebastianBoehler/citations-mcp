#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[citations-mcp] %s\n' "$*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    printf '[citations-mcp] error: %s command not found in PATH.\n' "${cmd}" >&2
    exit 1
  fi
}

# Configuration
LOCAL_PORT="${MCP_SSE_LOCAL_PORT:-8000}"
NGROK_BIN="${MCP_SSE_NGROK_BIN:-ngrok}"
NGROK_AUTH_TOKEN="${NGROK_AUTHTOKEN:-${MCP_SSE_NGROK_AUTHTOKEN:-}}"

# Required commands
require_cmd uv
require_cmd "${NGROK_BIN}"
require_cmd jq

# Temporary directory for logs
TMP_DIR=$(mktemp -d -t citations-mcp.XXXXXX)
NGROK_LOG="${TMP_DIR}/ngrok.log"

# PIDs for cleanup
server_pid=""
ngrok_pid=""

# Cleanup function
cleanup() {
  if [[ -n "${ngrok_pid}" ]]; then
    kill "${ngrok_pid}" >/dev/null 2>&1 || true
    wait "${ngrok_pid}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi

  rm -rf "${TMP_DIR}"
}

trap cleanup EXIT INT TERM

# Start the MCP SSE server
log "starting Research Citations MCP server on port ${LOCAL_PORT}"
uv run uvicorn src.main:app --host 0.0.0.0 --port "${LOCAL_PORT}" "$@" &
server_pid=$!

# Give the server time to start
sleep 2

# Start ngrok tunnel
log "starting ngrok tunnel"

ngrok_cmd=("${NGROK_BIN}" http --log=stdout --log-format=json "${LOCAL_PORT}")
if [[ -n "${NGROK_AUTH_TOKEN}" ]]; then
  ngrok_cmd+=(--authtoken "${NGROK_AUTH_TOKEN}")
fi

"${ngrok_cmd[@]}" >"${NGROK_LOG}" 2>&1 &
ngrok_pid=$!

# Wait for ngrok to report the public URL
public_url=""
start_time=$(date +%s)
timeout=30

while true; do
  if [[ -f "${NGROK_LOG}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line}" ]] && continue
      url=$(jq -r 'select(.msg == "started tunnel") | .url // empty' <<<"${line}" 2>/dev/null || true)
      if [[ -n "${url}" ]]; then
        public_url="${url}"
        break 2
      fi
    done <"${NGROK_LOG}"
  fi

  now=$(date +%s)
  if (( now - start_time > timeout )); then
    log "error: ngrok tunnel did not start within ${timeout}s"
    exit 1
  fi
  if ! kill -0 "${ngrok_pid}" >/dev/null 2>&1; then
    log "error: ngrok process exited unexpectedly"
    [[ -f "${NGROK_LOG}" ]] && log "ngrok log:" && cat "${NGROK_LOG}" >&2
    exit 1
  fi
  sleep 0.5
done

# Display connection information
log "✅ Server is running!"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Public URL: ${public_url}"
log "MCP SSE Endpoint: ${public_url}/mcp/sse"
log "Health Check: ${public_url}/health"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "Add this to ChatGPT MCP config:"
log "  URL: ${public_url}/mcp/sse"
log ""
log "Press Ctrl+C to stop the server"

# Wait for the server process
wait "${server_pid}"

# Cleanup ngrok if server exits
if [[ -n "${ngrok_pid}" ]]; then
  kill "${ngrok_pid}" >/dev/null 2>&1 || true
  wait "${ngrok_pid}" >/dev/null 2>&1 || true
fi
