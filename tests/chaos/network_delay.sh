#!/usr/bin/env bash
# Chaos test: add 500ms network latency between gateway and engine,
# verify timeouts and error handling work correctly.
#
# Usage:
#   sudo ./tests/chaos/network_delay.sh [compose-project-name]
#
# Requires: iproute2 (tc), root/sudo for network namespace manipulation.
# Prerequisites:
#   docker compose -f docker-compose.yml \
#       -f tests/integration/docker-compose.test.yml up -d

set -euo pipefail

PROJECT="${1:-}"
COMPOSE_ARGS=(-f docker-compose.yml -f tests/integration/docker-compose.test.yml)
if [[ -n "$PROJECT" ]]; then
    COMPOSE_ARGS=(-p "$PROJECT" "${COMPOSE_ARGS[@]}")
fi

GATEWAY_URL="http://localhost:8000"
DELAY_MS=500
HOLD_SECONDS=30

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Get engine container ID
ENGINE_CID=$(docker compose "${COMPOSE_ARGS[@]}" ps -q engine)
if [[ -z "$ENGINE_CID" ]]; then
    log "ERROR: Engine container not found"
    exit 1
fi

# Get engine PID for network namespace
ENGINE_PID=$(docker inspect --format '{{.State.Pid}}' "$ENGINE_CID")
if [[ -z "$ENGINE_PID" || "$ENGINE_PID" == "0" ]]; then
    log "ERROR: Could not get engine container PID"
    exit 1
fi

# Get the veth interface for the engine container
ENGINE_IFACE=$(nsenter -t "$ENGINE_PID" -n ip -o link show | awk -F': ' '/eth0/{print $2}')
if [[ -z "$ENGINE_IFACE" ]]; then
    ENGINE_IFACE="eth0"
fi

log "Engine container: $ENGINE_CID (PID $ENGINE_PID)"
log "Adding ${DELAY_MS}ms delay on interface $ENGINE_IFACE inside engine container"

# Add latency
nsenter -t "$ENGINE_PID" -n tc qdisc add dev "$ENGINE_IFACE" root netem delay "${DELAY_MS}ms" 50ms distribution normal 2>/dev/null || \
    nsenter -t "$ENGINE_PID" -n tc qdisc change dev "$ENGINE_IFACE" root netem delay "${DELAY_MS}ms" 50ms distribution normal

log "Delay active. Sending test requests for ${HOLD_SECONDS}s..."

# Send a few test requests during the delay period
ERRORS=0
TOTAL=0
DEADLINE=$((SECONDS + HOLD_SECONDS))

while [[ $SECONDS -lt $DEADLINE ]]; do
    TOTAL=$((TOTAL + 1))
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$GATEWAY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"Qwen/Qwen2-0.5B","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
        --max-time 30 2>/dev/null || echo "000")

    if [[ "$HTTP_CODE" != "200" ]]; then
        ERRORS=$((ERRORS + 1))
        log "Request $TOTAL: HTTP $HTTP_CODE"
    else
        log "Request $TOTAL: HTTP 200 (OK with delay)"
    fi
    sleep 3
done

# Remove latency
log "Removing network delay..."
nsenter -t "$ENGINE_PID" -n tc qdisc del dev "$ENGINE_IFACE" root 2>/dev/null || true

log "Delay removed. Verifying recovery..."
sleep 5

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL/readyz" 2>/dev/null || echo "000")

log "Results: $TOTAL requests sent, $ERRORS errors"
log "Post-recovery /readyz: HTTP $STATUS"

if [[ "$STATUS" == "200" ]]; then
    log "SUCCESS: System handled network delay and recovered"
    exit 0
else
    log "FAIL: System did not recover after delay removal (readyz=$STATUS)"
    exit 1
fi
