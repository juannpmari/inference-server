#!/usr/bin/env bash
# Chaos test: randomly kill one of the 3 containers, verify system recovers.
#
# Usage:
#   ./tests/chaos/kill_random.sh [compose-project-name]
#
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
SERVICES=("gateway" "engine" "sidecar")
RECOVERY_TIMEOUT=120  # seconds

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Pick a random service
TARGET="${SERVICES[$((RANDOM % ${#SERVICES[@]}))]}"
log "Selected target: $TARGET"

# Verify baseline
log "Checking baseline health..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL/healthz" || echo "000")
if [[ "$STATUS" != "200" ]]; then
    log "ERROR: Gateway not healthy before test (status=$STATUS). Aborting."
    exit 1
fi
log "Baseline OK"

# Kill the target
log "Killing $TARGET..."
docker compose "${COMPOSE_ARGS[@]}" kill "$TARGET"
log "$TARGET killed"

# Wait briefly
sleep 5

# Restart the target
log "Restarting $TARGET..."
docker compose "${COMPOSE_ARGS[@]}" start "$TARGET"

# Wait for recovery
log "Waiting up to ${RECOVERY_TIMEOUT}s for recovery..."
DEADLINE=$((SECONDS + RECOVERY_TIMEOUT))
RECOVERED=false

while [[ $SECONDS -lt $DEADLINE ]]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL/readyz" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        RECOVERED=true
        break
    fi
    sleep 2
done

if $RECOVERED; then
    ELAPSED=$((SECONDS))
    log "SUCCESS: System recovered after killing $TARGET (${ELAPSED}s elapsed)"
    exit 0
else
    log "FAIL: System did not recover within ${RECOVERY_TIMEOUT}s after killing $TARGET"
    docker compose "${COMPOSE_ARGS[@]}" ps
    docker compose "${COMPOSE_ARGS[@]}" logs --tail=20 "$TARGET"
    exit 1
fi
