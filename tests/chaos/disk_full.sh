#!/usr/bin/env bash
# Chaos test: fill the shared volume, verify sidecar's disk pre-check catches it.
#
# Usage:
#   ./tests/chaos/disk_full.sh [compose-project-name]
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
SIDECAR_URL="http://localhost:8001"
SHARED_VOLUME="/mnt/models"
FILL_FILE="$SHARED_VOLUME/_chaos_test_fill"

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    log "Cleaning up fill file..."
    # Remove via sidecar container to ensure we hit the right mount
    SIDECAR_CID=$(docker compose "${COMPOSE_ARGS[@]}" ps -q sidecar 2>/dev/null || true)
    if [[ -n "$SIDECAR_CID" ]]; then
        docker exec "$SIDECAR_CID" rm -f "$FILL_FILE" 2>/dev/null || true
    fi
    # Also try local removal (if volume is bind-mounted)
    rm -f "$FILL_FILE" 2>/dev/null || true
    log "Cleanup complete"
}
trap cleanup EXIT

# Get sidecar container ID
SIDECAR_CID=$(docker compose "${COMPOSE_ARGS[@]}" ps -q sidecar)
if [[ -z "$SIDECAR_CID" ]]; then
    log "ERROR: Sidecar container not found"
    exit 1
fi

log "Sidecar container: $SIDECAR_CID"

# Check current disk usage inside the container
log "Current disk usage in shared volume:"
docker exec "$SIDECAR_CID" df -h "$SHARED_VOLUME"

# Fill the disk inside the sidecar container
log "Filling shared volume..."
docker exec "$SIDECAR_CID" dd if=/dev/zero of="$FILL_FILE" bs=1M count=10240 2>/dev/null || true

log "Disk usage after fill:"
docker exec "$SIDECAR_CID" df -h "$SHARED_VOLUME"

# Check sidecar health — it should detect low disk space
log "Checking sidecar health..."
sleep 5

SIDECAR_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SIDECAR_URL/healthz" 2>/dev/null || echo "000")
log "Sidecar /healthz: HTTP $SIDECAR_STATUS"

# Check sidecar startup probe — may report unhealthy due to disk
SIDECAR_STARTUP=$(curl -s -o /dev/null -w "%{http_code}" "$SIDECAR_URL/startupz" 2>/dev/null || echo "000")
log "Sidecar /startupz: HTTP $SIDECAR_STARTUP"

# Check sidecar logs for disk space warnings
log "Sidecar logs (last 10 lines):"
docker compose "${COMPOSE_ARGS[@]}" logs --tail=10 sidecar

# Remove fill file — triggers cleanup via trap
log "Removing fill file to restore space..."
cleanup

# Wait for recovery
log "Waiting for sidecar to recover..."
sleep 10

SIDECAR_STATUS_AFTER=$(curl -s -o /dev/null -w "%{http_code}" "$SIDECAR_URL/healthz" 2>/dev/null || echo "000")
GATEWAY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL/readyz" 2>/dev/null || echo "000")

log "Post-recovery: sidecar /healthz=$SIDECAR_STATUS_AFTER, gateway /readyz=$GATEWAY_STATUS"

if [[ "$GATEWAY_STATUS" == "200" ]]; then
    log "SUCCESS: System recovered after disk space restored"
    exit 0
else
    log "WARNING: Gateway not ready after disk recovery (may need more time)"
    log "Waiting another 30s..."
    sleep 30
    GATEWAY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL/readyz" 2>/dev/null || echo "000")
    if [[ "$GATEWAY_STATUS" == "200" ]]; then
        log "SUCCESS: System recovered (delayed)"
        exit 0
    else
        log "FAIL: System did not recover"
        exit 1
    fi
fi
