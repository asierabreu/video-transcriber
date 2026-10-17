#!/usr/bin/env bash
# Restore a bundle written by export_models.sh into this host's Docker volumes.
set -euo pipefail

IN="${1:-models-bundle.tar.gz}"
PROJECT="${COMPOSE_PROJECT_NAME:-transcriber}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

tar -C "$WORK" -xzf "$IN"
for vol in models ollama; do
  echo "importing volume ${PROJECT}_${vol}"
  docker volume create "${PROJECT}_${vol}" >/dev/null
  docker run --rm -v "${PROJECT}_${vol}:/dst" -v "$WORK:/in:ro" alpine:3.20 \
    sh -c "tar -C /dst -xf /in/${vol}.tar && chown -R 10001:10001 /dst 2>/dev/null || true"
done
echo "imported $IN"
