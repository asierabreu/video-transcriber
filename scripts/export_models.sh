#!/usr/bin/env bash
# Bundle the models and Ollama volumes into one tarball for air-gapped hosts (§13.4).
set -euo pipefail

OUT="${1:-models-bundle.tar.gz}"
PROJECT="${COMPOSE_PROJECT_NAME:-transcriber}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for vol in models ollama; do
  echo "exporting volume ${PROJECT}_${vol}"
  docker run --rm -v "${PROJECT}_${vol}:/src:ro" -v "$WORK:/out" alpine:3.20 \
    tar -C /src -cf "/out/${vol}.tar" .
done
tar -C "$WORK" -czf "$OUT" models.tar ollama.tar
echo "wrote $OUT"
