#!/usr/bin/env bash
# Generate small sample videos (tone + test pattern) for manual API testing.
# The automated tests generate their own media; this is for curl experiments.
set -euo pipefail

OUT="${1:-tests/fixtures/generated}"
mkdir -p "$OUT"
for ext in mp4 mkv mov avi; do
  acodec=aac
  [ "$ext" = avi ] && acodec=pcm_s16le
  ffmpeg -nostdin -loglevel error -y \
    -f lavfi -i "testsrc=size=320x240:rate=10:duration=10" \
    -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=10" \
    -c:v mpeg4 -c:a "$acodec" -shortest "$OUT/sample.$ext"
done
ffmpeg -nostdin -loglevel error -y -f lavfi -i "testsrc=size=320x240:rate=10:duration=5" \
  -c:v mpeg4 "$OUT/no_audio.mp4"
head -c 4096 /dev/urandom > "$OUT/corrupt.mp4"
echo "fixtures written to $OUT"
