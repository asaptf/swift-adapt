#!/usr/bin/env bash
# Rebuild the Metal library fixture used by AdaptTrainTests.
#
# mlx-swift 0.31.x defines SWIFTPM_BUNDLE="mlx-swift_Cmlx" and expects
# default.metallib inside that bundle at runtime. SPM does not produce the
# metallib automatically for Cmlx; this script compiles the vendored
# mlx-generated .metal sources into Tests/AdaptTrainTests/MetalSupport/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift"
METAL_SRC="$CHECKOUT/Source/Cmlx/mlx-generated/metal"
OUT_DIR="$ROOT/Tests/AdaptTrainTests/MetalSupport/mlx-swift_Cmlx.bundle"
TMP="$(mktemp -d)"

if [[ ! -d "$METAL_SRC" ]]; then
  echo "error: $METAL_SRC not found — run 'swift package resolve' first" >&2
  exit 1
fi

airs=()
for f in "$METAL_SRC"/*.metal; do
  base="$(basename "$f" .metal)"
  echo "metal: $base"
  xcrun -sdk macosx metal -c "$f" -I"$METAL_SRC" -o "$TMP/$base.air"
  airs+=("$TMP/$base.air")
done

mkdir -p "$OUT_DIR"
xcrun -sdk macosx metallib "${airs[@]}" -o "$OUT_DIR/default.metallib"
echo "wrote $OUT_DIR/default.metallib ($(du -h "$OUT_DIR/default.metallib" | awk '{print $1}'))"
