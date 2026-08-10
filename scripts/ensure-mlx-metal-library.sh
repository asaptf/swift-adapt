#!/usr/bin/env bash
# Build (or reuse a cached) Metal library for AdaptTrainTests.
#
# Why this exists
# ---------------
# mlx-swift's Cmlx expects a `default.metallib` inside a bundle named
# `mlx-swift_Cmlx` (SWIFTPM_BUNDLE). SwiftPM does not compile/package that
# metallib for the Cmlx target, so under `swift test` MLX cannot load kernels.
#
# We used to vendor the blob under Tests/AdaptTrainTests/MetalSupport/. That is
# unsafe: Package.swift allows mlx-swift to float within 0.31.x, so a committed
# metallib silently drifts from the resolved sources. This script instead builds
# from the *resolved* checkout and caches the result, keyed by git revision.
#
# Cache layout (untracked — lives under .build/):
#   .build/mlx-metallib-cache/<mlx-swift-revision>/default.metallib
#   .build/mlx-metallib-cache/<mlx-swift-revision>/source-revision   # stamp
#   .build/mlx-metallib-cache/CURRENT                                # path to active metallib
#
# Invoked automatically by Tests/AdaptTrainTests/MetalBootstrap.swift on first
# MLX test setup. Safe to run by hand.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift"
METAL_SRC="$CHECKOUT/Source/Cmlx/mlx-generated/metal"
CACHE_ROOT="$ROOT/.build/mlx-metallib-cache"

die() {
  echo "error: $*" >&2
  exit 1
}

# --- Preconditions -----------------------------------------------------------

if [[ ! -d "$CHECKOUT" ]]; then
  die "mlx-swift checkout not found at $CHECKOUT
Run 'swift package resolve' (or 'swift test') first so SwiftPM populates
.build/checkouts/mlx-swift. AdaptTrainTests need that checkout to compile the
Metal kernels that match the resolved MLX version."
fi

if [[ ! -d "$METAL_SRC" ]]; then
  die "Metal sources not found at $METAL_SRC
The resolved mlx-swift checkout is missing Source/Cmlx/mlx-generated/metal.
Check that the mlx-swift pin is intact (Package.swift / Package.resolved)."
fi

# Full Xcode ships the Metal compiler; bare Command Line Tools do not.
if ! xcrun -sdk macosx metal -help >/dev/null 2>&1; then
  die "Apple Metal compiler (xcrun metal) is not available.

AdaptTrainTests must compile Cmlx's .metal sources into default.metallib so MLX
kernels load under 'swift test'. The Metal toolchain is part of full Xcode, not
the Command Line Tools alone.

Fix:
  1. Install Xcode from the App Store (or developer.apple.com)
  2. sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  3. xcodebuild -runFirstLaunch   # accept license / install components if prompted
  4. Re-run: swift test

Refusing to skip MLX tests — a green run that never exercised Metal is worse
than a loud failure here."
fi

# --- Resolve identity of sources --------------------------------------------

if ! REV="$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null)"; then
  die "could not read git revision of $CHECKOUT
SwiftPM checkouts are git clones; without a revision we cannot stamp the cache
and would risk using a stale metallib silently."
fi

VERSION="$(git -C "$CHECKOUT" describe --tags --always 2>/dev/null || echo unknown)"
CACHE_DIR="$CACHE_ROOT/$REV"
METALLIB="$CACHE_DIR/default.metallib"
STAMP="$CACHE_DIR/source-revision"

# --- Cache hit --------------------------------------------------------------

if [[ -f "$METALLIB" && -f "$STAMP" ]] && [[ "$(cat "$STAMP")" == "$REV" ]]; then
  # Refresh CURRENT pointer for consumers that do not know the revision.
  mkdir -p "$CACHE_ROOT"
  printf '%s\n' "$METALLIB" >"$CACHE_ROOT/CURRENT"
  printf '%s\n' "$REV" >"$CACHE_ROOT/CURRENT.revision"
  echo "mlx metallib cache hit (mlx-swift $VERSION / $REV)"
  echo "path: $METALLIB"
  exit 0
fi

# --- Compile ----------------------------------------------------------------

echo "building mlx metallib from mlx-swift $VERSION ($REV) …"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

airs=()
shopt -s nullglob
metal_files=("$METAL_SRC"/*.metal)
if [[ ${#metal_files[@]} -eq 0 ]]; then
  die "no .metal files under $METAL_SRC"
fi

for f in "${metal_files[@]}"; do
  base="$(basename "$f" .metal)"
  echo "  metal: $base"
  if ! xcrun -sdk macosx metal -c "$f" -I"$METAL_SRC" -o "$TMP/$base.air"; then
    die "metal compile failed for $base.metal
If this persists after a clean 'swift package resolve', the resolved mlx-swift
sources may need a different Metal SDK. See Tests/AdaptTrainTests/README.md."
  fi
  airs+=("$TMP/$base.air")
done

mkdir -p "$CACHE_DIR"
if ! xcrun -sdk macosx metallib "${airs[@]}" -o "$METALLIB"; then
  die "metallib link failed while writing $METALLIB"
fi

printf '%s\n' "$REV" >"$STAMP"
printf '%s\n' "$METALLIB" >"$CACHE_ROOT/CURRENT"
printf '%s\n' "$REV" >"$CACHE_ROOT/CURRENT.revision"

size="$(du -h "$METALLIB" | awk '{print $1}')"
echo "wrote $METALLIB ($size) for mlx-swift $VERSION ($REV)"
echo "path: $METALLIB"
