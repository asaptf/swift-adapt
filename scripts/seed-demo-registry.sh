#!/usr/bin/env bash
# seed-demo-registry.sh — seven real overnight training runs for the demo registry.
#
# Each night resumes from the previous night's adapter (separate process) and
# trains on that night's *new* examples. After each night, held-out mean
# cross-entropy is measured and recorded on the version's EvalReport.
#
# Does NOT implement the M3 promotion gate — measurement only.
#
# The registry is derived, not vendored (~200 MB of weights). Path is gitignored.
#
# Usage:
#   bash scripts/seed-demo-registry.sh
#   DEMO_REGISTRY=/tmp/my-reg STEPS_PER_NIGHT=40 bash scripts/seed-demo-registry.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REG="${DEMO_REGISTRY:-$ROOT/.build/demo-registry}"
SLICES="${DEMO_SLICES:-$ROOT/.build/demo-slices}"
CORPUS="$ROOT/Tools/adapt-cli/Fixtures/nix-caldera-seven-nights.jsonl"
MODEL="${DEMO_MODEL:-mlx-community/Qwen3-4B-4bit}"
TASK="${DEMO_TASK:-style-mirror}"
# Modest per-night steps. 300 total on 50 examples collapsed loss; keep nights short.
STEPS="${STEPS_PER_NIGHT:-40}"
RANK="${DEMO_RANK:-8}"
NUM_LAYERS="${DEMO_NUM_LAYERS:-8}"
LR="${DEMO_LR:-1e-4}"
SEED="${DEMO_SEED:-42}"
NIGHTS=7
HELD_OUT=30
# One version per night: checkpoint only at the end of the night's steps.
CHECKPOINT_EVERY="$STEPS"

if [[ ! -f "$CORPUS" ]]; then
  echo "error: missing corpus $CORPUS" >&2
  exit 1
fi

echo "=== seed-demo-registry ==="
echo "  registry:  $REG"
echo "  model:     $MODEL"
echo "  steps/night: $STEPS  rank=$RANK  layers=$NUM_LAYERS  lr=$LR"
echo "  nights:    $NIGHTS  held-out: $HELD_OUT"
echo

echo "Building adapt-cli (release)…"
swift build -c release --product adapt-cli
CLI=(swift run -c release adapt-cli)

rm -rf "$REG" "$SLICES"
mkdir -p "$REG" "$SLICES"

echo
echo "Exporting night + held-out slices…"
"${CLI[@]}" export-demo-nights \
  --data "$CORPUS" \
  --out-dir "$SLICES" \
  --nights "$NIGHTS" \
  --held-out "$HELD_OUT"

COMMON_TRAIN=(
  --model "$MODEL"
  --task "$TASK"
  --rank "$RANK"
  --num-layers "$NUM_LAYERS"
  --keys attention
  --batch-size 1
  --learning-rate "$LR"
  --seed "$SEED"
  --steps "$STEPS"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --registry "$REG"
  --promote
)

COMMON_MEASURE=(
  --model "$MODEL"
  --task "$TASK"
  --rank "$RANK"
  --num-layers "$NUM_LAYERS"
  --keys attention
  --registry "$REG"
  --data "$SLICES/held-out.jsonl"
  --record
)

# shellcheck disable=SC2034
declare -a NIGHT_SCORES=()
declare -a NIGHT_VERSIONS=()

for night in $(seq 1 "$NIGHTS"); do
  echo
  echo "======== Night $night / $NIGHTS ========"
  night_data="$SLICES/night-${night}.jsonl"
  if [[ ! -f "$night_data" ]]; then
    echo "error: missing $night_data" >&2
    exit 1
  fi

  echo "Training on $(basename "$night_data") (resume if prior night exists)…"
  "${CLI[@]}" train \
    --data "$night_data" \
    "${COMMON_TRAIN[@]}"

  # Latest version number = night index when one version is written per night.
  version="$night"
  echo "Measuring held-out loss for v${version}…"
  measure_out="$("${CLI[@]}" measure --version "$version" "${COMMON_MEASURE[@]}" 2>&1)" || {
    echo "$measure_out" >&2
    echo "error: measure failed for v${version}" >&2
    exit 1
  }
  echo "$measure_out"

  score="$(printf '%s\n' "$measure_out" | sed -n 's/.*mean_cross_entropy_nats=\([0-9.]*\).*/\1/p' | head -1)"
  if [[ -z "$score" ]]; then
    echo "error: could not parse mean_cross_entropy_nats from measure output" >&2
    exit 1
  fi
  NIGHT_SCORES+=("$score")
  NIGHT_VERSIONS+=("$version")
done

echo
echo "======== Seven-night summary (held-out mean CE, nats/token; lower is better) ========"
printf '%-8s  %-10s  %s\n' "night" "version" "held_out_ce_nats"
prev=""
for i in $(seq 0 $((NIGHTS - 1))); do
  night=$((i + 1))
  ver="${NIGHT_VERSIONS[$i]}"
  score="${NIGHT_SCORES[$i]}"
  delta=""
  if [[ -n "$prev" ]]; then
    # awk for floating delta (bash cannot)
    d="$(awk -v a="$score" -v b="$prev" 'BEGIN { printf "%+.4f", a - b }')"
    delta="  (Δ $d vs previous night)"
  fi
  printf '%-8s  v%-9s  %s%s\n' "$night" "$ver" "$score" "$delta"
  prev="$score"
done

echo
echo "Registry left at: $REG"
echo "Inspect with:"
echo "  swift run -c release adapt-cli inspect --registry \"$REG\""
echo
echo "Note: these numbers are measurements, not a promotion gate. Do not retune"
echo "the recipe to force a rising curve — report the shape as measured."
