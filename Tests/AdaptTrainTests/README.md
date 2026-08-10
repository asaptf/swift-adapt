# AdaptTrainTests

Offline, model-free suite for `AdaptTrain`. Suites use a tiny synthetic `Module`
and a stub `Tokenizer` — no network, no real weights.

## Metal library (`default.metallib`)

### The problem

MLX’s Cmlx layer loads GPU kernels from a Metal library named `default.metallib`,
expected inside a bundle named `mlx-swift_Cmlx` (`SWIFTPM_BUNDLE`). SwiftPM does
**not** compile or package that metallib for the Cmlx product, so a plain
`swift test` run cannot load MLX kernels.

### What we do **not** do

We do **not** commit a prebuilt `default.metallib`.

`Package.swift` pins mlx-swift as `.upToNextMinor(from: "0.31.4")`, so SwiftPM
may resolve any 0.31.x. A vendored metallib freezes at the revision that built
it. Tests would then run against kernels that no longer match the resolved
sources — a silent mismatch worse than a missing-file error.

### Mechanism

1. **Generate on demand.** The first AdaptTrain test that needs MLX calls
   `TestSupport.prepareMLX()` → `MetalBootstrap.ensureMetallib()`.
2. **Build script.** `MetalBootstrap` runs
   [`scripts/ensure-mlx-metal-library.sh`](../../scripts/ensure-mlx-metal-library.sh), which
   compiles every `.metal` file under the resolved checkout:
   `.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal/`.
3. **Revision-keyed cache** (untracked, under `.build/`):
   ```
   .build/mlx-metallib-cache/<mlx-swift-git-rev>/default.metallib
   .build/mlx-metallib-cache/<mlx-swift-git-rev>/source-revision
   .build/mlx-metallib-cache/CURRENT            # absolute path of active metallib
   .build/mlx-metallib-cache/CURRENT.revision
   ```
   If `source-revision` matches the checkout’s `HEAD`, the script is a no-op.
   A different resolved mlx-swift revision misses the cache and rebuilds.
4. **Install for Cmlx.** `MetalBootstrap` copies the cached metallib next to the
   test executable (`default.metallib`, `mlx.metallib`, and under
   `Resources/mlx-swift_Cmlx.bundle/`) so upstream load paths succeed.

No SPM `resources:` entry is required; nothing under `Tests/` is generated into
the tree.

### From a clean checkout

```bash
swift test   # resolve → build → bootstrap compiles metallib → tests run
```

No manual step. Regenerating after deleting the cache:

```bash
rm -rf .build/mlx-metallib-cache
swift test   # rebuilds metallib, then passes
```

### When generation fails

The script **exits non-zero** with a concrete fix. Common cases:

| Symptom | Cause | Fix |
|---|---|---|
| `xcrun metal` unavailable | Only Command Line Tools installed | Install full Xcode; `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` |
| Checkout missing | `.build/checkouts/mlx-swift` not populated | `swift package resolve` or just `swift test` from repo root |
| Metal sources missing | Unexpected mlx-swift layout | Check pin / re-resolve |

We deliberately **do not** skip MLX tests when generation fails. A green run
that never exercised Metal is the failure mode this design exists to prevent.

### Manual rebuild

```bash
./scripts/ensure-mlx-metal-library.sh
```

Useful when debugging the Metal toolchain without running the full suite.
