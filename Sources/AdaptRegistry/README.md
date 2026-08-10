# AdaptRegistry

Versioned on-device store for LoRA adapter weights and metadata, with atomic promote/rollback and integrity verification.

## Contract

- **Actor-isolated.** All mutation and shared state go through `AdapterRegistry`. Callers never touch the filesystem layout directly.
- **Layout** under a configurable root (default Application Support/Adapt):

  ```
  <root>/<lineageID>/v<N>/adapter_config.json   # LoRAConfig (upstream-compatible)
  <root>/<lineageID>/v<N>/adapters.safetensors  # opaque weight bytes
  <root>/<lineageID>/v<N>/version.json          # AdapterVersion metadata
  <root>/<lineageID>/state.json                 # active version pointer
  ```

- **At most one active version per lineage** (zero means base-model behavior). Enforced by the actor; `state.json` is the source of truth.
- **Rollback is O(1)** — only the pointer flips; weight files stay put until GC/archive.
- **Atomic metadata writes** — temp file + replace/rename. A crash mid-promote leaves a consistent registry (old or new active, never a corrupt pointer).
- **Integrity** — `weightsDigest` (SHA-256) is stored in metadata and verified when reading a version.
- **Data Protection** — files use `.completeUntilFirstUserAuthentication` on iOS; no-op on macOS.
- **No user data** — only metadata and weight blobs; never training examples.
- **GC** — `gc(keepLast:)` never deletes the active version.

This module depends only on `AdaptCore` and system frameworks. It does not train models or load MLX.
