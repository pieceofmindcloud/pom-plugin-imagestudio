# ComfyUI for POM

A POM plugin (`plugin_code: comfyui`) that runs [ComfyUI](https://github.com/comfy-org/ComfyUI)
inside POM, on demand, behind POM's plugin proxy, with nodes that call the
models the POM cluster serves. User guide: [`docs/getting-started.md`](docs/getting-started.md).

## Layout

| Path | What it is |
|---|---|
| `src/abi.rs` | C ABI v1 boundary (`create`, `ingest`, `query`, `free_buffer`, `shutdown`) |
| `src/lib.rs` | entry `pom_comfyui_plugin_v1` and the operation dispatch |
| `src/context.rs` | `host.configure`: gateway, workspace root, host API, preferences |
| `src/installer.rs` | pinned, SHA-256 verified ComfyUI source and `uv`; managed Python venv; PyTorch for the machine's GPU |
| `src/supervisor.rs` | install thread, launch flags, the bridge child and its status |
| `runtime/bridge.py` | per-launch token check in front of ComfyUI's loopback server, HTTP and WebSocket relay, process-group stop |
| `runtime/pom_nodes.py` | the `POM` node category installed into `custom_nodes/pom_nodes` |
| `ui/` | the screen: maps the status to POM's loader, then the full-bleed iframe |
| `release/manifest.json` | release contract: ABI, `feature_set`, preferences (`low_vram`, `cpu_only`) |
| `tests/contract.rs` | static contract checks of both manifests |

## Build and verify

```bash
scripts/build-ui.sh
POM_UI_REQUIRE_DIST=1 cargo test --locked
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
tests/test_ci_plan.sh
# full install + bridge + POM nodes (several GB, network):
POM_COMFYUI_E2E_DIR=/path/with/space cargo test --locked live:: -- --ignored --nocapture
scripts/package.sh --platform linux-x86_64 --version 0.1.0
```

## Updating ComfyUI

Change `VERSION`, `SOURCE_URL` and `SOURCE_SHA` in `src/installer.rs`
(`curl -sL <url> | sha256sum`). An installed node updates on its next opening;
the user's files under the workspace's `.comfyui/` are kept.

## Release

Run the **Publish plugin release** workflow with the platforms and a
`vX.Y.Z` tag per platform. A prerelease stays on GitHub; a stable release is
also uploaded to the license server (`POM_RELEASE_TOKEN`, `POM_RELEASE_API`).
