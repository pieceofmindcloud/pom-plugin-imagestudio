//! ComfyUI inside POM: the node-based image, video and audio studio, installed
//! from a pinned, verified release and served through POM's plugin proxy.
//!
//! `pom-plugin-host` loads this library through the C ABI in `abi.rs` and
//! sends every node request to `Plugin::query` as `{"operation": ..}`
//! (contract: POM `docs/subsystems/plugins-ui.md`). `query` answers at once:
//! the install and the ComfyUI server belong to `supervisor.rs`'s threads.

mod abi;
mod components;
mod context;
mod installer;
mod supervisor;

use context::Context;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use supervisor::Supervisor;

pub const PLUGIN_CODE: &str = "imagestudio";
const UI_MANIFEST: &str = include_str!("../ui/manifest.json");

include!(concat!(env!("OUT_DIR"), "/ui_assets.rs"));

/// The symbol `pom-plugin-host` resolves from the manifest's `plugin_code`.
///
/// # Safety
///
/// Returns a pointer to a static; the host only reads it.
#[no_mangle]
pub unsafe extern "C" fn pom_imagestudio_plugin_v1() -> *const abi::PluginApiV1 {
    &abi::API
}

pub struct Plugin {
    supervisor: Arc<Supervisor>,
}

impl Drop for Plugin {
    fn drop(&mut self) {
        self.supervisor.stop();
    }
}

impl Plugin {
    /// `config` is the host's creation JSON; it may be null.
    pub fn new(_config: &Value) -> Result<Self, String> {
        Ok(Self {
            supervisor: Supervisor::new(),
        })
    }

    /// The node only ingests into the `logs` plugin today; accept and ignore.
    pub fn ingest(&self, event: &Value) -> Result<(), String> {
        event["type"].as_str().ok_or("event has no type")?;
        Ok(())
    }

    pub fn query(&self, request: &Value) -> Result<Value, String> {
        match request["operation"].as_str().unwrap_or_default() {
            "host.configure" => {
                self.supervisor.configure(Context::from_configure(request)?);
                Ok(json!({"status": "ok"}))
            }
            "host.event" => Ok(json!({"status": "ok"})),
            "host.services" => {
                let status = self.supervisor.status();
                let mut services = vec![json!({
                    "id": "comfyui", "kind": "app", "name": "ComfyUI",
                    "status": status.phase, "version": installer::VERSION,
                    "detail": status.detail,
                })];
                // The image service tells POM's deploy screen this machine
                // runs ComfyUI, and lists what it serves to the network.
                if let Ok(served) = self.supervisor.served() {
                    let phase = if self.supervisor.installed() {
                        status.phase.as_str()
                    } else {
                        "waiting"
                    };
                    services.push(served.service(phase, ""));
                }
                Ok(json!({ "services": services }))
            }
            "component.serve" => {
                let served = self.supervisor.served()?;
                match served.serve(request, self.supervisor.installed()) {
                    Ok(()) => {
                        // Answer within the query budget; ComfyUI starts behind.
                        self.supervisor.launch(false);
                        Ok(json!({"status": "accepted"}))
                    }
                    Err(error) => Ok(json!({"status": "refused", "error": error})),
                }
            }
            "component.release" => {
                let deploy_id = request["deploy_id"]
                    .as_str()
                    .ok_or("deploy_id is missing")?;
                let component = request["component"].as_str();
                self.supervisor.served()?.release(deploy_id, component);
                Ok(json!({"status": "accepted"}))
            }
            "component.upstream" => {
                let deploy_id = request["deploy_id"].as_str().unwrap_or_default();
                let component = request["component"].as_str().unwrap_or_default();
                if !self.supervisor.served()?.serves(deploy_id, component) {
                    return Err(format!("{component} of {deploy_id} is not served here"));
                }
                Ok(self.supervisor.status().upstream())
            }
            "ui.manifest" => ui_manifest(),
            "ui.asset" => {
                let path = request["path"].as_str().ok_or("asset path is missing")?;
                let (_, content_type, bytes) = ui_asset(path).ok_or("unknown asset")?;
                Ok(json!({"content_type": content_type, "base64": encode_base64(bytes)}))
            }
            "ui.upstream" => Ok(self.supervisor.status().upstream()),
            "rpc.status" => {
                let mut status = self.supervisor.status().public();
                status["version"] = json!(installer::VERSION);
                status["context"] = self.supervisor.context_public();
                Ok(status)
            }
            // Plugins are local to their node: any profile may install and
            // open the application on its own machine.
            "rpc.install" => Ok(json!({
                "status": if self.supervisor.launch(false) { "starting" } else { "busy" }
            })),
            _ => Err("unknown operation".into()),
        }
    }
}

/// `<directory of POM_PLUGIN_DB>/<plugin_code>`: fixed files of this plugin
/// (downloaded runtimes, logs, databases). Dynamic user data goes to the
/// workspace instead (`Context::workspace_state_dir`).
pub fn plugin_directory() -> Result<PathBuf, String> {
    let parent = std::env::var_os("POM_PLUGIN_DB")
        .map(PathBuf::from)
        .and_then(|database| database.parent().map(Path::to_path_buf))
        .filter(|parent| !parent.as_os_str().is_empty());
    let parent = match parent {
        Some(parent) => parent,
        None => std::env::current_dir().map_err(|error| error.to_string())?,
    };
    Ok(parent.join(PLUGIN_CODE))
}

fn ui_asset(path: &str) -> Option<&'static (&'static str, &'static str, &'static [u8])> {
    UI_ASSETS.iter().find(|(name, _, _)| *name == path)
}

fn ui_manifest() -> Result<Value, String> {
    let manifest: Value = serde_json::from_str(UI_MANIFEST).map_err(|error| error.to_string())?;
    let assets = manifest["assets"]
        .as_array()
        .ok_or("manifest has no assets")?;
    for asset in assets.iter().filter_map(Value::as_str) {
        if ui_asset(asset).is_none() {
            return Err(format!("asset is not embedded: {asset}"));
        }
    }
    Ok(manifest)
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let value = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for index in 0..4 {
            if index <= chunk.len() {
                output.push(TABLE[((value >> (18 - 6 * index)) & 63) as usize] as char);
            } else {
                output.push('=');
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin() -> Plugin {
        Plugin::new(&Value::Null).unwrap()
    }

    #[test]
    fn entry_point_is_named_after_the_plugin_code() {
        let api = unsafe { &*pom_imagestudio_plugin_v1() };
        assert_eq!(api.abi_version, abi::ABI_VERSION);
    }

    #[test]
    fn base64_uses_standard_padding() {
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
    }

    #[test]
    fn manifest_resolves_every_embedded_asset() {
        if ui_asset("ui/screens.js").is_some() {
            assert!(ui_manifest().is_ok());
        } else {
            assert!(ui_manifest().is_err());
        }
    }

    #[test]
    fn no_proxy_and_no_secret_before_install() {
        let plugin = plugin();
        assert_eq!(
            plugin.query(&json!({"operation": "ui.upstream"})).unwrap()["status"],
            "starting"
        );
        let status = plugin.query(&json!({"operation": "rpc.status"})).unwrap();
        assert_eq!(status["status"], "waiting");
        assert!(!status.to_string().contains("token"));
        let services = plugin
            .query(&json!({"operation": "host.services"}))
            .unwrap();
        assert_eq!(services["services"][0]["kind"], "app");
    }

    #[test]
    fn ingest_requires_a_typed_event() {
        let plugin = plugin();
        assert!(plugin.ingest(&json!({"type": "begin"})).is_ok());
        assert!(plugin.ingest(&json!({})).is_err());
    }

    #[test]
    fn unknown_operations_fail() {
        assert!(plugin().query(&json!({"operation": "nope"})).is_err());
    }
}
