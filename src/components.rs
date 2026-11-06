//! The image model components this node serves to the POM network (POM's
//! `component.serve`). The list is kept in the plugin directory and mirrored,
//! paths included, to `<home>/pom-served.json`, which ComfyUI's POM nodes read
//! to know what to answer on `/pom/component/...`.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const COMPONENTS: [&str; 3] = ["diffusion", "text_encoder", "vae"];
const LIST_FILE: &str = "served-components.json";
/// Read by ComfyUI's POM nodes (`runtime/pom_components.py`).
pub const COMFY_FILE: &str = "pom-served.json";

pub struct Served {
    list: PathBuf,
    comfy: PathBuf,
    items: Mutex<Vec<Value>>,
}

fn write_atomic(path: &Path, value: &Value) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let partial = path.with_extension("json.part");
    if fs::write(&partial, value.to_string()).is_ok() {
        let _ = fs::rename(&partial, path);
    }
}

impl Served {
    /// `state`: the plugin's own directory; `home`: ComfyUI's base directory.
    pub fn new(state: &Path, home: &Path) -> Self {
        let list = state.join(LIST_FILE);
        let items = fs::read(&list)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Vec<Value>>(&bytes).ok())
            .unwrap_or_default();
        Self {
            list,
            comfy: home.join(COMFY_FILE),
            items: Mutex::new(items),
        }
    }

    fn save(&self, items: &[Value]) {
        let all = Value::Array(items.to_vec());
        write_atomic(&self.list, &all);
        let for_comfy: Vec<Value> = items
            .iter()
            .map(|item| {
                json!({
                    "deploy_id": item["deploy_id"], "component": item["component"],
                    "paths": item["paths"], "options": item["options"],
                })
            })
            .collect();
        write_atomic(&self.comfy, &Value::Array(for_comfy));
    }

    /// Rewrites ComfyUI's copy (after ComfyUI's home moved, or at start).
    /// Nothing is written while nothing was ever served.
    pub fn publish(&self) {
        let items = self.items.lock().unwrap().clone();
        if !items.is_empty() || self.comfy.exists() {
            self.save(&items);
        }
    }

    /// Serve one component; idempotent (the same component of the same
    /// deployment replaces the earlier one).
    pub fn serve(&self, assignment: &Value, installed: bool) -> Result<(), String> {
        if !installed {
            return Err("Open ComfyUI once on this machine to install it".into());
        }
        let deploy_id = assignment["deploy_id"].as_str().filter(|s| !s.is_empty());
        let component = assignment["component"].as_str().unwrap_or_default();
        let Some(deploy_id) = deploy_id else {
            return Err("an image component needs its deployment".into());
        };
        if !COMPONENTS.contains(&component) {
            return Err(format!("unknown component {component}"));
        }
        let paths: Vec<&str> = assignment["paths"]
            .as_array()
            .map(|paths| paths.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if paths.is_empty() {
            return Err("an image component needs its file".into());
        }
        if let Some(missing) = paths.iter().find(|path| !Path::new(path).is_file()) {
            let name = Path::new(missing)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            return Err(format!("{name} is not on this machine"));
        }
        let mut items = self.items.lock().unwrap();
        items.retain(|item| !(item["deploy_id"] == deploy_id && item["component"] == component));
        items.push(json!({
            "deploy_id": deploy_id,
            "name": assignment["name"].as_str().unwrap_or(deploy_id),
            "component": component,
            "paths": paths,
            "options": assignment.get("options").cloned().unwrap_or(json!({})),
        }));
        self.save(&items);
        Ok(())
    }

    pub fn release(&self, deploy_id: &str) {
        let mut items = self.items.lock().unwrap();
        items.retain(|item| item["deploy_id"] != deploy_id);
        self.save(&items);
    }

    pub fn serves(&self, deploy_id: &str, component: &str) -> bool {
        self.items
            .lock()
            .unwrap()
            .iter()
            .any(|item| item["deploy_id"] == deploy_id && item["component"] == component)
    }

    /// The `image` service for `host.services`: file names, never paths.
    pub fn service(&self, phase: &str, detail: &str) -> Value {
        let status = match phase {
            "ready" => "ready",
            "error" => "error",
            _ => "loading",
        };
        let components: Vec<Value> = self
            .items
            .lock()
            .unwrap()
            .iter()
            .map(|item| {
                let files: Vec<String> = item["paths"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .filter_map(|p| {
                        Path::new(p)
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                    })
                    .collect();
                json!({
                    "deploy_id": item["deploy_id"], "name": item["name"],
                    "component": item["component"], "files": files, "status": status,
                })
            })
            .collect();
        let service_status = match phase {
            "ready" | "error" | "installing" | "waiting" => phase,
            _ => "starting",
        };
        let mut service = json!({
            "id": "comfyui-image", "kind": "image", "name": "ComfyUI",
            "status": service_status, "components": components,
        });
        if !detail.is_empty() {
            service["detail"] = json!(detail);
        }
        service
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A scratch tree removed when dropped.
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn dirs() -> (Scratch, PathBuf, PathBuf, PathBuf) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let root =
            std::env::temp_dir().join(format!("pom-comfyui-served-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let state = root.join("plugin");
        let home = root.join("home");
        let model = root.join("ae.safetensors");
        std::fs::write(&model, b"x").unwrap();
        (Scratch(root), state, home, model)
    }

    fn assignment(deploy_id: &str, component: &str, path: &Path) -> Value {
        json!({
            "deploy_id": deploy_id, "name": "Z-Image", "component": component,
            "paths": [path], "plugin_code": "comfyui", "options": {}
        })
    }

    #[test]
    fn serving_stores_and_publishes_the_component() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        let reloaded = Served::new(&state, &home);
        let service = reloaded.service("ready", "");
        assert_eq!(service["kind"], "image");
        assert_eq!(service["components"][0]["deploy_id"], "img-1");
        assert_eq!(service["components"][0]["component"], "vae");
        assert_eq!(service["components"][0]["status"], "ready");
        assert_eq!(
            reloaded.service("starting", "")["components"][0]["status"],
            "loading"
        );
        assert_eq!(
            reloaded.service("error", "x")["components"][0]["status"],
            "error"
        );
    }

    #[test]
    fn serving_requires_an_installed_comfyui() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        let error = served
            .serve(&assignment("img-1", "vae", &model), false)
            .unwrap_err();
        assert!(error.contains("Open ComfyUI once"), "{error}");
        let error = served
            .serve(&assignment("img-1", "unet", &model), true)
            .unwrap_err();
        assert!(error.contains("component"), "{error}");
        let error = served
            .serve(
                &assignment("img-1", "vae", Path::new("/nowhere/x.safetensors")),
                true,
            )
            .unwrap_err();
        assert!(error.contains("x.safetensors"), "{error}");
        assert!(served.service("ready", "")["components"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn release_drops_only_that_deployment() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        served
            .serve(&assignment("img-1", "diffusion", &model), true)
            .unwrap();
        served
            .serve(&assignment("img-2", "vae", &model), true)
            .unwrap();
        // The same component again replaces it.
        served
            .serve(&assignment("img-2", "vae", &model), true)
            .unwrap();
        served.release("img-1");
        let components = served.service("ready", "")["components"].clone();
        assert_eq!(components.as_array().unwrap().len(), 1);
        assert_eq!(components[0]["deploy_id"], "img-2");
    }

    #[test]
    fn upstream_answers_only_for_served_components() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        assert!(served.serves("img-1", "vae"));
        assert!(!served.serves("img-1", "diffusion"));
        assert!(!served.serves("img-2", "vae"));
    }

    #[test]
    fn the_served_file_is_written_for_comfyui() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        let written: Value =
            serde_json::from_slice(&std::fs::read(home.join("pom-served.json")).unwrap()).unwrap();
        assert_eq!(written[0]["deploy_id"], "img-1");
        assert_eq!(written[0]["component"], "vae");
        assert_eq!(written[0]["paths"][0], json!(model));
        served.release("img-1");
        let written: Value =
            serde_json::from_slice(&std::fs::read(home.join("pom-served.json")).unwrap()).unwrap();
        assert_eq!(written, json!([]));
    }

    #[test]
    fn nothing_is_written_before_anything_is_served() {
        let (_root, state, home, _model) = dirs();
        let served = Served::new(&state, &home);
        served.publish();
        let _ = served.service("ready", "");
        assert!(!state.exists() && !home.exists());
    }

    #[test]
    fn services_never_leak_paths_or_tokens() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        let text = served.service("ready", "").to_string();
        assert!(!text.contains(&*model.to_string_lossy()), "{text}");
        assert!(text.contains("\"ae.safetensors\""), "{text}");
    }
}
