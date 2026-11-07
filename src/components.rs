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

/// Written by ComfyUI's POM nodes: calls being served per `<deploy_id>/<component>`.
const IN_FLIGHT_FILE: &str = "pom-inflight.json";
/// Workflows an admin published for the Chat (`runtime/pom_workflows.py`).
const WORKFLOWS_DIR: &str = "user/pom-workflows";

/// ComfyUI's copy of this node's list, in the plugin's own directory: the
/// nodes of one machine share one ComfyUI home (the workspace), so a file
/// there would hold whichever node wrote last. ComfyUI finds it through
/// `POM_SERVED_FILE` (`supervisor::bridge_env`).
pub fn comfy_file(state: &Path) -> PathBuf {
    state.join(COMFY_FILE)
}

/// Where ComfyUI counts this node's calls in flight (`POM_INFLIGHT_FILE`).
pub fn in_flight_file(state: &Path) -> PathBuf {
    state.join(IN_FLIGHT_FILE)
}

pub struct Served {
    list: PathBuf,
    comfy: PathBuf,
    in_flight: PathBuf,
    home: PathBuf,
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
            comfy: comfy_file(state),
            in_flight: in_flight_file(state),
            home: home.to_path_buf(),
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

    /// Stop serving a deployment: one component of it, or all of them.
    pub fn release(&self, deploy_id: &str, component: Option<&str>) {
        let mut items = self.items.lock().unwrap();
        items.retain(|item| {
            item["deploy_id"] != deploy_id || component.is_some_and(|c| item["component"] != c)
        });
        self.save(&items);
    }

    pub fn serves(&self, deploy_id: &str, component: &str) -> bool {
        self.items
            .lock()
            .unwrap()
            .iter()
            .any(|item| item["deploy_id"] == deploy_id && item["component"] == component)
    }

    /// The Chat's workflows here: a built-in text to image for each
    /// deployment this machine serves a part of, and the published ones.
    pub fn workflows(&self) -> Vec<Value> {
        let mut list: Vec<Value> = Vec::new();
        for item in self.items.lock().unwrap().iter() {
            let Some(deploy_id) = item["deploy_id"].as_str() else {
                continue;
            };
            let id = format!("builtin:{deploy_id}");
            if list.iter().any(|w| w["id"] == id.as_str()) {
                continue;
            }
            list.push(json!({
                "id": id, "name": item["name"], "deploy_id": deploy_id,
                "inputs": ["prompt", "negative", "width", "height", "seed", "image"],
            }));
        }
        let mut published: Vec<PathBuf> = fs::read_dir(self.home.join(WORKFLOWS_DIR))
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "json"))
            .collect();
        published.sort();
        for path in published.into_iter().take(24) {
            let Some(record) = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            else {
                continue;
            };
            if record["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("pub:"))
            {
                list.push(json!({
                    "id": record["id"], "name": record["name"], "inputs": record["inputs"],
                }));
            }
        }
        list
    }

    /// The `image` service for `host.services`: file names, never paths.
    pub fn service(&self, phase: &str, detail: &str) -> Value {
        let status = match phase {
            "ready" => "ready",
            "error" => "error",
            _ => "loading",
        };
        let in_flight: Value = fs::read(&self.in_flight)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or(Value::Null);
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
                let key = format!(
                    "{}/{}",
                    item["deploy_id"].as_str().unwrap_or_default(),
                    item["component"].as_str().unwrap_or_default()
                );
                json!({
                    "deploy_id": item["deploy_id"], "name": item["name"],
                    "component": item["component"], "files": files, "status": status,
                    "in_flight": in_flight.get(&key).and_then(Value::as_u64).unwrap_or(0),
                })
            })
            .collect();
        let workflows = self.workflows();
        let service_status = match phase {
            "ready" | "error" | "installing" | "waiting" => phase,
            _ => "starting",
        };
        let mut service = json!({
            "id": "comfyui-image", "kind": "image", "name": "ComfyUI",
            "status": service_status, "components": components,
            "workflows": workflows,
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
    fn the_service_lists_in_flight_and_workflows() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        served
            .serve(&assignment("img-1", "diffusion", &model), true)
            .unwrap();
        std::fs::write(in_flight_file(&state), r#"{"img-1/vae": 2}"#).unwrap();
        std::fs::create_dir_all(home.join(WORKFLOWS_DIR)).unwrap();
        std::fs::write(
            home.join(WORKFLOWS_DIR).join("sketch.json"),
            r#"{"id": "pub:sketch", "name": "Sketch", "inputs": ["prompt"], "graph": {}}"#,
        )
        .unwrap();
        let service = served.service("ready", "");
        let parts = service["components"].as_array().unwrap();
        let vae = parts.iter().find(|c| c["component"] == "vae").unwrap();
        assert_eq!(vae["in_flight"], 2);
        let diffusion = parts
            .iter()
            .find(|c| c["component"] == "diffusion")
            .unwrap();
        assert_eq!(diffusion["in_flight"], 0);
        let workflows = service["workflows"].as_array().unwrap();
        assert_eq!(
            workflows.len(),
            2,
            "one built-in per deployment, then the published"
        );
        assert_eq!(workflows[0]["id"], "builtin:img-1");
        assert_eq!(workflows[1]["id"], "pub:sketch");
        assert!(workflows[1].get("graph").is_none(), "the graph stays here");
    }

    #[test]
    fn release_of_one_component_keeps_the_others() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        served
            .serve(&assignment("img-1", "text_encoder", &model), true)
            .unwrap();
        served.release("img-1", Some("vae"));
        assert!(!served.serves("img-1", "vae"));
        assert!(served.serves("img-1", "text_encoder"));
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
        served.release("img-1", None);
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
    fn two_nodes_sharing_a_home_keep_their_own_lists() {
        let (_root, state, home, model) = dirs();
        let other_state = state.with_file_name("other-plugin");
        let gpu1 = Served::new(&state, &home);
        let gpu2 = Served::new(&other_state, &home);
        gpu1.serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        gpu2.serve(&assignment("img-1", "diffusion", &model), true)
            .unwrap();
        let read = |dir: &Path| -> Value {
            serde_json::from_slice(&std::fs::read(comfy_file(dir)).unwrap()).unwrap()
        };
        assert_eq!(read(&state)[0]["component"], "vae");
        assert_eq!(read(&other_state)[0]["component"], "diffusion");
        assert!(
            !home.join(COMFY_FILE).exists(),
            "nothing shared in ComfyUI's home"
        );
    }

    #[test]
    fn the_served_file_is_written_for_comfyui() {
        let (_root, state, home, model) = dirs();
        let served = Served::new(&state, &home);
        served
            .serve(&assignment("img-1", "vae", &model), true)
            .unwrap();
        let written: Value =
            serde_json::from_slice(&std::fs::read(comfy_file(&state)).unwrap()).unwrap();
        assert_eq!(written[0]["deploy_id"], "img-1");
        assert_eq!(written[0]["component"], "vae");
        assert_eq!(written[0]["paths"][0], json!(model));
        served.release("img-1", None);
        let written: Value =
            serde_json::from_slice(&std::fs::read(comfy_file(&state)).unwrap()).unwrap();
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
