//! Background install and the supervised ComfyUI server. IPC never waits for
//! a download: `query` reads the status snapshot and returns at once.
use crate::{
    context::Context,
    installer::{self, InstallMode, Step},
    plugin_directory,
};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

const BRIDGE: &str = include_str!("../runtime/bridge.py");
const POM_NODES: &str = include_str!("../runtime/pom_nodes.py");
const POM_NODES_WEB: &str = include_str!("../runtime/web/pom.js");

struct Running {
    child: Child,
    stdin: ChildStdin,
}

#[derive(Clone)]
pub struct Status {
    pub phase: String,
    pub detail: String,
    port: Option<u16>,
    token: Option<String>,
    /// While installing: the current step, when it started and, for a
    /// download, the fraction done.
    step: Option<(Step, Instant, Option<f64>)>,
    /// While installing: `install` or `update`.
    mode: Option<&'static str>,
}
impl Status {
    fn new(phase: &str, detail: &str) -> Self {
        Self {
            phase: phase.into(),
            detail: detail.into(),
            port: None,
            token: None,
            step: None,
            mode: None,
        }
    }
    pub fn public(&self) -> Value {
        let mut public = json!({"status": self.phase, "detail": self.detail});
        if self.phase == "installing" {
            if let Some((step, since, measured)) = self.step {
                public["step"] = json!(step.code());
                public["progress"] = json!(progress_at(step, since.elapsed(), measured));
            }
            if let Some(mode) = self.mode {
                public["mode"] = json!(mode);
            }
        }
        public
    }
    pub fn upstream(&self) -> Value {
        if self.phase == "ready" {
            json!({"status":"ready", "port":self.port, "token":self.token})
        } else if self.phase == "error" {
            json!({"status":"error", "error":self.detail})
        } else {
            json!({"status":"starting"})
        }
    }
}

/// Each step's share of the bar, `(start, end)`, and for an unmeasured step
/// the time constant of its creep. PyTorch with its GPU libraries is several
/// gigabytes, so the packages own most of the bar.
fn weights(step: Step) -> (f64, f64, f64) {
    match step {
        Step::Source => (0.0, 5.0, 0.0),
        Step::Uv => (5.0, 8.0, 0.0),
        Step::Python => (8.0, 15.0, 20.0),
        Step::Packages => (15.0, 100.0, 300.0),
    }
}

/// The bar while `step` runs: a download follows its bytes, any other step
/// creeps toward its end with time and never reaches it before it finishes.
pub fn progress_at(step: Step, elapsed: Duration, measured: Option<f64>) -> u8 {
    let (start, end, tau) = weights(step);
    let fraction = match measured {
        Some(done) => done.clamp(0.0, 1.0),
        None if tau > 0.0 => 1.0 - (-elapsed.as_secs_f64() / tau).exp(),
        None => 0.0,
    };
    (start + (end - start) * 0.95 * fraction).floor() as u8
}

pub struct Supervisor {
    status: Mutex<Status>,
    process: Mutex<Option<Running>>,
    installer: Mutex<Option<Child>>,
    context: Mutex<Context>,
    busy: Mutex<bool>,
}
impl Supervisor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(Status::new("waiting", "Install ComfyUI to begin")),
            process: Mutex::new(None),
            installer: Mutex::new(None),
            context: Mutex::new(Context::default()),
            busy: Mutex::new(false),
        })
    }
    pub fn status(&self) -> Status {
        self.status.lock().unwrap().clone()
    }
    /// What the screen may see of the node's configuration (no secret).
    pub fn context_public(&self) -> Value {
        self.context.lock().unwrap().public_json()
    }
    fn set(&self, phase: &str, detail: &str) {
        *self.status.lock().unwrap() = Status::new(phase, detail);
    }
    /// The launch flags of the running server depend on the preferences; a
    /// change of them (or of the gateway key) restarts a running server.
    pub fn configure(self: &Arc<Self>, context: Context) {
        let changed = {
            let mut current = self.context.lock().unwrap();
            let changed = current.launch_identity() != context.launch_identity();
            *current = context;
            changed
        };
        // The plugin is on demand: the host runs only while the screen is
        // open, so an installed ComfyUI starts right away. A fresh install
        // waits for the screen's `rpc.install`: opening it is the opt-in.
        if let Ok(base) = plugin_directory() {
            if base.join(".installed").exists() {
                self.launch(changed);
            }
        }
    }
    /// `None` while another launch runs; `Some(false)` when the server is
    /// already up and nothing has to start; `Some(true)` when the caller
    /// starts it, the phase already off any previous error.
    fn begin_launch(&self, restart: bool) -> Option<bool> {
        let mut busy = self.busy.lock().unwrap();
        if *busy {
            return None;
        }
        if !restart && self.status().phase == "ready" {
            return Some(false);
        }
        *busy = true;
        self.set("starting", "Starting ComfyUI");
        Some(true)
    }
    pub fn launch(self: &Arc<Self>, restart: bool) -> bool {
        match self.begin_launch(restart) {
            None => return false,
            Some(false) => return true,
            Some(true) => {}
        }
        let this = Arc::clone(self);
        thread::spawn(move || {
            if let Err(error) = this.start() {
                this.stop();
                // Never subprocess output, environment or credentials.
                this.set("error", &error);
            }
            *this.busy.lock().unwrap() = false;
        });
        true
    }
    fn start(&self) -> Result<(), String> {
        let base = plugin_directory()?;
        fs::create_dir_all(&base).map_err(|_| "Cannot create ComfyUI runtime directory")?;
        private_directory(&base)?;
        self.stop();
        let context = self.context.lock().unwrap().clone();
        // Models, custom nodes, inputs, outputs and workflows belong to the
        // operator: they live in the workspace, not next to the runtime.
        let home = context
            .workspace_state_dir()
            .unwrap_or_else(|| base.join("data"));
        fs::create_dir_all(home.join("custom_nodes"))
            .map_err(|_| "Cannot create ComfyUI data directory")?;
        let backend = if context.feature("cpu_only") {
            "cpu"
        } else {
            installer::torch_backend()
        };
        let mode = match installer::install_mode(&base, backend) {
            InstallMode::Fresh => Some("install"),
            InstallMode::Update => Some("update"),
            InstallMode::Installed => None,
        };
        if mode.is_some() {
            self.set("installing", "Installing ComfyUI (see install.log)");
        }
        let report = |step: Step, measured: Option<f64>| {
            let Some(mode) = mode else { return };
            let mut status = self.status.lock().unwrap();
            let since = match status.step {
                Some((current, since, _)) if current == step => since,
                _ => Instant::now(),
            };
            status.step = Some((step, since, measured));
            status.mode = Some(mode);
        };
        let (python, source) = installer::ensure(
            &base,
            backend,
            |command| self.run_install_step(command, &base),
            &report,
        )?;
        install_pom_nodes(&home)?;
        self.set("starting", "Starting ComfyUI");

        let bridge = base.join("bridge.py");
        fs::write(&bridge, BRIDGE).map_err(|_| "Cannot prepare ComfyUI bridge")?;
        let log =
            fs::File::create(base.join("bridge.log")).map_err(|_| "Cannot open bridge log")?;
        let mut cmd = Command::new(&python);
        cmd.arg("-I")
            .arg(&bridge)
            .arg(&base)
            .arg(&source)
            .arg(&home)
            .current_dir(&base)
            .env_remove("PYTHONPATH")
            .env("POM_COMFYUI_ARGS", comfy_args(&context).join("\n"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log);
        for (key, value) in bridge_env(&context) {
            match value {
                Some(value) => cmd.env(key, value),
                None => cmd.env_remove(key),
            };
        }
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = cmd.spawn().map_err(|_| "Cannot start the ComfyUI bridge")?;
        let stdin = child.stdin.take().ok_or("Bridge stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("Bridge stdout unavailable")?;
        *self.process.lock().unwrap() = Some(Running { child, stdin });
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .map_err(|_| "Bridge failed to start; see bridge.log")?;
        let reply: Value =
            serde_json::from_str(&line).map_err(|_| "ComfyUI failed to start; see comfyui.log")?;
        let port = reply["port"]
            .as_u64()
            .and_then(|p| u16::try_from(p).ok())
            .filter(|p| *p > 0)
            .ok_or("Invalid bridge port")?;
        let token = reply["token"]
            .as_str()
            .filter(|s| s.len() >= 16)
            .ok_or("Missing bridge token")?;
        *self.status.lock().unwrap() = Status {
            phase: "ready".into(),
            detail: "ComfyUI is running".into(),
            port: Some(port),
            token: Some(token.into()),
            step: None,
            mode: None,
        };
        loop {
            thread::sleep(Duration::from_secs(2));
            let mut process = self.process.lock().unwrap();
            let Some(running) = process.as_mut() else {
                break;
            };
            if let Ok(Some(_)) = running.child.try_wait() {
                *process = None;
                drop(process);
                self.set("error", "ComfyUI stopped; see comfyui.log");
                break;
            }
        }
        Ok(())
    }
    fn run_install_step(&self, mut command: Command, base: &Path) -> Result<(), String> {
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(base.join("install.log"))
            .map_err(|_| "Cannot open ComfyUI install log")?;
        let errors = log
            .try_clone()
            .map_err(|_| "Cannot open ComfyUI install log")?;
        command.stdin(Stdio::null()).stdout(log).stderr(errors);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
            #[cfg(target_os = "linux")]
            unsafe {
                command.pre_exec(|| {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        *self.installer.lock().unwrap() = Some(
            command
                .spawn()
                .map_err(|_| "Cannot start the ComfyUI installer; see install.log")?,
        );
        loop {
            let outcome = {
                let mut installer = self.installer.lock().unwrap();
                match installer.as_mut() {
                    None => return Err("ComfyUI installation interrupted".into()),
                    Some(child) => match child.try_wait() {
                        Ok(Some(exit)) => {
                            *installer = None;
                            Some(exit)
                        }
                        Ok(None) => None,
                        Err(_) => return Err("Cannot monitor ComfyUI installation".into()),
                    },
                }
            };
            if let Some(exit) = outcome {
                return if exit.success() {
                    Ok(())
                } else {
                    Err("ComfyUI installation failed; see install.log".into())
                };
            }
            thread::sleep(Duration::from_millis(500));
        }
    }
    pub(crate) fn stop(&self) {
        if let Some(mut child) = self.installer.lock().unwrap().take() {
            #[cfg(unix)]
            unsafe {
                libc::killpg(child.id() as i32, libc::SIGTERM);
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &child.id().to_string()])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(mut running) = self.process.lock().unwrap().take() {
            // Closing stdin makes the bridge stop ComfyUI's whole process
            // group; give its grace time (5 s) to finish.
            drop(running.stdin);
            for _ in 0..70 {
                if matches!(running.child.try_wait(), Ok(Some(_))) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
            let _ = running.child.kill();
            let _ = running.child.wait();
        }
    }
}
impl Drop for Supervisor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The node's secrets and addresses for ComfyUI's POM nodes: set (`Some`) or
/// removed so a stale value is never inherited (`None`). They travel only in
/// the child's private environment, never argv, IPC or a file.
pub fn bridge_env(context: &Context) -> Vec<(&'static str, Option<String>)> {
    let gateway = context.gateway.as_ref();
    let host_api = context.host_api.as_ref();
    let model_dirs = context
        .model_dirs
        .iter()
        .map(|dir| dir.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("\n");
    vec![
        ("POM_OPENAI_BASE_URL", gateway.map(|g| g.openai_base_url.clone())),
        ("POM_API_KEY", gateway.map(|g| g.api_key.clone())),
        ("POM_MODEL_DIRS", (!model_dirs.is_empty()).then_some(model_dirs)),
        ("POM_HOST_API_URL", host_api.map(|h| h.base_url.clone())),
        ("POM_HOST_API_TOKEN", host_api.map(|h| h.token.clone())),
    ]
}

/// ComfyUI's command line after the fixed `--listen`/`--port`, one per line.
pub fn comfy_args(context: &Context) -> Vec<String> {
    let mut args = Vec::new();
    if context.feature("cpu_only") {
        args.push("--cpu".to_owned());
    } else if context.feature("low_vram") {
        // POM's own models usually hold the GPU too: keep ComfyUI's weights
        // in RAM and move them in per step.
        args.push("--lowvram".to_owned());
    }
    args
}

/// The POM nodes are this plugin's own directory inside `custom_nodes`;
/// rewriting it on every start keeps it the plugin's version.
fn install_pom_nodes(home: &Path) -> Result<(), String> {
    let directory = home.join("custom_nodes/pom_nodes");
    let failed = |_| "Cannot install the POM nodes".to_owned();
    fs::create_dir_all(directory.join("web")).map_err(failed)?;
    fs::write(directory.join("__init__.py"), POM_NODES).map_err(failed)?;
    fs::write(directory.join("web/pom.js"), POM_NODES_WEB).map_err(failed)
}

fn private_directory(path: &Path) -> Result<(), String> {
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|_| "Cannot secure ComfyUI directory")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SECOND: Duration = Duration::from_secs(1);

    #[test]
    fn an_unmeasured_step_never_reaches_its_end() {
        assert_eq!(progress_at(Step::Packages, Duration::ZERO, None), 15);
        assert!(progress_at(Step::Packages, SECOND * 100_000, None) < 100);
        let mut last = 0;
        for seconds in 0..1200 {
            let now = progress_at(Step::Packages, SECOND * seconds, None);
            assert!(now >= last, "went back at {seconds}s");
            last = now;
        }
        assert!(last > 80, "still crawling at 20 minutes: {last}");
    }

    #[test]
    fn a_measured_step_follows_its_fraction() {
        assert_eq!(progress_at(Step::Source, Duration::ZERO, Some(0.5)), 2);
        assert_eq!(progress_at(Step::Uv, Duration::ZERO, Some(1.0)), 7);
    }

    #[test]
    fn an_accepted_launch_leaves_the_error_at_once() {
        let supervisor = Supervisor::new();
        supervisor.set("error", "Runtime download interrupted");
        assert_eq!(supervisor.begin_launch(false), Some(true));
        assert_eq!(supervisor.status().phase, "starting");
        assert_eq!(
            supervisor.begin_launch(false),
            None,
            "a second launch while busy"
        );
    }

    #[test]
    fn status_never_leaks_token() {
        let status = Status {
            phase: "ready".into(),
            detail: "ok".into(),
            port: Some(42),
            token: Some("secret-secret-secret".into()),
            step: None,
            mode: None,
        };
        assert!(!status.public().to_string().contains("secret"));
        assert_eq!(status.upstream()["token"], "secret-secret-secret");
    }

    #[test]
    fn the_bridge_gets_the_model_dirs_and_host_api() {
        let context = Context::from_configure(&json!({
            "model_dirs": ["/models/full", "/mnt/disk2/models"],
            "host_api": {"base_url": "http://127.0.0.1:8080/api/plugin-host", "token": "host-secret-token"}
        }))
        .unwrap();
        let env: std::collections::HashMap<_, _> = bridge_env(&context).into_iter().collect();
        assert_eq!(
            env["POM_MODEL_DIRS"].as_deref(),
            Some("/models/full\n/mnt/disk2/models")
        );
        assert_eq!(
            env["POM_HOST_API_URL"].as_deref(),
            Some("http://127.0.0.1:8080/api/plugin-host")
        );
        assert_eq!(env["POM_HOST_API_TOKEN"].as_deref(), Some("host-secret-token"));
        assert!(!context.public_json().to_string().contains("host-secret-token"));
        let bare: std::collections::HashMap<_, _> =
            bridge_env(&Context::default()).into_iter().collect();
        assert_eq!(bare["POM_HOST_API_TOKEN"], None, "removed, not inherited");
    }

    #[test]
    fn preferences_choose_the_launch_flags() {
        let with = |features: Value| {
            Context::from_configure(&json!({"preferences": {"features": features}})).unwrap()
        };
        assert!(comfy_args(&with(json!({}))).is_empty());
        assert_eq!(comfy_args(&with(json!({"low_vram": true}))), ["--lowvram"]);
        assert_eq!(
            comfy_args(&with(json!({"low_vram": true, "cpu_only": true}))),
            ["--cpu"]
        );
    }

    #[test]
    fn the_pom_nodes_are_rewritten_in_their_own_directory() {
        let home = std::env::temp_dir().join(format!("pom-comfyui-nodes-{}", std::process::id()));
        let _ = fs::remove_dir_all(&home);
        fs::create_dir_all(home.join("custom_nodes/pom_nodes")).unwrap();
        fs::write(home.join("custom_nodes/pom_nodes/__init__.py"), "old").unwrap();
        install_pom_nodes(&home).unwrap();
        let text = fs::read_to_string(home.join("custom_nodes/pom_nodes/__init__.py")).unwrap();
        assert_eq!(text, POM_NODES);
        let web = fs::read_to_string(home.join("custom_nodes/pom_nodes/web/pom.js")).unwrap();
        assert_eq!(web, POM_NODES_WEB);
        assert!(!text.contains("sk-"), "no key in the node source");
        fs::remove_dir_all(home).unwrap();
    }
}
