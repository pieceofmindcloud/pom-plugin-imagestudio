//! Self-contained, verified bootstrap of ComfyUI. Nothing from the host is
//! required (no Python, git, curl or pip): the pinned ComfyUI source and a
//! pinned `uv` are downloaded and SHA-256 verified, `uv` installs a managed
//! Python into a private virtual environment, and PyTorch comes from the
//! PyTorch index that matches the machine's GPU.
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process::Command,
    time::Duration,
};

/// The pinned ComfyUI release. Updating it means a new tag, URL and digest
/// (`curl -sL <url> | sha256sum`); an installed node then updates on open.
pub const VERSION: &str = "0.43.0";
const SOURCE_URL: &str = "https://codeload.github.com/comfy-org/ComfyUI/tar.gz/refs/tags/v0.43.0";
const SOURCE_SHA: &str = "8a496a7d82bd343cafdb2b867857fe8c5140e3cd9ec5cf46c6277ccb7466f202";
const MAX_SOURCE: u64 = 256 * 1024 * 1024;
const MAX_UV: u64 = 100 * 1024 * 1024;
/// Python series ComfyUI recommends; `uv` installs its managed build.
const PYTHON: &str = "3.12";

#[derive(Clone, Copy)]
struct Artifact {
    url: &'static str,
    sha: &'static str,
    zipped: bool,
}
fn uv_artifact() -> Result<Artifact, String> {
    let target = (std::env::consts::OS, std::env::consts::ARCH);
    let (url, sha, zipped) = match target {
        ("linux", "x86_64") => ("https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-unknown-linux-gnu.tar.gz", "f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f", false),
        ("linux", "aarch64") => ("https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-unknown-linux-gnu.tar.gz", "4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403", false),
        ("macos", "aarch64") => ("https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-apple-darwin.tar.gz", "743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73", false),
        ("macos", "x86_64") => ("https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-apple-darwin.tar.gz", "b0f52b0e3945b4107d89024ea38e625683011fbfc05a0d98f9fa2d6dcee28c66", false),
        ("windows", "x86_64") => ("https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip", "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a", true),
        ("windows", "aarch64") => ("https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-pc-windows-msvc.zip", "09f47edf90726084a70f9426f6e3ca4f46e9bf7cbf23b1716ec64d60b055fae3", true),
        _ => return Err("ComfyUI does not support this operating system/architecture".into()),
    };
    Ok(Artifact { url, sha, zipped })
}

/// The PyTorch build for this machine, as `uv pip --torch-backend` names it.
/// NVIDIA wins over AMD on a machine with both (CUDA is ComfyUI's main path).
pub fn torch_backend() -> &'static str {
    let exists = |path: &str| Path::new(path).exists();
    match std::env::consts::OS {
        "macos" => "auto",
        "linux" if exists("/proc/driver/nvidia/version") || exists("/dev/nvidiactl") => {
            cuda_backend(nvidia_driver().as_deref())
        }
        "linux" if exists("/dev/kfd") => "rocm6.4",
        "windows" if exists(r"C:\Windows\System32\nvidia-smi.exe") => {
            cuda_backend(nvidia_driver().as_deref())
        }
        _ => "cpu",
    }
}

/// ComfyUI needs a `cu130` PyTorch for its optimized CUDA kernels (comfy-kitchen:
/// the int8/fp8/nvfp4 weights of current image models); CUDA 13.0 needs an
/// NVIDIA driver from the 580 series. Older or unknown drivers get `cu128`,
/// the first build with Blackwell (sm_120) kernels.
fn cuda_backend(driver: Option<&str>) -> &'static str {
    let major = driver
        .and_then(|version| version.split('.').next())
        .and_then(|major| major.parse::<u32>().ok());
    match major {
        Some(major) if major >= 580 => "cu130",
        _ => "cu128",
    }
}

/// The NVIDIA driver version (`580.178.04`): the kernel module's on Linux,
/// `nvidia-smi`'s otherwise.
fn nvidia_driver() -> Option<String> {
    if let Ok(text) = fs::read_to_string("/proc/driver/nvidia/version") {
        return driver_in(&text);
    }
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    driver_in(&String::from_utf8_lossy(&output.stdout))
}

/// The first `NNN.NN[.NN]` token, as `/proc/driver/nvidia/version` and
/// `nvidia-smi` print the driver version.
fn driver_in(text: &str) -> Option<String> {
    text.split_whitespace()
        .find(|word| {
            let parts: Vec<&str> = word.split('.').collect();
            parts.len() >= 2
                && parts
                    .iter()
                    .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
                && parts[0].len() == 3
        })
        .map(str::to_owned)
}

/// The stages of a fresh install, in order, as the screen reports them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    Source,
    Uv,
    Python,
    Packages,
}

impl Step {
    pub fn code(self) -> &'static str {
        match self {
            Step::Source => "source",
            Step::Uv => "uv",
            Step::Python => "python",
            Step::Packages => "packages",
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum InstallMode {
    /// Nothing installed yet.
    Fresh,
    /// Another pinned release, another PyTorch build, or not finished.
    Update,
    /// The pinned release is installed for this machine's PyTorch build.
    Installed,
}

/// What `.installed` holds: the source digest and the PyTorch build, so a
/// machine whose GPU changed reinstalls its packages.
fn stamp(backend: &str) -> String {
    format!("{SOURCE_SHA} {backend}")
}

pub fn install_mode(base: &Path, backend: &str) -> InstallMode {
    match fs::read_to_string(base.join(".installed")) {
        Err(_) => InstallMode::Fresh,
        Ok(found) if found == stamp(backend) && venv_python(base).is_file() => {
            InstallMode::Installed
        }
        Ok(_) => InstallMode::Update,
    }
}

pub fn venv_python(base: &Path) -> PathBuf {
    if cfg!(windows) {
        base.join("venv/Scripts/python.exe")
    } else {
        base.join("venv/bin/python")
    }
}

fn download(
    url: &str,
    target: &Path,
    sha: &str,
    max: u64,
    progress: &mut dyn FnMut(f64),
) -> Result<(), String> {
    if target.is_file() && hash_file(target).is_ok_and(|hash| hash == sha) {
        return Ok(());
    }
    let part = target.with_extension("download");
    let response = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        .redirects(5)
        .build()
        .get(url)
        .call()
        .map_err(|err| match err {
            ureq::Error::Status(code, _) => {
                format!("Verified runtime download failed (HTTP {code})")
            }
            ureq::Error::Transport(transport) => format!(
                "Verified runtime download failed ({:?}); check network access",
                transport.kind()
            ),
        })?;
    let total = response
        .header("Content-Length")
        .and_then(|length| length.parse::<u64>().ok())
        .filter(|length| *length > 0)
        .unwrap_or(max);
    let mut reader = response.into_reader();
    let mut out = fs::File::create(&part).map_err(|_| "Cannot write runtime download")?;
    let mut hash = Sha256::new();
    let mut size = 0u64;
    let mut bytes = [0u8; 256 * 1024];
    let result = (|| -> Result<(), String> {
        loop {
            let read = reader
                .read(&mut bytes)
                .map_err(|_| "Runtime download interrupted")?;
            if read == 0 {
                break;
            }
            size += read as u64;
            if size > max {
                return Err("Runtime download exceeds size limit".into());
            }
            out.write_all(&bytes[..read])
                .map_err(|_| "Cannot write runtime download")?;
            hash.update(&bytes[..read]);
            progress((size as f64 / total as f64).min(1.0));
        }
        out.sync_all()
            .map_err(|_| "Cannot commit runtime download")?;
        if format!("{:x}", hash.finalize()) != sha {
            return Err("Runtime download checksum mismatch".into());
        }
        if target.exists() {
            fs::remove_file(target).map_err(|_| "Cannot replace invalid runtime download")?;
        }
        fs::rename(&part, target).map_err(|_| "Cannot publish runtime download".to_owned())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
}
fn hash_file(path: &Path) -> Result<String, io::Error> {
    let mut hash = Sha256::new();
    io::copy(&mut fs::File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}
fn safe_file(path: &Path) -> bool {
    !path.as_os_str().is_empty() && path.components().all(|p| matches!(p, Component::Normal(_)))
}

/// Extracts the verified archive into `source/`, replacing an earlier pinned
/// release (the source holds no user data: models, nodes, inputs and outputs
/// live in the base directory under the workspace).
fn extract_source(archive: &Path, base: &Path) -> Result<PathBuf, String> {
    let source = base.join("source");
    if fs::read_to_string(source.join(".verified")).is_ok_and(|sha| sha == SOURCE_SHA)
        && source.join("main.py").is_file()
    {
        return Ok(source);
    }
    let stage = base.join("source.partial");
    let _ = fs::remove_dir_all(&stage);
    fs::create_dir(&stage).map_err(|_| "Cannot prepare ComfyUI source")?;
    let root = format!("ComfyUI-{VERSION}/");
    let file = fs::File::open(archive).map_err(|_| "Cannot open verified ComfyUI archive")?;
    let mut tar = tar::Archive::new(GzDecoder::new(file));
    for entry in tar.entries().map_err(|_| "Invalid ComfyUI archive")? {
        let mut entry = entry.map_err(|_| "Invalid ComfyUI archive member")?;
        let kind = entry.header().entry_type();
        if kind.is_pax_global_extensions() || kind.is_pax_local_extensions() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|_| "Invalid ComfyUI archive path")?
            .into_owned();
        if path == Path::new(root.trim_end_matches('/')) && kind.is_dir() {
            continue;
        }
        let relative = path
            .strip_prefix(&root)
            .map_err(|_| format!("Unexpected ComfyUI archive path: {}", path.display()))?;
        if !safe_file(relative) {
            return Err("Unsafe ComfyUI archive path".into());
        }
        let dest = stage.join(relative);
        if kind.is_dir() {
            fs::create_dir_all(dest).map_err(|_| "Cannot create ComfyUI directory")?;
        } else if kind.is_file() {
            fs::create_dir_all(dest.parent().ok_or("Invalid ComfyUI path")?)
                .map_err(|_| "Cannot create ComfyUI directory")?;
            let mut out = fs::File::create(&dest).map_err(|_| "Cannot write ComfyUI source")?;
            io::copy(&mut entry, &mut out).map_err(|_| "Cannot extract ComfyUI source")?;
            #[cfg(unix)]
            if let Ok(mode) = entry.header().mode() {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&dest, fs::Permissions::from_mode(mode & 0o755))
                    .map_err(|_| "Cannot set ComfyUI file permissions")?;
            }
        } else if kind.is_symlink() {
            // The release ships no link the server needs; never follow one.
            continue;
        } else {
            return Err("ComfyUI archive contains unsupported file type".into());
        }
    }
    if !stage.join("main.py").is_file() || !stage.join("requirements.txt").is_file() {
        return Err("Verified ComfyUI source is incomplete".into());
    }
    fs::write(stage.join(".verified"), SOURCE_SHA).map_err(|_| "Cannot mark ComfyUI source")?;
    if source.exists() {
        fs::remove_dir_all(&source).map_err(|_| "Cannot replace the earlier ComfyUI source")?;
    }
    fs::rename(&stage, &source).map_err(|_| "Cannot publish ComfyUI source")?;
    Ok(source)
}

fn extract_uv(archive: &Path, base: &Path, artifact: Artifact) -> Result<PathBuf, String> {
    let target = base.join(if cfg!(windows) { "uv.exe" } else { "uv" });
    let marker = base.join("uv.sha256");
    if target.is_file()
        && hash_file(&target).is_ok_and(|digest| {
            fs::read_to_string(&marker).ok().as_deref() == Some(digest.as_str())
        })
    {
        return Ok(target);
    }
    let stage = base.join("uv.partial");
    if artifact.zipped {
        let file = fs::File::open(archive).map_err(|_| "Cannot read uv archive")?;
        let mut zip = zip::ZipArchive::new(file).map_err(|_| "Invalid uv zip")?;
        let mut uv = zip.by_name("uv.exe").map_err(|_| "uv executable missing")?;
        let mut out = fs::File::create(&stage).map_err(|_| "Cannot write uv")?;
        io::copy(&mut uv, &mut out).map_err(|_| "Cannot extract uv")?;
    } else {
        let file = fs::File::open(archive).map_err(|_| "Cannot read uv archive")?;
        let mut tar = tar::Archive::new(GzDecoder::new(file));
        let mut found = false;
        for entry in tar.entries().map_err(|_| "Invalid uv archive")? {
            let mut entry = entry.map_err(|_| "Invalid uv archive member")?;
            let path = entry.path().map_err(|_| "Invalid uv path")?;
            if path.file_name().is_some_and(|name| name == "uv")
                && entry.header().entry_type().is_file()
            {
                let mut out = fs::File::create(&stage).map_err(|_| "Cannot write uv")?;
                io::copy(&mut entry, &mut out).map_err(|_| "Cannot extract uv")?;
                found = true;
                break;
            }
        }
        if !found {
            return Err("uv executable missing".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&stage, fs::Permissions::from_mode(0o700))
                .map_err(|_| "Cannot execute uv")?;
        }
    }
    if target.exists() {
        fs::remove_file(&target).map_err(|_| "Cannot replace uv")?;
    }
    fs::rename(&stage, &target).map_err(|_| "Cannot publish uv")?;
    fs::write(
        marker,
        hash_file(&target).map_err(|_| "Cannot verify uv executable")?,
    )
    .map_err(|_| "Cannot mark uv extraction")?;
    Ok(target)
}

/// `uv` keeps everything it downloads under the plugin's own directory.
fn environment(cmd: &mut Command, base: &Path, source: &Path) {
    let boot = base.join("bootstrap");
    cmd.current_dir(source)
        .env("UV_PYTHON_INSTALL_DIR", boot.join("pythons"))
        .env("UV_CACHE_DIR", boot.join("cache"))
        .env("UV_NO_CONFIG", "1")
        .env("UV_NO_PROJECT", "1")
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        .env("UV_LINK_MODE", "copy")
        .env_remove("VIRTUAL_ENV")
        .env_remove("PYTHONPATH")
        .env("CI", "1");
}

/// `run` owns children so a plugin-host shutdown can stop installation.
/// `report` hears each step as it starts and, for the downloads, the fraction done.
/// Returns the virtual environment's Python and the ComfyUI source.
pub fn ensure(
    base: &Path,
    backend: &str,
    mut run: impl FnMut(Command) -> Result<(), String>,
    mut report: impl FnMut(Step, Option<f64>),
) -> Result<(PathBuf, PathBuf), String> {
    let source = base.join("source");
    if install_mode(base, backend) == InstallMode::Installed && source.join("main.py").is_file() {
        return Ok((venv_python(base), source));
    }
    let marker = base.join(".installed");
    let _ = fs::remove_file(&marker);
    let source_archive = base.join("source.tar.gz");
    report(Step::Source, Some(0.0));
    download(
        SOURCE_URL,
        &source_archive,
        SOURCE_SHA,
        MAX_SOURCE,
        &mut |done| report(Step::Source, Some(done)),
    )?;
    let source = extract_source(&source_archive, base)?;
    let artifact = uv_artifact()?;
    let boot = base.join("bootstrap");
    fs::create_dir_all(&boot).map_err(|_| "Cannot prepare managed runtime")?;
    let uv_archive = boot.join(if artifact.zipped {
        "uv.zip"
    } else {
        "uv.tar.gz"
    });
    report(Step::Uv, Some(0.0));
    download(
        artifact.url,
        &uv_archive,
        artifact.sha,
        MAX_UV,
        &mut |done| report(Step::Uv, Some(done)),
    )?;
    let uv = extract_uv(&uv_archive, &boot, artifact)?;

    report(Step::Python, None);
    let venv = base.join("venv");
    if !venv_python(base).is_file() {
        let _ = fs::remove_dir_all(&venv);
        let mut create = Command::new(&uv);
        create
            .args(["venv", "--seed", "--python", PYTHON])
            .arg(&venv);
        environment(&mut create, base, &source);
        run(create)?;
    }
    let python = venv_python(base);
    if !python.is_file() {
        return Err("Managed Python was not installed; see install.log".into());
    }

    // One resolution for PyTorch and ComfyUI's requirements: the torch
    // packages come from the PyTorch index of this machine's build and every
    // other package from PyPI, so a later step never swaps the GPU build out.
    report(Step::Packages, None);
    let mut packages = Command::new(&uv);
    packages
        .args(["pip", "install", "--python"])
        .arg(&python)
        .args(["--torch-backend", backend])
        .args([
            "torch",
            "torchvision",
            "torchaudio",
            "-r",
            "requirements.txt",
        ]);
    environment(&mut packages, base, &source);
    run(packages)?;

    fs::write(&marker, stamp(backend)).map_err(|_| "Cannot mark ComfyUI installation")?;
    let _ = fs::remove_file(source_archive);
    Ok((python, source))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_mode_tells_fresh_update_and_installed_apart() {
        let base = std::env::temp_dir().join(format!("pom-comfyui-mode-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(venv_python(&base).parent().unwrap()).unwrap();
        assert_eq!(install_mode(&base, "cpu"), InstallMode::Fresh);
        fs::write(base.join(".installed"), "another-sha cpu").unwrap();
        assert_eq!(install_mode(&base, "cpu"), InstallMode::Update);
        fs::write(base.join(".installed"), stamp("cpu")).unwrap();
        assert_eq!(install_mode(&base, "cpu"), InstallMode::Update, "no venv");
        fs::write(venv_python(&base), "").unwrap();
        assert_eq!(install_mode(&base, "cpu"), InstallMode::Installed);
        assert_eq!(
            install_mode(&base, "cu128"),
            InstallMode::Update,
            "another PyTorch build reinstalls the packages"
        );
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn pins_are_complete_and_paths_are_confined() {
        let artifact = uv_artifact().unwrap();
        assert_eq!(artifact.sha.len(), 64);
        assert_eq!(SOURCE_SHA.len(), 64);
        assert!(SOURCE_URL.ends_with(&format!("v{VERSION}")));
        assert!(safe_file(Path::new("comfy/sd.py")));
        assert!(!safe_file(Path::new("../private")));
        assert!(!safe_file(Path::new("/private")));
        assert!(!safe_file(Path::new("")));
    }

    #[test]
    fn torch_backend_is_one_uv_knows() {
        assert!(["auto", "cu130", "cu128", "rocm6.4", "cpu"].contains(&torch_backend()));
    }

    #[test]
    fn the_driver_picks_the_cuda_build() {
        let proc = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.178.04  Release Build  (dvs-builder@U22)  Tue Jul  7 12:18:12 UTC 2026\nGCC version:  gcc version 13.3.0\n";
        assert_eq!(driver_in(proc).as_deref(), Some("580.178.04"));
        assert_eq!(driver_in("570.86.15\n").as_deref(), Some("570.86.15"));
        assert_eq!(driver_in("no driver"), None);
        assert_eq!(cuda_backend(Some("580.178.04")), "cu130");
        assert_eq!(cuda_backend(Some("590.1")), "cu130");
        assert_eq!(cuda_backend(Some("570.86.15")), "cu128");
        assert_eq!(cuda_backend(None), "cu128");
    }
}

#[cfg(test)]
mod live {
    use super::*;
    use std::io::BufRead;

    /// Full install into `$POM_COMFYUI_E2E_DIR` (default: a temp dir), then the
    /// bridge, ComfyUI's page under the proxy prefix, its API and the POM nodes.
    #[test]
    #[ignore = "downloads ComfyUI and PyTorch (several gigabytes)"]
    fn installs_and_serves_behind_the_bridge() {
        let base = std::env::var_os("POM_COMFYUI_E2E_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("pom-comfyui-e2e"));
        let home = base.join("home");
        // The POM nodes as a start writes them (nodes, web, components, remote).
        crate::supervisor::install_pom_nodes(&home).unwrap();
        let (python, source) = ensure(
            &base,
            torch_backend(),
            |mut command| {
                let log = fs::File::create(base.join("e2e-install.log")).unwrap();
                command.stdout(log.try_clone().unwrap()).stderr(log);
                let status = command.status().map_err(|e| e.to_string())?;
                status
                    .success()
                    .then_some(())
                    .ok_or(format!("installer exited: {status}"))
            },
            |_, _| {},
        )
        .unwrap();
        // A fake node API whose image catalog holds one VAE file.
        let store = base.join("store");
        fs::create_dir_all(&store).unwrap();
        let vae = store.join("ae.safetensors");
        fs::write(&vae, b"not real weights").unwrap();
        let catalog = serde_json::json!({"image_files": [{
            "path": vae, "name": "a/b/ae.safetensors", "category": "vae", "metadata_error": null
        }]})
        .to_string();
        // With a real VAE named, the fake node also lists it as a network
        // component and forwards its ops to the bridge, as a POM node would.
        let real_vae = std::env::var_os("POM_COMFYUI_E2E_VAE");
        let components = if real_vae.is_some() {
            serde_json::json!({"components": [{
                "deploy_id": "img-e2e", "name": "E2E", "component": "vae",
                "files": ["ae.safetensors"], "status": "ready",
                "node_id": "self", "hostname": "this-machine", "plugin_code": "comfyui"
            }]})
        } else {
            serde_json::json!({"components": []})
        }
        .to_string();
        let upstream: std::sync::Arc<std::sync::Mutex<Option<(u64, String)>>> = Default::default();
        let forward_to = std::sync::Arc::clone(&upstream);
        let node = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let node_port = node.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in node.incoming().flatten() {
                let (catalog, components, forward_to) = (
                    catalog.clone(),
                    components.clone(),
                    std::sync::Arc::clone(&forward_to),
                );
                std::thread::spawn(move || {
                    let mut stream = stream;
                    let mut reader = io::BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    io::BufRead::read_line(&mut reader, &mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
                    let mut length = 0usize;
                    loop {
                        let mut header = String::new();
                        io::BufRead::read_line(&mut reader, &mut header).unwrap();
                        if header.trim().is_empty() {
                            break;
                        }
                        if let Some(value) =
                            header.to_ascii_lowercase().strip_prefix("content-length:")
                        {
                            length = value.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut body = vec![0u8; length];
                    io::Read::read_exact(&mut reader, &mut body).unwrap();
                    let (status, kind, answer) = if path.ends_with("/catalog/images") {
                        (200, "application/json", catalog.into_bytes())
                    } else if path.ends_with("/components") {
                        (200, "application/json", components.into_bytes())
                    } else if let Some(op) = path.strip_prefix("/api/plugin-host/component/") {
                        let (port, token) = forward_to.lock().unwrap().clone().unwrap();
                        let url = format!("http://127.0.0.1:{port}/pom/component/{op}");
                        match ureq::post(&url)
                            .set("x-pom-plugin-token", &token)
                            .send_bytes(&body)
                        {
                            Ok(response) => {
                                let mut bytes = Vec::new();
                                io::Read::read_to_end(&mut response.into_reader(), &mut bytes)
                                    .unwrap();
                                (200, "application/octet-stream", bytes)
                            }
                            Err(error) => (
                                502,
                                "application/json",
                                serde_json::json!({"error": error.to_string()})
                                    .to_string()
                                    .into_bytes(),
                            ),
                        }
                    } else {
                        (200, "application/json", b"{}".to_vec())
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status} X\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        answer.len()
                    );
                    let _ = io::Write::write_all(&mut stream, &answer);
                });
            }
        });
        let log = fs::File::create(base.join("e2e-bridge.log")).unwrap();
        let mut bridge = Command::new(&python)
            .arg("-I")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime/bridge.py"))
            .args([&base, &source, &home])
            .env("POM_OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
            .env("POM_API_KEY", "sk-e2e")
            .env(
                "POM_HOST_API_URL",
                format!("http://127.0.0.1:{node_port}/api/plugin-host"),
            )
            .env("POM_HOST_API_TOKEN", "host-e2e")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(log)
            .spawn()
            .unwrap();
        let mut line = String::new();
        io::BufReader::new(bridge.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let ready: serde_json::Value =
            serde_json::from_str(&line).expect("ComfyUI did not start; see comfyui.log");
        let port = ready["port"].as_u64().unwrap();
        let token = ready["token"].as_str().unwrap();
        *upstream.lock().unwrap() = Some((port, token.to_owned()));
        let agent = ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(60))
            .build();
        let get = |path: &str| {
            agent
                .get(&format!("http://127.0.0.1:{port}{path}"))
                .set("x-pom-plugin-token", token)
                .set("origin", "http://pom.example:8080")
                .call()
                .map_err(|error| error.to_string())
                .and_then(|response| response.into_string().map_err(|e| e.to_string()))
        };
        assert!(matches!(
            agent.get(&format!("http://127.0.0.1:{port}/")).call(),
            Err(ureq::Error::Status(401, _))
        ));
        assert!(get("/").unwrap().contains("<html"));
        let info = get("/api/object_info/POMChat").unwrap();
        assert!(info.contains("POM gateway unavailable"), "{info}");
        // The catalog's VAE is linked under `vae/pom/` and listed by ComfyUI.
        let mut listed = String::new();
        for _ in 0..30 {
            listed = get("/api/models/vae").unwrap();
            if listed.contains("pom/a/b/ae.safetensors") {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        assert!(listed.contains("pom/a/b/ae.safetensors"), "{listed}");
        // A served VAE answers its ops through the bridge (the path POM's node
        // uses), with the operator's real VAE when one is named.
        if let Some(vae) = std::env::var_os("POM_COMFYUI_E2E_VAE") {
            fs::write(
                home.join("pom-served.json"),
                serde_json::json!([{"deploy_id": "img-e2e", "component": "vae",
                                    "paths": [vae.to_string_lossy()], "options": {}}])
                .to_string(),
            )
            .unwrap();
            let post = |op: &str, body: &[u8]| -> Result<Vec<u8>, String> {
                let response = agent
                    .post(&format!(
                        "http://127.0.0.1:{port}/pom/component/img-e2e/vae/{op}"
                    ))
                    .set("x-pom-plugin-token", token)
                    .send_bytes(body)
                    .map_err(|error| match error {
                        ureq::Error::Status(code, response) => {
                            format!(
                                "{op}: {code} {}",
                                response.into_string().unwrap_or_default()
                            )
                        }
                        other => other.to_string(),
                    })?;
                let mut bytes = Vec::new();
                io::Read::read_to_end(&mut response.into_reader(), &mut bytes)
                    .map_err(|error| error.to_string())?;
                Ok(bytes)
            };
            let header = |body: &[u8]| -> serde_json::Value {
                let len = u64::from_le_bytes(body[..8].try_into().unwrap()) as usize;
                serde_json::from_slice(&body[8..8 + len]).unwrap()
            };
            let empty = br#"{"__metadata__":{"args":"{}","lists":"{}"}}"#;
            let mut body = (empty.len() as u64).to_le_bytes().to_vec();
            body.extend_from_slice(empty);
            let loaded = header(&post("load", &body).unwrap());
            let args: serde_json::Value =
                serde_json::from_str(loaded["__metadata__"]["args"].as_str().unwrap()).unwrap();
            let channels = args["latent_channels"].as_u64().unwrap() as usize;
            let bytes = channels * 8 * 8 * 4;
            let meta = format!(
                r#"{{"samples":{{"dtype":"F32","shape":[1,{channels},8,8],"data_offsets":[0,{bytes}]}},"__metadata__":{{"args":"{{}}","lists":"{{}}"}}}}"#
            );
            let mut body = (meta.len() as u64).to_le_bytes().to_vec();
            body.extend_from_slice(meta.as_bytes());
            body.resize(body.len() + bytes, 0);
            let decoded = header(&post("decode", &body).unwrap());
            assert_eq!(
                decoded["images"]["shape"],
                serde_json::json!([1, 64, 64, 3]),
                "{decoded}"
            );
            // ComfyUI's own VAE Decode on the network VAE, as a person would
            // run it: the name is listed by Load VAE and the decode is remote.
            let listed = get("/api/object_info/VAELoader").unwrap();
            assert!(listed.contains("E2E"), "{listed}");
            let workflow = serde_json::json!({"prompt": {
                "1": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
                "2": {"class_type": "VAELoader", "inputs": {"vae_name": "POM ▸ E2E ▸ ae.safetensors"}},
                "3": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["2", 0]}},
                "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}}
            }});
            let queued: serde_json::Value = match agent
                .post(&format!("http://127.0.0.1:{port}/prompt"))
                .set("x-pom-plugin-token", token)
                .set("origin", "http://pom.example:8080")
                .set("content-type", "application/json")
                .send_string(&workflow.to_string())
            {
                Ok(response) => serde_json::from_str(&response.into_string().unwrap()).unwrap(),
                Err(ureq::Error::Status(code, response)) => {
                    panic!(
                        "prompt refused {code}: {}",
                        response.into_string().unwrap_or_default()
                    )
                }
                Err(error) => panic!("{error}"),
            };
            let id = queued["prompt_id"].as_str().unwrap().to_owned();
            let mut history = serde_json::Value::Null;
            for _ in 0..120 {
                history =
                    serde_json::from_str(&get(&format!("/api/history/{id}")).unwrap()).unwrap();
                if history[&id]["status"]["completed"] == true
                    || history[&id]["status"]["status_str"] == "error"
                {
                    break;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            assert_eq!(history[&id]["status"]["status_str"], "success", "{history}");
            assert!(
                history[&id]["outputs"]["4"]["images"]
                    .as_array()
                    .is_some_and(|images| !images.is_empty()),
                "{history}"
            );
        }
        drop(bridge.stdin.take());
        assert!(bridge.wait().unwrap().success());
    }
}
