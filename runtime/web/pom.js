// POM: ComfyUI runs on the POM node, not on the viewer's computer. The
// missing-model "Download" button would otherwise save the file in the
// browser; ComfyUI lets a host take that button over, so the POM nodes
// download the model into the server's models directory instead.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const downloads = new Map();
let panel;

function toast(severity, summary, detail) {
  app.extensionManager?.toast?.add({ severity, summary, detail, life: severity === "error" ? 0 : 6000 });
}

function size(bytes) {
  return bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.round(bytes / 2 ** 20)} MB`;
}

function render() {
  const active = [...downloads.values()].filter((state) => state.status === "downloading");
  if (!active.length) {
    panel?.remove();
    panel = undefined;
    return;
  }
  if (!panel) {
    panel = document.createElement("div");
    panel.style.cssText =
      "position:fixed;right:16px;bottom:16px;z-index:10000;max-width:360px;padding:10px 12px;" +
      "border-radius:8px;background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);" +
      "font:12px/1.5 sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.4);pointer-events:none";
    document.body.append(panel);
  }
  panel.replaceChildren(
    ...active.map((state) => {
      const line = document.createElement("div");
      const percent = state.total ? ` ${Math.floor((100 * state.received) / state.total)}%` : "";
      line.textContent = `⬇ ${state.name}${percent} (${size(state.received)}${state.total ? ` / ${size(state.total)}` : ""})`;
      return line;
    }),
  );
}

function update(state) {
  const key = `${state.directory}/${state.name}`;
  const previous = downloads.get(key);
  downloads.set(key, state);
  render();
  if (previous?.status !== "downloading" || state.status === "downloading") return;
  if (state.status === "done") {
    toast("success", "Model downloaded", state.path || `${state.directory}/${state.name}`);
    app.refreshComboInNodes?.();
  } else {
    toast("error", "Model download failed", `${state.name}: ${state.error}`);
  }
}

async function downloadModel(url, name, directory) {
  const response = await api.fetchApi("/pom/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, name, directory }),
  });
  const state = await response.json();
  if (!response.ok) {
    toast("error", "Model download failed", state.error || `HTTP ${response.status}`);
    return false;
  }
  if (state.status === "done") {
    toast("info", "Model already on the server", state.path || `${directory}/${name}`);
    app.refreshComboInNodes?.();
  } else {
    toast("info", "Downloading on the POM server", `${directory}/${name}`);
    update(state);
  }
  return true;
}

app.registerExtension({
  name: "pom.modelDownloads",
  async setup() {
    window.__comfyDesktop2 = Object.assign(window.__comfyDesktop2 || {}, {
      downloadModel,
      isRemote: () => false,
    });
    api.addEventListener("pom-download", ({ detail }) => update(detail));
    try {
      const response = await api.fetchApi("/pom/models/downloads");
      if (response.ok) for (const state of await response.json()) update(state);
    } catch {
      // The list only restores the progress panel after a reload.
    }
  },
});
