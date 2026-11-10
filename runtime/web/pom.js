// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 The POM authors. Runs inside ComfyUI; see runtime/LICENSE.
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
    // POM's catalog may have changed since ComfyUI started (a file copied or
    // downloaded on the node): relink it and reload the model lists.
    try {
      const synced = await api.fetchApi("/pom/models/sync", { method: "POST" });
      if (synced.ok) app.refreshComboInNodes?.();
    } catch {
      // Without the node's API the lists keep ComfyUI's own folders.
    }
    try {
      const response = await api.fetchApi("/pom/models/downloads");
      if (response.ok) for (const state of await response.json()) update(state);
    } catch {
      // The list only restores the progress panel after a reload.
    }
  },
});

// --- Deploy on POM (admins) --------------------------------------------------
// An admin node deploys an image model by component from here, as on POM's
// deploy screen: per component a file of that kind and a machine that holds
// it. `/pom/deploy/*` relays to the node, which checks the role and choices.

const COMPONENTS = [
  ["diffusion", "Diffusion model"],
  ["text_encoder", "Text encoder"],
  ["vae", "VAE"],
];

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

async function deployCall(action, body) {
  const response = await api.fetchApi(`/pom/deploy/${action}`, body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const answer = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(answer.error || `HTTP ${response.status}`);
  return answer;
}

/** The choices of one component: each machine with ComfyUI and a file that fits. */
function choicesFor(component, machines) {
  const choices = [];
  for (const machine of machines) {
    if (!machine.has_image_service) continue;
    for (const file of machine.image_files || []) {
      if (file.category === component || file.category === "checkpoint") {
        choices.push({ node_id: machine.node_id, path: file.path, label: `${file.name} · ${machine.hostname}` });
      }
    }
  }
  return choices;
}

async function openDeployDialog() {
  const field = "width:100%;box-sizing:border-box;margin:4px 0 10px;padding:6px;border-radius:6px;" +
    "border:1px solid var(--border-color,#444);background:var(--comfy-input-bg,#111);color:inherit";
  const status = element("p", { style: "min-height:1.2em;margin:6px 0;color:var(--error-text,#f88)" });
  const deployed = element("div");
  const body = element("div", {}, "Loading the network's image files…");
  const close = () => overlay.remove();
  const box = element(
    "div",
    {
      role: "dialog",
      "aria-label": "Deploy on POM",
      style: "width:min(560px,92vw);max-height:86vh;overflow:auto;padding:16px 18px;border-radius:10px;" +
        "background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);font:13px/1.5 sans-serif",
    },
    element("h3", { style: "margin:0 0 8px" }, "Deploy an image model on POM"),
    element("p", { style: "margin:0 0 12px;opacity:.75" },
      "Each part runs on a machine of the network that already holds its file. Nothing is copied."),
    body,
    status,
    element("h4", { style: "margin:14px 0 6px" }, "Served image models"),
    deployed,
  );
  const overlay = element(
    "div",
    { style: "position:fixed;inset:0;z-index:10001;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.5)",
      onclick: (event) => { if (event.target === overlay) close(); } },
    box,
  );
  document.body.append(overlay);

  async function listDeployed() {
    try {
      const { deployments = [] } = await deployCall("deployments");
      deployed.replaceChildren(
        ...(deployments.length ? deployments.map((deployment) => element(
          "div",
          { style: "display:flex;justify-content:space-between;gap:8px;align-items:center;padding:6px 0;border-top:1px solid var(--border-color,#333)" },
          element("span", {}, `${deployment.name} - ${deployment.components.map((part) => `${part.component} on ${part.hostname} (${part.status})`).join(", ")}`),
          element("button", { type: "button", onclick: async () => {
            try {
              await deployCall("undeploy", { deploy_id: deployment.deploy_id });
              toast("success", "Undeployed", deployment.name);
              listDeployed();
            } catch (error) {
              status.textContent = error.message;
            }
          } }, "Undeploy"),
        )) : [element("p", { style: "opacity:.7;margin:0" }, "None yet.")]),
      );
    } catch (error) {
      deployed.replaceChildren(element("p", {}, error.message));
    }
  }

  let machines = [];
  try {
    ({ machines = [] } = await deployCall("files"));
  } catch (error) {
    body.replaceChildren(element("p", {}, error.message));
    listDeployed();
    return;
  }
  const name = element("input", { style: field, placeholder: "Z-Image Turbo", "aria-label": "Name" });
  const selects = COMPONENTS.map(([component, label]) => {
    const choices = choicesFor(component, machines);
    const select = element("select", { style: field, "aria-label": label },
      element("option", { value: "" }, choices.length ? "Not used" : "No machine with ComfyUI holds one"),
      ...choices.map((choice, index) => element("option", { value: String(index) }, choice.label)));
    return { component, label, select, choices };
  });
  const submit = element("button", { type: "button", style: "padding:6px 14px", onclick: async () => {
    const components = selects
      .filter(({ select }) => select.value !== "")
      .map(({ component, select, choices }) => {
        const { node_id, path } = choices[Number(select.value)];
        return { component, node_id, path };
      });
    status.textContent = "";
    submit.disabled = true;
    try {
      const { deploy_id } = await deployCall("deploy", { name: name.value.trim(), components });
      toast("success", "Deployed on POM", `${name.value.trim()} (${deploy_id})`);
      app.refreshComboInNodes?.();
      listDeployed();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  } }, "Deploy");
  body.replaceChildren(
    element("label", {}, "Name", name),
    ...selects.map(({ label, select }) => element("label", {}, label, select)),
    element("div", { style: "display:flex;gap:8px;justify-content:flex-end" },
      element("button", { type: "button", style: "padding:6px 14px", onclick: close }, "Close"),
      submit),
  );
  listDeployed();
}

// "Publish to Chat": the open workflow becomes a function of POM's Chat (and
// of its API). Every editable input of the graph is listed; each one checked
// becomes a parameter the Chat fills, with its type, bounds and options read
// from the node's widget. The node checks the role and the shape again.
function slugOf(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40) || "input";
}

function candidatesOf(output) {
  const found = [];
  for (const [id, node] of Object.entries(output || {})) {
    const widgets = app.graph?.getNodeById?.(Number(id))?.widgets || [];
    for (const [input, value] of Object.entries(node?.inputs || {})) {
      if (Array.isArray(value) || value === null || typeof value === "object") continue;
      const widget = widgets.find((w) => w.name === input);
      const options = widget?.options || {};
      const title = node._meta?.title || node.class_type;
      const param = { node: id, input, label: input === "text" ? title : `${title} ${input}`, default: value };
      if (node.class_type === "LoadImage" && input === "image") param.type = "image";
      else if (typeof value === "boolean") param.type = "boolean";
      else if (Array.isArray(options.values) && options.values.length) {
        param.type = "choice";
        param.options = options.values.slice(0, 200);
      } else if (typeof value === "number") {
        // ComfyUI's INT widgets have precision 0 (round 1); FLOAT ones a fraction.
        const whole = options.precision === 0 || options.round === 1 ||
          (options.precision === undefined && options.round === undefined && Number.isInteger(value));
        param.type = whole ? "integer" : "number";
        if (typeof options.min === "number") param.min = options.min;
        if (typeof options.max === "number") param.max = options.max;
      } else if (typeof value === "string") param.type = "text";
      else continue;
      // Text prompts and the image loaders are what a Chat usually fills.
      param.checked = param.type === "image" || (param.type === "text" && input === "text");
      found.push(param);
    }
  }
  return found;
}

async function openPublishDialog() {
  const field = "width:100%;box-sizing:border-box;margin:4px 0 10px;padding:6px;border-radius:6px;" +
    "border:1px solid var(--border-color,#444);background:var(--comfy-input-bg,#111);color:inherit";
  const small = "box-sizing:border-box;padding:3px 6px;border-radius:5px;" +
    "border:1px solid var(--border-color,#444);background:var(--comfy-input-bg,#111);color:inherit";
  const { output } = await app.graphToPrompt();
  const candidates = candidatesOf(output);
  const status = element("p", { style: "min-height:1.2em;margin:6px 0;color:var(--error-text,#f88)" });
  const name = element("input", { style: field, placeholder: "What this workflow makes" });
  const description = element("textarea", { style: field + ";min-height:52px", rows: "2",
    placeholder: "What it does and when to use it (the Chat's model reads this)" });
  const rows = candidates.map((param) => {
    const check = element("input", { type: "checkbox" });
    check.checked = param.checked;
    const label = element("input", { style: small + ";flex:1;min-width:0", value: param.label });
    const hint = element("input", { style: small + ";flex:1;min-width:0", placeholder: "Hint (optional)" });
    const row = element("div", { style: "display:flex;gap:6px;align-items:center;margin:3px 0" },
      check, label, hint,
      element("span", { style: "opacity:.7;font-size:11px;width:110px;text-align:right" },
        `#${param.node} ${param.type}`));
    return { param, check, label, hint, row };
  });
  const close = () => overlay.remove();
  const submit = element("button", { type: "button", style: "padding:6px 14px", onclick: async () => {
    status.textContent = "";
    const keys = new Set();
    const params = rows.filter((r) => r.check.checked).map(({ param, label, hint }) => {
      let key = param.type === "image" && !keys.has("image") ? "image" : slugOf(label.value);
      for (let n = 2; keys.has(key); n += 1) key = `${slugOf(label.value)}_${n}`;
      keys.add(key);
      const out = { key, label: label.value.trim() || param.label, node: param.node, input: param.input, type: param.type };
      if (hint.value.trim()) out.description = hint.value.trim();
      for (const field of ["min", "max", "options"]) if (param[field] !== undefined) out[field] = param[field];
      if (param.type !== "image") out.default = param.default;
      return out;
    });
    if (!params.length) {
      status.textContent = "Check at least one input the Chat fills.";
      return;
    }
    submit.disabled = true;
    try {
      const response = await api.fetchApi("/pom/workflow/publish", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: name.value.trim(), description: description.value.trim(), graph: output, params }),
      });
      const answer = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(answer.error || `HTTP ${response.status}`);
      toast("success", "Published to POM's Chat", answer.id);
      close();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  } }, "Publish");
  const box = element(
    "div",
    { role: "dialog", "aria-label": "Publish to Chat",
      style: "width:min(640px,94vw);max-height:90vh;overflow:auto;padding:16px 18px;border-radius:10px;" +
        "background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);font:13px/1.5 sans-serif" },
    element("h3", { style: "margin:0 0 8px" }, "Publish to POM's Chat"),
    candidates.length === 0
      ? element("p", {}, "This workflow has no input the Chat could fill.")
      : element("div", {},
          element("label", {}, "Name", name),
          element("label", {}, "Description", description),
          element("p", { style: "margin:4px 0" }, "Inputs the Chat fills (the others keep their values):"),
          ...rows.map((r) => r.row)),
    status,
    element("div", { style: "display:flex;gap:8px;justify-content:flex-end" },
      element("button", { type: "button", style: "padding:6px 14px", onclick: close }, "Close"),
      ...(candidates.length ? [submit] : [])),
  );
  const overlay = element(
    "div",
    { style: "position:fixed;inset:0;z-index:10001;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.5)",
      onclick: (event) => { if (event.target === overlay) close(); } },
    box,
  );
  document.body.append(overlay);
}

// Remove the published copy from Chat without changing the ComfyUI graph.
async function openUnpublishDialog() {
  const status = element("p", { style: "min-height:1.2em;margin:6px 0;color:var(--error-text,#f88)" });
  const list = element("div", {}, "Loading published workflows…");
  const close = () => overlay.remove();
  const box = element(
    "div",
    { role: "dialog", "aria-label": "Unpublish from Chat",
      style: "width:min(560px,92vw);max-height:86vh;overflow:auto;padding:16px 18px;border-radius:10px;" +
        "background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);font:13px/1.5 sans-serif" },
    element("h3", { style: "margin:0 0 8px" }, "Unpublish from Chat"),
    element("p", { style: "margin:0 0 12px;opacity:.75" },
      "This removes the published copy from POM Chat. The original workflow stays in ComfyUI."),
    list,
    status,
    element("div", { style: "display:flex;gap:8px;justify-content:flex-end" },
      element("button", { type: "button", onclick: close }, "Close")),
  );
  const overlay = element(
    "div",
    { style: "position:fixed;inset:0;z-index:10001;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.5)",
      onclick: (event) => { if (event.target === overlay) close(); } },
    box,
  );
  document.body.append(overlay);

  try {
    const response = await api.fetchApi("/pom/workflow/list");
    const answer = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(answer.error || `HTTP ${response.status}`);
    const workflows = Array.isArray(answer.workflows) ? answer.workflows : [];
    if (!workflows.length) {
      list.replaceChildren(element("p", { style: "margin:0;opacity:.7" }, "No workflows are published to Chat."));
      return;
    }
    list.replaceChildren(...workflows.map((workflow) => {
      const button = element("button", { type: "button", onclick: async () => {
        if (!window.confirm(`Unpublish "${workflow.name}" from POM Chat? The original ComfyUI workflow will stay unchanged.`)) return;
        status.textContent = "";
        button.disabled = true;
        try {
          const removed = await api.fetchApi("/pom/workflow/unpublish", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: workflow.id }),
          });
          const result = await removed.json().catch(() => ({}));
          if (!removed.ok) throw new Error(result.error || `HTTP ${removed.status}`);
          toast("success", "Unpublished from Chat", workflow.name);
          row.remove();
          if (!list.children.length) {
            list.replaceChildren(element("p", { style: "margin:0;opacity:.7" }, "No workflows are published to Chat."));
          }
        } catch (error) {
          status.textContent = error.message;
        } finally {
          button.disabled = false;
        }
      } }, "Unpublish");
      const row = element(
        "div",
        { style: "display:flex;justify-content:space-between;gap:8px;align-items:center;padding:8px 0;border-top:1px solid var(--border-color,#333)" },
        element("div", {},
          element("strong", {}, workflow.name || workflow.id),
          workflow.description ? element("div", { style: "opacity:.7" }, workflow.description) : document.createTextNode("")),
        button,
      );
      return row;
    }));
  } catch (error) {
    list.replaceChildren(element("p", { style: "margin:0" }, error.message));
  }
}

// Offered only when this node may deploy; the node checks again on every call.
let deployRole = { admin: false };
try {
  const response = await api.fetchApi("/pom/deploy/role");
  if (response.ok) deployRole = await response.json();
} catch {
  // An older node, or none: no deploy command.
}
if (deployRole.admin) {
  app.registerExtension({
    name: "pom.deploy",
    commands: [
      { id: "pom.deploy.open", label: "Deploy on POM", icon: "pi pi-cloud-upload", function: openDeployDialog },
      { id: "pom.workflow.publish", label: "Publish to Chat", icon: "pi pi-comments", function: openPublishDialog },
      { id: "pom.workflow.unpublish", label: "Unpublish from Chat", icon: "pi pi-trash", function: openUnpublishDialog },
    ],
    menuCommands: [{ path: ["POM"], commands: ["pom.deploy.open", "pom.workflow.publish", "pom.workflow.unpublish"] }],
  });
}
