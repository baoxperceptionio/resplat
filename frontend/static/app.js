import { SplatViewer } from "./viewer.js?v=20260523-uploadfix2";

const form = document.querySelector("#jobForm");
const submitButton = document.querySelector("#submitButton");
const filesInput = document.querySelector("#files");
const modelSelect = document.querySelector("#model");
const jobsEl = document.querySelector("#jobs");
const refreshButton = document.querySelector("#refreshJobs");
const activeTitle = document.querySelector("#activeTitle");
const activeMeta = document.querySelector("#activeMeta");
const logsEl = document.querySelector("#logs");
const videoPreview = document.querySelector("#videoPreview");
const downloadVideo = document.querySelector("#downloadVideo");
const downloadPly = document.querySelector("#downloadPly");
const toggleOutputPane = document.querySelector("#toggleOutputPane");
const pipelineActions = document.querySelector(".actions");
const viewerEmpty = document.querySelector("#viewerEmpty");
const panoramaForm = document.querySelector("#panoramaForm");
const panoramaButton = document.querySelector("#panoramaButton");
const panoramaGallery = document.querySelector("#panoramaGallery");
const panoramaTitle = document.querySelector("#panoramaTitle");
const panoramaMeta = document.querySelector("#panoramaMeta");
const downloadPanoramaZip = document.querySelector("#downloadPanoramaZip");
const seedanceForm = document.querySelector("#seedanceForm");
const seedanceButton = document.querySelector("#seedanceButton");
const seedanceStatus = document.querySelector("#seedanceStatus");
const pipelineWorkspace = document.querySelector("#pipelineWorkspace");
const panoramaWorkspace = document.querySelector("#panoramaWorkspace");
const viewerResize = document.querySelector("#viewerResize");
const outputResize = document.querySelector("#outputResize");
const outputPane = document.querySelector(".output-pane");

let jobs = [];
let activeJobId = null;
let viewer = null;
let loadedPlyUrl = null;
let activeTab = "pipeline";
let activePanoramaId = null;
let autoSelectJob = true;
const validTabs = new Set(["pipeline", "panorama"]);

const statusText = {
  queued: "排队",
  running: "运行",
  succeeded: "完成",
  failed: "失败",
  interrupted: "中断",
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function applySavedSplits() {
  const viewerHeight = localStorage.getItem("resplat.viewerHeight.v3");
  const previewWidth = localStorage.getItem("resplat.previewWidth");
  if (viewerHeight && Number.parseInt(viewerHeight, 10) >= 560) {
    pipelineWorkspace.style.setProperty("--viewer-height", viewerHeight);
  } else {
    localStorage.removeItem("resplat.viewerHeight");
    localStorage.removeItem("resplat.viewerHeight.v2");
  }
  if (previewWidth) {
    pipelineWorkspace.style.setProperty("--preview-width", previewWidth);
  }
}

function installDragSplit(handle, onMove) {
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    handle.classList.add("active");
    document.body.classList.add("resizing");
    handle.setPointerCapture(event.pointerId);

    const move = (moveEvent) => {
      onMove(moveEvent);
      requestAnimationFrame(() => viewer?.resize());
    };
    const stop = () => {
      handle.classList.remove("active");
      document.body.classList.remove("resizing");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", stop);
      handle.removeEventListener("pointercancel", stop);
    };

    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  });
}

function installResizableLayout() {
  applySavedSplits();

  installDragSplit(viewerResize, (event) => {
    const rect = pipelineWorkspace.getBoundingClientRect();
    const height = clamp(event.clientY - rect.top, 560, rect.height - 80);
    const value = `${Math.round(height)}px`;
    pipelineWorkspace.style.setProperty("--viewer-height", value);
    localStorage.setItem("resplat.viewerHeight.v3", value);
  });

  installDragSplit(outputResize, (event) => {
    const rect = outputPane.getBoundingClientRect();
    if (window.matchMedia("(max-width: 900px)").matches) {
      return;
    }
    const width = clamp(event.clientX - rect.left, 220, rect.width - 320);
    const value = `${Math.round(width)}px`;
    pipelineWorkspace.style.setProperty("--preview-width", value);
    localStorage.setItem("resplat.previewWidth", value);
  });
}

function setOutputCollapsed(collapsed) {
  pipelineWorkspace.classList.toggle("viewer-maximized", collapsed);
  toggleOutputPane.textContent = collapsed ? "显示 video/log" : "展开 3DGS";
  requestAnimationFrame(() => viewer?.resize());
}

function setDownload(link, url) {
  if (url) {
    link.href = url;
    link.classList.remove("disabled");
  } else {
    link.href = "#";
    link.classList.add("disabled");
  }
}

function clearPipelineArtifacts(message = "") {
  loadedPlyUrl = null;
  setDownload(downloadVideo, null);
  setDownload(downloadPly, null);
  videoPreview.removeAttribute("src");
  videoPreview.load();
  logsEl.textContent = message;
  viewer?.clear();
  viewerEmpty.classList.remove("hidden");
}

function jobUploadCount(job) {
  return job.upload_count ?? job.uploads?.length ?? 0;
}

function buildJobFormData() {
  const selectedFiles = Array.from(filesInput.files ?? []);
  if (selectedFiles.length === 0) {
    throw new Error("请至少选择一个视频");
  }

  const formData = new FormData();
  for (const file of selectedFiles) {
    formData.append("files", file, file.name);
  }
  formData.set("model_id", modelSelect.value);
  formData.set("sample_fps", document.querySelector("#sampleFps").value);
  formData.set("render_chunk_size", document.querySelector("#chunkSize").value);

  const startTime = document.querySelector("#startTime").value;
  const endTime = document.querySelector("#endTime").value;
  if (startTime) {
    formData.set("start_time", startTime);
  }
  if (endTime) {
    formData.set("end_time", endTime);
  }
  return formData;
}

function setTab(tab) {
  if (!validTabs.has(tab)) {
    tab = "pipeline";
  }
  activeTab = tab;
  document.querySelectorAll(".tab-button").forEach((button) => {
    const selected = button.dataset.tab === tab;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const selected = panel.dataset.panel === tab;
    panel.classList.toggle("active", selected);
    panel.hidden = !selected;
  });
  const pipelineSelected = tab === "pipeline";
  pipelineWorkspace.classList.toggle("active", pipelineSelected);
  panoramaWorkspace.classList.toggle("active", !pipelineSelected);
  pipelineWorkspace.hidden = !pipelineSelected;
  panoramaWorkspace.hidden = pipelineSelected;
  pipelineActions.hidden = !pipelineSelected;

  const desiredUrl = `/?tool=${tab}`;
  if (window.location.pathname === "/" && window.location.search !== `?tool=${tab}`) {
    history.replaceState(null, "", desiredUrl);
  }

  if (pipelineSelected) {
    activeTitle.textContent = activeJobId || "等待任务";
    activeMeta.textContent = activeJobId ? "任务详情" : "上传视频后会在这里显示产物";
    refreshActiveJob();
  } else {
    activeTitle.textContent = "全景切图";
    activeMeta.textContent = "60 度水平视角，起点每 20 度生成一张";
  }
}

function renderJobs() {
  jobsEl.innerHTML = "";
  if (jobs.length === 0) {
    jobsEl.innerHTML = '<div class="job-meta">暂无任务</div>';
    return;
  }

  for (const job of jobs) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `job-card ${job.id === activeJobId ? "active" : ""}`;
    card.innerHTML = `
      <div class="job-top">
        <div class="job-id">${job.id}</div>
        <span class="status ${job.status}">${statusText[job.status] ?? job.status}</span>
      </div>
      <div class="job-meta">${job.sample_fps} FPS · ${jobUploadCount(job)} video</div>
    `;
    card.addEventListener("click", () => selectJob(job.id));
    jobsEl.appendChild(card);
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  modelSelect.innerHTML = "";
  for (const model of data.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.label} · ${model.resolution}`;
    option.disabled = !model.checkpoint_exists;
    modelSelect.appendChild(option);
  }
}

async function refreshJobs() {
  const response = await fetch("/api/jobs", { cache: "no-store" });
  const data = await response.json();
  const previousActive = activeJobId;
  jobs = data.jobs;
  if (!activeJobId && jobs.length > 0 && autoSelectJob) {
    activeJobId = jobs[0].id;
  } else if (previousActive && !jobs.some((job) => job.id === previousActive)) {
    activeJobId = autoSelectJob ? jobs[0]?.id ?? null : null;
    loadedPlyUrl = null;
  }
  renderJobs();
  if (activeJobId) {
    await refreshActiveJob();
  }
}

async function selectJob(jobId) {
  autoSelectJob = true;
  activeJobId = jobId;
  loadedPlyUrl = null;
  renderJobs();
  await refreshActiveJob();
}

async function refreshActiveJob() {
  if (activeTab !== "pipeline") {
    return;
  }
  const job = jobs.find((item) => item.id === activeJobId);
  if (!job) {
    return;
  }

  activeTitle.textContent = job.id;
  activeMeta.textContent = `${statusText[job.status] ?? job.status} · ${jobUploadCount(job)} video · ${job.sample_fps} FPS`;
  setDownload(downloadVideo, job.artifacts.video);
  setDownload(downloadPly, job.artifacts.ply);

  if (job.artifacts.video) {
    videoPreview.src = job.artifacts.video;
  } else {
    videoPreview.removeAttribute("src");
    videoPreview.load();
  }

  const logsResponse = await fetch(`/api/jobs/${job.id}/logs`, { cache: "no-store" });
  logsEl.textContent = await logsResponse.text();
  logsEl.scrollTop = logsEl.scrollHeight;

  if (job.artifacts.ply && loadedPlyUrl !== job.artifacts.ply) {
    viewer ??= new SplatViewer(document.querySelector("#viewer"));
    viewerEmpty.classList.add("hidden");
    try {
      await viewer.load(job.artifacts.ply);
      loadedPlyUrl = job.artifacts.ply;
    } catch (error) {
      logsEl.textContent += `\n[viewer] ${error.message}\n`;
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "启动中";
  autoSelectJob = false;
  activeJobId = null;
  renderJobs();
  clearPipelineArtifacts(`正在上传 ${filesInput.files?.length ?? 0} 个视频并创建新 pipeline 任务...`);

  try {
    const formData = buildJobFormData();
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Job creation failed");
    }
    autoSelectJob = true;
    activeJobId = data.job.id;
    clearPipelineArtifacts(`任务 ${data.job.id} 已创建，等待结果...`);
    form.reset();
    document.querySelector("#sampleFps").value = "4";
    document.querySelector("#chunkSize").value = "2";
    await refreshJobs();
  } catch (error) {
    logsEl.textContent = `[frontend] ${error.message}`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "启动 pipeline";
  }
});

refreshButton.addEventListener("click", refreshJobs);
toggleOutputPane.addEventListener("click", () => {
  setOutputCollapsed(!pipelineWorkspace.classList.contains("viewer-maximized"));
});
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setTab(button.dataset.tab));
});
window.addEventListener("popstate", () => {
  const params = new URLSearchParams(window.location.search);
  setTab(params.get("tool") || document.body.dataset.tool || "pipeline");
});

panoramaForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  panoramaButton.disabled = true;
  panoramaButton.textContent = "生成中";
  panoramaGallery.innerHTML = '<div class="empty-inline">正在投影生成 rectified views</div>';
  setDownload(downloadPanoramaZip, null);

  try {
    const response = await fetch("/api/panorama", {
      method: "POST",
      body: new FormData(panoramaForm),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Panorama conversion failed");
    }

    const panorama = data.panorama;
    activePanoramaId = panorama.id;
    panoramaTitle.textContent = panorama.source;
    panoramaMeta.textContent = `${panorama.files.length} 张 · ${panorama.output_size.width}x${panorama.output_size.height} · ${panorama.hfov_deg}° FOV`;
    setDownload(downloadPanoramaZip, panorama.zip);
    seedanceForm.hidden = false;
    seedanceStatus.textContent = `${panorama.files.length} 张 rectified 图片已准备好`;
    panoramaGallery.innerHTML = "";

    for (const file of panorama.files) {
      const card = document.createElement("div");
      card.className = "pano-card";
      card.innerHTML = `
        <img src="${file.url}" alt="${file.name}" loading="lazy" />
        <a href="${file.url}" download>${file.name}</a>
      `;
      panoramaGallery.appendChild(card);
    }
  } catch (error) {
    panoramaGallery.innerHTML = `<div class="empty-inline">${error.message}</div>`;
  } finally {
    panoramaButton.disabled = false;
    panoramaButton.textContent = "生成 rectified 图片";
  }
});

seedanceForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activePanoramaId) {
    seedanceStatus.textContent = "请先生成 rectified 图片";
    return;
  }

  seedanceButton.disabled = true;
  seedanceButton.textContent = "提交中";
  seedanceStatus.textContent = "正在逐张提交 Seedance 任务...";

  const formData = new FormData(seedanceForm);
  formData.set("public_base_url", window.location.origin);
  if (!formData.has("generate_audio")) {
    formData.set("generate_audio", "false");
  }

  try {
    const response = await fetch(`/api/panorama/${activePanoramaId}/seedance`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Seedance submission failed");
    }
    const results = data.seedance.results;
    const okCount = results.filter((item) => item.ok).length;
    seedanceStatus.textContent = `已提交 ${okCount}/${results.length} 个任务\n` +
      results.map((item) => `${item.ok ? "OK" : "ERR"} ${item.image}`).join("\n");
  } catch (error) {
    seedanceStatus.textContent = `[seedance] ${error.message}`;
  } finally {
    seedanceButton.disabled = false;
    seedanceButton.textContent = "对 rectified 图片运行 Seedance";
  }
});

await loadModels();
installResizableLayout();
const params = new URLSearchParams(window.location.search);
setTab(params.get("tool") || document.body.dataset.tool || "pipeline");
await refreshJobs();
setInterval(refreshJobs, 3000);
