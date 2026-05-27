import { SplatViewer } from "./viewer.js?v=20260527-camera-up-v2";

const form = document.querySelector("#jobForm");
const submitButton = document.querySelector("#submitButton");
const datasetChunkSize = document.querySelector("#datasetChunkSize");
const colmapDatasetSelect = document.querySelector("#colmapDataset");
const batchMerge = document.querySelector("#batchMerge");
const modelSelect = document.querySelector("#model");
const jobsEl = document.querySelector("#jobs");
const refreshButton = document.querySelector("#refreshJobs");
const cpuMetric = document.querySelector("#cpuMetric");
const gpuMetric = document.querySelector("#gpuMetric");
const activeTitle = document.querySelector("#activeTitle");
const activeMeta = document.querySelector("#activeMeta");
const pipelineToolbar = document.querySelector("#pipelineToolbar");
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
const colmapWorkspace = document.querySelector("#colmapWorkspace");
const colmapForm = document.querySelector("#colmapForm");
const colmapFilesInput = document.querySelector("#colmapFiles");
const colmapVideoMode = document.querySelector("#colmapVideoMode");
const colmapUserVideoField = document.querySelector("#colmapUserVideoField");
const colmapUserVideo = document.querySelector("#colmapUserVideo");
const colmapButton = document.querySelector("#colmapButton");
const colmapUploadProgress = document.querySelector("#colmapUploadProgress");
const colmapJobsEl = document.querySelector("#colmapJobs");
const refreshColmapButton = document.querySelector("#refreshColmapJobs");
const refreshColmapPreview = document.querySelector("#refreshColmapPreview");
const colmapTitle = document.querySelector("#colmapTitle");
const colmapMeta = document.querySelector("#colmapMeta");
const candidateGallery = document.querySelector("#candidateGallery");
const colmapLogs = document.querySelector("#colmapLogs");
const viewerResize = document.querySelector("#viewerResize");
const outputResize = document.querySelector("#outputResize");

let jobs = [];
let colmapJobs = [];
let colmapDatasets = [];
let userVideos = [];
let activeJobId = null;
let activeBatchIndex = null;
let activeColmapJobId = null;
let viewer = null;
let loadedPlyUrl = null;
let loadingPlyUrl = null;
let loadedCameraKey = null;
let viewerLoadToken = 0;
let activeColmapRenderKey = null;
let activeTab = "colmap";
let activePanoramaId = null;
let autoSelectJob = true;
const validTabs = new Set(["colmap", "pipeline", "panorama"]);
const UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024;

const statusText = {
  queued: "排队",
  running: "运行",
  proposed: "待选择",
  aligning: "保存中",
  ready: "可用",
  succeeded: "完成",
  failed: "失败",
  interrupted: "中断",
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function applySavedSplits() {
  const logWidth = localStorage.getItem("resplat.pipelineLogWidth");
  const videoHeight = localStorage.getItem("resplat.pipelineVideoHeight");
  localStorage.removeItem("resplat.viewerHeight");
  localStorage.removeItem("resplat.viewerHeight.v2");
  localStorage.removeItem("resplat.viewerHeight.v3");
  localStorage.removeItem("resplat.previewWidth");
  if (logWidth) {
    pipelineWorkspace.style.setProperty("--log-width", logWidth);
  }
  if (videoHeight) {
    pipelineWorkspace.style.setProperty("--video-height", videoHeight);
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
    const preview = pipelineWorkspace.querySelector(".pipeline-preview");
    const rect = preview.getBoundingClientRect();
    const maxHeight = Math.max(140, rect.height - 240);
    const height = clamp(rect.bottom - event.clientY, 120, maxHeight);
    const value = `${Math.round(height)}px`;
    pipelineWorkspace.style.setProperty("--video-height", value);
    localStorage.setItem("resplat.pipelineVideoHeight", value);
  });

  installDragSplit(outputResize, (event) => {
    if (window.matchMedia("(max-width: 900px)").matches) {
      return;
    }
    const rect = pipelineWorkspace.getBoundingClientRect();
    const width = clamp(rect.right - event.clientX, 320, rect.width - 520);
    const value = `${Math.round(width)}px`;
    pipelineWorkspace.style.setProperty("--log-width", value);
    localStorage.setItem("resplat.pipelineLogWidth", value);
  });
}

function setOutputCollapsed(collapsed) {
  pipelineWorkspace.classList.toggle("viewer-maximized", collapsed);
  toggleOutputPane.textContent = collapsed ? "显示 log" : "展开 3DGS";
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

function setVideoPreviewSource(url) {
  const currentUrl = videoPreview.getAttribute("src");
  if (url) {
    if (currentUrl !== url) {
      videoPreview.src = url;
    }
  } else if (currentUrl) {
    videoPreview.removeAttribute("src");
    videoPreview.load();
  }
}

function setVideoMode(modeSelect, fileInput, userField) {
  const useUpload = modeSelect.value === "upload";
  fileInput.closest("label").hidden = !useUpload;
  fileInput.required = useUpload;
  userField.hidden = useUpload;
}

function updateVideoModes() {
  setVideoMode(colmapVideoMode, colmapFilesInput, colmapUserVideoField);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatUtilization(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(0)}%`;
}

function formatGpuMetric(gpu) {
  if (!gpu?.available) {
    return "--";
  }
  const utilization = Number(gpu.utilization);
  if (!Number.isNaN(utilization) && utilization > 0) {
    return `${utilization.toFixed(0)}%`;
  }
  const memoryPercent = Number(gpu.memory?.percent);
  if (!Number.isNaN(memoryPercent) && memoryPercent > 0) {
    return `显存${memoryPercent.toFixed(0)}%`;
  }
  return formatUtilization(gpu.utilization);
}

function formatGpuMetricTitle(gpu) {
  if (!gpu?.available) {
    return "GPU metrics unavailable";
  }
  const parts = [];
  const utilization = Number(gpu.utilization);
  if (!Number.isNaN(utilization)) {
    parts.push(`compute ${utilization.toFixed(0)}%`);
  }
  const usedMb = Number(gpu.memory?.used_mb);
  const totalMb = Number(gpu.memory?.total_mb);
  if (!Number.isNaN(usedMb) && !Number.isNaN(totalMb) && totalMb > 0) {
    parts.push(`VRAM ${(usedMb / 1024).toFixed(1)} / ${(totalMb / 1024).toFixed(1)} GB`);
  }
  if (Number(gpu.process_count) > 0) {
    parts.push(`${gpu.process_count} compute process${Number(gpu.process_count) === 1 ? "" : "es"}`);
  }
  return parts.join(" · ") || "GPU idle";
}

async function refreshSystemMetrics() {
  try {
    const response = await fetch("/api/system-metrics", {cache: "no-store"});
    if (!response.ok) {
      throw new Error("metrics unavailable");
    }
    const metrics = await response.json();
    cpuMetric.textContent = formatUtilization(metrics.cpu?.utilization);
    gpuMetric.textContent = formatGpuMetric(metrics.gpu);
    gpuMetric.parentElement.title = formatGpuMetricTitle(metrics.gpu);
  } catch (error) {
    cpuMetric.textContent = "--";
    gpuMetric.textContent = "--";
    gpuMetric.parentElement.title = "GPU metrics unavailable";
  }
}

function renderUploadProgress(progressEl, files, activeIndex, activeBytes, committedBytes, totalBytes, label) {
  progressEl.hidden = false;
  const overall = totalBytes > 0 ? Math.min(100, ((committedBytes + activeBytes) / totalBytes) * 100) : 0;
  const rows = files.map((file, index) => {
    const doneBefore = index < activeIndex;
    const isActive = index === activeIndex;
    const pct = doneBefore ? 100 : (isActive ? Math.min(100, (activeBytes / file.size) * 100) : 0);
    return `
      <div class="upload-progress-row">
        <div class="upload-progress-text">${file.name} · ${pct.toFixed(0)}%</div>
        <div class="upload-progress-track"><div class="upload-progress-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  }).join("");
  progressEl.innerHTML = `
    <div class="upload-progress-row">
      <div class="upload-progress-text">${label} · ${overall.toFixed(0)}% · ${formatBytes(committedBytes + activeBytes)} / ${formatBytes(totalBytes)}</div>
      <div class="upload-progress-track"><div class="upload-progress-fill" style="width:${overall}%"></div></div>
    </div>
    ${rows}
  `;
}

async function uploadChunk(uploadId, index, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/uploads/${uploadId}/chunks?index=${index}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(xhr.responseText || `Chunk upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error("网络错误，分片上传失败"));
    xhr.send(blob);
  });
}

async function uploadFilesInChunks(files, progressEl, label = "上传中") {
  const fileList = Array.from(files);
  if (fileList.length === 0) {
    throw new Error("请至少选择一个视频");
  }

  const totalBytes = fileList.reduce((sum, file) => sum + file.size, 0);
  let committedBytes = 0;
  const uploadIds = [];

  for (let fileIndex = 0; fileIndex < fileList.length; fileIndex += 1) {
    const file = fileList[fileIndex];
    const totalChunks = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK_SIZE));
    const initResponse = await fetch("/api/uploads/init", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        filename: file.name,
        total_size: file.size,
        total_chunks: totalChunks,
      }),
    });
    const initData = await initResponse.json();
    if (!initResponse.ok) {
      throw new Error(initData.detail || "创建分片上传失败");
    }

    let fileCommitted = 0;
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += 1) {
      const start = chunkIndex * UPLOAD_CHUNK_SIZE;
      const end = Math.min(file.size, start + UPLOAD_CHUNK_SIZE);
      const blob = file.slice(start, end);
      await uploadChunk(initData.upload_id, chunkIndex, blob, (loaded) => {
        renderUploadProgress(
          progressEl,
          fileList,
          fileIndex,
          fileCommitted + loaded,
          committedBytes,
          totalBytes,
          label,
        );
      });
      fileCommitted += blob.size;
      renderUploadProgress(progressEl, fileList, fileIndex, fileCommitted, committedBytes, totalBytes, label);
    }

    const completeResponse = await fetch(`/api/uploads/${initData.upload_id}/complete`, {method: "POST"});
    const completeData = await completeResponse.json();
    if (!completeResponse.ok) {
      throw new Error(completeData.detail || "合并分片失败");
    }
    uploadIds.push(initData.upload_id);
    committedBytes += file.size;
    renderUploadProgress(progressEl, fileList, fileIndex + 1, 0, committedBytes, totalBytes, label);
  }

  return uploadIds;
}

function clearPipelineArtifacts(message = "") {
  loadedPlyUrl = null;
  loadingPlyUrl = null;
  loadedCameraKey = null;
  viewerLoadToken += 1;
  setDownload(downloadVideo, null);
  setDownload(downloadPly, null);
  setVideoPreviewSource(null);
  logsEl.textContent = message;
  viewer?.clear();
  viewerEmpty.textContent = "3DGS viewer";
  viewerEmpty.classList.remove("hidden");
}

function jobUploadCount(job) {
  return job.upload_count ?? job.uploads?.length ?? 0;
}

function jobSourceMeta(job) {
  if (job.source_type === "colmap") {
    const dataset = job.colmap_dataset;
    const mode = job.resplat_mode === "batch_merge" ? "分批合并" : "单次";
    const batches = job.resplat_mode === "batch_merge" && job.batch_count
      ? ` · ${job.batch_count} batches`
      : "";
    const fps = dataset?.sample_fps ?? job.sample_fps;
    const fpsText = fps ? ` · ${fps} FPS` : "";
    return `COLMAP · ${mode} · ${dataset?.frame_count ?? 0} images${fpsText}${batches}`;
  }
  return `${job.sample_fps} FPS · ${jobUploadCount(job)} video`;
}

function batchMeta(batch) {
  const start = Number(batch.start ?? 0);
  const end = Number(batch.end ?? 0);
  const coreStart = Number(batch.core_start ?? start);
  const coreEnd = Number(batch.core_end ?? end);
  const count = Number(batch.kept_gaussians ?? 0);
  const countText = count > 0 ? `${(count / 1000).toFixed(0)}k` : "0";
  const artifacts = [
    batch.artifacts?.viewer_ply || batch.artifacts?.ply ? "3DGS" : null,
    batch.artifacts?.video ? "video" : null,
  ].filter(Boolean).join(" + ");
  const artifactText = artifacts || "结果生成中";
  return `${artifactText} · frames ${start}..${Math.max(start, end - 1)} · core ${coreStart}..${Math.max(coreStart, coreEnd - 1)} · ${countText} GS`;
}

function activeJobSelection(job) {
  if (activeBatchIndex === null || activeBatchIndex === undefined) {
    return {
      title: job.id,
      meta: `${statusText[job.status] ?? job.status} · ${jobSourceMeta(job)}`,
      artifacts: job.artifacts,
    };
  }
  const batch = (job.batches ?? []).find((item) => Number(item.index) === Number(activeBatchIndex));
  if (!batch) {
    activeBatchIndex = null;
    return {
      title: job.id,
      meta: `${statusText[job.status] ?? job.status} · ${jobSourceMeta(job)}`,
      artifacts: job.artifacts,
    };
  }
  return {
    title: `${job.id} · ${batch.label}`,
    meta: batchMeta(batch),
    artifacts: batch.artifacts,
  };
}

function buildJobFormData(uploadIds = []) {
  const formData = new FormData();
  if (uploadIds.length > 0) {
    throw new Error("视频 pipeline 只能使用已保存的 COLMAP dataset");
  }
  if (!colmapDatasetSelect.value) {
    throw new Error("请先在 COLMAP tab 保存一个 dataset");
  }
  formData.set("source_type", "colmap");
  formData.set("dataset_id", colmapDatasetSelect.value);
  formData.set("render_chunk_size", datasetChunkSize.value);
  formData.set("resplat_mode", batchMerge.checked ? "batch_merge" : "single");
  formData.set("model_id", modelSelect.value);
  return formData;
}

function setTab(tab) {
  if (!validTabs.has(tab)) {
    tab = "colmap";
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
  const panoramaSelected = tab === "panorama";
  const colmapSelected = tab === "colmap";
  pipelineWorkspace.classList.toggle("active", pipelineSelected);
  panoramaWorkspace.classList.toggle("active", panoramaSelected);
  colmapWorkspace.classList.toggle("active", colmapSelected);
  pipelineWorkspace.hidden = !pipelineSelected;
  panoramaWorkspace.hidden = !panoramaSelected;
  colmapWorkspace.hidden = !colmapSelected;
  pipelineActions.hidden = !pipelineSelected;
  pipelineToolbar.hidden = !pipelineSelected;

  const desiredUrl = `/?tool=${tab}`;
  if (window.location.pathname === "/" && window.location.search !== `?tool=${tab}`) {
    history.replaceState(null, "", desiredUrl);
  }

  if (pipelineSelected) {
    activeTitle.textContent = activeJobId || "等待任务";
    activeMeta.textContent = activeJobId ? "任务详情" : "上传视频后会在这里显示产物";
    refreshActiveJob();
  } else if (panoramaSelected) {
    activeTitle.textContent = "全景切图";
    activeMeta.textContent = "60 度水平视角，起点每 20 度生成一张";
  } else {
    activeTitle.textContent = "COLMAP";
    activeMeta.textContent = "生成地面候选并保存为 ReSplat dataset";
    refreshActiveColmapJob();
  }
}

function renderJobs() {
  jobsEl.innerHTML = "";
  if (jobs.length === 0) {
    jobsEl.innerHTML = '<div class="job-meta">暂无任务</div>';
    return;
  }

  for (const job of jobs) {
    const wrapper = document.createElement("div");
    wrapper.className = "job-group";
    const card = document.createElement("button");
    card.type = "button";
    card.className = `job-card ${job.id === activeJobId && activeBatchIndex === null ? "active" : ""}`;
    card.innerHTML = `
      <div class="job-top">
        <div class="job-id">${job.id}</div>
        <span class="status ${job.status}">${statusText[job.status] ?? job.status}</span>
      </div>
      <div class="job-meta">${jobSourceMeta(job)}</div>
    `;
    card.addEventListener("click", () => selectJob(job.id));
    wrapper.appendChild(card);

    if (job.id === activeJobId && job.batches?.length) {
      const batches = document.createElement("div");
      batches.className = "batch-list";

      if (job.artifacts?.viewer_ply || job.artifacts?.ply || job.artifacts?.video) {
        const merged = document.createElement("button");
        merged.type = "button";
        merged.className = `batch-card ${activeBatchIndex === null ? "active" : ""}`;
        merged.innerHTML = `
          <span>合并结果</span>
          <small>${job.batch_manifest?.merged_gaussians ? `${(job.batch_manifest.merged_gaussians / 1000000).toFixed(2)}M GS` : "merged PLY"}</small>
        `;
        merged.addEventListener("click", () => selectJob(job.id));
        batches.appendChild(merged);
      }

      for (const batch of job.batches) {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `batch-card ${Number(activeBatchIndex) === Number(batch.index) ? "active" : ""}`;
        item.innerHTML = `
          <span>${batch.label}</span>
          <small>${batchMeta(batch)}</small>
        `;
        item.addEventListener("click", () => selectBatch(job.id, batch.index));
        batches.appendChild(item);
      }
      wrapper.appendChild(batches);
    }
    jobsEl.appendChild(wrapper);
  }
}

function renderColmapJobs() {
  colmapJobsEl.innerHTML = "";
  if (colmapJobs.length === 0) {
    colmapJobsEl.innerHTML = '<div class="job-meta">暂无 COLMAP 任务</div>';
    return;
  }

  for (const job of colmapJobs) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `job-card ${job.id === activeColmapJobId ? "active" : ""}`;
    const candidateText = job.selected_candidate_id ?? "未选";
    card.innerHTML = `
      <div class="job-top">
        <div class="job-id">${job.id}</div>
        <span class="status ${job.status}">${statusText[job.status] ?? job.status}</span>
      </div>
      <div class="job-meta">${job.sample_fps} FPS · ${job.upload_count ?? 0} video · candidate ${candidateText}</div>
    `;
    card.addEventListener("click", () => selectColmapJob(job.id));
    colmapJobsEl.appendChild(card);
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  modelSelect.innerHTML = "";
  let preferredOption = null;
  let firstEnabledOption = null;
  for (const model of data.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.label} · ${model.resolution}`;
    option.disabled = !model.checkpoint_exists;
    modelSelect.appendChild(option);
    if (model.checkpoint_exists && !firstEnabledOption) {
      firstEnabledOption = option;
    }
    if (model.checkpoint_exists && model.id === "resplat-base-dl3dv-540x960-view16") {
      preferredOption = option;
    }
  }
  if (preferredOption || firstEnabledOption) {
    modelSelect.value = (preferredOption || firstEnabledOption).value;
  }
}

async function loadColmapDatasets() {
  const current = colmapDatasetSelect.value;
  const response = await fetch("/api/colmap-datasets", { cache: "no-store" });
  const data = await response.json();
  colmapDatasets = data.datasets ?? [];
  colmapDatasetSelect.innerHTML = "";
  if (colmapDatasets.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "暂无已保存 dataset";
    colmapDatasetSelect.appendChild(option);
    return;
  }
  for (const dataset of colmapDatasets) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.label} · ${dataset.frame_count} images`;
    colmapDatasetSelect.appendChild(option);
  }
  if (current && colmapDatasets.some((dataset) => dataset.id === current)) {
    colmapDatasetSelect.value = current;
  }
}

function renderUserVideoOptions(selectEl, currentValue) {
  selectEl.innerHTML = "";
  if (userVideos.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "users 目录下暂无视频";
    selectEl.appendChild(option);
    return;
  }
  for (const video of userVideos) {
    const option = document.createElement("option");
    option.value = video.path;
    option.textContent = `${video.path} · ${formatBytes(video.size)}`;
    selectEl.appendChild(option);
  }
  if (currentValue && userVideos.some((video) => video.path === currentValue)) {
    selectEl.value = currentValue;
  }
}

async function loadUserVideos() {
  const previousColmap = colmapUserVideo.value;
  const response = await fetch("/api/user-videos", {cache: "no-store"});
  const data = await response.json();
  userVideos = data.videos ?? [];
  renderUserVideoOptions(colmapUserVideo, previousColmap);
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
    activeBatchIndex = null;
    loadingPlyUrl = null;
    loadedPlyUrl = null;
    loadedCameraKey = null;
    viewerLoadToken += 1;
  }
  renderJobs();
  if (activeJobId) {
    await refreshActiveJob();
  }
}

async function refreshColmapJobs() {
  const response = await fetch("/api/colmap-jobs", { cache: "no-store" });
  const data = await response.json();
  const previousActive = activeColmapJobId;
  colmapJobs = data.jobs ?? [];
  if (!activeColmapJobId && colmapJobs.length > 0) {
    activeColmapJobId = colmapJobs[0].id;
    activeColmapRenderKey = null;
  } else if (previousActive && !colmapJobs.some((job) => job.id === previousActive)) {
    activeColmapJobId = colmapJobs[0]?.id ?? null;
    activeColmapRenderKey = null;
  }
  renderColmapJobs();
  await loadColmapDatasets();
  if (activeColmapJobId) {
    await refreshActiveColmapJob();
  }
}

async function selectJob(jobId) {
  autoSelectJob = true;
  activeJobId = jobId;
  activeBatchIndex = null;
  loadingPlyUrl = null;
  loadedPlyUrl = null;
  loadedCameraKey = null;
  viewerLoadToken += 1;
  viewer?.clear();
  renderJobs();
  await refreshActiveJob();
}

async function selectBatch(jobId, batchIndex) {
  autoSelectJob = true;
  activeJobId = jobId;
  activeBatchIndex = Number(batchIndex);
  loadingPlyUrl = null;
  loadedPlyUrl = null;
  loadedCameraKey = null;
  viewerLoadToken += 1;
  viewer?.clear();
  renderJobs();
  await refreshActiveJob();
}

async function selectColmapJob(jobId) {
  activeColmapJobId = jobId;
  activeColmapRenderKey = null;
  renderColmapJobs();
  await refreshActiveColmapJob();
}

function colmapRenderKey(job) {
  return JSON.stringify({
    id: job.id,
    status: job.status,
    selected: job.selected_candidate_id ?? null,
    overview: job.overview_url ?? null,
    candidates: (job.candidates ?? []).map((candidate) => [
      candidate.id,
      candidate.image_url,
      candidate.inliers,
      candidate.inlier_ratio,
      candidate.rms_distance,
      candidate.angle_to_camera_up_deg,
      candidate.camera_yz_track_angle_deg,
    ]),
    debug_reprojections: (job.debug_reprojections ?? []).map((item) => [
      item.image_url,
      item.source_image,
      item.projected_count,
    ]),
  });
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
  const selection = activeJobSelection(job);
  activeTitle.textContent = selection.title;
  activeMeta.textContent = selection.meta;
  setDownload(downloadVideo, selection.artifacts.video);
  setDownload(downloadPly, selection.artifacts.ply);

  setVideoPreviewSource(selection.artifacts.video);

  const logsResponse = await fetch(`/api/jobs/${job.id}/logs`, { cache: "no-store" });
  logsEl.textContent = await logsResponse.text();
  logsEl.scrollTop = logsEl.scrollHeight;

  const viewerPly = selection.artifacts.viewer_ply ?? selection.artifacts.ply;
  const initialCamera = selection.artifacts.initial_camera ?? null;
  const initialCameraKey = initialCamera ? JSON.stringify(initialCamera) : null;
  if (viewerPly && loadedPlyUrl !== viewerPly && loadingPlyUrl !== viewerPly) {
    viewer ??= new SplatViewer(document.querySelector("#viewer"));
    const loadToken = ++viewerLoadToken;
    loadingPlyUrl = viewerPly;
    viewerEmpty.textContent = "正在加载 3DGS...";
    viewerEmpty.classList.remove("hidden");
    try {
      const loaded = await viewer.load(viewerPly, { initialCamera });
      if (loadToken === viewerLoadToken && loaded) {
        loadedPlyUrl = viewerPly;
        loadedCameraKey = initialCameraKey;
        viewerEmpty.classList.add("hidden");
      }
    } catch (error) {
      if (loadToken === viewerLoadToken) {
        loadedPlyUrl = null;
        loadedCameraKey = null;
        viewerEmpty.textContent = "3DGS 加载失败";
        viewerEmpty.classList.remove("hidden");
        logsEl.textContent += `\n[viewer] ${error.message}\n`;
      }
    } finally {
      if (loadToken === viewerLoadToken) {
        loadingPlyUrl = null;
      }
    }
  } else if (viewerPly && viewer && initialCameraKey && loadedCameraKey !== initialCameraKey) {
    if (viewer.applyInitialCamera(initialCamera)) {
      loadedCameraKey = initialCameraKey;
    }
  } else if (!viewerPly) {
    viewer?.clear();
    loadingPlyUrl = null;
    loadedPlyUrl = null;
    loadedCameraKey = null;
    viewerLoadToken += 1;
    viewerEmpty.textContent = "暂无 3DGS";
    viewerEmpty.classList.remove("hidden");
  }
}

function renderCandidates(job) {
  candidateGallery.innerHTML = "";
  if (job.overview_url) {
    const overview = document.createElement("div");
    overview.className = "candidate-card";
    overview.innerHTML = `
      <img src="${job.overview_url}" alt="overview" loading="lazy" />
      <div class="job-meta">overview</div>
    `;
    candidateGallery.appendChild(overview);
  }

  if (!job.candidates?.length) {
    candidateGallery.innerHTML = '<div class="empty-inline">等待地面候选图</div>';
    return;
  }

  for (const candidate of job.candidates) {
    const card = document.createElement("div");
    card.className = `candidate-card ${candidate.id === job.selected_candidate_id ? "selected" : ""}`;
    const ratio = Number(candidate.inlier_ratio ?? 0).toFixed(3);
    const rms = Number(candidate.rms_distance ?? 0).toFixed(4);
    const angle = candidate.angle_to_camera_up_deg === null || candidate.angle_to_camera_up_deg === undefined
      ? "n/a"
      : `${Number(candidate.angle_to_camera_up_deg).toFixed(1)}°`;
    const track = candidate.camera_yz_track_angle_deg === null || candidate.camera_yz_track_angle_deg === undefined
      ? "n/a"
      : `${Number(candidate.camera_yz_track_angle_deg).toFixed(1)}°`;
    card.innerHTML = `
      <img src="${candidate.image_url}" alt="candidate ${candidate.id}" loading="lazy" />
      <div class="job-top">
        <div class="job-id">candidate ${candidate.id}</div>
        <span class="status ${candidate.id === job.selected_candidate_id ? "succeeded" : "queued"}">${candidate.id === job.selected_candidate_id ? "已保存" : "候选"}</span>
      </div>
      <div class="job-meta">inliers ${candidate.inliers} · ratio ${ratio} · rms ${rms} · angle ${angle} · track ${track}</div>
      <button type="button">保存为 dataset</button>
    `;
    const button = card.querySelector("button");
    button.disabled = job.status === "running" || job.status === "aligning";
    button.addEventListener("click", () => alignCandidate(job.id, candidate.id));
    candidateGallery.appendChild(card);
  }

  if (job.debug_reprojections?.length) {
    for (const item of job.debug_reprojections) {
      const card = document.createElement("div");
      card.className = "candidate-card reprojection-card";
      const count = Number(item.projected_count ?? 0);
      card.innerHTML = `
        <img src="${item.image_url}" alt="reprojection ${item.source_image ?? ""}" loading="lazy" />
        <div class="job-top">
          <div class="job-id">重投影 debug</div>
          <span class="status succeeded">已生成</span>
        </div>
        <div class="job-meta">${item.source_image ?? item.image} · ${count.toLocaleString()} projected points</div>
      `;
      candidateGallery.appendChild(card);
    }
  }
}

async function refreshActiveColmapJob() {
  if (activeTab !== "colmap") {
    return;
  }
  const job = colmapJobs.find((item) => item.id === activeColmapJobId);
  if (!job) {
    colmapTitle.textContent = "Ground Candidates";
    colmapMeta.textContent = "等待 COLMAP 任务";
    candidateGallery.innerHTML = '<div class="empty-inline">等待 COLMAP 任务</div>';
    colmapLogs.textContent = "";
    activeColmapRenderKey = null;
    return;
  }

  colmapTitle.textContent = job.id;
  colmapMeta.textContent = `${statusText[job.status] ?? job.status} · ${job.frame_count ?? 0} images · ${job.proposal_count ?? job.candidates?.length ?? 0} candidates`;
  const nextRenderKey = colmapRenderKey(job);
  if (nextRenderKey !== activeColmapRenderKey) {
    renderCandidates(job);
    activeColmapRenderKey = nextRenderKey;
  }

  const logsResponse = await fetch(`/api/colmap-jobs/${job.id}/logs`, { cache: "no-store" });
  colmapLogs.textContent = await logsResponse.text();
  colmapLogs.scrollTop = colmapLogs.scrollHeight;
}

async function alignCandidate(jobId, candidateId) {
  const formData = new FormData();
  formData.set("candidate_id", candidateId);
  colmapLogs.textContent += `\n[frontend] 保存 candidate ${candidateId} 为 dataset...\n`;
  const response = await fetch(`/api/colmap-jobs/${jobId}/align`, {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) {
    colmapLogs.textContent += `\n[frontend] ${data.detail || "Alignment failed"}\n`;
    return;
  }
  activeColmapJobId = data.job.id;
  await refreshColmapJobs();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "启动中";
  autoSelectJob = false;
  activeJobId = null;
  activeBatchIndex = null;
  renderJobs();
  const sourceText = batchMerge.checked
    ? "正在用 COLMAP dataset 创建分批高清合并任务..."
    : "正在用 COLMAP dataset 创建 ReSplat 任务...";
  clearPipelineArtifacts(sourceText);

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
    activeBatchIndex = null;
    clearPipelineArtifacts(`任务 ${data.job.id} 已创建，等待结果...`);
    await refreshJobs();
  } catch (error) {
    logsEl.textContent = `[frontend] ${error.message}`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "启动 pipeline";
  }
});

refreshButton.addEventListener("click", refreshJobs);
refreshColmapButton.addEventListener("click", refreshColmapJobs);
refreshColmapPreview.addEventListener("click", refreshColmapJobs);
colmapVideoMode.addEventListener("change", updateVideoModes);
toggleOutputPane.addEventListener("click", () => {
  setOutputCollapsed(!pipelineWorkspace.classList.contains("viewer-maximized"));
});

colmapForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  colmapButton.disabled = true;
  colmapButton.textContent = "启动中";
  candidateGallery.innerHTML = '<div class="empty-inline">正在上传视频并创建 COLMAP 任务</div>';
  colmapLogs.textContent = "";

  try {
    const formData = new FormData(colmapForm);
    if (colmapVideoMode.value === "upload") {
      const uploadIds = await uploadFilesInChunks(
        Array.from(colmapFilesInput.files ?? []),
        colmapUploadProgress,
        "上传 COLMAP 视频",
      );
      formData.delete("files");
      for (const uploadId of uploadIds) {
        formData.append("upload_ids", uploadId);
      }
    } else {
      if (!colmapUserVideo.value) {
        throw new Error("请选择 users 目录下的视频");
      }
      formData.delete("files");
      formData.append("existing_videos", colmapUserVideo.value);
    }
    const response = await fetch("/api/colmap-jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "COLMAP job creation failed");
    }
    activeColmapJobId = data.job.id;
    colmapForm.reset();
    colmapUploadProgress.hidden = true;
    colmapUploadProgress.innerHTML = "";
    await refreshColmapJobs();
  } catch (error) {
    colmapLogs.textContent = `[frontend] ${error.message}`;
  } finally {
    colmapButton.disabled = false;
    colmapButton.textContent = "启动 COLMAP";
  }
});
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setTab(button.dataset.tab));
});
window.addEventListener("popstate", () => {
  const params = new URLSearchParams(window.location.search);
  setTab(params.get("tool") || document.body.dataset.tool || "colmap");
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
await loadColmapDatasets();
await loadUserVideos();
updateVideoModes();
installResizableLayout();
const params = new URLSearchParams(window.location.search);
setTab(params.get("tool") || document.body.dataset.tool || "colmap");
await refreshSystemMetrics();
await refreshJobs();
await refreshColmapJobs();
setInterval(refreshSystemMetrics, 3000);
setInterval(refreshJobs, 3000);
setInterval(refreshColmapJobs, 3000);
