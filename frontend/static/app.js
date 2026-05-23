import { SplatViewer } from "./viewer.js";

const form = document.querySelector("#jobForm");
const submitButton = document.querySelector("#submitButton");
const modelSelect = document.querySelector("#model");
const jobsEl = document.querySelector("#jobs");
const refreshButton = document.querySelector("#refreshJobs");
const activeTitle = document.querySelector("#activeTitle");
const activeMeta = document.querySelector("#activeMeta");
const logsEl = document.querySelector("#logs");
const videoPreview = document.querySelector("#videoPreview");
const downloadVideo = document.querySelector("#downloadVideo");
const downloadPly = document.querySelector("#downloadPly");
const viewerEmpty = document.querySelector("#viewerEmpty");

let jobs = [];
let activeJobId = null;
let viewer = null;
let loadedPlyUrl = null;

const statusText = {
  queued: "排队",
  running: "运行",
  succeeded: "完成",
  failed: "失败",
  interrupted: "中断",
};

function setDownload(link, url) {
  if (url) {
    link.href = url;
    link.classList.remove("disabled");
  } else {
    link.href = "#";
    link.classList.add("disabled");
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
      <div class="job-meta">${job.model_id} · ${job.sample_fps} FPS · ${job.uploads.length} video</div>
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
  const response = await fetch("/api/jobs");
  const data = await response.json();
  jobs = data.jobs;
  if (!activeJobId && jobs.length > 0) {
    activeJobId = jobs[0].id;
  }
  renderJobs();
  if (activeJobId) {
    await refreshActiveJob();
  }
}

async function selectJob(jobId) {
  activeJobId = jobId;
  loadedPlyUrl = null;
  renderJobs();
  await refreshActiveJob();
}

async function refreshActiveJob() {
  const job = jobs.find((item) => item.id === activeJobId);
  if (!job) {
    return;
  }

  activeTitle.textContent = job.id;
  activeMeta.textContent = `${statusText[job.status] ?? job.status} · ${job.model_id}`;
  setDownload(downloadVideo, job.artifacts.video);
  setDownload(downloadPly, job.artifacts.ply);

  if (job.artifacts.video) {
    videoPreview.src = job.artifacts.video;
  } else {
    videoPreview.removeAttribute("src");
    videoPreview.load();
  }

  const logsResponse = await fetch(`/api/jobs/${job.id}/logs`);
  logsEl.textContent = await logsResponse.text();
  logsEl.scrollTop = logsEl.scrollHeight;

  if (job.artifacts.ply && loadedPlyUrl !== job.artifacts.ply) {
    viewer ??= new SplatViewer(document.querySelector("#viewer"));
    viewerEmpty.classList.add("hidden");
    loadedPlyUrl = job.artifacts.ply;
    try {
      await viewer.load(job.artifacts.ply);
    } catch (error) {
      logsEl.textContent += `\n[viewer] ${error.message}\n`;
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "启动中";

  const formData = new FormData(form);
  for (const field of ["start_time", "end_time"]) {
    if (!formData.get(field)) {
      formData.delete(field);
    }
  }

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Job creation failed");
    }
    activeJobId = data.job.id;
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

await loadModels();
await refreshJobs();
setInterval(refreshJobs, 3000);
