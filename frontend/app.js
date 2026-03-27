/* ── State ──────────────────────────────────────────────────── */
const state = {
  file: null,
  selectedTargets: new Set(),
  jobId: null,
  pollTimer: null,
};

const TARGET_META = {
  vocals: { icon: "🎤", label: "Vocals" },
  drums:  { icon: "🥁", label: "Drums" },
  bass:   { icon: "🎸", label: "Bass" },
  other:  { icon: "🎹", label: "Other" },
};

/* ── DOM refs ───────────────────────────────────────────────── */
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const browseBtn     = document.getElementById("browse-btn");
const fileInfo      = document.getElementById("file-info");
const fileName      = document.getElementById("file-name");
const fileSize      = document.getElementById("file-size");
const clearFileBtn  = document.getElementById("clear-file-btn");
const targetCards   = document.querySelectorAll(".target-card");
const separateBtn   = document.getElementById("separate-btn");
const btnLabel      = document.getElementById("btn-label");
const btnSpinner    = document.getElementById("btn-spinner");
const progressSec   = document.getElementById("progress-section");
const progressFill  = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const resultsSec    = document.getElementById("results-section");
const resultsGrid   = document.getElementById("results-grid");
const resetBtn      = document.getElementById("reset-btn");
const errorBanner   = document.getElementById("error-banner");
const errorMessage  = document.getElementById("error-message");
const errorDismiss  = document.getElementById("error-dismiss");

/* ── Helpers ────────────────────────────────────────────────── */
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }

function showError(msg) {
  errorMessage.textContent = msg;
  show(errorBanner);
}

function hideError() {
  hide(errorBanner);
  errorMessage.textContent = "";
}

function updateSeparateBtn() {
  separateBtn.disabled = !(state.file && state.selectedTargets.size > 0);
}

/* ── File handling ──────────────────────────────────────────── */
const ALLOWED_EXTS = new Set([".wav", ".mp3", ".flac"]);
const ALLOWED_MIME = new Set(["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/x-flac"]);
const MAX_BYTES = 100 * 1024 * 1024;

function acceptFile(file) {
  hideError();
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTS.has(ext) && !ALLOWED_MIME.has(file.type)) {
    showError(`Unsupported file type "${ext}". Please use WAV, MP3, or FLAC.`);
    return;
  }
  if (file.size > MAX_BYTES) {
    showError(`File is too large (${formatBytes(file.size)}). Maximum is 100 MB.`);
    return;
  }
  state.file = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  show(fileInfo);
  dropZone.classList.add("has-file");
  updateSeparateBtn();
}

function clearFile() {
  state.file = null;
  fileInput.value = "";
  hide(fileInfo);
  dropZone.classList.remove("has-file");
  updateSeparateBtn();
}

/* ── Drag and drop ──────────────────────────────────────────── */
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
["dragleave", "dragend"].forEach((ev) =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove("drag-over"))
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) acceptFile(file);
});
dropZone.addEventListener("click", (e) => {
  if (e.target !== browseBtn) fileInput.click();
});
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) acceptFile(fileInput.files[0]);
});
clearFileBtn.addEventListener("click", clearFile);

/* ── Target selection ───────────────────────────────────────── */
targetCards.forEach((card) => {
  card.addEventListener("click", () => {
    const target = card.dataset.target;
    if (state.selectedTargets.has(target)) {
      state.selectedTargets.delete(target);
      card.classList.remove("selected");
    } else {
      state.selectedTargets.add(target);
      card.classList.add("selected");
    }
    updateSeparateBtn();
  });
});

/* ── Separate button ────────────────────────────────────────── */
separateBtn.addEventListener("click", submitJob);

async function submitJob() {
  hideError();
  btnLabel.textContent = "Uploading...";
  show(btnSpinner);
  separateBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("targets", JSON.stringify([...state.selectedTargets]));

  let jobId;
  try {
    const res = await fetch("/api/separate", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed.");
    jobId = data.job_id;
  } catch (err) {
    showError(err.message);
    btnLabel.textContent = "Separate";
    hide(btnSpinner);
    separateBtn.disabled = false;
    return;
  }

  state.jobId = jobId;
  btnLabel.textContent = "Separate";
  hide(btnSpinner);

  // Show progress UI
  show(progressSec);
  progressFill.style.width = "0%";
  progressLabel.textContent = "Starting...";

  startPolling(jobId);
}

/* ── Polling ────────────────────────────────────────────────── */
function startPolling(jobId) {
  state.pollTimer = setInterval(() => pollStatus(jobId), 2000);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function pollStatus(jobId) {
  let data;
  try {
    const res = await fetch(`/api/status/${jobId}`);
    data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Status check failed.");
  } catch (err) {
    stopPolling();
    hide(progressSec);
    showError(err.message);
    separateBtn.disabled = false;
    return;
  }

  if (data.status === "processing" && data.progress) {
    const { completed, total, current_target } = data.progress;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    progressFill.style.width = pct + "%";
    const targetLabel = current_target ? TARGET_META[current_target]?.label || current_target : "…";
    progressLabel.textContent = `Extracting ${targetLabel} (${completed + 1} of ${total})…`;
  } else if (data.status === "queued") {
    progressLabel.textContent = "Queued — waiting to start…";
  } else if (data.status === "done") {
    stopPolling();
    progressFill.style.width = "100%";
    progressLabel.textContent = "Done!";
    setTimeout(() => {
      hide(progressSec);
      renderResults(data.results, jobId);
    }, 600);
  } else if (data.status === "error") {
    stopPolling();
    hide(progressSec);
    showError(data.error || "Separation failed. Please try again.");
    separateBtn.disabled = false;
  }
}

/* ── Results ────────────────────────────────────────────────── */
function renderResults(results, jobId) {
  resultsGrid.innerHTML = "";
  results.forEach((r, i) => {
    const meta = TARGET_META[r.target] || { icon: "🎵", label: r.target };
    const card = document.createElement("div");
    card.className = "result-card";
    card.style.animationDelay = `${i * 80}ms`;
    card.innerHTML = `
      <div class="result-header">
        <span class="result-icon">${meta.icon}</span>
        <span class="result-name">${meta.label}</span>
      </div>
      <audio controls src="${r.download_url}" preload="metadata"></audio>
      <a class="download-btn" href="${r.download_url}" download="${r.target}.wav">
        <svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <path d="M10 3v10M5 13l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M3 18h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        Download WAV
      </a>
    `;
    resultsGrid.appendChild(card);
  });
  show(resultsSec);
}

/* ── Reset ──────────────────────────────────────────────────── */
resetBtn.addEventListener("click", resetToIdle);

function resetToIdle() {
  stopPolling();
  state.file = null;
  state.jobId = null;
  state.selectedTargets.clear();
  fileInput.value = "";

  hide(fileInfo);
  dropZone.classList.remove("has-file", "drag-over");

  targetCards.forEach((c) => c.classList.remove("selected"));

  hide(progressSec);
  hide(resultsSec);
  resultsGrid.innerHTML = "";
  hideError();

  progressFill.style.width = "0%";
  btnLabel.textContent = "Separate";
  hide(btnSpinner);
  separateBtn.disabled = true;
}

/* ── Error dismiss ──────────────────────────────────────────── */
errorDismiss.addEventListener("click", hideError);
