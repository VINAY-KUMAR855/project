// ================== CONFIG ==================
/*
Set your prediction API endpoint here.
Expected request: multipart/form-data with field name "file".
Expected response JSON:
  { "breed": "Gir", "accuracy": 0.9532 }
Adjust if your keys differ (map below).
*/
const PREDICT_URL = "http://localhost:8000/predict"; // change if needed
const FIELD_NAME  = "file";          
const KEY_BREED   = "breed";         
const KEY_ACC     = "accuracy";      

// Optional: request timeout (ms)
const REQUEST_TIMEOUT = 30000;

// ================ ELEMENTS =================
const dropzone   = document.getElementById("dropzone");
const fileInput  = document.getElementById("file-input");
const preview    = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const clearBtn   = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const statusEl   = document.getElementById("status");
const resultEl   = document.getElementById("result");
const breedEl    = document.getElementById("breed");
const accEl      = document.getElementById("accuracy");

let selectedFile = null;

// ============== HELPERS ====================
function setStatus(msg) { statusEl.textContent = msg || ""; }
function showResult(breed, accuracy) {
  breedEl.textContent = breed;
  if (typeof accuracy === "number") {
    const pct = (accuracy <= 1 && accuracy >= 0) ? (accuracy * 100) : accuracy;
    accEl.textContent = `${pct.toFixed(2)}%`;
  } else {
    accEl.textContent = accuracy ?? "—";
  }
  resultEl.classList.remove("hidden");
}

function clearAll() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  preview.classList.add("hidden");
  predictBtn.disabled = true;
  resultEl.classList.add("hidden");
  setStatus("");
}

function isImage(file) {
  return file && file.type && file.type.startsWith("image/");
}

function pickFile(file) {
  if (!file) return;
  if (!isImage(file)) {
    setStatus("Please select an image file (jpg, png, etc.).");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    previewImg.src = reader.result;
    preview.classList.remove("hidden");
    predictBtn.disabled = false;
    resultEl.classList.add("hidden");
    setStatus("");
  };
  reader.readAsDataURL(file);

  // ✅ Reset input so user can reselect same file
  fileInput.value = "";
}

// ============== DROPZONE EVENTS =============
["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
    setStatus("Drop the image to select it.");
  });
});
["dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
    if (evt === "dragleave") setStatus("");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  pickFile(file);
});

// ⚠️ Removed dropzone click to avoid double file picker
// dropzone.addEventListener("click", () => fileInput.click());

// ============== FILE INPUT ==================
fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  pickFile(file);
});

// ============== CLEAR =======================
clearBtn.addEventListener("click", clearAll);

// ============== PREDICT =====================
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  setStatus("Uploading and predicting…");
  resultEl.classList.add("hidden");

  const form = new FormData();
  form.append(FIELD_NAME, selectedFile, selectedFile.name);

  const controller = new AbortController();
  const to = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  try {
    const res = await fetch(PREDICT_URL, {
      method: "POST",
      body: form,
      signal: controller.signal
    });
    clearTimeout(to);

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Server error ${res.status}: ${text}`);
    }

    const data = await res.json();
    const breed = data[KEY_BREED];
    const acc   = data[KEY_ACC];

    if (!breed || (acc === undefined || acc === null)) {
      throw new Error("Unexpected response. Please check JSON keys in app.js.");
    }

    showResult(String(breed), Number(acc));
    setStatus("Done.");
  } catch (err) {
    const msg = err?.name === "AbortError"
      ? "Request timed out. Try again."
      : (err?.message || "Something went wrong.");
    setStatus(msg);
    predictBtn.disabled = false;
  } finally {
    predictBtn.disabled = false;
  }
});

