let mode = "image";
let selectedFile = null;
let liveRunning = false;
let livePollId = null;
let lastBeepAt = 0;
let audioCtx = null;
let enableLive = true;

function getEl(id) {
    return document.getElementById(id);
}

function setStatus(msg) {
    const el = getEl("status");
    if (el) el.textContent = msg || "";
}

function stopLiveStream() {
    const liveOut = getEl("liveOut");
    if (liveOut) liveOut.src = "";
    liveRunning = false;

    const startBtn = getEl("liveDetectBtn");
    const stopBtn = getEl("stopLiveBtn");
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    if (livePollId) {
        clearInterval(livePollId);
        livePollId = null;
    }

    const alertBox = getEl("liveAlert");
    const alertText = getEl("liveAlertText");
    if (alertBox) alertBox.classList.remove("defect");
    if (alertText) alertText.textContent = "No defect";
}

function playDangerSound() {
    // Requires user gesture: startLive is triggered by a click.
    if (!audioCtx) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        audioCtx = new Ctx();
    }
    if (audioCtx.state === "suspended") {
        audioCtx.resume().catch(() => {});
    }

    const now = Date.now();
    if (now - lastBeepAt < 2500) return;
    lastBeepAt = now;

    // Siren-like sweep (no external audio assets): triangle wave + frequency ramps.
    const start = audioCtx.currentTime;
    const duration = 0.9;

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "triangle";

    // Fade in/out to avoid clicks.
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.22, start + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);

    // Frequency sweep pattern.
    const baseLow = 520;
    const baseHigh = 980;
    const steps = 6;
    for (let i = 0; i < steps; i++) {
        const t0 = start + (i * duration) / steps;
        const t1 = start + ((i + 1) * duration) / steps;
        const up = i % 2 === 0;
        osc.frequency.setValueAtTime(up ? baseLow : baseHigh, t0);
        osc.frequency.linearRampToValueAtTime(up ? baseHigh : baseLow, t1);
    }

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(start);
    osc.stop(start + duration + 0.02);
}

async function pollLiveStatus() {
    try {
        const res = await fetch(`/live-status?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        const isDefect = Boolean(data.defect);
        const alertBox = getEl("liveAlert");
        const alertText = getEl("liveAlertText");

        if (alertBox) alertBox.classList.toggle("defect", isDefect);
        if (alertText) {
            const score = typeof data.score === "number" ? data.score.toFixed(3) : "?";
            alertText.textContent = isDefect ? `DEFECT (score: ${score})` : "No defect";
        }

        if (isDefect) {
            playDangerSound();
        }
    } catch {
        // ignore
    }
}

function syncControls() {
    const fileControls = getEl("fileControls");
    const liveControls = getEl("liveControls");

    if (fileControls) fileControls.style.display = mode === "live" ? "none" : "flex";
    if (liveControls) liveControls.style.display = mode === "live" ? "flex" : "none";
}

function setMode(nextMode) {
    const requested = nextMode === "video" ? "video" : nextMode === "live" ? "live" : "image";
    mode = requested === "live" && !enableLive ? "image" : requested;

    const imgOut = getEl("imgOut");
    const vidOut = getEl("vidOut");
    const liveOut = getEl("liveOut");
    const fileInput = getEl("file");

    setStatus("");
    stopLiveStream();

    if (imgOut) imgOut.style.display = "none";
    if (vidOut) vidOut.style.display = "none";
    if (liveOut) liveOut.style.display = "none";

    if (fileInput) {
        fileInput.value = "";
        selectedFile = null;
        fileInput.accept = mode === "video" ? "video/*" : "image/*";
    }

    const label = document.querySelector(".fileLabel");
    if (label) label.textContent = "Choose file";

    syncControls();

    const tImg = getEl("modeImage");
    const tVid = getEl("modeVideo");
    const tLiv = getEl("modeLive");
    if (tImg) tImg.classList.toggle("active", mode === "image");
    if (tVid) tVid.classList.toggle("active", mode === "video");
    if (tLiv) tLiv.classList.toggle("active", mode === "live");
}

async function detect() {
    if (mode === "live") {
        startLive();
        return;
    }

    const fileInput = getEl("file");
    const file = selectedFile || (fileInput && fileInput.files ? fileInput.files[0] : null);
    if (!file) {
        alert("Select a file first");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const url = mode === "video" ? "/upload-video" : "/upload-image";
    const res = await fetch(url, { method: "POST", body: formData });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
    }

    const defect = res.headers.get("X-Defect-Detected") === "1";
    const score = res.headers.get("X-Heatmap-Max");
    setStatus(defect ? `DEFECT (score: ${score ?? "?"})` : `No defect (score: ${score ?? "?"})`);

    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);

    const imgOut = getEl("imgOut");
    const vidOut = getEl("vidOut");
    const liveOut = getEl("liveOut");
    stopLiveStream();
    if (liveOut) liveOut.style.display = "none";

    if (mode === "video") {
        if (vidOut) {
            vidOut.src = objectUrl;
            vidOut.style.display = "block";
            vidOut.load();
        }
        if (imgOut) imgOut.style.display = "none";
    } else {
        if (imgOut) {
            imgOut.src = objectUrl;
            imgOut.style.display = "block";
        }
        if (vidOut) vidOut.style.display = "none";
    }
}

function startLive() {
    if (mode !== "live") return;
    if (!enableLive) {
        setStatus("Live camera is not available on this deployment.");
        return;
    }

    const liveOut = getEl("liveOut");
    const imgOut = getEl("imgOut");
    const vidOut = getEl("vidOut");
    const startBtn = getEl("liveDetectBtn");
    const stopBtn = getEl("stopLiveBtn");

    if (imgOut) imgOut.style.display = "none";
    if (vidOut) vidOut.style.display = "none";
    if (liveOut) {
        liveOut.style.display = "block";
        // Cache-bust to avoid stale connections.
        liveOut.src = `/live?t=${Date.now()}`;
    }

    liveRunning = true;
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    setStatus("Live camera started");

    if (!livePollId) {
        pollLiveStatus();
        livePollId = setInterval(pollLiveStatus, 500);
    }
}

function stopLive() {
    if (mode !== "live") return;
    stopLiveStream();
    const liveOut = getEl("liveOut");
    if (liveOut) liveOut.style.display = "none";
    setStatus("Live camera stopped");
}

// Initialize defaults
setMode("image");

window.addEventListener("DOMContentLoaded", () => {
    // Enable/disable Live Cam for cloud deployments.
    try {
        const v = document.body && document.body.dataset ? document.body.dataset.enableLive : null;
        enableLive = v !== "0";
    } catch {
        enableLive = true;
    }

    const liveTab = getEl("modeLive");
    if (liveTab) {
        liveTab.style.display = enableLive ? "" : "none";
    }

    const fileInput = getEl("file");
    if (fileInput) {
        fileInput.addEventListener("change", () => {
            selectedFile = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
            const label = document.querySelector(".fileLabel");
            if (label) label.textContent = selectedFile ? selectedFile.name : "Choose file";
        });
    }
});
