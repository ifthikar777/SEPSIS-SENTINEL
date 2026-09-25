// SepsisGuard Dashboard Client Script
let staticSocket = null;
let liveSocket = null;
let isMuted = false;
let currentUtterance = null;

// Session & Usability Log State
// Session identifier uses the platform CSPRNG rather than Math.random(), which is
// not suitable for generating identifiers that end up in URLs and stored logs.
const sessionId = "sess_" + Array.from(crypto.getRandomValues(new Uint8Array(8)))
    .map(b => b.toString(16).padStart(2, "0")).join("");
let layoutOrder = ""; // Counterbalanced order: 3col_first or 1col_first
let currentLayout = "3-column"; // 3-column or 1-column
let taskStartTimestamp = Date.now();
let currentPatientId = "";
let currentSepsisRisk = 0.0;
let currentShapSnapshot = null;

// Decision thresholds. Overwritten by the server's patients_list message so the
// dashboard alerts at exactly the thresholds evaluation.py selected.
let ALERT_THRESHOLD = 0.30;
let HIGH_RISK_THRESHOLD = 0.70;

// Replay State
let isPlaying = false;
let liveTimelineData = [];

// Clinical Variables Metadata (Norm ranges)
const vitalsNorms = {
    HR: { low: 50, high: 100 },
    O2Sat: { low: 92, high: 100 },
    Temp: { low: 36.0, high: 38.0 },
    SBP: { low: 100, high: 180 },
    MAP: { low: 65, high: 110 },
    Resp: { low: 8, high: 22 }
};

// DOM Elements
const patientSelect = document.getElementById("patient-select");
const patientSearch = document.getElementById("patient-search");
const patientCountHint = document.getElementById("patient-count-hint");
const sampleDataBadge = document.getElementById("sample-data-badge");
const sampleDataBadgeText = document.getElementById("sample-data-badge-text");
const toastContainer = document.getElementById("toast-container");
const toggleMuteBtn = document.getElementById("toggle-mute-btn");
const muteIcon = document.getElementById("mute-icon");
const ttsStatusText = document.querySelector(".tts-status-text");

const riskBadge = document.getElementById("risk-badge");
const riskPercentage = document.getElementById("risk-percentage");
const gaugeFill = document.getElementById("gauge-fill");
const modelDecisionCertainty = document.getElementById("model-decision-certainty");
const decisionCertaintyBar = document.getElementById("decision-certainty-bar");
const dataCompletenessText = document.getElementById("data-completeness");
const completenessBar = document.getElementById("completeness-bar");
const riskCategory = document.getElementById("risk-category");
const groundTruth = document.getElementById("ground-truth");

const qsofaVal = document.getElementById("qsofa-val");
const qsofaAlertBadge = document.getElementById("qsofa-alert-badge");
const news2Val = document.getElementById("news2-val");
const news2AlertBadge = document.getElementById("news2-alert-badge");

const timelineSlider = document.getElementById("timeline-slider");
const currentHourDisplay = document.getElementById("current-hour-display");
const maxHourLabel = document.getElementById("max-hour-label");

const variablesForm = document.getElementById("variables-form");
const shapBarsContainer = document.getElementById("shap-bars-container");
const activeAlertsContainer = document.getElementById("active-alerts-container");

const consultAdvisorBtn = document.getElementById("consult-advisor-btn");
const advisorWelcomeMsg = document.getElementById("advisor-welcome-msg");
const advisorResponsePanel = document.getElementById("advisor-response-panel");
const explanationTextBox = document.getElementById("explanation-text-box");
const readAloudBtn = document.getElementById("read-aloud-btn");

const wsIndicator = document.getElementById("ws-indicator");
const wsStatusText = document.getElementById("ws-status-text");

// Replay Panel Elements
const playBtn = document.getElementById("play-btn");
const pauseBtn = document.getElementById("pause-btn");
const resetBtn = document.getElementById("reset-btn");
const playbackSpeed = document.getElementById("playback-speed");
const avgLatencyDisplay = document.getElementById("avg-latency-display");
const liveReplayBadge = document.getElementById("live-replay-badge");

// Modals
const aboutModal = document.getElementById("about-modal");
const aboutModalTrigger = document.getElementById("about-modal-trigger");
const aboutModalClose = document.getElementById("about-modal-close");
const aboutModalOk = document.getElementById("about-modal-ok");
const modalMetricsTableBody = document.getElementById("modal-metrics-table-body");
const modalSubgroupGrid = document.getElementById("modal-subgroup-grid");
const modalDatasetMode = document.getElementById("modal-dataset-mode");
const modalLeakageTableBody = document.getElementById("modal-leakage-table-body");
const modalLatencyGrid = document.getElementById("modal-latency-grid");
const modalAlarmEvidence = document.getElementById("modal-alarm-evidence");
const modalSessionId = document.getElementById("modal-session-id");
const exportSessionBtn = document.getElementById("export-session-btn");
const exportAllBtn = document.getElementById("export-all-btn");

const actionModal = document.getElementById("action-modal");
const actionModalTitle = document.getElementById("action-modal-title");
const actionModalBody = document.getElementById("action-modal-body");
const actionModalFooter = document.getElementById("action-modal-footer");
const actionModalClose = document.getElementById("action-modal-close");

// Quick Actions
const btnOrderCultures = document.getElementById("btn-order-cultures");
const btnOrderFluids = document.getElementById("btn-order-fluids");
const btnOrderAntibiotics = document.getElementById("btn-order-antibiotics");
const btnDismissAlert = document.getElementById("btn-dismiss-alert");

// Layout Switcher
const layoutToggleBtn = document.getElementById("layout-toggle-btn");
const layoutToggleText = document.getElementById("layout-toggle-text");

// 0. Toast notifications
// Replaces window.alert(), which blocks the event loop and stalls the replay
// stream mid-stride — unacceptable while a live timeline is playing.
const TOAST_ICONS = {
    success: "fa-circle-check",
    info: "fa-circle-info",
    warning: "fa-triangle-exclamation",
    error: "fa-circle-exclamation"
};

function showToast(message, kind = "success", timeoutMs = 4000) {
    if (!toastContainer) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${kind}`;
    toast.innerHTML = `
        <i class="fa-solid ${TOAST_ICONS[kind] || TOAST_ICONS.info}" aria-hidden="true"></i>
        <span class="toast-message"></span>
        <button class="toast-close" aria-label="Dismiss notification">&times;</button>
    `;
    toast.querySelector(".toast-message").textContent = message;

    const dismiss = () => {
        toast.classList.add("toast-leaving");
        setTimeout(() => toast.remove(), 220);
    };
    toast.querySelector(".toast-close").addEventListener("click", dismiss);

    toastContainer.appendChild(toast);
    // Force a reflow, then flip the class. requestAnimationFrame is throttled in
    // background tabs, which would leave the toast permanently invisible.
    void toast.offsetWidth;
    toast.classList.add("toast-visible");
    if (timeoutMs > 0) setTimeout(dismiss, timeoutMs);
}

// Format a bootstrap p-value honestly. With 1,000 resamples the smallest
// resolvable value is 0.001, so anything at the floor is reported as "< 0.001"
// rather than an exponential that implies far more precision than exists.
function formatPValue(p) {
    if (p === null || p === undefined || Number.isNaN(p)) return "—";
    if (p < 0.001) return "p < 0.001";
    return `p = ${p.toFixed(3)}`;
}

// 1. Initial counterbalancing assignment
function initCounterbalancing() {
    // Randomize layout order assignment: 3col_first or 1col_first
    layoutOrder = Math.random() < 0.5 ? "3col_first" : "1col_first";
    console.log(`Usability counterbalancing assigned: ${layoutOrder}`);
    
    // Set initial layout based on order
    if (layoutOrder === "1col_first") {
        setInterfaceLayout("1-column");
    } else {
        setInterfaceLayout("3-column");
    }
}

// Set layout representation
function setInterfaceLayout(mode) {
    currentLayout = mode;
    const body = document.body;
    const mainGrid = document.getElementById("dashboard-main");
    
    if (mode === "1-column") {
        body.className = "layout-flat-body";
        mainGrid.className = "dashboard-grid layout-1col-grid";
        layoutToggleText.textContent = "1-Column (Baseline)";
        layoutToggleBtn.className = "toggle-switch-btn baseline-layout";
    } else {
        body.className = "layout-3col";
        mainGrid.className = "dashboard-grid layout-3col-grid";
        layoutToggleText.textContent = "3-Column (HCI optimized)";
        layoutToggleBtn.className = "toggle-switch-btn";
    }
    
    // Log event
    logUsabilityEvent("layout_switch", "click", `Switched layout to ${mode}`);
}

// Log clinical/usability events to backend
function logUsabilityEvent(taskId, actionType, actionDetail = "") {
    const endTimestamp = Date.now();
    const payload = {
        session_id: sessionId,
        layout_order: layoutOrder,
        layout_condition: currentLayout,
        task_id: taskId,
        start_timestamp: taskStartTimestamp,
        end_timestamp: endTimestamp,
        patient_id: currentPatientId || "None",
        risk_score: currentSepsisRisk,
        action_type: actionType,
        action_detail: actionDetail,
        shap_snapshot: currentShapSnapshot
    };
    
    // Reset task timer to current time for sequential task duration computation
    taskStartTimestamp = endTimestamp;
    
    fetch("/api/log_event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).catch(err => console.error("Error writing usability log:", err));
}

// 2. WebSocket Connections
function initStaticWebSocket() {
    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsScheme}://${window.location.host}/ws`;
    
    staticSocket = new WebSocket(wsUrl);
    
    staticSocket.onopen = () => {
        wsIndicator.className = "status-indicator connected";
        wsStatusText.textContent = "Connected to Local WebSocket";
    };
    
    staticSocket.onclose = () => {
        wsIndicator.className = "status-indicator";
        wsStatusText.textContent = "Disconnected (Retrying...)";
        setTimeout(initStaticWebSocket, 3000);
    };
    
    staticSocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
            case "patients_list":
                if (typeof message.alert_threshold === "number") ALERT_THRESHOLD = message.alert_threshold;
                if (typeof message.high_risk_threshold === "number") HIGH_RISK_THRESHOLD = message.high_risk_threshold;
                populateDropdown(message.patients, message.total_patients, message.truncated);
                break;
            case "patient_timeline":
                handleStaticTimelineReceived(message.timeline);
                break;
            case "prediction_result":
                isSimulationMode = true;
                currentSepsisRisk = message.probability;
                currentShapSnapshot = message.feature_contribs;
                
                updateRiskDisplay(message.probability, message.decision_certainty, message.data_completeness, message.risk_level);
                renderSHAPBars(message.feature_contribs);
                evaluateBedsideAlarms(message.probability, getFormValues());
                break;
            case "advisor_response":
                handleAdvisorResponse(message.explanation);
                break;
        }
    };
}

// Populate Patient ID Dropdown.
// The server caps how many IDs it sends (the full PhysioNet corpus has ~40k
// patients), so the count hint tells the user when they need to narrow the filter.
function populateDropdown(patients, totalPatients, truncated) {
    const previousSelection = currentPatientId;

    patientSelect.innerHTML = '<option value="" disabled selected>Select Patient...</option>';
    const fragment = document.createDocumentFragment();
    patients.forEach(pid => {
        const opt = document.createElement("option");
        opt.value = pid;
        opt.textContent = pid;
        if (pid === previousSelection) opt.selected = true;
        fragment.appendChild(opt);
    });
    patientSelect.appendChild(fragment);

    if (patientCountHint) {
        const total = (totalPatients === undefined) ? patients.length : totalPatients;
        if (total === 0) {
            patientCountHint.textContent = "No matching patients";
            patientCountHint.classList.add("hint-empty");
        } else if (truncated) {
            patientCountHint.textContent = `Showing ${patients.length} of ${total.toLocaleString()} — refine the filter`;
            patientCountHint.classList.remove("hint-empty");
        } else {
            patientCountHint.textContent = `${total.toLocaleString()} patient${total === 1 ? "" : "s"}`;
            patientCountHint.classList.remove("hint-empty");
        }
    }
}

// Debounced server-side patient search
let patientSearchTimer = null;
function requestPatientSearch(query) {
    if (!staticSocket || staticSocket.readyState !== WebSocket.OPEN) return;
    staticSocket.send(JSON.stringify({ type: "search_patients", query: query }));
}

// Static Timeline Loader
let staticTimeline = [];
function handleStaticTimelineReceived(timeline) {
    staticTimeline = timeline;
    
    // Setup slider
    timelineSlider.disabled = false;
    timelineSlider.min = 0;
    timelineSlider.max = timeline.length - 1;
    timelineSlider.value = 0;
    maxHourLabel.textContent = `Hour ${Math.round(timeline[timeline.length - 1].ICULOS)}`;
    
    // Load first step
    loadStaticTimelineStep(0);
    
    // Enable Replay Buttons
    playBtn.disabled = false;
    resetBtn.disabled = false;
}

function loadStaticTimelineStep(index) {
    if (index < 0 || index >= staticTimeline.length) return;
    
    const step = staticTimeline[index];
    currentPatientId = step.PatientID;
    currentSepsisRisk = step.PredictedRisk;
    currentShapSnapshot = step.PredictedSHAP;
    
    currentHourDisplay.textContent = `ICU Hour: ${Math.round(step.ICULOS)}`;
    
    // Load Form values
    populateForm(step);
    
    // Update displays
    updateRiskDisplay(step.PredictedRisk, step.PredictedDecisionCertainty, step.PredictedDataCompleteness, step.PredictedRiskLevel);
    renderSHAPBars(step.PredictedSHAP);
    evaluateBedsideAlarms(step.PredictedRisk, step);
    
    // Draw trajectory graph
    const trajectoryPoints = staticTimeline.slice(Math.max(0, index - 11), index + 1).map(s => s.PredictedRisk);
    drawRiskTrajectory(trajectoryPoints);
    
    // Update ground truth
    if (step.SepsisLabel === 1) {
        groundTruth.textContent = "Sepsis Detected";
        groundTruth.style.color = "#ff7b72";
    } else {
        groundTruth.textContent = "No Sepsis";
        groundTruth.style.color = "#c9d1d9";
    }
    
    // Enable Advisor Consulting
    consultAdvisorBtn.disabled = false;
    enableActionButtons(true);
    resetAdvisorPanel();
}

function populateForm(data) {
    // Populate form fields
    for (const key in data) {
        const input = document.getElementById(key);
        if (input) {
            if (input.tagName === "SELECT") {
                input.value = data[key] !== null ? data[key].toString() : "1";
            } else {
                input.value = data[key] !== null ? data[key] : "";
            }
        }
    }
    
    // Update demog labels in Left Column
    document.getElementById("patient-id-display").textContent = data.PatientID;
    document.getElementById("demog-age").textContent = `${Math.round(data.Age)} yrs`;
    document.getElementById("demog-sex").textContent = data.Gender === 1 ? "Male" : "Female";
    document.getElementById("demog-dataset").textContent = data.Dataset;
    
    // Update bedside vitals tiles in Left Column
    updateVitalsTiles(data);
}

function updateVitalsTiles(data) {
    for (const key in vitalsNorms) {
        const valSpan = document.getElementById(`tile-${key.toLowerCase()}-val`);
        const tile = document.getElementById(`tile-${key.toLowerCase()}`);
        if (!valSpan || !tile) continue;
        
        const val = data[key];
        if (val !== undefined && val !== null) {
            valSpan.textContent = typeof val === "number" ? val.toFixed(1) : val;
            
            // Check norms
            const limits = vitalsNorms[key];
            if (val > limits.high) {
                tile.className = "vital-tile abnormal-high";
            } else if (val < limits.low) {
                tile.className = "vital-tile abnormal-low";
            } else {
                tile.className = "vital-tile";
            }
        } else {
            valSpan.textContent = "N/A";
            tile.className = "vital-tile";
        }
    }
}

// 3. Simulated Live Replay WebSocket Connection
// Patient IDs are "<A|B>_p<digits>" — the same shape the server enforces before it
// touches the filesystem. Validating here too keeps an unexpected value out of the
// socket URL rather than relying on the server to reject it.
const PATIENT_ID_PATTERN = /^[AB]_p\d{1,10}$/;

function startLiveReplay(patientId, speed) {
    stopLiveReplay();

    if (!PATIENT_ID_PATTERN.test(String(patientId))) {
        showToast("Invalid patient ID — cannot start replay.", "error");
        return;
    }

    // Playback speed is constrained to the values the UI offers.
    const allowedSpeeds = [1, 10, 60];
    const safeSpeed = allowedSpeeds.includes(Number(speed)) ? Number(speed) : 10;

    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsScheme}://${window.location.host}/ws/live/`
        + `${encodeURIComponent(patientId)}?playback_speed=${safeSpeed}`;

    isPlaying = true;
    liveTimelineData = [];
    liveReplayBadge.classList.remove("hidden");
    
    playBtn.classList.add("hidden");
    pauseBtn.classList.remove("hidden");
    
    // Disable timeline slider manual dragging during live playback
    timelineSlider.disabled = true;
    
    liveSocket = new WebSocket(wsUrl);
    
    liveSocket.onopen = () => {
        console.log(`Live Replay connection opened for patient ${patientId}`);
    };
    
    liveSocket.onclose = () => {
        console.log("Live Replay connection closed.");
        stopLiveReplay();
    };
    
    liveSocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === "live_reading") {
            currentPatientId = patientId;
            currentSepsisRisk = message.probability;
            currentShapSnapshot = message.feature_contribs;
            
            // Add to timeline
            liveTimelineData.push(message.probability);
            if (liveTimelineData.length > 12) {
                liveTimelineData.shift();
            }
            
            // Update UI elements
            currentHourDisplay.textContent = `ICU Hour: ${Math.round(message.hour)}`;
            
            // Load slider value
            timelineSlider.max = message.total_steps - 1;
            timelineSlider.value = message.step;
            
            // Populate form & vitals
            populateForm(message.features);
            
            // Update risk display & SHAP
            updateRiskDisplay(message.probability, message.decision_certainty, message.data_completeness, message.risk_level);
            renderSHAPBars(message.feature_contribs);
            evaluateBedsideAlarms(message.probability, message.features, message.alert_status);
            
            // Draw Trajectory
            drawRiskTrajectory(liveTimelineData);
            
            // Update ground truth
            if (message.sepsis_label === 1) {
                groundTruth.textContent = "Sepsis Detected";
                groundTruth.style.color = "#ff7b72";
            } else {
                groundTruth.textContent = "No Sepsis";
                groundTruth.style.color = "#c9d1d9";
            }
            
            // Log event
            logUsabilityEvent("live_replay_tick", "receive", `Emitted hour ${message.hour} at latency ${message.latency_ms.toFixed(1)}ms`);
            
            // Retrieve average latency stats periodically
            fetchLatencyStats();
            
        } else if (message.type === "live_complete") {
            console.log("Live Replay simulation complete.");
            stopLiveReplay();
        } else if (message.type === "live_error") {
            console.error("Live Replay error:", message.message);
            stopLiveReplay();
        }
    };
}

function stopLiveReplay() {
    isPlaying = false;
    if (liveSocket) {
        liveSocket.close();
        liveSocket = null;
    }
    
    playBtn.classList.remove("hidden");
    pauseBtn.classList.add("hidden");
    liveReplayBadge.classList.add("hidden");
    
    // Enable timeline slider when paused/stopped
    if (staticTimeline.length > 0) {
        timelineSlider.disabled = false;
    }
}

// 4. GUI Rendering Functions
function updateRiskDisplay(probability, decisionCertainty, dataCompleteness, risk_lvl) {
    const percent = Math.round(probability * 100);
    riskPercentage.textContent = `${percent}%`;
    
    // SVG radial stroke-dashoffset: max is 283
    const strokeOffset = 283 * (1.0 - probability);
    gaugeFill.style.strokeDashoffset = strokeOffset;
    
    let color = "#2ea043"; // green
    let badgeClass = "badge badge-success";
    
    if (risk_lvl === "High") {
        color = "#ff7b72"; // red
        badgeClass = "badge badge-danger";
    } else if (risk_lvl === "Medium") {
        color = "#e3b341"; // yellow
        badgeClass = "badge badge-warning";
    }
    
    gaugeFill.style.stroke = color;
    riskCategory.textContent = `${risk_lvl} Risk`;
    riskCategory.style.color = color;
    
    riskBadge.textContent = isPlaying ? "Live Replay" : "Static Profile";
    riskBadge.className = isPlaying ? "badge badge-info" : badgeClass;
    
    // Render decision certainty
    modelDecisionCertainty.textContent = `${(decisionCertainty * 100).toFixed(1)}%`;
    decisionCertaintyBar.style.width = `${decisionCertainty * 100}%`;
    decisionCertaintyBar.style.backgroundColor = color;
    
    // Render data completeness
    const completenessPct = (dataCompleteness || 0.0) * 100.0;
    dataCompletenessText.textContent = `${completenessPct.toFixed(1)}%`;
    completenessBar.style.width = `${completenessPct}%`;
}

// Draw SHAP Horizontal Bar Charts
function renderSHAPBars(contribs) {
    shapBarsContainer.innerHTML = "";
    if (!contribs) return;
    
    // Sort features by absolute contribution descending
    const sorted = Object.keys(contribs)
        .map(key => ({ name: key, value: contribs[key] }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
        
    // Find max absolute value to scale width
    const maxVal = Math.max(...sorted.map(s => Math.abs(s.value))) || 1.0;
    
    sorted.forEach(item => {
        const row = document.createElement("div");
        row.className = "shap-row";
        
        const nameSpan = document.createElement("span");
        nameSpan.className = "shap-feat-name";
        nameSpan.textContent = item.name;
        
        const track = document.createElement("div");
        track.className = "shap-bar-track";
        
        const bar = document.createElement("div");
        const valPct = (Math.abs(item.value) / maxVal) * 50.0; // scale to max 50% on either side
        
        bar.style.width = `${valPct}%`;
        if (item.value >= 0) {
            bar.className = "shap-bar positive";
            bar.style.left = "50%";
        } else {
            bar.className = "shap-bar negative";
            bar.style.left = `${50.0 - valPct}%`;
        }
        
        track.appendChild(bar);
        
        const valSpan = document.createElement("span");
        valSpan.className = "shap-feat-val";
        valSpan.textContent = item.value >= 0 ? `+${item.value.toFixed(2)}` : item.value.toFixed(2);
        
        row.appendChild(nameSpan);
        row.appendChild(track);
        row.appendChild(valSpan);
        
        shapBarsContainer.appendChild(row);
    });
}

// Draw Risk Trajectory using HTML5 2D Canvas
function drawRiskTrajectory(points) {
    const canvas = document.getElementById("trajectory-canvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, w, h);
    
    if (points.length < 2) {
        ctx.fillStyle = "#8b949e";
        ctx.font = "10px Outfit";
        ctx.fillText("Insufficient history data.", 10, h / 2);
        return;
    }
    
    // Draw grid lines
    ctx.strokeStyle = "rgba(240, 246, 252, 0.04)";
    ctx.lineWidth = 1;
    for (let yGrid = 0.3; yGrid <= 0.7; yGrid += 0.4) {
        const yVal = h - (yGrid * h);
        ctx.beginPath();
        ctx.moveTo(0, yVal);
        ctx.lineTo(w, yVal);
        ctx.stroke();
    }
    
    // Plot trajectory line
    const xStep = w / 11;
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    
    // Setup gradient color based on final risk
    const lastRisk = points[points.length - 1];
    let strokeColor = "#2ea043"; // green
    if (lastRisk >= HIGH_RISK_THRESHOLD) strokeColor = "#ff7b72"; // red
    else if (lastRisk >= ALERT_THRESHOLD) strokeColor = "#e3b341"; // yellow
    
    ctx.strokeStyle = strokeColor;
    
    points.forEach((val, i) => {
        // x coordinate: map points to the rightmost columns
        const startXIndex = 12 - points.length;
        const x = (startXIndex + i) * xStep;
        const y = h - (val * h); // inverted y scale
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // Gradient fill under curve
    ctx.lineTo((12 - points.length + points.length - 1) * xStep, h);
    ctx.lineTo((12 - points.length) * xStep, h);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, strokeColor + "20"); // 12% alpha
    fillGrad.addColorStop(1, strokeColor + "00"); // transparent
    ctx.fillStyle = fillGrad;
    ctx.fill();
    
    // Draw dots on values
    points.forEach((val, i) => {
        const startXIndex = 12 - points.length;
        const x = (startXIndex + i) * xStep;
        const y = h - (val * h);
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = strokeColor;
        ctx.fill();
    });
}

// 5. Evaluate Bedside Alarms & Smart Suppression
function evaluateBedsideAlarms(probability, features, overrideStatus = null) {
    let alertStatus = "NORMAL";
    
    if (overrideStatus) {
        alertStatus = overrideStatus;
    } else {
        // Calculate qSOFA & NEWS2
        const s_qsofa = calculate_qsofa_points(features);
        const s_news2 = calculate_news2_points(features);
        
        qsofaVal.textContent = `${s_qsofa} / 2`;
        news2Val.textContent = `${s_news2} / 20`;
        
        // s-qSOFA alarm badge
        if (s_qsofa >= 2) {
            qsofaAlertBadge.className = "badge badge-danger";
            qsofaAlertBadge.textContent = "High Risk";
        } else {
            qsofaAlertBadge.className = "badge badge-success";
            qsofaAlertBadge.textContent = "Normal";
        }
        
        // s-NEWS2 alarm badge
        if (s_news2 >= 5) {
            news2AlertBadge.className = "badge badge-danger";
            news2AlertBadge.textContent = "High Risk";
        } else {
            news2AlertBadge.className = "badge badge-success";
            news2AlertBadge.textContent = "Normal";
        }
        
        // Rule-based suppression logic for local static edits
        if (probability >= ALERT_THRESHOLD) {
            const lac = features.Lactate || 0.0;
            const o2 = features.O2Sat || 100.0;
            const severe_organ = (lac > 2.0 && o2 < 90.0);
            
            // Check if sustained (static mode assumes baseline is sustained for demo, or checks organ marker)
            if (severe_organ) {
                alertStatus = probability >= HIGH_RISK_THRESHOLD ? "CRITICAL_ALARM" : "WARNING_ALARM";
            } else {
                alertStatus = "SUPPRESSED_WARNING"; // Suppress transient spikes
            }
        }
    }
    
    // Update alert status banner in Left Column
    const alertBanner = document.getElementById("alert-banner");
    const alertBannerIcon = document.getElementById("alert-banner-icon");
    const alertBannerText = document.getElementById("alert-banner-text");
    
    if (alertStatus === "CRITICAL_ALARM") {
        alertBanner.className = "alert-status-banner alert-critical";
        alertBannerIcon.className = "fa-solid fa-triangle-exclamation";
        alertBannerText.textContent = "CRITICAL ALARM: SEVERE SEPSIS RISK";
        
        // Speak alert once if active
        if (!isMuted && !isPlaying) {
            speakText("Critical Alert. Sustained Sepsis Risk detected. Please initiate resuscitation protocols immediately.");
        }
    } else if (alertStatus === "WARNING_ALARM") {
        alertBanner.className = "alert-status-banner alert-warning";
        alertBannerIcon.className = "fa-solid fa-circle-exclamation";
        alertBannerText.textContent = "WARNING: SUSTAINED RISK IN PROGRESS";
    } else if (alertStatus === "SUPPRESSED_WARNING") {
        alertBanner.className = "alert-status-banner alert-warning";
        alertBannerIcon.className = "fa-solid fa-filter";
        alertBannerText.textContent = "STATUS: TRANSIENT SPIKE SUPPRESSED";
    } else {
        alertBanner.className = "alert-status-banner alert-normal";
        alertBannerIcon.className = "fa-solid fa-circle-check";
        alertBannerText.textContent = "STATUS: NORMAL";
    }
    
    // Render alerts list in Right Column
    renderAlertsList(probability, features, alertStatus);
}

// Render dynamic expandable alert cards
function renderAlertsList(prob, features, status) {
    activeAlertsContainer.innerHTML = "";
    
    if (prob < ALERT_THRESHOLD) {
        activeAlertsContainer.innerHTML = '<p class="section-desc"><i class="fa-solid fa-circle-check"></i> No active clinical alerts for this patient.</p>';
        return;
    }
    
    // Alert 1: Sepsis Risk alert card
    const card = document.createElement("div");
    card.className = `alert-item-card ${prob >= HIGH_RISK_THRESHOLD ? 'critical' : 'warning'}`;
    
    const header = document.createElement("div");
    header.className = "alert-item-header";
    
    const titleBox = document.createElement("div");
    titleBox.className = `alert-title-box ${prob >= HIGH_RISK_THRESHOLD ? 'critical' : 'warning'}`;
    titleBox.innerHTML = `<i class="fa-solid fa-bell"></i> Sepsis Risk Alert: ${(prob*100).toFixed(0)}%`;
    
    const chevron = document.createElement("i");
    chevron.className = "fa-solid fa-chevron-down chevron-icon";
    
    header.appendChild(titleBox);
    header.appendChild(chevron);
    
    const body = document.createElement("div");
    body.className = "alert-item-body";
    
    const isSuppressed = status === "SUPPRESSED_WARNING";
    // Feature values reach this card straight from the socket payload, so they are
    // escaped before being interpolated into innerHTML. Numeric values are also
    // formatted rather than printed raw, which keeps the card readable.
    const fmtFeature = (v) => (v === null || v === undefined || Number.isNaN(v))
        ? "N/A"
        : escapeHtml(typeof v === "number" ? v.toFixed(1) : v);

    body.innerHTML = `
        <p><strong>Assessment:</strong> Sepsis probability is elevated at ${(prob*100).toFixed(1)}%.</p>
        <p><strong>Smart Suppression status:</strong> ${isSuppressed ? '<span class="text-red">SUPPRESSED (Transient Spike)</span>' : 'ACTIVE (Sustained/Organ flags active)'}</p>
        <p><strong>Rationale:</strong> Lactate levels are ${fmtFeature(features.Lactate)} mmol/L and oxygen saturation is ${fmtFeature(features.O2Sat)}%. Real-time SHAP analysis attributes risk mainly to these variables.</p>
    `;
    
    // Toggle expander on click
    header.addEventListener("click", () => {
        const isExp = card.classList.toggle("expanded");
        body.style.display = isExp ? "block" : "none";
        logUsabilityEvent("alert_expand", "click", `Toggled alert expand state: ${isExp}`);
    });
    
    card.appendChild(header);
    card.appendChild(body);
    activeAlertsContainer.appendChild(card);
}

// Bedside calculations of clinical scoring points
function calculate_qsofa_points(features) {
    let score = 0;
    if (features.Resp >= 22) score++;
    if (features.SBP <= 100) score++;
    return score;
}

function calculate_news2_points(features) {
    let score = 0;
    
    // RR
    const rr = features.Resp;
    if (rr !== undefined && rr !== null) {
        if (rr <= 8) score += 3;
        else if (rr <= 11) score += 1;
        else if (rr <= 20) score += 0;
        else if (rr <= 24) score += 1;
        else score += 3;
    }
    
    // O2Sat
    const o2 = features.O2Sat;
    if (o2 !== undefined && o2 !== null) {
        if (o2 <= 91) score += 3;
        else if (o2 <= 93) score += 2;
        else if (o2 <= 95) score += 1;
        else score += 0;
    }
    
    // Temp
    const temp = features.Temp;
    if (temp !== undefined && temp !== null) {
        if (temp <= 35.0) score += 3;
        else if (temp <= 36.0) score += 1;
        else if (temp <= 38.0) score += 0;
        else if (temp <= 39.0) score += 1;
        else score += 2;
    }
    
    // SBP
    const sbp = features.SBP;
    if (sbp !== undefined && sbp !== null) {
        if (sbp <= 90) score += 3;
        else if (sbp <= 100) score += 2;
        else if (sbp <= 110) score += 1;
        else if (sbp <= 219) score += 0;
        else score += 3;
    }
    
    // HR
    const hr = features.HR;
    if (hr !== undefined && hr !== null) {
        if (hr <= 40) score += 3;
        else if (hr <= 50) score += 1;
        else if (hr <= 90) score += 0;
        else if (hr <= 110) score += 1;
        else if (hr <= 130) score += 2;
        else score += 3;
    }
    
    return score;
}

// 6. Consult Sepsis Advisor Output
function handleAdvisorResponse(explanation) {
    consultAdvisorBtn.disabled = false;
    consultAdvisorBtn.innerHTML = '<i class="fa-solid fa-brain"></i> Consult Sepsis Advisor';
    
    // Parse simple markdown to HTML
    const formatted = parseMarkdown(explanation);
    
    advisorWelcomeMsg.classList.add("hidden");
    advisorResponsePanel.classList.remove("hidden");
    explanationTextBox.innerHTML = formatted;
    
    // Speak response if unmuted
    if (!isMuted) {
        speakText(explanation);
    }
}

function resetAdvisorPanel() {
    stopSpeaking();
    advisorWelcomeMsg.classList.remove("hidden");
    advisorResponsePanel.classList.add("hidden");
    explanationTextBox.innerHTML = "";
}

// Escape HTML before any markup is added. Advisor text arrives from an external
// LLM endpoint, so it is untrusted input to innerHTML.
function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
    ));
}

function parseMarkdown(text) {
    let html = escapeHtml(text).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/^\d\.\s(.*)/gm, '<li>$1</li>');
    html = html.replace(/<li>(.*)<\/li>/gs, '<ul>$&</ul>');
    return html;
}

// 7. Text-To-Speech Web Speech API
function speakText(text) {
    stopSpeaking();
    
    if (!window.speechSynthesis) return;
    
    // Strip markdown tags
    const clean = text.replace(/\*\*/g, "").replace(/\*/g, "").replace(/#/g, "").trim();
    
    currentUtterance = new SpeechSynthesisUtterance(clean);
    currentUtterance.lang = "en-US";
    
    const voices = window.speechSynthesis.getVoices();
    const naturalVoice = voices.find(v => v.lang.startsWith("en") && (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Microsoft")));
    if (naturalVoice) {
        currentUtterance.voice = naturalVoice;
    }
    
    currentUtterance.rate = 1.0;
    
    currentUtterance.onend = () => {
        currentUtterance = null;
    };
    
    window.speechSynthesis.speak(currentUtterance);
}

function stopSpeaking() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
}

// 8. REST APIs Fetchers (Startup details)
// Render an AUROC-difference comparison cell. A confidence interval that spans
// zero means "no detectable difference" — the interface says so plainly rather
// than letting a bare p-value imply superiority.
function renderComparison(baseline) {
    const p = baseline.auroc_difference_adjusted_p_value;
    const ci = baseline.auroc_difference_95_ci;
    const pText = formatPValue(p);

    if (!ci) return pText;

    const spansZero = ci[0] <= 0 && ci[1] >= 0;
    const ciText = `Δ ${ci[0] >= 0 ? "+" : ""}${ci[0].toFixed(3)} to ${ci[1] >= 0 ? "+" : ""}${ci[1].toFixed(3)}`;
    const verdict = spansZero
        ? '<span class="sig-badge sig-none">no significant difference</span>'
        : '<span class="sig-badge sig-yes">significant</span>';

    return `${pText}<br><span class="ci-subtext">${ciText}</span><br>${verdict}`;
}

function fetchLeakageAudit() {
    fetch("/api/leakage_audit")
        .then(res => res.json())
        .then(data => {
            if (!modalLeakageTableBody) return;
            if (data.error) {
                modalLeakageTableBody.innerHTML = `<tr><td colspan="5">${escapeHtml(data.error)}</td></tr>`;
                return;
            }
            const rows = Object.entries(data).map(([feature, r]) => `
                <tr>
                    <td><strong>${escapeHtml(feature)}</strong></td>
                    <td>${r.correlation_t0.toFixed(3)}</td>
                    <td>${r.correlation_t_minus_6.toFixed(3)}</td>
                    <td>${r.correlation_difference >= 0 ? "+" : ""}${r.correlation_difference.toFixed(3)}</td>
                    <td><span class="risk-chip risk-${escapeHtml(r.leakage_risk_level).toLowerCase()}">${escapeHtml(r.leakage_risk_level)}</span></td>
                </tr>
            `).join("");
            modalLeakageTableBody.innerHTML = rows || `<tr><td colspan="5">No audit entries.</td></tr>`;
        })
        .catch(() => {
            if (modalLeakageTableBody) {
                modalLeakageTableBody.innerHTML = `<tr><td colspan="5">Could not load leakage audit.</td></tr>`;
            }
        });
}

function fetchLatencyDetail() {
    fetch("/api/latency_stats")
        .then(res => res.json())
        .then(data => {
            if (!modalLatencyGrid) return;
            if (!data.total_runs) {
                modalLatencyGrid.innerHTML = `<span class="evidence-loading">No inference runs recorded yet — start a replay to collect latency samples.</span>`;
                return;
            }
            modalLatencyGrid.innerHTML = `
                <div class="evidence-tile"><span class="evidence-label">Mean</span><strong>${data.mean_ms.toFixed(1)} ms</strong></div>
                <div class="evidence-tile"><span class="evidence-label">p50</span><strong>${data.p50_ms.toFixed(1)} ms</strong></div>
                <div class="evidence-tile"><span class="evidence-label">p95</span><strong>${data.p95_ms.toFixed(1)} ms</strong></div>
                <div class="evidence-tile"><span class="evidence-label">p99</span><strong>${data.p99_ms.toFixed(1)} ms</strong></div>
                <div class="evidence-tile"><span class="evidence-label">Samples</span><strong>${data.total_runs.toLocaleString()}</strong></div>
            `;
        })
        .catch(() => {
            if (modalLatencyGrid) {
                modalLatencyGrid.innerHTML = `<span class="evidence-loading">Could not load latency statistics.</span>`;
            }
        });
}

function runAlarmReplay() {
    if (!modalAlarmEvidence) return;
    modalAlarmEvidence.innerHTML = `<span class="evidence-loading"><i class="fa-solid fa-spinner fa-spin"></i> Replaying every patient-hour through suppression logic…</span>`;

    fetch("/api/replay_results")
        .then(res => res.json())
        .then(d => {
            modalAlarmEvidence.innerHTML = `
                <div class="evidence-tile"><span class="evidence-label">Raw alarms</span><strong>${d.total_unsuppressed_alarms.toLocaleString()}</strong></div>
                <div class="evidence-tile"><span class="evidence-label">After suppression</span><strong>${d.total_suppressed_alarms.toLocaleString()}</strong></div>
                <div class="evidence-tile evidence-good"><span class="evidence-label">Alarms avoided</span><strong>${d.alarms_avoided_count.toLocaleString()} (${d.suppression_effectiveness_pct.toFixed(1)}%)</strong></div>
                <div class="evidence-tile"><span class="evidence-label">False-alarm rate before</span><strong>${d.false_alarm_rate_before_suppression_pct.toFixed(2)}%</strong></div>
                <div class="evidence-tile evidence-good"><span class="evidence-label">False-alarm rate after</span><strong>${d.false_alarm_rate_after_suppression_pct.toFixed(2)}%</strong></div>
            `;
            showToast(`Suppression avoided ${d.alarms_avoided_count.toLocaleString()} alarms (${d.suppression_effectiveness_pct.toFixed(1)}%)`, "success");
        })
        .catch(() => {
            modalAlarmEvidence.innerHTML = `<span class="evidence-loading">Replay failed. Check the server log.</span>`;
            showToast("Alarm replay failed", "error");
        });
}

// Export anonymized usability-study events
function exportSessionLogs(onlyThisSession) {
    const url = onlyThisSession ? `/api/session_logs?session_id=${encodeURIComponent(sessionId)}` : "/api/session_logs";
    fetch(url)
        .then(res => res.json())
        .then(data => {
            if (!data.count) {
                showToast("No session events recorded yet", "warning");
                return;
            }
            const blob = new Blob([JSON.stringify(data.events, null, 2)], { type: "application/json" });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = onlyThisSession ? `sepsisguard_${sessionId}.json` : "sepsisguard_all_sessions.json";
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(link.href);
            showToast(`Exported ${data.count} event${data.count === 1 ? "" : "s"}`, "success");
        })
        .catch(() => showToast("Export failed", "error"));
}

function fetchAcademicMetrics() {
    fetch("/api/metrics")
        .then(res => res.json())
        .then(data => {
            // Render metrics table inside About Modal
            modalMetricsTableBody.innerHTML = `
                <tr>
                    <td><strong>LightGBM Classifier (Ours)</strong></td>
                    <td>${data.model_performance.auroc.toFixed(3)} (95% CI: ${data.model_performance.auroc_95_ci[0].toFixed(3)}–${data.model_performance.auroc_95_ci[1].toFixed(3)})</td>
                    <td>${data.model_performance.auprc.toFixed(3)} (95% CI: ${data.model_performance.auprc_95_ci[0].toFixed(3)}–${data.model_performance.auprc_95_ci[1].toFixed(3)})</td>
                    <td>${data.model_performance.brier_score.toFixed(3)}</td>
                    <td>— (Reference Model)</td>
                </tr>
                <tr>
                    <td><strong>NEWS2 Score Baseline</strong></td>
                    <td>${data.baseline_comparison.news2.auroc.toFixed(3)} (95% CI: ${data.baseline_comparison.news2.auroc_95_ci[0].toFixed(3)}–${data.baseline_comparison.news2.auroc_95_ci[1].toFixed(3)})</td>
                    <td>${data.baseline_comparison.news2.auprc.toFixed(3)}</td>
                    <td>—</td>
                    <td>${renderComparison(data.baseline_comparison.news2)}</td>
                </tr>
                <tr>
                    <td><strong>qSOFA Score Baseline</strong></td>
                    <td>${data.baseline_comparison.qsofa.auroc.toFixed(3)} (95% CI: ${data.baseline_comparison.qsofa.auroc_95_ci[0].toFixed(3)}–${data.baseline_comparison.qsofa.auroc_95_ci[1].toFixed(3)})</td>
                    <td>${data.baseline_comparison.qsofa.auprc.toFixed(3)}</td>
                    <td>—</td>
                    <td>${renderComparison(data.baseline_comparison.qsofa)}</td>
                </tr>
            `;

            // Dataset provenance badge + modal line, driven by real metadata so the
            // interface stays truthful when running against the full corpus.
            const meta = data.metadata || {};
            if (sampleDataBadgeText && meta.sample_label) {
                sampleDataBadgeText.textContent = meta.sample_label;
                sampleDataBadge.setAttribute(
                    "data-tooltip",
                    `${meta.total_patients?.toLocaleString() ?? "?"} patients · ` +
                    `${meta.total_rows?.toLocaleString() ?? "?"} rows · ` +
                    `${meta.ever_sepsis_patients?.toLocaleString() ?? "?"} ever-sepsis`
                );
                sampleDataBadge.classList.toggle("badge-full-corpus", meta.dataset_mode === "full");
            }
            if (modalDatasetMode) {
                modalDatasetMode.textContent =
                    `${meta.sample_label ?? "unknown"} — ${meta.total_patients?.toLocaleString() ?? "?"} patients, ` +
                    `${meta.ever_sepsis_patients?.toLocaleString() ?? "?"} of whom develop sepsis ` +
                    `(split seed ${meta.split_seed ?? "?"}, model seed ${meta.model_seed ?? "?"})`;
            }

            // Render subgroup grids
            modalSubgroupGrid.innerHTML = "";
            for (const subgroup in data.subgroups) {
                const sub = data.subgroups[subgroup];
                const tile = document.createElement("div");
                tile.className = `subgroup-tile ${sub.underpowered ? 'underpowered' : ''}`;
                
                tile.innerHTML = `
                    <div class="subgroup-tile-title">${escapeHtml(subgroup).replace(/_/g, " ")}</div>
                    <div>N: <strong>${sub.N}</strong> | Positive Cases: <strong>${sub.positive_cases}</strong></div>
                    <div>AUROC: <strong>${sub.auroc !== null ? sub.auroc.toFixed(3) : 'N/A'}</strong></div>
                    ${sub.underpowered ? '<span class="subgroup-warning-lbl"><i class="fa-solid fa-triangle-exclamation"></i> underpowered — interpret with caution</span>' : ''}
                `;
                modalSubgroupGrid.appendChild(tile);
            }
        })
        .catch(err => console.error("Error loading metrics:", err));
}

function fetchLatencyStats() {
    fetch("/api/latency_stats")
        .then(res => res.json())
        .then(data => {
            if (data.total_runs > 0) {
                avgLatencyDisplay.textContent = `${data.p50_ms.toFixed(1)}ms`;
            }
        })
        .catch(err => console.error("Error reading latency stats:", err));
}

// 9. Quick Actions Protocol Confirmation
function openActionModal(title, bodyText, onConfirm) {
    actionModalTitle.textContent = title;
    actionModalBody.innerHTML = `<p>${bodyText}</p>`;
    
    actionModalFooter.innerHTML = "";
    
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "outline-btn compact-btn";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => actionModal.classList.add("hidden");
    
    const confirmBtn = document.createElement("button");
    confirmBtn.className = "primary-btn compact-btn";
    confirmBtn.textContent = "Confirm Order";
    confirmBtn.onclick = () => {
        onConfirm();
        actionModal.classList.add("hidden");
    };
    
    actionModalFooter.appendChild(cancelBtn);
    actionModalFooter.appendChild(confirmBtn);
    
    actionModal.classList.remove("hidden");
}

function enableActionButtons(enable) {
    btnOrderCultures.disabled = !enable;
    btnOrderFluids.disabled = !enable;
    btnOrderAntibiotics.disabled = !enable;
    btnDismissAlert.disabled = !enable;
}

// Gather form values
function getFormValues() {
    const vals = {};
    variablesForm.querySelectorAll("input, select").forEach(input => {
        const id = input.id;
        const val = input.value.trim ? input.value.trim() : input.value;
        vals[id] = val === "" ? null : parseFloat(val);
    });
    return vals;
}

// Event Listeners
patientSelect.addEventListener("change", (e) => {
    const pid = e.target.value;
    stopLiveReplay();
    resetAdvisorPanel();
    
    // Set active patient
    currentPatientId = pid;
    
    if (staticSocket && staticSocket.readyState === WebSocket.OPEN) {
        staticSocket.send(JSON.stringify({
            type: "get_patient_timeline",
            patient_id: pid
        }));
    }
});

// Timeline slider navigation
timelineSlider.addEventListener("input", (e) => {
    const idx = parseInt(e.target.value);
    loadStaticTimelineStep(idx);
    
    // Log event
    logUsabilityEvent("timeline_slider_drag", "drag", `Slider moved to index ${idx}`);
});

// What-If input modifications trigger real-time predictions
variablesForm.addEventListener("input", () => {
    if (!staticSocket || staticSocket.readyState !== WebSocket.OPEN) return;
    
    const features = getFormValues();
    staticSocket.send(JSON.stringify({
        type: "predict_risk",
        features: features
    }));
});

// Replay Buttons
playBtn.addEventListener("click", () => {
    if (!currentPatientId) return;
    const speed = parseInt(playbackSpeed.value);
    startLiveReplay(currentPatientId, speed);
    logUsabilityEvent("live_replay_start", "click", `Replay started at ${speed}x speed`);
});

pauseBtn.addEventListener("click", () => {
    stopLiveReplay();
    logUsabilityEvent("live_replay_pause", "click", "Replay paused");
});

resetBtn.addEventListener("click", () => {
    stopLiveReplay();
    if (staticTimeline.length > 0) {
        loadStaticTimelineStep(0);
        timelineSlider.value = 0;
    }
    logUsabilityEvent("live_replay_reset", "click", "Replay reset to hour 0");
});

// Layout Mode Toggle Button
layoutToggleBtn.addEventListener("click", () => {
    const nextLayout = currentLayout === "3-column" ? "1-column" : "3-column";
    setInterfaceLayout(nextLayout);
});

// Quick Action Order clicks
btnOrderCultures.addEventListener("click", () => {
    openActionModal(
        "Confirm Blood Cultures Order",
        "This will order bilateral blood cultures. Protocol requires drawing cultures before administering new empiric antibiotics.",
        () => {
            logUsabilityEvent("protocol_action", "order", "Blood Cultures Ordered");
            showToast("Blood cultures ordered", "success");
        }
    );
});

btnOrderFluids.addEventListener("click", () => {
    openActionModal(
        "Confirm IV Fluid Resuscitation",
        "Initiate crystalloid fluid resuscitation protocol (30 mL/kg). Target MAP is >= 65 mmHg.",
        () => {
            logUsabilityEvent("protocol_action", "order", "IV Fluid Resuscitation Initiated");
            showToast("IV fluid resuscitation initiated", "success");
        }
    );
});

btnOrderAntibiotics.addEventListener("click", () => {
    openActionModal(
        "Confirm Empiric Antibiotics Order",
        "Order broad-spectrum empiric antimicrobials. Ensure blood cultures have been ordered/drawn first.",
        () => {
            logUsabilityEvent("protocol_action", "order", "Empiric Antibiotics Ordered");
            showToast("Empiric antibiotics ordered", "success");
        }
    );
});

btnDismissAlert.addEventListener("click", () => {
    // Open action modal for dismissal with reason selection
    actionModalTitle.textContent = "Dismiss Sepsis Alert";
    actionModalBody.innerHTML = `
        <label for="dismiss-reason" style="font-size:0.8rem;color:#8b949e;display:block;margin-bottom:0.4rem;">Select Justification for Dismissal:</label>
        <select id="dismiss-reason" class="dropdown-control" style="width:100%;">
            <option value="transient_spike">Transient reading spike / artifact</option>
            <option value="alternative_diag">Alternative clinical diagnosis</option>
            <option value="patient_stable">Patient clinically stable / improving</option>
            <option value="already_treated">Already receiving guideline-adherent treatment</option>
        </select>
    `;
    
    actionModalFooter.innerHTML = "";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "outline-btn compact-btn";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => actionModal.classList.add("hidden");
    
    const confirmBtn = document.createElement("button");
    confirmBtn.className = "primary-btn compact-btn";
    confirmBtn.textContent = "Dismiss Alert";
    confirmBtn.onclick = () => {
        const reason = document.getElementById("dismiss-reason").value;
        logUsabilityEvent("protocol_action", "dismiss", `Alert dismissed: ${reason}`);
        actionModal.classList.add("hidden");
        showToast("Sepsis alert dismissed", "info");
    };
    
    actionModalFooter.appendChild(cancelBtn);
    actionModalFooter.appendChild(confirmBtn);
    actionModal.classList.remove("hidden");
});

// Advisor consult
consultAdvisorBtn.addEventListener("click", () => {
    if (!staticSocket || staticSocket.readyState !== WebSocket.OPEN) return;
    
    consultAdvisorBtn.disabled = true;
    consultAdvisorBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Consulting AI...';
    
    const features = getFormValues();
    staticSocket.send(JSON.stringify({
        type: "ask_advisor",
        features: features
    }));
    
    logUsabilityEvent("advisor_consult", "click", "Consulted clinical AI advisor");
});

// TTS master controls
toggleMuteBtn.addEventListener("click", () => {
    isMuted = !isMuted;
    if (isMuted) {
        stopSpeaking();
        muteIcon.className = "fa-solid fa-volume-mute";
        ttsStatusText.textContent = "Voice Muted";
        toggleMuteBtn.parentElement.classList.add("muted");
    } else {
        muteIcon.className = "fa-solid fa-volume-up";
        ttsStatusText.textContent = "Voice Active";
        toggleMuteBtn.parentElement.classList.remove("muted");
        if (!advisorResponsePanel.classList.contains("hidden")) {
            speakText(explanationTextBox.innerText);
        }
    }
});

readAloudBtn.addEventListener("click", () => {
    speakText(explanationTextBox.innerText);
});

// Modals Open/Close, with focus management so keyboard and screen-reader users
// are not left tabbing around behind an open dialog.
let lastFocusedElement = null;

function openModal(modal) {
    const active = document.activeElement;
    // document.body is not focusable, so remember it only if it can actually
    // take focus back — otherwise focus would be stranded on a hidden element.
    lastFocusedElement = (active && active !== document.body && typeof active.focus === "function")
        ? active
        : null;

    modal.classList.remove("hidden");
    const focusable = modal.querySelector(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusable) focusable.focus();
}

function closeModal(modal) {
    // Move focus out before hiding, so it never rests on a display:none element.
    const target = (lastFocusedElement && document.contains(lastFocusedElement))
        ? lastFocusedElement
        : aboutModalTrigger;
    if (target && typeof target.focus === "function") target.focus();

    modal.classList.add("hidden");
    lastFocusedElement = null;
}

function openAboutModal() {
    fetchAcademicMetrics();
    fetchLeakageAudit();
    fetchLatencyDetail();
    if (modalSessionId) modalSessionId.textContent = sessionId;
    openModal(aboutModal);
}

aboutModalTrigger.addEventListener("click", openAboutModal);
aboutModalClose.addEventListener("click", () => closeModal(aboutModal));
aboutModalOk.addEventListener("click", () => closeModal(aboutModal));
actionModalClose.addEventListener("click", () => closeModal(actionModal));

// Click the backdrop to dismiss
[aboutModal, actionModal].forEach(modal => {
    modal.addEventListener("mousedown", (e) => {
        if (e.target === modal) closeModal(modal);
    });
});

// Keep Tab focus inside an open dialog
document.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const modal = [actionModal, aboutModal].find(m => !m.classList.contains("hidden"));
    if (!modal) return;

    const focusables = Array.from(modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )).filter(el => !el.disabled && el.offsetParent !== null);
    if (!focusables.length) return;

    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
    }
});

// Patient filter (server-side search; the full corpus is far too large to filter
// client-side, and the server caps how many IDs it returns).
if (patientSearch) {
    patientSearch.addEventListener("input", () => {
        clearTimeout(patientSearchTimer);
        patientSearchTimer = setTimeout(() => requestPatientSearch(patientSearch.value), 180);
    });
    // Enter selects the sole remaining match
    patientSearch.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        e.preventDefault();
        const options = Array.from(patientSelect.options).filter(o => o.value);
        if (options.length === 1) {
            patientSelect.value = options[0].value;
            patientSelect.dispatchEvent(new Event("change"));
        }
    });
}

// Study-modal actions
document.addEventListener("click", (e) => {
    if (e.target.closest("#run-alarm-replay-btn")) runAlarmReplay();
});
if (exportSessionBtn) exportSessionBtn.addEventListener("click", () => exportSessionLogs(true));
if (exportAllBtn) exportAllBtn.addEventListener("click", () => exportSessionLogs(false));

// Global keyboard shortcuts
document.addEventListener("keydown", (e) => {
    // Escape always closes the topmost dialog, even while typing
    if (e.key === "Escape") {
        if (!actionModal.classList.contains("hidden")) return closeModal(actionModal);
        if (!aboutModal.classList.contains("hidden")) return closeModal(aboutModal);
        return;
    }

    // Never hijack keys while the user is typing into a field
    const tag = (e.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "select" || tag === "textarea" || e.target.isContentEditable;
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;

    switch (e.key) {
        case "/":
            e.preventDefault();
            if (patientSearch) patientSearch.focus();
            break;
        case "?":
            e.preventDefault();
            openAboutModal();
            break;
        case " ":
            if (!playBtn.disabled || !pauseBtn.classList.contains("hidden")) {
                e.preventDefault();
                (isPlaying ? pauseBtn : playBtn).click();
            }
            break;
        case "ArrowLeft":
            if (!timelineSlider.disabled) {
                e.preventDefault();
                timelineSlider.value = Math.max(+timelineSlider.min, +timelineSlider.value - 1);
                timelineSlider.dispatchEvent(new Event("input"));
            }
            break;
        case "ArrowRight":
            if (!timelineSlider.disabled) {
                e.preventDefault();
                timelineSlider.value = Math.min(+timelineSlider.max, +timelineSlider.value + 1);
                timelineSlider.dispatchEvent(new Event("input"));
            }
            break;
        case "l":
        case "L":
            layoutToggleBtn.click();
            break;
        case "m":
        case "M":
            toggleMuteBtn.click();
            break;
    }
});

// Init connections on load
window.onload = () => {
    initCounterbalancing();
    initStaticWebSocket();
    fetchLatencyStats();
    // Populates the dataset provenance badge from real metadata on first paint
    fetchAcademicMetrics();

    // Chrome initial voice fetch requirement
    if (window.speechSynthesis) {
        window.speechSynthesis.getVoices();
    }
};

/* ---------------------------------------------------------------------------
   Tab-lifetime heartbeat.

   The server has no way to know a browser tab closed — HTTP is stateless and
   there is no browser-to-process signal for it. This socket stays open for the
   life of the page; the server treats its closure (after a short grace period,
   so a refresh does not count) as the tab being closed.
   --------------------------------------------------------------------------- */
let heartbeatSocket = null;
let heartbeatTimer = null;

function startHeartbeat() {
    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    try {
        heartbeatSocket = new WebSocket(`${wsScheme}://${window.location.host}/ws/heartbeat`);
    } catch (e) {
        return;
    }

    heartbeatSocket.onopen = () => {
        heartbeatTimer = setInterval(() => {
            if (heartbeatSocket && heartbeatSocket.readyState === WebSocket.OPEN) {
                heartbeatSocket.send("ping");
            }
        }, 3000);
    };

    heartbeatSocket.onclose = () => {
        if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
    };
}

startHeartbeat();
