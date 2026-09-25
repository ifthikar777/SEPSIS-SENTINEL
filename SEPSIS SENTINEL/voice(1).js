/* ============================================================================
   Voice commands — offline speech recognition
   ----------------------------------------------------------------------------
   Recognition runs entirely on this machine. The browser's built-in
   SpeechRecognition API is deliberately NOT used: Chrome implements it by
   streaming microphone audio to a cloud speech service, which both fails on an
   air-gapped machine and would send anything spoken near the microphone off the
   box. There is no fallback to it — if local recognition is unavailable, voice
   input is disabled and says so.

   Audio path: getUserMedia -> Web Audio -> downsample to 16 kHz mono PCM16 ->
   WebSocket /ws/voice -> Vosk on the server -> transcript back.

   Loaded after client.js, so it reuses that file's element references,
   showToast() and logUsabilityEvent().

   Safety: commands that place clinical orders or dismiss an alert do NOT act
   directly. They open the same confirmation dialog a click would and require a
   spoken "confirm"; "cancel" aborts.
   ========================================================================== */

const voiceListenBtn = document.getElementById("voice-listen-btn");
const voiceIcon = document.getElementById("voice-icon");
const voiceStatusText = document.getElementById("voice-status-text");
const voiceInputControl = document.getElementById("voice-input-control");
const voiceHud = document.getElementById("voice-hud");
const voiceHudTranscript = document.getElementById("voice-hud-transcript");
const voiceHudClose = document.getElementById("voice-hud-close");
const voiceHelpModal = document.getElementById("voice-help-modal");
const voiceHelpClose = document.getElementById("voice-help-close");
const voiceHelpOk = document.getElementById("voice-help-ok");
const voiceCmdGrid = document.getElementById("voice-cmd-grid");

// Target rate for the server-side recogniser.
const VOICE_SAMPLE_RATE = 16000;

let voiceListening = false;
let voiceSocket = null;
let voiceAudioContext = null;
let voiceMicStream = null;
let voiceProcessor = null;
let voiceSourceNode = null;
let voiceLocalAvailable = null;   // null until /api/voice_status has answered
let voiceUnavailableReason = null;

// Spoken digits, so "patient a p one five" resolves as well as "a p 15".
const DIGIT_WORDS = {
    zero: "0", oh: "0", o: "0", one: "1", won: "1", two: "2", to: "2", too: "2",
    three: "3", four: "4", for: "4", five: "5", six: "6", seven: "7",
    eight: "8", ate: "8", nine: "9"
};

const TEEN_WORDS = {
    ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
    sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19
};

const TENS_WORDS = {
    twenty: 20, thirty: 30, forty: 40, fourty: 40, fifty: 50,
    sixty: 60, seventy: 70, eighty: 80, ninety: 90
};

function normaliseSpeech(text) {
    return String(text || "").toLowerCase().trim().replace(/[.,!?;:]/g, "");
}

// Turn "a p zero zero one five", "a p 15" or "a fifteen" into a set letter and a
// numeric patient index.
function extractPatientToken(phrase) {
    const words = phrase.split(/\s+/);
    let set = "";
    let digitRun = "";
    let wholeValue = null;
    let pendingTens = 0;

    function flushTens() {
        if (pendingTens) {
            wholeValue = (wholeValue || 0) + pendingTens;
            pendingTens = 0;
        }
    }

    for (const w of words) {
        if (!set && (w === "a" || w === "alpha" || w === "eh")) { set = "A"; continue; }
        if (!set && (w === "b" || w === "bravo" || w === "be" || w === "bee")) { set = "B"; continue; }
        if (w === "p" || w === "patient" || w === "pee") continue;

        if (/^\d+$/.test(w)) { flushTens(); digitRun += w; continue; }
        if (TENS_WORDS[w] !== undefined) { flushTens(); pendingTens = TENS_WORDS[w]; continue; }
        if (TEEN_WORDS[w] !== undefined) { flushTens(); wholeValue = (wholeValue || 0) + TEEN_WORDS[w]; continue; }

        if (DIGIT_WORDS[w] !== undefined) {
            if (pendingTens) {
                wholeValue = (wholeValue || 0) + pendingTens + parseInt(DIGIT_WORDS[w], 10);
                pendingTens = 0;
            } else {
                digitRun += DIGIT_WORDS[w];
            }
        }
    }
    flushTens();

    if (!set) return null;
    let value = null;
    if (digitRun) value = parseInt(digitRun, 10);
    else if (wholeValue !== null) value = wholeValue;
    if (value === null || Number.isNaN(value)) return null;

    return { set: set, value: value };
}

// Match a spoken patient reference against whatever is in the dropdown now.
function resolvePatientFromSpeech(phrase) {
    const token = extractPatientToken(phrase);
    if (!token) return null;
    const options = Array.from(patientSelect.options).map(o => o.value).filter(Boolean);
    for (const v of options) {
        if (!v.startsWith(token.set + "_")) continue;
        if (parseInt(v.replace(/^[AB]_p/, ""), 10) === token.value) return v;
    }
    return null;
}

function setVoiceStatus(text, active) {
    if (voiceStatusText) voiceStatusText.textContent = text;
    if (voiceInputControl) voiceInputControl.classList.toggle("listening", !!active);
    if (voiceIcon) voiceIcon.className = active ? "fa-solid fa-microphone" : "fa-solid fa-microphone-slash";
    if (voiceListenBtn) voiceListenBtn.setAttribute("aria-pressed", active ? "true" : "false");
    if (voiceHud) voiceHud.classList.toggle("hidden", !active);
}

/** Disable the control outright — used when local recognition cannot run. */
function markVoiceUnavailable(reason) {
    voiceLocalAvailable = false;
    voiceUnavailableReason = reason || "Local speech recognition is unavailable.";
    if (voiceInputControl) voiceInputControl.classList.add("unavailable");
    if (voiceListenBtn) {
        voiceListenBtn.disabled = true;
        voiceListenBtn.setAttribute("data-tooltip", voiceUnavailableReason);
        voiceListenBtn.setAttribute("aria-label", "Voice input unavailable: " + voiceUnavailableReason);
    }
    setVoiceStatus("Voice Unavailable", false);
}

// Command table. Checked in order, so specific phrases must precede general ones.
const VOICE_COMMANDS = [
    { say: "help", test: function (p) { return /^(help|commands|what can i say)$/.test(p); },
      run: function () { return openVoiceHelp(); } },

    { say: "select patient A 15", test: function (p) { return /(patient|select|open|show)/.test(p) && !!extractPatientToken(p); },
      run: function (p) {
          const pid = resolvePatientFromSpeech(p);
          if (!pid) return "No matching patient in the current list";
          patientSelect.value = pid;
          patientSelect.dispatchEvent(new Event("change"));
          return "Selected " + pid;
      } },

    { say: "start replay", test: function (p) { return /^(start|play|begin)( replay| playback)?$/.test(p); },
      run: function () { if (playBtn.disabled) return "Select a patient first"; playBtn.click(); return "Replay started"; } },
    { say: "pause", test: function (p) { return /^(pause|halt)( replay| playback)?$/.test(p); },
      run: function () { if (pauseBtn.classList.contains("hidden")) return "Replay is not running"; pauseBtn.click(); return "Replay paused"; } },
    { say: "reset", test: function (p) { return /^reset( replay| playback)?$/.test(p); },
      run: function () { if (resetBtn.disabled) return "Nothing to reset"; resetBtn.click(); return "Replay reset"; } },

    { say: "next hour / previous hour", test: function (p) { return /^(next|previous|prev|back|forward)( hour| step)?$/.test(p); },
      run: function (p) {
          if (timelineSlider.disabled) return "Select a patient first";
          const fwd = /next|forward/.test(p);
          timelineSlider.value = fwd
              ? Math.min(+timelineSlider.max, +timelineSlider.value + 1)
              : Math.max(+timelineSlider.min, +timelineSlider.value - 1);
          timelineSlider.dispatchEvent(new Event("input"));
          return "Hour " + timelineSlider.value;
      } },
    { say: "go to hour 12", test: function (p) { return /^(go to |jump to )?hour \d+$/.test(p); },
      run: function (p) {
          if (timelineSlider.disabled) return "Select a patient first";
          const h = parseInt(p.match(/hour (\d+)/)[1], 10);
          timelineSlider.value = Math.max(+timelineSlider.min, Math.min(+timelineSlider.max, h));
          timelineSlider.dispatchEvent(new Event("input"));
          return "Hour " + timelineSlider.value;
      } },

    { say: "consult advisor", test: function (p) { return /(consult|ask)( the)?( sepsis)?( ai)?( advisor| doctor)/.test(p); },
      run: function () { if (consultAdvisorBtn.disabled) return "Select a patient first"; consultAdvisorBtn.click(); return "Consulting advisor"; } },
    { say: "read aloud", test: function (p) { return /^(read( it)?( aloud| out)?|speak)$/.test(p); },
      run: function () { readAloudBtn.click(); return "Reading assessment"; } },

    // The four below open the confirmation dialog instead of acting directly.
    { say: "order blood cultures", test: function (p) { return /blood culture/.test(p); },
      run: function () { if (btnOrderCultures.disabled) return "Select a patient first"; btnOrderCultures.click(); return "Confirm blood cultures?"; } },
    { say: "start IV fluids", test: function (p) { return /(fluid|resuscitation)/.test(p); },
      run: function () { if (btnOrderFluids.disabled) return "Select a patient first"; btnOrderFluids.click(); return "Confirm IV fluids?"; } },
    { say: "order antibiotics", test: function (p) { return /(antibiotic|antimicrobial)/.test(p); },
      run: function () { if (btnOrderAntibiotics.disabled) return "Select a patient first"; btnOrderAntibiotics.click(); return "Confirm antibiotics?"; } },
    { say: "dismiss alert", test: function (p) { return /(dismiss|silence)( the)?( alert| alarm)?/.test(p); },
      run: function () { if (btnDismissAlert.disabled) return "Select a patient first"; btnDismissAlert.click(); return "Confirm dismissal?"; } },

    { say: "confirm", test: function (p) { return /^(confirm|yes|accept|proceed)$/.test(p); },
      run: function () {
          if (actionModal.classList.contains("hidden")) return "Nothing to confirm";
          const buttons = Array.from(document.querySelectorAll("#action-modal-footer button"));
          const btn = buttons.filter(function (b) { return !/cancel/i.test(b.textContent); })[0];
          if (!btn) return "Nothing to confirm";
          btn.click();
          return "Confirmed";
      } },
    { say: "cancel", test: function (p) { return /^(cancel|no|never mind|nevermind|abort|stop)$/.test(p); },
      run: function () {
          if (!actionModal.classList.contains("hidden")) { closeModal(actionModal); return "Cancelled"; }
          if (!voiceHelpModal.classList.contains("hidden")) { closeVoiceHelp(); return "Closed"; }
          if (!aboutModal.classList.contains("hidden")) { closeModal(aboutModal); return "Closed"; }
          return "Nothing to cancel";
      } },

    { say: "switch layout", test: function (p) { return /(switch|toggle|change)( the)?( layout| view| columns?)/.test(p); },
      run: function () { layoutToggleBtn.click(); return "Layout: " + currentLayout; } },
    { say: "study info", test: function (p) { return /(study info|show metrics|show study|open study)/.test(p); },
      run: function () { openAboutModal(); return "Study info open"; } },
    { say: "mute / unmute", test: function (p) { return /^(mute|unmute)( voice| readback)?$/.test(p); },
      run: function () { toggleMuteBtn.click(); return isMuted ? "Readback muted" : "Readback active"; } },
    { say: "stop listening", test: function (p) { return /^(stop listening|voice off|sleep)$/.test(p); },
      run: function () { stopVoiceListening(); return "Voice commands off"; } }
];

function handleVoiceTranscript(rawText) {
    const phrase = normaliseSpeech(rawText);
    if (voiceHudTranscript) voiceHudTranscript.textContent = "“" + rawText + "”";
    if (!phrase) return;

    for (const cmd of VOICE_COMMANDS) {
        let matched = false;
        try { matched = !!cmd.test(phrase); } catch (e) { matched = false; }
        if (!matched) continue;

        let feedback;
        try { feedback = cmd.run(phrase); } catch (e) { feedback = "Command failed"; }
        showToast(feedback || "Done", "info", 3000);
        logUsabilityEvent("voice_command", "voice", phrase + " -> " + (feedback || "ok"));
        return;
    }

    showToast("Not recognised: “" + rawText + "”. Say “help” for commands.", "warning", 4000);
    logUsabilityEvent("voice_command", "voice", "unrecognised: " + phrase);
}

/** Ask the server whether local recognition can run at all. */
async function checkVoiceAvailability() {
    try {
        const res = await fetch("/api/voice_status");
        const data = await res.json();
        if (!data.local_available) {
            markVoiceUnavailable(data.reason || "Offline speech model not installed.");
            return false;
        }
        voiceLocalAvailable = true;
        return true;
    } catch (e) {
        markVoiceUnavailable("Could not reach the local speech service.");
        return false;
    }
}

/**
 * Linear-interpolation downsample to 16 kHz and convert to signed 16-bit PCM,
 * which is what the server-side recogniser expects.
 */
function toPcm16(float32, inputRate) {
    const ratio = inputRate / VOICE_SAMPLE_RATE;
    const outLength = Math.floor(float32.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
        const src = i * ratio;
        const lo = Math.floor(src);
        const hi = Math.min(lo + 1, float32.length - 1);
        const sample = float32[lo] + (float32[hi] - float32[lo]) * (src - lo);
        const clamped = Math.max(-1, Math.min(1, sample));
        out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return out;
}

async function startVoiceListening() {
    if (voiceListening) return;

    if (voiceLocalAvailable === null) {
        const ok = await checkVoiceAvailability();
        if (!ok) return;
    }
    if (!voiceLocalAvailable) {
        showToast(voiceUnavailableReason, "error", 7000);
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        markVoiceUnavailable("This browser cannot capture microphone audio.");
        return;
    }

    try {
        voiceMicStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
        });
    } catch (e) {
        showToast("Microphone permission denied.", "error", 5000);
        return;
    }

    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    voiceSocket = new WebSocket(`${wsScheme}://${window.location.host}/ws/voice`);
    voiceSocket.binaryType = "arraybuffer";

    voiceSocket.onmessage = (event) => {
        let msg;
        try { msg = JSON.parse(event.data); } catch (e) { return; }

        if (msg.type === "transcript" && msg.text) {
            handleVoiceTranscript(msg.text);
        } else if (msg.type === "voice_unavailable") {
            markVoiceUnavailable(msg.reason || "Local speech recognition is unavailable.");
            stopVoiceListening();
            showToast(msg.reason || "Local speech recognition is unavailable.", "error", 7000);
        } else if (msg.type === "voice_ready") {
            voiceListening = true;
            setVoiceStatus("Listening…", true);
            showToast("Voice commands on (offline). Say “help” for the list.", "success", 3500);
        } else if (msg.type === "voice_error") {
            showToast("Voice error: " + (msg.message || "unknown"), "error", 5000);
            stopVoiceListening();
        }
    };

    voiceSocket.onerror = () => {
        showToast("Could not connect to the local speech service.", "error", 5000);
        stopVoiceListening();
    };

    voiceSocket.onclose = () => {
        if (voiceListening) stopVoiceListening();
    };

    voiceSocket.onopen = () => {
        voiceAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        voiceSourceNode = voiceAudioContext.createMediaStreamSource(voiceMicStream);

        // ScriptProcessor is deprecated but is the widest-support option that needs
        // no separate worklet file — which matters for an offline, file-local deploy.
        voiceProcessor = voiceAudioContext.createScriptProcessor(4096, 1, 1);
        voiceProcessor.onaudioprocess = (e) => {
            if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;
            const pcm = toPcm16(e.inputBuffer.getChannelData(0), voiceAudioContext.sampleRate);
            voiceSocket.send(pcm.buffer);
        };

        voiceSourceNode.connect(voiceProcessor);
        // Route to a muted gain node so the processor runs without echoing audio back.
        const sink = voiceAudioContext.createGain();
        sink.gain.value = 0;
        voiceProcessor.connect(sink);
        sink.connect(voiceAudioContext.destination);
    };
}

function stopVoiceListening() {
    voiceListening = false;

    if (voiceProcessor) { try { voiceProcessor.disconnect(); } catch (e) {} voiceProcessor = null; }
    if (voiceSourceNode) { try { voiceSourceNode.disconnect(); } catch (e) {} voiceSourceNode = null; }
    if (voiceAudioContext) { try { voiceAudioContext.close(); } catch (e) {} voiceAudioContext = null; }
    if (voiceMicStream) {
        voiceMicStream.getTracks().forEach(t => { try { t.stop(); } catch (e) {} });
        voiceMicStream = null;
    }
    if (voiceSocket) {
        try { if (voiceSocket.readyState === WebSocket.OPEN) voiceSocket.close(); } catch (e) {}
        voiceSocket = null;
    }

    if (voiceLocalAvailable === false) {
        setVoiceStatus("Voice Unavailable", false);
    } else {
        setVoiceStatus("Voice Cmds Off", false);
    }
}

function toggleVoiceListening() {
    if (voiceListening) { stopVoiceListening(); } else { startVoiceListening(); }
}

function openVoiceHelp() {
    if (voiceCmdGrid && !voiceCmdGrid.childElementCount) {
        voiceCmdGrid.innerHTML = VOICE_COMMANDS.map(function (c) {
            return '<div class="voice-cmd-row"><span class="voice-cmd-phrase">“'
                + escapeHtml(c.say) + '”</span></div>';
        }).join("");
    }
    openModal(voiceHelpModal);
    return "Command list open";
}

function closeVoiceHelp() { closeModal(voiceHelpModal); }

if (voiceListenBtn) voiceListenBtn.addEventListener("click", toggleVoiceListening);
if (voiceHudClose) voiceHudClose.addEventListener("click", stopVoiceListening);
if (voiceHelpClose) voiceHelpClose.addEventListener("click", closeVoiceHelp);
if (voiceHelpOk) voiceHelpOk.addEventListener("click", closeVoiceHelp);
if (voiceHelpModal) {
    voiceHelpModal.addEventListener("mousedown", function (e) {
        if (e.target === voiceHelpModal) closeVoiceHelp();
    });
}

// "V" toggles listening; Escape also closes the voice help dialog.
document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && voiceHelpModal && !voiceHelpModal.classList.contains("hidden")) {
        closeVoiceHelp();
        return;
    }
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea" || e.target.isContentEditable) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "v" || e.key === "V") toggleVoiceListening();
});

// Resolve availability on load so the control shows its true state before the
// user clicks it.
checkVoiceAvailability();
