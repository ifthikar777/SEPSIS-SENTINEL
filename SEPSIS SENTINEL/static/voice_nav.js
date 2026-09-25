/* ============================================================================
   Voice-only operation
   ----------------------------------------------------------------------------
   Driving the dashboard by voice alone needs more than commands: the system has to
   answer back. A user who is not looking at the screen gets nothing from a toast,
   so these commands speak their result aloud through the existing speechSynthesis
   readback, and cover the controls that previously had no spoken equivalent
   (playback speed, stepping through patients, what-if values).

   Loaded after voice.js; these entries are placed ahead of the base command table
   so phrases like "read patient" are not swallowed by the broader patient-selection
   rule.
   ========================================================================== */

// Speak and toast together, so the answer lands whether or not the user is looking.
function respond(text) {
    if (!isMuted) speakText(text);
    return text;
}

function readEl(id, fallback) {
    const el = document.getElementById(id);
    if (!el) return fallback || "not available";
    const raw = (el.tagName === "INPUT" || el.tagName === "SELECT") ? el.value : el.textContent;
    const t = String(raw || "").trim();
    return (!t || t === "—" || t === "N/A") ? (fallback || "not available") : t;
}

// Spoken names for the what-if inputs.
const FEATURE_ALIASES = [
    [/heart rate|pulse/, "HR", "heart rate"],
    [/oxygen|o two|o2|saturation|sat\b/, "O2Sat", "oxygen saturation"],
    [/temperature|temp\b/, "Temp", "temperature"],
    [/systolic|blood pressure|s b p/, "SBP", "systolic blood pressure"],
    [/mean arterial|\bmap\b/, "MAP", "mean arterial pressure"],
    [/respiration|respiratory|resp\b/, "Resp", "respiration rate"],
    [/white blood|white cell|\bwbc\b|w b c/, "WBC", "white cell count"],
    [/creatinine/, "Creatinine", "creatinine"],
    [/bilirubin/, "Bilirubin_total", "bilirubin"],
    [/platelet/, "Platelets", "platelets"],
    [/glucose|sugar/, "Glucose", "glucose"],
    [/lactate|lactic/, "Lactate", "lactate"],
    [/\bp h\b|\bph\b/, "pH", "pH"],
    [/carbon dioxide|c o two|paco2|pa c o/, "PaCO2", "PaCO2"],
    [/\bbun\b|urea/, "BUN", "BUN"],
    [/hemoglobin|haemoglobin|\bhgb\b/, "Hgb", "hemoglobin"],
    [/\bptt\b|thromboplastin/, "PTT", "PTT"],
    [/bicarbonate|hco3|h c o/, "HCO3", "bicarbonate"],
    [/\bage\b/, "Age", "age"]
];

// "four point two", "4.2", "one twenty" -> a number.
function parseSpokenNumber(text) {
    const direct = text.match(/-?\d+(\.\d+)?/);
    if (direct) return parseFloat(direct[0]);

    let whole = null;
    let decimals = "";
    let afterPoint = false;
    let pendingTens = 0;

    for (const w of text.split(/\s+/)) {
        if (w === "point" || w === "decimal") { afterPoint = true; continue; }

        let val = null;
        if (DIGIT_WORDS[w] !== undefined) val = parseInt(DIGIT_WORDS[w], 10);
        else if (TEEN_WORDS[w] !== undefined) val = TEEN_WORDS[w];
        else if (TENS_WORDS[w] !== undefined) { pendingTens = TENS_WORDS[w]; continue; }
        else if (w === "hundred" && whole !== null) { whole *= 100; continue; }
        if (val === null) continue;

        if (afterPoint) { decimals += String(val); continue; }
        if (pendingTens) { whole = (whole || 0) + pendingTens + val; pendingTens = 0; continue; }
        whole = (whole === null) ? val : whole * 10 + val;
    }
    if (pendingTens) whole = (whole || 0) + pendingTens;
    if (whole === null && !decimals) return null;
    return parseFloat(String(whole === null ? 0 : whole) + (decimals ? "." + decimals : ""));
}

const VOICE_ONLY_COMMANDS = [
    // ---- Spoken readback: the dashboard answers back --------------------
    { say: "what is the risk", test: p => /what.*(risk|score)|read (the )?risk|how bad/.test(p),
      run: () => respond(
          "Sepsis risk is " + readEl("risk-percentage") + ", " + readEl("risk-category", "unknown")
          + ". Decision certainty " + readEl("model-decision-certainty")
          + ". Data completeness " + readEl("data-completeness") + ".") },

    { say: "read vitals", test: p => /(read|say|what are)( the)? vitals/.test(p),
      run: () => respond(
          "Heart rate " + readEl("tile-hr-val")
          + ". Oxygen saturation " + readEl("tile-o2sat-val") + " percent"
          + ". Temperature " + readEl("tile-temp-val")
          + ". Systolic " + readEl("tile-sbp-val")
          + ". Mean arterial pressure " + readEl("tile-map-val")
          + ". Respiration " + readEl("tile-resp-val") + ".") },

    { say: "read patient", test: p => /read( the)? patient|who is( the)? patient|patient details/.test(p),
      run: () => respond(
          "Patient " + readEl("patient-id-display", "none selected")
          + ". Age " + readEl("demog-age")
          + ". Gender " + readEl("demog-sex")
          + ". Site " + readEl("demog-dataset")
          + ". " + readEl("current-hour-display", "") + ".") },

    { say: "why / read top factors", test: p => /^(why|explain)$|read (the )?(top )?(factors|attributions|shap)/.test(p),
      run: () => {
          const rows = Array.from(document.querySelectorAll("#shap-bars-container .shap-row")).slice(0, 3);
          if (!rows.length) return respond("No feature attributions yet. Select a patient first.");
          const parts = rows.map(r => {
              const name = r.querySelector(".shap-feat-name");
              const val = r.querySelector(".shap-feat-val");
              const dir = r.querySelector(".shap-bar.positive") ? "increasing" : "decreasing";
              return (name ? name.textContent.trim() : "unknown") + ", " + dir + " risk, "
                  + (val ? val.textContent.trim() : "");
          });
          return respond("Top contributing factors: " + parts.join(". ") + ".");
      } },

    { say: "alert status", test: p => /alert status|any alerts|read( the)? alerts?/.test(p),
      run: () => {
          const banner = readEl("alert-banner-text", "unknown");
          const cards = document.querySelectorAll("#active-alerts-container .alert-item-card").length;
          return respond(banner + ". " + (cards
              ? cards + " active alert card" + (cards > 1 ? "s" : "")
              : "No active clinical alerts") + ".");
      } },

    { say: "where am I", test: p => /^(where am i|read screen|summarise|summarize|status)$/.test(p),
      run: () => respond(
          "Patient " + readEl("patient-id-display", "none selected")
          + ", " + readEl("current-hour-display", "")
          + ". Risk " + readEl("risk-percentage") + ", " + readEl("risk-category", "unknown")
          + ". " + readEl("alert-banner-text", "")
          + ". Layout is " + currentLayout + ".") },

    // ---- Controls that had no spoken equivalent -------------------------
    { say: "next patient / previous patient", test: p => /^(next|previous|prev|last) patient$/.test(p),
      run: p => {
          const opts = Array.from(patientSelect.options).filter(o => o.value);
          if (!opts.length) return respond("No patients loaded.");
          const idx = opts.findIndex(o => o.value === patientSelect.value);
          const step = /next/.test(p) ? 1 : -1;
          const target = Math.min(opts.length - 1, Math.max(0, (idx < 0 ? 0 : idx) + step));
          patientSelect.value = opts[target].value;
          patientSelect.dispatchEvent(new Event("change"));
          return respond("Patient " + opts[target].value);
      } },

    { say: "set speed to 60", test: p => /(speed|playback rate|rate)/.test(p) && /\b(1|10|60|one|ten|sixty)\b/.test(p),
      run: p => {
          const n = /sixty|60/.test(p) ? "60" : (/\bten\b|\b10\b/.test(p) ? "10" : "1");
          playbackSpeed.value = n;
          playbackSpeed.dispatchEvent(new Event("change"));
          return respond("Playback speed " + n + " times");
      } },

    { say: "set lactate to 4.2", test: p => /^(set|change|make)\s/.test(p) && FEATURE_ALIASES.some(a => a[0].test(p)),
      run: p => {
          const match = FEATURE_ALIASES.find(a => a[0].test(p));
          if (!match) return respond("I did not catch which value to change.");
          const id = match[1];
          const label = match[2];
          const field = document.getElementById(id);
          if (!field) return respond("That field is not available.");

          // Remove the feature words first, so digits inside them (o2, paco2, p h)
          // are not mistaken for the value.
          const valuePart = p.replace(match[0], " ")
                             .replace(/^(set|change|make)\s+/, "")
                             .replace(/\bto\b/g, " ");
          const num = parseSpokenNumber(valuePart);
          if (num === null || Number.isNaN(num)) {
              return respond("I did not catch a number for " + label + ".");
          }

          field.value = String(num);
          field.dispatchEvent(new Event("input", { bubbles: true }));
          field.dispatchEvent(new Event("change", { bubbles: true }));

          // Give the model a moment to score the edited vector before reporting.
          setTimeout(() => {
              if (!isMuted) speakText("Risk is now " + readEl("risk-percentage") + ".");
          }, 700);
          return label + " set to " + num;
      } }
];

// Placed ahead of the base table so readback phrases win over broader rules.
VOICE_COMMANDS.unshift.apply(VOICE_COMMANDS, VOICE_ONLY_COMMANDS);
