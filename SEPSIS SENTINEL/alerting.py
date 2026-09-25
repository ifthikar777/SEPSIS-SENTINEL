"""
Alarm-suppression rule — single source of truth.

The rule was previously implemented twice: vectorised in app.py's
_compute_replay_counts() and row-by-row in live_stream.py's ReplayEngine.step().
Two copies of a clinical decision rule is exactly the kind of duplication that
drifts, so both now call into this module.

The rule itself is unchanged:

    An alert is raised when probability >= alert_threshold.
    It is *not suppressed* when either
        - it is sustained: the previous hour for the same patient also
          reached alert_threshold, or
        - severe organ dysfunction is present: Lactate > 2.0 AND O2Sat < 90.0
    Otherwise it is suppressed as a transient spike.

Two forms are provided from that one definition:
    * evaluate_series()  - vectorised, for scoring a whole corpus at once
    * AlertState         - scalar/incremental, for streaming one hour at a time
"""

# Organ-dysfunction thresholds. Named so the numbers appear once.
SEVERE_LACTATE = 2.0
SEVERE_O2SAT = 90.0

# Fallbacks used when a value is absent, matching the original implementations:
# a missing lactate cannot indicate hyperlactataemia, and a missing saturation
# cannot indicate hypoxia.
MISSING_LACTATE = 0.0
MISSING_O2SAT = 100.0


def is_severe_organ_dysfunction(lactate, o2sat):
    """Severe organ dysfunction marker: raised lactate AND low saturation."""
    lac = MISSING_LACTATE if lactate is None else lactate
    o2 = MISSING_O2SAT if o2sat is None else o2sat
    return bool(lac > SEVERE_LACTATE and o2 < SEVERE_O2SAT)


def classify(probability, alert_threshold, high_risk_threshold, is_sustained, severe_organ):
    """
    Map one hour to an alert status string.

    Returns one of NORMAL, SUPPRESSED_WARNING, WARNING_ALARM, CRITICAL_ALARM.
    """
    if probability < alert_threshold:
        return "NORMAL"
    if is_sustained or severe_organ:
        return "CRITICAL_ALARM" if probability >= high_risk_threshold else "WARNING_ALARM"
    return "SUPPRESSED_WARNING"


class AlertState:
    """
    Incremental form, for streaming a patient one hour at a time.

    Holds only the previous hour's probability, which is all "sustained" needs.
    """

    def __init__(self, alert_threshold, high_risk_threshold):
        self.alert_threshold = float(alert_threshold)
        self.high_risk_threshold = float(high_risk_threshold)
        self._previous_probability = None

    def step(self, probability, lactate=None, o2sat=None):
        """
        Advance one hour. Returns (status, raw_alert, is_sustained, severe_organ).

        `raw_alert` is whether the threshold was crossed at all, before suppression.
        """
        raw_alert = bool(probability >= self.alert_threshold)

        is_sustained = bool(
            self._previous_probability is not None
            and self._previous_probability >= self.alert_threshold
            and probability >= self.alert_threshold
        )
        severe_organ = is_severe_organ_dysfunction(lactate, o2sat)
        status = classify(
            probability, self.alert_threshold, self.high_risk_threshold,
            is_sustained, severe_organ
        )

        self._previous_probability = probability
        return status, raw_alert, is_sustained, severe_organ


def evaluate_series(probabilities, patient_ids, lactate, o2sat, alert_threshold):
    """
    Vectorised form, for scoring a whole corpus at once.

    Expects pandas Series aligned on a common index and already ordered by
    patient then hour. Returns (unsuppressed, suppressed) boolean Series.
    """
    previous = probabilities.groupby(patient_ids).shift(1)
    is_sustained = (previous >= alert_threshold) & (probabilities >= alert_threshold)

    severe_organ = (
        (lactate.fillna(MISSING_LACTATE) > SEVERE_LACTATE)
        & (o2sat.fillna(MISSING_O2SAT) < SEVERE_O2SAT)
    )

    unsuppressed = probabilities >= alert_threshold
    suppressed = unsuppressed & (is_sustained | severe_organ)
    return unsuppressed, suppressed
