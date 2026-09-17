"""The two drift filters the env loop runs in parallel.

DriftFilterV1 : rectified-excess EWMA estimator (from bnn_fl_cem_surprise_drift.py).
                Shadow filter, logged for comparison.
DriftFilterV2 : equal-weight, SIGNED drift filter with an empirical baseline (from
                mcts_drift_copy_v2.py).  The REAL filter -- its delta_bar drives
                forget_dirichlet.

Both are extracted verbatim; only the constant imports changed (now from config).
"""
from collections import deque

import numpy as np

from config import ETA, GAMMA_UNCERTAINTY, KAPPA


class DriftFilterV1:
    """Rectified-excess EWMA estimator of the squared drift lambda.

    lambda_hat = EWMA([delta_n - 1]_+); optionally an EWMA of the excess's second
    moment for a moment-matched std.  Reset at the change notification.
    """

    def __init__(self, eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY):
        self.eta = eta
        self.gamma_uncertainty = gamma_uncertainty
        self.reset()

    def reset(self):
        self.lambda_hat = 0.0      # filtered squared drift (per-dim, excess units)
        self._m2 = 0.0             # EWMA of excess^2 (for the Gamma-view variance)

    def update(self, delta_n):
        excess = max(delta_n - 1.0, 0.0)
        self.lambda_hat = (1.0 - self.eta) * self.lambda_hat + self.eta * excess
        if self.gamma_uncertainty:
            self._m2 = (1.0 - self.eta) * self._m2 + self.eta * (excess ** 2)
        return self.lambda_hat

    @property
    def delta_bar(self):
        """Filtered, calibration-relative surprise (baseline 1)."""
        return 1.0 + self.lambda_hat

    @property
    def lambda_sd(self):
        """Moment-matched standard deviation of the filtered excess."""
        if not self.gamma_uncertainty:
            return 0.0
        var = max(self._m2 - self.lambda_hat ** 2, 0.0)
        # EWMA of an i.i.d. stream has variance ~ eta/(2-eta) of the sample var.
        return float(np.sqrt(var * self.eta / (2.0 - self.eta)))

    def drift_estimate(self, kappa=KAPPA):
        """Point (risk-neutral) or risk-averse (kappa>0) squared-drift estimate."""
        return max(self.lambda_hat + kappa * self.lambda_sd, 0.0)


class DriftFilterV2:
    """Equal-weight, SIGNED drift filter with an EMPIRICAL baseline (v2).

    Every update() BEFORE the first reset() (the pre-change phase) is treated as
    unchanged and folded into the baseline b; reset() (fired at the change
    notification) freezes b and switches to detection, where lambda_hat is the
    signed equal-weight mean of (delta_n - b).

    window=None (default) averages over ALL post-change samples (the original
    cumulative behavior).  window=k averages over only the LAST k samples --
    set k to the forget period so each forget application consumes exactly the
    evidence gathered since the previous one (no sample drives two
    applications).  Only sensible when k is large enough to average out the
    per-sample noise in delta_n; with k ~ 1 the estimate is a single draw and
    the (shrink-only) forgetting compounds noise instead of canceling it.
    """

    def __init__(self, eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY,
                 default_baseline=1.0, window=None):
        # eta kept for call-site compatibility; unused (equal weights, not EWMA).
        self.gamma_uncertainty = gamma_uncertainty
        self.default_baseline = default_baseline   # used until b is calibrated
        self.window = window
        self._b_sum = 0.0
        self._b_n = 0
        self._calibrating = True
        self._reset_detection()

    @property
    def baseline(self):
        return self._b_sum / self._b_n if self._b_n > 0 else self.default_baseline

    def _reset_detection(self):
        self._sum = 0.0
        self._sumsq = 0.0
        self._n = 0
        self._buf = deque(maxlen=self.window) if self.window else None

    def reset(self):
        # Change notification: stop calibrating, KEEP the learned baseline, and
        # clear the post-change drift accumulators.
        self._calibrating = False
        self._reset_detection()

    def update(self, delta_n):
        if self._calibrating:
            # pre-change == known-unchanged: fold into the empirical baseline b
            self._b_sum += delta_n
            self._b_n += 1
            return self.lambda_hat
        excess = delta_n - self.baseline     # SIGNED, EMPIRICAL baseline (not 1.0)
        if self._buf is not None:
            self._buf.append(excess)         # windowed: only the last k samples
        else:
            self._sum += excess
            self._sumsq += excess * excess
            self._n += 1
        return self.lambda_hat

    @property
    def lambda_hat(self):
        if self._buf is not None:
            return float(np.mean(self._buf)) if self._buf else 0.0
        return self._sum / self._n if self._n > 0 else 0.0

    @property
    def delta_bar(self):
        # May be < 1 when the net post-shift evidence says "mostly unchanged".
        return 1.0 + self.lambda_hat

    @property
    def lambda_sd(self):
        if self._buf is not None:
            if not self.gamma_uncertainty or len(self._buf) < 2:
                return 0.0
            return float(np.sqrt(np.var(self._buf) / len(self._buf)))
        if not self.gamma_uncertainty or self._n < 2:
            return 0.0
        mean = self.lambda_hat
        var = max(self._sumsq / self._n - mean * mean, 0.0)
        return float(np.sqrt(var / self._n))   # std error of the equal-weight mean

    def drift_estimate(self, kappa=KAPPA):
        # SIGNED point estimate (no max(...,0)); the value reflects ALL evidence,
        # and forget() handles the non-negativity of the actual inflation.
        return self.lambda_hat + kappa * self.lambda_sd


class DualDriftFilter:
    """Dual-channel drift filter: mean shift (raw MSE nu2) + variance shift (delta_n).

    Tracks both the raw prediction residual (captures mean shifts) and the
    variance-normalised delta_n (captures variance/scale shifts).  The max of the
    two drift estimates drives forgetting, so either type of non-stationarity
    triggers re-inflation.

    Both channels calibrate their baselines during a warmup period (pre-change or
    early post-change).  Call calibrate() BEFORE reset() — the filter starts in
    calibration mode, reset() freezes the baselines and switches to detection.
    """

    def __init__(self, eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY):
        self.mean_channel = DriftFilterV2(eta=eta, gamma_uncertainty=gamma_uncertainty)
        self.var_channel = DriftFilterV2(eta=eta, gamma_uncertainty=gamma_uncertainty)
        # Start in calibration mode (DriftFilterV2 default)

    def reset(self):
        """Freeze calibrated baselines and switch to detection mode."""
        self.mean_channel.reset()
        self.var_channel.reset()

    def update(self, nu2, delta_n):
        """Feed both the raw squared error (nu2) and the variance-scaled delta_n."""
        self.mean_channel.update(nu2)
        self.var_channel.update(delta_n)
        return self.lambda_hat

    @property
    def is_calibrating(self):
        return self.mean_channel._calibrating and self.var_channel._calibrating

    @property
    def lambda_hat(self):
        return max(self.mean_channel.lambda_hat, self.var_channel.lambda_hat)

    @property
    def delta_bar(self):
        return 1.0 + self.lambda_hat

    @property
    def lambda_sd(self):
        return max(self.mean_channel.lambda_sd, self.var_channel.lambda_sd)

    def drift_estimate(self, kappa=KAPPA):
        m = self.mean_channel.drift_estimate(kappa)
        v = self.var_channel.drift_estimate(kappa)
        trigger = "mean" if m >= v else "var"
        return max(m, v), {"mean_ch": m, "var_ch": v, "trigger": trigger,
                           "mean_baseline": self.mean_channel.baseline,
                           "var_baseline": self.var_channel.baseline}
