"""Bayesian Online Change Point Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm with a Normal-inverse-gamma
conjugate prior for detecting distribution changes (drift) in sequential
data.  All computations are done in log-space for numerical stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class ChangePoint:
    """A detected change point in sequential data."""

    index: int
    probability: float
    prior_mean: float
    posterior_mean: float


# ── Helpers ────────────────────────────────────────────────────────────


def _logsumexp(log_values: list[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not log_values:
        return float("-inf")
    max_val = max(log_values)
    if max_val == float("-inf"):
        return float("-inf")
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


# ── BOCPD ──────────────────────────────────────────────────────────────


class BOCPD:
    """Bayesian Online Change Point Detection.

    Uses a Normal-inverse-gamma conjugate prior to model the data
    within each segment, and a constant hazard function for the
    change point prior.

    Parameters
    ----------
    hazard_rate : float
        Expected run length before a change point (default 100).
        The hazard function is ``H(r) = 1 / hazard_rate``.
    mu0 : float
        Prior mean of the Normal-inverse-gamma (default 0.0).
    kappa0 : float
        Prior precision scaling (default 1.0).
    alpha0 : float
        Prior shape parameter for inverse-gamma (default 1.0).
    beta0 : float
        Prior rate parameter for inverse-gamma (default 1.0).
    """

    def __init__(
        self,
        hazard_rate: float = 100.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Sufficient statistics per run length (growing lists)
        self._mu: list[float] = [mu0]
        self._kappa: list[float] = [kappa0]
        self._alpha: list[float] = [alpha0]
        self._beta: list[float] = [beta0]

        # Log run-length posterior (initially 100 % at r=0)
        self._log_run_length: list[float] = [0.0]  # log(1.0) = 0

        # Track change-point probabilities and MAP run lengths
        self._change_probs: list[float] = []
        self._map_run_lengths: list[int] = []

        self._t = 0  # time step counter

    def _hazard(self, r: int) -> float:
        """Constant hazard function: P(change) = 1 / hazard_rate."""
        return 1.0 / self.hazard_rate

    def _student_t_logpdf(
        self, x: float, mu: float, var: float, df: float
    ) -> float:
        """Log PDF of Student-t distribution.

        Parameters
        ----------
        x : float
            Observation.
        mu : float
            Location parameter.
        var : float
            Scale parameter (variance-like, not the t-distribution scale).
        df : float
            Degrees of freedom.
        """
        if var <= 0 or df <= 0:
            return float("-inf")
        return (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi * var)
            - (df + 1.0) / 2.0 * math.log(1.0 + (x - mu) ** 2 / (df * var))
        )

    def update(self, x: float) -> None:
        """Process one observation and update the run-length posterior.

        Implements the BOCPD message-passing algorithm:
        1. Compute predictive probabilities under Student-t for each run length.
        2. Compute growth probabilities (no change point).
        3. Compute change-point probability (run length resets to 0).
        4. Normalise the run-length distribution.
        5. Update sufficient statistics for each run length.
        """
        n_rl = len(self._log_run_length)

        # Step 1: predictive probabilities for each current run length
        log_pred = []
        for i in range(n_rl):
            # Predictive distribution is Student-t
            df = 2.0 * self._alpha[i]
            var = self._beta[i] * (self._kappa[i] + 1.0) / (
                self._alpha[i] * self._kappa[i]
            )
            log_p = self._student_t_logpdf(x, self._mu[i], var, df)
            log_pred.append(log_p)

        # Step 2: growth probabilities (run length grows by 1)
        log_growth = []
        for i in range(n_rl):
            h = self._hazard(i)
            log_h_comp = math.log(max(1.0 - h, 1e-300))
            log_growth.append(
                log_pred[i] + log_h_comp + self._log_run_length[i]
            )

        # Step 3: change-point probability (run length resets to 0)
        log_cp_terms = []
        for i in range(n_rl):
            h = self._hazard(i)
            log_h = math.log(max(h, 1e-300))
            log_cp_terms.append(
                log_pred[i] + log_h + self._log_run_length[i]
            )
        log_cp = _logsumexp(log_cp_terms)

        # Step 4: normalise
        new_log_rl = [log_cp] + log_growth
        log_evidence = _logsumexp(new_log_rl)

        if log_evidence > float("-inf"):
            new_log_rl = [v - log_evidence for v in new_log_rl]

        # Compute MAP run length for this time step
        map_rl = 0
        max_log = new_log_rl[0]
        for j in range(1, len(new_log_rl)):
            if new_log_rl[j] > max_log:
                max_log = new_log_rl[j]
                map_rl = j
        self._map_run_lengths.append(map_rl)

        # Store the cumulative probability of short run lengths (r <= 1)
        # This spikes at change points when the posterior concentrates
        # on short run lengths.
        short_rl_prob = 0.0
        for j in range(min(2, len(new_log_rl))):
            if new_log_rl[j] > -700:
                short_rl_prob += math.exp(new_log_rl[j])
        self._change_probs.append(short_rl_prob)

        # Step 5: update sufficient statistics
        # New run length r=0 gets the prior
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]

        # Existing run lengths update their stats
        for i in range(n_rl):
            kappa_new = self._kappa[i] + 1.0
            mu_new = (self._kappa[i] * self._mu[i] + x) / kappa_new
            alpha_new = self._alpha[i] + 0.5
            beta_new = (
                self._beta[i]
                + 0.5
                * self._kappa[i]
                * (x - self._mu[i]) ** 2
                / kappa_new
            )
            new_mu.append(mu_new)
            new_kappa.append(kappa_new)
            new_alpha.append(alpha_new)
            new_beta.append(beta_new)

        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta
        self._log_run_length = new_log_rl
        self._t += 1

    def detect(
        self, data: list[float], threshold: float = 0.5
    ) -> list[ChangePoint]:
        """Run BOCPD on a full sequence and return detected change points.

        Detection uses the MAP run length: a change point is flagged when
        the MAP run length drops below a short threshold, indicating the
        posterior has reset.  The ``threshold`` parameter controls how
        much cumulative probability on short run lengths (r <= 1) is
        required to declare a change point.

        Parameters
        ----------
        data : list[float]
            Sequential observations.
        threshold : float
            Minimum cumulative probability of short run lengths to flag
            as a change point (default 0.5).

        Returns
        -------
        list[ChangePoint]
            Detected change points.
        """
        # Reset state
        self._mu = [self.mu0]
        self._kappa = [self.kappa0]
        self._alpha = [self.alpha0]
        self._beta = [self.beta0]
        self._log_run_length = [0.0]
        self._change_probs = []
        self._map_run_lengths = []
        self._t = 0

        for x in data:
            self.update(x)

        if not self._map_run_lengths:
            return []

        change_points: list[ChangePoint] = []

        # Detect change points via MAP run length drops.
        # When MAP RL drops to 0 (after being > some minimum) and the
        # cumulative probability of short run lengths exceeds the threshold,
        # we flag it.
        min_run_before_cp = 3  # need at least 3 observations before flagging

        for i in range(1, len(self._map_run_lengths)):
            map_rl = self._map_run_lengths[i]
            prev_map_rl = self._map_run_lengths[i - 1]
            short_rl_prob = self._change_probs[i]

            # Change point: MAP RL drops significantly and short RL has
            # high probability
            is_drop = (
                map_rl <= 1
                and prev_map_rl >= min_run_before_cp
                and short_rl_prob > threshold
            )

            if is_drop:
                # Prior mean from the model before the change
                prior_mean = self.mu0

                # Posterior mean from the current r=0 or r=1 model
                posterior_mean = self._mu[0] if self._mu else self.mu0
                # If r=1 has higher probability, use that
                if len(self._mu) > 1 and map_rl == 1:
                    posterior_mean = self._mu[1]

                change_points.append(ChangePoint(
                    index=i,
                    probability=short_rl_prob,
                    prior_mean=prior_mean,
                    posterior_mean=posterior_mean,
                ))

        return change_points

    def get_run_length_posterior(self) -> list[float]:
        """Return the current run-length distribution as probabilities."""
        result = []
        for log_p in self._log_run_length:
            if log_p > -700:
                result.append(math.exp(log_p))
            else:
                result.append(0.0)
        return result
