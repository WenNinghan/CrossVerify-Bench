"""Exact analytical Markov chain theory for iterative multi-agent cross-verification.

Models the iterative ``Generator -> Verifier -> Conditional Repair`` pipeline on state
space ``{C (Correct), I (Incorrect)}`` (both in the homogeneous re-verification regime
and in the absorbing early-exit regime used by conditional repair orchestrators).

Key Theoretical Results
-----------------------
1. **Single-Step & Homogeneous Net-Gain Condition (Verification Harm Theorem)**:
   Given initial generator accuracy ``p_0``, verifier Type-I false rejection
   ``alpha = P(reject | C)``, Type-II false acceptance ``beta = P(accept | I)``,
   repair recovery ``q_{I->C} = P(C | I, repair)``, and false-critique corruption
   ``q_{C->I} = P(I | C, repair)``:

       Delta p_1 = p_1 - p_0 = (1 - p_0)(1 - beta) q_{I->C} - p_0 * alpha * q_{C->I}

   Verification strictly improves expected accuracy at step 1 (and asymptotically in
   the homogeneous chain) if and only if:

       alpha / (1 - beta) < ((1 - p_0) / p_0) * (q_{I->C} / q_{C->I})

   Consequently, as generator capability ``p_0 -> 1``, the tolerable false-rejection
   ratio shrinks as ``O(1 - p_0)``: pairing a strong generator with an uncalibrated
   verifier actively degrades accuracy while multiplying token expenditure.

2. **Closed-Form T-Step Accuracy & Expected Token Cost**:
   - *Homogeneous chain* (re-verified every round ``t = 1..T``):
         p_T = p* + (p_0 - p*) * (1 - u - v)^T
     where ``u = alpha * q_{C->I}``, ``v = (1 - beta) * q_{I->C}``, and
     ``p* = v / (u + v)``.
   - *Early-stopping conditional repair chain* (halts immediately upon verifier
     acceptance, executing at most ``T`` rounds):
     Transient open-candidate distribution evolves as ``s_t = Q^t s_0`` with exact
     2x2 sub-stochastic operator ``Q``, yielding exact closed-form ``p_T`` and
     ``E[Cost_T]`` without numerical rollout error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VerificationParams:
    """Parameters governing a Generator -> Verifier -> Conditional Repair Markov chain.

    Attributes:
        p0: Initial generator accuracy P(X_0 = C) in [0, 1].
        alpha: Verifier Type-I false-rejection rate P(reject | C) in [0, 1].
        beta: Verifier Type-II false-acceptance rate P(accept | I) in [0, 1].
        q_i2c: Conditional repair recovery probability P(X_t = C | X_{t-1} = I, rejected).
        q_c2i: Conditional repair corruption probability P(X_t = I | X_{t-1} = C, rejected).
        cost_gen: Expected token cost of initial candidate generation.
        cost_ver: Expected token cost of a single verifier invocation.
        cost_rep: Expected token cost of a single conditional repair invocation.
    """

    p0: float
    alpha: float
    beta: float
    q_i2c: float
    q_c2i: float
    cost_gen: float = 600.0
    cost_ver: float = 350.0
    cost_rep: float = 650.0

    def __post_init__(self) -> None:
        for name in ("p0", "alpha", "beta", "q_i2c", "q_c2i"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Probability parameter '{name}' must be in [0, 1], got {val}.")
        for name in ("cost_gen", "cost_ver", "cost_rep"):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(f"Cost parameter '{name}' must be non-negative, got {val}.")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class MarkovVerifierChain:
    """Exact analytical solver and Monte Carlo validator for cross-verification loops."""

    def __init__(self, params: VerificationParams) -> None:
        self.params = params

    @property
    def u_c_to_i(self) -> float:
        """Per-round effective corruption rate u = P(I | C) = alpha * q_{C->I}."""
        return self.params.alpha * self.params.q_c2i

    @property
    def v_i_to_c(self) -> float:
        """Per-round effective recovery rate v = P(C | I) = (1 - beta) * q_{I->C}."""
        return (1.0 - self.params.beta) * self.params.q_i2c

    def transition_matrix_homogeneous(self) -> np.ndarray:
        """2x2 column-stochastic transition matrix on [C, I]^T for homogeneous rounds."""
        u = self.u_c_to_i
        v = self.v_i_to_c
        return np.array(
            [
                [1.0 - u, v],
                [u, 1.0 - v],
            ],
            dtype=np.float64,
        )

    def transient_matrix_early_stop(self) -> np.ndarray:
        """2x2 sub-stochastic operator Q mapping open state s_{t-1} -> s_t after rejection+repair."""
        a = self.params.alpha
        b = self.params.beta
        q_ic = self.params.q_i2c
        q_ci = self.params.q_c2i
        return np.array(
            [
                [a * (1.0 - q_ci), (1.0 - b) * q_ic],
                [a * q_ci, (1.0 - b) * (1.0 - q_ic)],
            ],
            dtype=np.float64,
        )

    def one_step_gain(self) -> float:
        """Exact single-round verification accuracy gain Delta p_1 = p_1 - p_0.

        Identical for both early-stopping and homogeneous chains at T = 1.
        """
        p0 = self.params.p0
        return (1.0 - p0) * self.v_i_to_c - p0 * self.u_c_to_i

    def is_net_positive_one_step(self) -> bool:
        """Return True iff adding a 1-step verifier strictly increases accuracy (Delta p_1 > 0)."""
        return self.one_step_gain() > 0.0

    def critical_alpha(self, beta: Optional[float] = None) -> float:
        """Maximum tolerable Type-I false rejection alpha*(beta) before Delta p_1 < 0.

        Values clamped to [0, 1] for phase-boundary plotting; returns raw threshold
        if unclamped is not needed, or exact clipped probability bound in [0, 1].
        """
        b = self.params.beta if beta is None else beta
        p0 = self.params.p0
        denom = p0 * self.params.q_c2i
        if denom <= 1e-15:
            return 1.0
        raw = ((1.0 - p0) * self.params.q_i2c * (1.0 - b)) / denom
        return float(np.clip(raw, 0.0, 1.0))

    def critical_alpha_unclipped(self, beta: Optional[float] = None) -> float:
        """Unclipped analytical zero-gain line alpha*(beta) = ((1-p0)q_{I->C}/(p0 q_{C->I}))(1-beta)."""
        b = self.params.beta if beta is None else beta
        p0 = self.params.p0
        denom = p0 * self.params.q_c2i
        if denom <= 1e-15:
            return float("inf")
        return float(((1.0 - p0) * self.params.q_i2c * (1.0 - b)) / denom)

    def critical_beta(self, alpha: Optional[float] = None) -> float:
        """Maximum tolerable Type-II false acceptance beta*(alpha) for net positive gain."""
        a = self.params.alpha if alpha is None else alpha
        p0 = self.params.p0
        num = p0 * self.params.q_c2i * a
        denom = (1.0 - p0) * self.params.q_i2c
        if denom <= 1e-15:
            return 0.0
        return float(np.clip(1.0 - num / denom, 0.0, 1.0))

    def critical_generator_accuracy(self) -> float:
        """Critical generator accuracy p_{0,crit} above which this verifier causes net harm.

        Note: In the homogeneous chain, p_{0,crit} equals the stationary accuracy p* = v / (u + v)!
        """
        u = self.u_c_to_i
        v = self.v_i_to_c
        if u + v <= 1e-15:
            return 1.0
        return float(v / (u + v))

    def stationary_accuracy(self, early_stop: bool = False) -> float:
        """Exact asymptotic accuracy as T -> infinity.

        Args:
            early_stop: If False, computes homogeneous stationary accuracy p* = v / (u + v).
                If True, computes exact absorbing accuracy p_inf^early via (I - Q)^{-1}.
        """
        p0 = self.params.p0
        if not early_stop:
            u = self.u_c_to_i
            v = self.v_i_to_c
            if u + v <= 1e-15:
                return p0
            return float(v / (u + v))

        a = self.params.alpha
        b = self.params.beta
        q_ic = self.params.q_i2c
        q_ci = self.params.q_c2i
        denom = (1.0 - a) * (b + (1.0 - b) * q_ic) + a * b * q_ci
        if denom <= 1e-15:
            # Degenerate absorbing corner (e.g., alpha=1 and beta=0: never accepts anything,
            # falls back to homogeneous stationary state on open candidates).
            return self.stationary_accuracy(early_stop=False)
        numer = (1.0 - a) * (p0 * b + (1.0 - b) * q_ic)
        return float(numer / denom)

    def accuracy_at_step(self, T: int, early_stop: bool = True) -> float:
        """Exact closed-form accuracy p_T after at most T verification-repair rounds."""
        if T < 0:
            raise ValueError(f"Number of verification rounds T must be >= 0, got {T}.")
        p0 = self.params.p0
        if T == 0:
            return p0

        if not early_stop:
            u = self.u_c_to_i
            v = self.v_i_to_c
            if u + v <= 1e-15:
                return p0
            p_star = v / (u + v)
            eig = 1.0 - u - v
            return float(p_star + (p0 - p_star) * (eig ** T))

        # Exact early-stopping closed-form on transient 2-vector s_0 = [p0, 1 - p0]^T
        Q = self.transient_matrix_early_stop()
        s = np.array([p0, 1.0 - p0], dtype=np.float64)
        acc_absorbed_c = 0.0
        accept_c_prob = 1.0 - self.params.alpha
        for _ in range(T):
            acc_absorbed_c += accept_c_prob * s[0]
            s = Q @ s
        return float(acc_absorbed_c + s[0])

    def expected_calls(self, T: int, early_stop: bool = True) -> Tuple[float, float]:
        """Return exact (E[verifier_calls], E[repair_calls]) for a budget of T rounds."""
        if T < 0:
            raise ValueError(f"T must be >= 0, got {T}.")
        if T == 0:
            return 0.0, 0.0

        p0 = self.params.p0
        a = self.params.alpha
        b = self.params.beta

        if not early_stop:
            u = self.u_c_to_i
            v = self.v_i_to_c
            if u + v <= 1e-15:
                sum_p = T * p0
            else:
                p_star = v / (u + v)
                eig = 1.0 - u - v
                sum_p = T * p_star + (p0 - p_star) * (1.0 - (eig ** T)) / (u + v)
            ver_calls = float(T)
            rep_calls = float(a * sum_p + (1.0 - b) * (T - sum_p))
            return ver_calls, rep_calls

        Q = self.transient_matrix_early_stop()
        s = np.array([p0, 1.0 - p0], dtype=np.float64)
        ver_calls = 0.0
        rep_calls = 0.0
        for _ in range(T):
            ver_calls += float(s[0] + s[1])
            s = Q @ s
            rep_calls += float(s[0] + s[1])
        return ver_calls, rep_calls

    def expected_cost(self, T: int, early_stop: bool = True) -> float:
        """Exact analytical expected token expenditure E[Cost_T]."""
        ver_calls, rep_calls = self.expected_calls(T=T, early_stop=early_stop)
        return float(
            self.params.cost_gen
            + ver_calls * self.params.cost_ver
            + rep_calls * self.params.cost_rep
        )

    def trajectory(self, max_T: int, early_stop: bool = True) -> Dict[str, np.ndarray]:
        """Compute exact accuracy and expected cost trajectories for t = 0..max_T."""
        steps = np.arange(0, max_T + 1, dtype=np.int64)
        accs = np.array([self.accuracy_at_step(int(t), early_stop=early_stop) for t in steps])
        costs = np.array([self.expected_cost(int(t), early_stop=early_stop) for t in steps])
        return {"steps": steps, "accuracy": accs, "expected_cost": costs}

    def simulate_monte_carlo(
        self,
        T: int,
        n_trials: int = 25000,
        early_stop: bool = True,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Vectorized Monte Carlo rollout to cross-check closed-form Markov equations."""
        rng = np.random.default_rng(seed)
        p = self.params

        is_correct = rng.random(n_trials) < p.p0
        active = np.ones(n_trials, dtype=bool)
        ver_count = np.zeros(n_trials, dtype=np.int64)
        rep_count = np.zeros(n_trials, dtype=np.int64)

        for _ in range(T):
            if early_stop and not np.any(active):
                break
            idx = np.flatnonzero(active) if early_stop else np.arange(n_trials)
            ver_count[idx] += 1

            curr_c = is_correct[idx]
            u_ver = rng.random(idx.size)
            rejected = np.where(curr_c, u_ver < p.alpha, u_ver < (1.0 - p.beta))

            if early_stop:
                accepted_idx = idx[~rejected]
                active[accepted_idx] = False

            rej_idx = idx[rejected]
            if rej_idx.size > 0:
                rep_count[rej_idx] += 1
                rej_c = is_correct[rej_idx]
                u_rep = rng.random(rej_idx.size)
                next_c = np.where(rej_c, u_rep >= p.q_c2i, u_rep < p.q_i2c)
                is_correct[rej_idx] = next_c

        emp_acc = float(np.mean(is_correct))
        emp_ver = float(np.mean(ver_count))
        emp_rep = float(np.mean(rep_count))
        emp_cost = float(p.cost_gen + emp_ver * p.cost_ver + emp_rep * p.cost_rep)

        ana_acc = self.accuracy_at_step(T=T, early_stop=early_stop)
        ana_cost = self.expected_cost(T=T, early_stop=early_stop)
        ana_ver, ana_rep = self.expected_calls(T=T, early_stop=early_stop)

        return {
            "T": float(T),
            "early_stop": float(early_stop),
            "n_trials": float(n_trials),
            "analytical_accuracy": ana_acc,
            "empirical_accuracy": emp_acc,
            "abs_accuracy_error": abs(emp_acc - ana_acc),
            "analytical_cost": ana_cost,
            "empirical_cost": emp_cost,
            "rel_cost_error": abs(emp_cost - ana_cost) / max(ana_cost, 1e-9),
            "analytical_ver_calls": ana_ver,
            "empirical_ver_calls": emp_ver,
            "analytical_rep_calls": ana_rep,
            "empirical_rep_calls": emp_rep,
        }


def compute_phase_diagram_grid(
    p0: float,
    q_i2c: float,
    q_c2i: float,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    T: int = 1,
    early_stop: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D verification net-gain surface Delta p_T(alpha, beta) = p_T - p_0.

    Returns:
        Tuple ``(AA, BB, delta_p)`` of shape ``(len(beta_grid), len(alpha_grid))``.
    """
    AA, BB = np.meshgrid(alpha_grid, beta_grid)
    delta_p = np.zeros_like(AA, dtype=np.float64)
    if T == 1:
        # Fully vectorized closed-form for T = 1
        delta_p = (1.0 - p0) * (1.0 - BB) * q_i2c - p0 * AA * q_c2i
        return AA, BB, delta_p

    for i in range(BB.shape[0]):
        for j in range(AA.shape[1]):
            chain = MarkovVerifierChain(
                VerificationParams(
                    p0=p0,
                    alpha=float(AA[i, j]),
                    beta=float(BB[i, j]),
                    q_i2c=q_i2c,
                    q_c2i=q_c2i,
                )
            )
            delta_p[i, j] = chain.accuracy_at_step(T=T, early_stop=early_stop) - p0
    return AA, BB, delta_p
