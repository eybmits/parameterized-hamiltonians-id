# -*- coding: utf-8 -*-
"""
ID(QAOA) & ID(VQE) + SPSA-λ-Baselines für Max-Cut
Hauptsimulation. Importiert plotting_utils.py.

VERGLEICH:
 - ID nutzt ADAM (weil der Gradient gut ist).
 - SPSA nutzt Standard-Decay (weil der Gradient verrauscht ist -> Baseline).
"""

import math
import argparse
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable

# IMPORT DER PLOTTING UTILS
import plotting_utils as pu

GRID_POINTS_EXACT = 201
SAFE_W_MAX = 100.0


# ======================================================================
# Utils (Simulation)
# ======================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def _renorm_state(psi: np.ndarray) -> np.ndarray:
    nrm2 = (psi.conj() * psi).real.sum()
    if (not np.isfinite(nrm2)) or (nrm2 <= 0):
        psi[:] = 1.0 / np.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm2)
    return psi


def _stabilize_phase(phase: np.ndarray) -> np.ndarray:
    return np.remainder(phase + np.pi, 2 * np.pi) - np.pi

def index_to_bitstring(idx: int, n: int) -> np.ndarray:
    """Basisindex -> Bitstring (LSB-first)."""
    bits = [(idx >> i) & 1 for i in range(n)]
    return np.array(bits, dtype=int)


# ======================================================================
# Graph & Z-Pattern
# ======================================================================

def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                A[i, j] = A[j, i] = 1
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
    return edges, A


def precompute_z_patterns(n: int) -> List[np.ndarray]:
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = []
    for i in range(n):
        Zi = 1 - 2 * ((idx >> i) & 1).astype(np.int8)  # ±1
        Z.append(Zi)
    return Z


# ======================================================================
# Antwortfunktion w(λ) und Grad J'(λ)
# ======================================================================

def make_response_params(edges,
                         rng: np.random.Generator,
                         resp_kind: str = "periodic",
                         lam_bounds: Tuple[float, float] = (-2.0, 2.0)) -> Dict:
    m = len(edges)
    par = {
        "kind": resp_kind,
        "w_init": rng.uniform(0.6, 1.4, size=m).astype(float),
        "lambda0": rng.uniform(-1.0, 1.0, size=m).astype(float),
    }
    if resp_kind == "linear":
        par["b"] = rng.uniform(-1.0, 1.0, size=m).astype(float)
    elif resp_kind == "quadratic":
        par["c"] = rng.uniform(-1.0, -0.2, size=m).astype(float)
    else:  # periodic
        A = rng.uniform(0.3, 1.2, size=m).astype(float)
        lam_min, lam_max = lam_bounds
        Delta = lam_max - lam_min
        cycles = rng.uniform(1.0, 6.0, size=m).astype(float)
        kappa = 2.0 * np.pi * cycles / max(1e-9, Delta)
        par.update({"A": A, "kappa": kappa, "w_floor": 0.05})
    return par


def _w_of_lambda(kind, lam, w0, lam0, p):
    if kind == "linear":
        return w0 + p["b"] * (lam - lam0)
    if kind == "quadratic":
        return w0 + p["c"] * ((lam - lam0) ** 2)
    val = w0 + p["A"] * math.cos(p["kappa"] * (lam - lam0))   # periodic
    wf = p.get("w_floor", None)
    return max(val, float(wf)) if wf is not None else val


def _dw_dlambda(kind, lam, lam0, p):
    if kind == "linear":
        return p["b"]
    if kind == "quadratic":
        return 2.0 * p["c"] * (lam - lam0)
    return -p["A"] * p["kappa"] * math.sin(p["kappa"] * (lam - lam0))  # periodic


def make_w_and_grad_1d(resp: Dict) -> Tuple[Callable[[float], np.ndarray],
                                            Callable[[float, np.ndarray], float]]:
    kind = resp["kind"]
    w0 = resp["w_init"]
    lam0 = resp["lambda0"]
    m = len(w0)
    pars = {k: v for k, v in resp.items() if k not in ("kind", "w_init", "lambda0")}

    def w_fun(lam: float) -> np.ndarray:
        w = np.empty(m, dtype=float)
        for e in range(m):
            p = {k: (v[e] if isinstance(v, np.ndarray) else v) for k, v in pars.items()}
            w[e] = _w_of_lambda(kind, float(lam), float(w0[e]), float(lam0[e]), p)
        return np.clip(w, -SAFE_W_MAX, SAFE_W_MAX)

    def gradJ(lam: float, p_cut: np.ndarray) -> float:
        """dJ/dλ für gegebenen Kanten-Cut-Vektor p_cut."""
        g = 0.0
        eps = 1e-12
        wfloor = pars.get("w_floor", None)
        for e in range(m):
            p = {k: (v[e] if isinstance(v, np.ndarray) else v) for k, v in pars.items()}
            w_e = _w_of_lambda(kind, float(lam), float(w0[e]), float(lam0[e]), p)
            if (wfloor is not None) and (w_e <= float(wfloor) + eps):
                continue
            g += _dw_dlambda(kind, float(lam), float(lam0[e]), p) * float(p_cut[e])
        return float(g)

    return w_fun, gradJ


# ======================================================================
# Cut/Erwartungen
# ======================================================================

def compute_cut_vals_for_w_on_the_fly(w: np.ndarray,
                                      edges: List[Tuple[int, int]],
                                      Z: List[np.ndarray]) -> np.ndarray:
    K = Z[0].shape[0]
    cut_vals = np.zeros(K, dtype=float)
    buf = np.empty(K, dtype=np.int8)
    for e, (i, j) in enumerate(edges):
        np.multiply(Z[i], Z[j], out=buf)
        cut_vals += 0.5 * (1.0 - buf.astype(float)) * w[e]
    return cut_vals


def z_expect_from_probs(probs: np.ndarray,
                        edges: List[Tuple[int, int]],
                        Z: List[np.ndarray]) -> np.ndarray:
    zexp = np.empty(len(edges), dtype=float)
    buf = np.empty_like(Z[0], dtype=np.int8)
    for e, (i, j) in enumerate(edges):
        np.multiply(Z[i], Z[j], out=buf)
        zexp[e] = probs.dot(buf.astype(float))
    return np.clip(zexp, -1.0, 1.0)


def z_expect_from_samples(sample_idx: np.ndarray,
                          edges: List[Tuple[int, int]],
                          Z: List[np.ndarray]) -> np.ndarray:
    n = len(Z)
    ZS = np.stack([Z[q][sample_idx].astype(float) for q in range(n)], axis=0)
    zexp = np.empty(len(edges), dtype=float)
    for e, (i, j) in enumerate(edges):
        zexp[e] = float(np.mean(ZS[i] * ZS[j]))
    return np.clip(zexp, -1.0, 1.0)


# ======================================================================
# QAOA
# ======================================================================

def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(psi, target, 0)
    block = psi.reshape(2, -1).astype(np.complex128, copy=False)
    np.nan_to_num(block, copy=False)
    two_by_M = gate @ block
    psi = two_by_M.reshape([2] + [2] * (n - 1))
    psi = np.moveaxis(psi, 0, target).reshape(-1)
    return _renorm_state(psi)


def _apply_2q(psi: np.ndarray, gate4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    if q1 == q2:
        return psi
    a, b = sorted((q1, q2))
    psi_r = psi.reshape([2] * n)
    psi_r = np.moveaxis(psi_r, (a, b), (0, 1))
    block = psi_r.reshape(4, -1).astype(np.complex128, copy=False)
    np.nan_to_num(block, copy=False)
    four_by_M = gate4 @ block
    psi = four_by_M.reshape(2, 2, *psi_r.shape[2:])
    psi = np.moveaxis(psi, (0, 1), (a, b)).reshape(-1)
    return _renorm_state(psi)


def _add_cost_phase_inplace(psi: np.ndarray, gamma: float,
                            w: np.ndarray, edges: List[Tuple[int, int]], Z: List[np.ndarray]) -> np.ndarray:
    phase = _stabilize_phase(-gamma * compute_cut_vals_for_w_on_the_fly(w, edges, Z))
    psi *= np.exp(1j * phase, dtype=np.complex128)
    return _renorm_state(psi)


def _qaoa_mixer_layer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = math.cos(beta), math.sin(beta)
    Rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    for q in range(n):
        psi = _apply_1q(psi, Rx, target=q, n=n)
    return _renorm_state(psi)


def qaoa_expectation_and_state_p(n: int, edges: List[Tuple[int, int]], Z: List[np.ndarray],
                                 w: np.ndarray, gammas: np.ndarray, betas: np.ndarray,
                                 shots: Optional[int] = None,
                                 rng: Optional[np.random.Generator] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    K = 1 << n
    psi = np.ones(K, dtype=np.complex128) / math.sqrt(K)
    for g, b in zip(gammas, betas):
        psi = _add_cost_phase_inplace(psi, float(g), w, edges, Z)
        psi = _qaoa_mixer_layer(psi, float(b), n)
    probs = (psi.conj() * psi).real.astype(float)
    s = probs.sum()
    probs = (probs / s) if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / probs.size)
    if (shots is None) or (shots <= 0):
        zexp = z_expect_from_probs(probs, edges, Z)
    else:
        rng = rng or np.random.default_rng()
        idx = rng.choice(np.arange(probs.size, dtype=np.int64), size=shots, replace=True, p=probs)
        zexp = z_expect_from_samples(idx, edges, Z)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    return J, psi, zexp


def qaoa_energy_p(n, edges, Z, w, theta_vec, shots=None, rng=None) -> float:
    p = len(theta_vec) // 2
    gammas = np.array(theta_vec[:p], dtype=float)
    betas = np.array(theta_vec[p:], dtype=float)
    J, _, _ = qaoa_expectation_and_state_p(n, edges, Z, w, gammas, betas, shots=shots, rng=rng)
    return -J


# ======================================================================
# VQE (RY–RZ + Ring-CNOT)
# ======================================================================

def _vqe_layer(psi: np.ndarray, n: int, ry: np.ndarray, rz: np.ndarray) -> np.ndarray:
    for q in range(n):
        cy, sy = math.cos(ry[q] / 2.0), math.sin(ry[q] / 2.0)
        RY = np.array([[cy, -sy], [sy, cy]], dtype=np.complex128)
        psi = _apply_1q(psi, RY, q, n)
        cz, sz = np.exp(-0.5j * rz[q]), np.exp(+0.5j * rz[q])
        RZ = np.array([[cz, 0], [0, sz]], dtype=np.complex128)
        psi = _apply_1q(psi, RZ, q, n)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=np.complex128)
    for q in range(n):
        psi = _apply_2q(psi, CNOT, q, (q + 1) % n, n)
    return _renorm_state(psi)


def vqe_expectation_and_state(n: int, edges: List[Tuple[int, int]], Z: List[np.ndarray],
                              w: np.ndarray, params: np.ndarray, L: int,
                              shots: Optional[int] = None,
                              rng: Optional[np.random.Generator] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    K = 1 << n
    psi = np.zeros(K, dtype=np.complex128)
    psi[0] = 1.0
    assert params.size == 2 * n * L
    for l in range(L):
        ry = params[l * (2 * n): l * (2 * n) + n]
        rz = params[l * (2 * n) + n: (l + 1) * (2 * n)]
        psi = _vqe_layer(psi, n, ry, rz)
    probs = (psi.conj() * psi).real.astype(float)
    s = probs.sum()
    probs = (probs / s) if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / probs.size)
    if (shots is None) or (shots <= 0):
        zexp = z_expect_from_probs(probs, edges, Z)
    else:
        rng = rng or np.random.default_rng()
        idx = rng.choice(np.arange(probs.size, dtype=np.int64), size=shots, replace=True, p=probs)
        zexp = z_expect_from_samples(idx, edges, Z)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    return J, psi, zexp


def vqe_energy(n, edges, Z, w, params, L, shots=None, rng=None) -> float:
    J, _, _ = vqe_expectation_and_state(n, edges, Z, w, params, L, shots=shots, rng=rng)
    return -J


# ======================================================================
# SPSA-Minimierung
# ======================================================================

def spsa_minimize(energy_fun: Callable[[np.ndarray], float],
                  params0: np.ndarray,
                  bounds: List[Tuple[float, float]],
                  iters: int = 120,
                  a: float = 0.25, c: float = 0.12, A: float = 60.0,
                  alpha: float = 0.602, gamma: float = 0.101,
                  num_starts: int = 1,
                  seed: int = 0) -> Tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(to_uint_seed(seed))
    D = len(params0)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    best_params = None
    best_E = None
    best_run_avg_grad = 0.0

    for s in range(num_starts):
        params = params0.astype(float).copy() if s == 0 else rng.uniform(lo, hi, size=D)
        best_E_run = None
        best_params_run = params.copy()
        
        grad_norms_run = []

        for k in range(1, iters + 1):
            ak = a / ((k + A) ** alpha)
            ck = c / (k ** gamma)
            delta = rng.choice([-1.0, 1.0], size=D)

            pp = np.clip(params + ck * delta, lo, hi)
            pm = np.clip(params - ck * delta, lo, hi)

            Ep = energy_fun(pp)
            Em = energy_fun(pm)

            diff = (Ep - Em) / (2.0 * ck) if ck > 0 else 0.0
            gk_vec = diff * delta
            grad_norm_k = float(np.linalg.norm(gk_vec))
            grad_norms_run.append(grad_norm_k)

            params = np.clip(params - ak * gk_vec, lo, hi)

            E = energy_fun(params)
            if (best_E_run is None) or (E < best_E_run):
                best_E_run = E
                best_params_run = params.copy()

        if (best_E is None) or (best_E_run < best_E):
            best_E = best_E_run
            best_params = best_params_run
            best_run_avg_grad = float(np.mean(grad_norms_run)) if grad_norms_run else 0.0

    return best_params, float(best_E), best_run_avg_grad


# ======================================================================
# QAOA-Init (Ramp Only)
# ======================================================================

def build_qaoa_theta_init(p: int,
                          gamma_star: float,
                          beta_star: float) -> np.ndarray:
    """Erzeugt QAOA Parameter-Init im 'ramp' Modus (lineare Interpolation)."""
    l = np.arange(1, p + 1, dtype=float)
    gammas0 = (l / float(p)) * gamma_star
    betas0 = (1.0 - l / float(p + 1.0)) * beta_star
    return np.concatenate([gammas0, betas0])


# ======================================================================
# Ground Truth & Hülle
# ======================================================================

def _parabolic_refine(xs: np.ndarray, ys: np.ndarray, i: int) -> Tuple[float, float]:
    if i <= 0 or i >= len(xs) - 1:
        return float(xs[i]), float(ys[i])
    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-18:
        return float(x1), float(y1)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    if abs(A) < 1e-18:
        return float(x1), float(y1)
    x_star = -B / (2 * A)
    y_star = A * x_star**2 + B * x_star + (y0 - A * x0**2 - B * x0)
    return float(x_star), float(y_star)


def ground_truth_lambda_1d(n: int, edges: List[Tuple[int, int]], Z: List[np.ndarray],
                           w_fun_1d: Callable[[float], np.ndarray],
                           lam_bounds: Tuple[float, float],
                           grid_points: int = GRID_POINTS_EXACT) -> Tuple[float, float]:
    xs = np.linspace(lam_bounds[0], lam_bounds[1], grid_points, dtype=float)
    Js = np.empty_like(xs)
    for t, lam in enumerate(xs):
        w = w_fun_1d(float(lam))
        Js[t] = compute_cut_vals_for_w_on_the_fly(w, edges, Z).max()
    i = int(np.argmax(Js))
    return _parabolic_refine(xs, Js, i)


def envelope_value_and_active_id(lam: float, edges, Z, w_fun_1d) -> Tuple[float, int]:
    v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(lam), edges, Z)
    idx = int(np.argmax(v))
    return float(v[idx]), idx


def compute_upper_envelope(edges, Z, w_fun_1d,
                           lam_bounds: Tuple[float, float],
                           grid_points: int = GRID_POINTS_EXACT):
    lam_grid = np.linspace(lam_bounds[0], lam_bounds[1], grid_points, dtype=float)
    J_star = np.empty_like(lam_grid)
    s_star = np.empty_like(lam_grid, dtype=int)
    for i, lam in enumerate(lam_grid):
        v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(float(lam)), edges, Z)
        J_star[i] = float(v.max())
        s_star[i] = int(v.argmax())
    return lam_grid, J_star, s_star


# ======================================================================
# NEU: Full Spectrum Calculation (für den Plot)
# ======================================================================

def compute_full_spectrum(n, edges, w_fun_1d, lam_grid):
    """
    Berechnet J für ALLE 2^n möglichen Patterns über das Lambda-Grid.
    Vektorisierte Berechnung für Geschwindigkeit.
    """
    print(f"\n[Spektrum] Berechne volles Spektrum ({2**n} Instanzen x {len(lam_grid)} Punkte)...")
    
    # 1. Z patterns (all 2^n)
    K = 1 << n
    indices = np.arange(K, dtype=np.uint32)
    # Shape: (K, n)
    bits = ((indices[:, None] & (1 << np.arange(n)[::-1])) > 0).astype(int)
    Z_all = 1 - 2 * bits

    # 2. Cut mask (Patterns x Edges) - vorberechnen
    m = len(edges)
    cut_mask = np.zeros((K, m), dtype=float)
    for e, (u, v) in enumerate(edges):
        # 0.5 * (1 - z_u*z_v) ist 1 wenn cut, sonst 0
        cut_mask[:, e] = 0.5 * (1.0 - Z_all[:, u] * Z_all[:, v])

    # 3. Sweep über Lambda
    # all_J: (Patterns, GridPoints)
    all_J = np.zeros((K, len(lam_grid)))
    J_star = np.zeros(len(lam_grid))
    active_ids = np.zeros(len(lam_grid), dtype=int)

    for t, lam in enumerate(lam_grid):
        w = w_fun_1d(lam) # shape (m,)
        # Vektorisierte Berechnung: Matrix(Patterns, Edges) @ Vector(Edges)
        vals = cut_mask @ w
        
        all_J[:, t] = vals
        best = np.argmax(vals)
        J_star[t] = vals[best]
        active_ids[t] = best

    # 4. Switch Points detektieren
    switch_indices = []
    for t in range(1, len(lam_grid)):
        if active_ids[t] != active_ids[t-1]:
            switch_indices.append(t)
            
    switch_lams = lam_grid[switch_indices]
    switch_vals = J_star[switch_indices]

    return all_J, J_star, active_ids, switch_lams, switch_vals


# ======================================================================
# Readout-Statistiken
# ======================================================================

def sample_readout_statistics(psi: np.ndarray,
                              w: np.ndarray,
                              edges: List[Tuple[int, int]],
                              Z: List[np.ndarray],
                              shots_readout: int,
                              rng: np.random.Generator) -> Tuple[float, int, float]:
    K = psi.size
    probs = (psi.conj() * psi).real.astype(float)
    s = probs.sum()
    if (not np.isfinite(s)) or (s <= 0):
        probs = np.full(K, 1.0 / K, dtype=float)
    else:
        probs /= s

    idx_all = np.arange(K, dtype=np.int64)
    idx_samples = rng.choice(idx_all, size=shots_readout, replace=True, p=probs)

    cut_vals = compute_cut_vals_for_w_on_the_fly(w, edges, Z)
    cut_samples = cut_vals[idx_samples]
    J_best = float(cut_samples.max())

    counts = np.bincount(idx_samples, minlength=K)
    mode_idx = int(counts.argmax())
    mode_cut = float(cut_vals[mode_idx])

    return J_best, mode_idx, mode_cut


# ======================================================================
# Optimization Loops (QAOA & VQE)
# ======================================================================

def optimize_lambda_with_ID_1d_qaoa(n, edges, Z, w_fun_1d, gradJ_1d,
                                    theta_init: np.ndarray, p_layers: int,
                                    lam0: float, lam_bounds: Tuple[float, float],
                                    outer_iters: int = 12, eta0: float = 0.25,
                                    inner_spsa_iters: int = 100, inner_num_starts: int = 1,
                                    seed: int = 0, shots: Optional[int] = None,
                                    spsa_theta_a: float = 0.25, spsa_theta_c: float = 0.12, spsa_theta_A: float = 60.0,
                                    spsa_theta_alpha: float = 0.602, spsa_theta_gamma: float = 0.101,
                                    idx_opt: Optional[int] = None,
                                    readout_shots_per_outer: int = 0):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))

    lam_min, lam_max = lam_bounds
    lam = float(lam0)
    theta = theta_init.astype(float).copy()
    p = p_layers
    theta_bounds = [(-math.pi, math.pi)] * p + [(-math.pi / 2, math.pi / 2)] * p

    m_adam = 0.0
    v_adam = 0.0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    eta = eta0

    best_J = -1e300
    best_lam = lam
    best_theta = theta.copy()

    # Histories
    lam_hist = []
    lam_hist_pre = []
    J_hist = []
    J_env_hist = []
    J_best_cut_hist = []
    J_mode_cut_hist = []
    
    grad_norm_lambda_hist = []
    grad_norm_theta_hist = []
    w_hist = []
    param_hist = []

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)
        w = w_fun_1d(lam)
        w_hist.append(w.copy())

        def energy_fun(params):
            return qaoa_energy_p(n, edges, Z, w, params, shots=None, rng=None)

        # Inner Loop
        theta, _, avg_grad_norm_inner = spsa_minimize(
            energy_fun, theta,
            bounds=theta_bounds,
            iters=inner_spsa_iters,
            a=spsa_theta_a, c=spsa_theta_c, A=spsa_theta_A,
            alpha=spsa_theta_alpha, gamma=spsa_theta_gamma,
            num_starts=inner_num_starts,
            seed=seed + 10 * t,
        )
        grad_norm_theta_hist.append(avg_grad_norm_inner)
        param_hist.append(theta.copy())

        # Expectation
        J_val, psi, zexp = qaoa_expectation_and_state_p(
            n, edges, Z, w,
            theta[:p], theta[p:], shots=shots, rng=rng_expect
        )
        J_hist.append(float(J_val))

        J_env, idx_active = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        if readout_shots_per_outer and readout_shots_per_outer > 0:
            J_best_samp_t, mode_idx_t, mode_cut_t = sample_readout_statistics(
                psi, w, edges, Z, readout_shots_per_outer, rng_readout
            )
            J_best_cut_hist.append(float(J_best_samp_t))
            J_mode_cut_hist.append(float(mode_cut_t))

        if J_val > best_J + 1e-12:
            best_J, best_lam, best_theta = J_val, lam, theta.copy()

        # Outer Gradient Step
        g = gradJ_1d(lam, 0.5 * (1.0 - zexp))
        grad_norm_lambda_hist.append(abs(g))

        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        m_hat = m_adam / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        step = eta * m_hat / (math.sqrt(v_hat) + eps)

        lam = float(np.clip(lam + step, lam_min, lam_max))
        lam_hist.append(lam)

        print(f"[ID(QAOA-Adam) {t:02d}] J={J_val:.6f}  λ={lam:.6f}  |∇λ|={abs(g):.2e} | best={best_J:.6f}")

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "J_best_cut": np.array(J_best_cut_hist, dtype=float),
        "J_mode_cut": np.array(J_mode_cut_hist, dtype=float),
        "grad_norm_lambda": np.array(grad_norm_lambda_hist, dtype=float),
        "grad_norm_theta": np.array(grad_norm_theta_hist, dtype=float),
        "w_hist": np.array(w_hist, dtype=float),
        "param_hist": np.array(param_hist, dtype=float)
    }
    evals_per_outer = inner_num_starts * inner_spsa_iters * 3 + 1
    hist["evals_cum"] = evals_per_outer * np.arange(1, outer_iters + 1, dtype=float)

    return best_lam, best_theta, hist


def optimize_lambda_with_ID_1d_vqe(n, edges, Z, w_fun_1d, gradJ_1d,
                                   phi_init: np.ndarray, L_layers: int,
                                   lam0: float, lam_bounds: Tuple[float, float],
                                   outer_iters: int = 12, eta0: float = 0.25,
                                   inner_spsa_iters: int = 100, inner_num_starts: int = 1,
                                   seed: int = 0, shots: Optional[int] = None,
                                   spsa_vqe_a: float = 0.25, spsa_vqe_c: float = 0.12, spsa_vqe_A: float = 60.0,
                                   spsa_vqe_alpha: float = 0.602, spsa_vqe_gamma: float = 0.101,
                                   idx_opt: Optional[int] = None,
                                   readout_shots_per_outer: int = 0):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))

    lam_min, lam_max = lam_bounds
    lam = float(lam0)
    phi = phi_init.astype(float).copy()
    D = len(phi)
    bounds = [(-math.pi, math.pi)] * D

    m_adam = 0.0
    v_adam = 0.0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    eta = eta0

    best_J = -1e300
    best_lam = lam
    best_phi = phi.copy()

    lam_hist = []
    lam_hist_pre = []
    J_hist = []
    J_env_hist = []
    J_best_cut_hist = []
    J_mode_cut_hist = []

    grad_norm_lambda_hist = []
    grad_norm_theta_hist = []
    w_hist = []
    param_hist = []

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)
        w = w_fun_1d(lam)
        w_hist.append(w.copy())

        def energy_fun(params):
            return vqe_energy(n, edges, Z, w, params, L_layers, shots=None, rng=None)

        phi, _, avg_grad_norm_inner = spsa_minimize(
            energy_fun, phi,
            bounds=bounds,
            iters=inner_spsa_iters,
            a=spsa_vqe_a, c=spsa_vqe_c, A=spsa_vqe_A,
            alpha=spsa_vqe_alpha, gamma=spsa_vqe_gamma,
            num_starts=inner_num_starts,
            seed=seed + 10 * t,
        )
        grad_norm_theta_hist.append(avg_grad_norm_inner)
        param_hist.append(phi.copy())

        J_val, psi, zexp = vqe_expectation_and_state(
            n, edges, Z, w, phi, L_layers, shots=shots, rng=rng_expect
        )
        J_hist.append(float(J_val))

        J_env, idx_active = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        if readout_shots_per_outer and readout_shots_per_outer > 0:
            J_best_samp_t, mode_idx_t, mode_cut_t = sample_readout_statistics(
                psi, w, edges, Z, readout_shots_per_outer, rng_readout
            )
            J_best_cut_hist.append(float(J_best_samp_t))
            J_mode_cut_hist.append(float(mode_cut_t))

        if J_val > best_J + 1e-12:
            best_J, best_lam, best_phi = J_val, lam, phi.copy()

        g = gradJ_1d(lam, 0.5 * (1.0 - zexp))
        grad_norm_lambda_hist.append(abs(g))

        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        m_hat = m_adam / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        step = eta * m_hat / (math.sqrt(v_hat) + eps)

        lam = float(np.clip(lam + step, lam_min, lam_max))
        lam_hist.append(lam)

        print(f"[ID(VQE-Adam)  {t:02d}] J={J_val:.6f}  λ={lam:.6f}  |∇λ|={abs(g):.2e} | best={best_J:.6f}")

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "J_best_cut": np.array(J_best_cut_hist, dtype=float),
        "J_mode_cut": np.array(J_mode_cut_hist, dtype=float),
        "grad_norm_lambda": np.array(grad_norm_lambda_hist, dtype=float),
        "grad_norm_theta": np.array(grad_norm_theta_hist, dtype=float),
        "w_hist": np.array(w_hist, dtype=float),
        "param_hist": np.array(param_hist, dtype=float)
    }
    evals_per_outer = inner_num_starts * inner_spsa_iters * 3 + 1
    hist["evals_cum"] = evals_per_outer * np.arange(1, outer_iters + 1, dtype=float)

    return best_lam, best_phi, hist


def optimize_lambda_with_SPSA_1d_qaoa(n, edges, Z, w_fun_1d,
                                      theta_init: np.ndarray, p_layers: int,
                                      lam0: float, lam_bounds: Tuple[float, float],
                                      outer_iters: int = 12,
                                      inner_spsa_iters: int = 100, inner_num_starts: int = 1,
                                      seed: int = 0, shots: Optional[int] = None,
                                      spsa_theta_a: float = 0.25, spsa_theta_c: float = 0.12, spsa_theta_A: float = 60.0,
                                      spsa_theta_alpha: float = 0.602, spsa_theta_gamma: float = 0.101,
                                      idx_opt: Optional[int] = None,
                                      readout_shots_per_outer: int = 0):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))

    lam_min, lam_max = lam_bounds
    lam = float(np.clip(lam0, lam_min, lam_max))

    theta = theta_init.astype(float).copy()
    p = p_layers
    theta_bounds = [(-math.pi, math.pi)] * p + [(-math.pi / 2, math.pi / 2)] * p

    # Standard SPSA Parameter
    Delta = lam_max - lam_min
    if Delta <= 0: Delta = 1.0
    lambda_a = 0.2 * Delta
    lambda_c = 0.05 * Delta
    lambda_A = 10.0
    lambda_alpha = 0.602
    lambda_gamma = 0.101

    best_J = -1e300
    best_lam = lam
    best_theta = theta.copy()

    lam_hist = []
    lam_hist_pre = []
    J_hist = []
    J_env_hist = []
    J_best_cut_hist = []
    J_mode_cut_hist = []

    best_cut_so_far = -1e300  
    inner_eval_cost = inner_num_starts * inner_spsa_iters * 3
    evals_cum = 0.0
    evals_cum_hist = []
    
    grad_norm_lambda_hist = []
    w_hist = []

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)
        
        # 1) Innerer SPSA Optimierer
        w_cur = w_fun_1d(lam)
        w_hist.append(w_cur.copy())

        def energy_fun_theta(params):
            return qaoa_energy_p(n, edges, Z, w_cur, params, shots=None, rng=None)

        theta, _, _ = spsa_minimize(
            energy_fun_theta, theta,
            bounds=theta_bounds,
            iters=inner_spsa_iters,
            a=spsa_theta_a, c=spsa_theta_c, A=spsa_theta_A,
            alpha=spsa_theta_alpha, gamma=spsa_theta_gamma,
            num_starts=inner_num_starts,
            seed=seed + 1000 * t,
        )

        J_val, psi, zexp = qaoa_expectation_and_state_p(
            n, edges, Z, w_cur,
            theta[:p], theta[p:], shots=shots, rng=rng_expect
        )
        J_hist.append(float(J_val))

        J_env, idx_active = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        if readout_shots_per_outer and readout_shots_per_outer > 0:
            J_best_samp_t, mode_idx_t, mode_cut_t = sample_readout_statistics(
                psi, w_cur, edges, Z, readout_shots_per_outer, rng_readout
            )
            best_cut_so_far = max(best_cut_so_far, J_best_samp_t)
            J_best_cut_hist.append(float(J_best_samp_t))
            J_mode_cut_hist.append(float(mode_cut_t))

        if J_val > best_J + 1e-12:
            best_J = J_val
            best_lam = lam
            best_theta = theta.copy()

        # 3) SPSA-Schritt in λ (STANDARD SPSA - KEIN ADAM)
        ck = lambda_c / (t ** lambda_gamma)
        ak = lambda_a / ((t + lambda_A) ** lambda_alpha)

        lam_p = float(np.clip(lam + ck, lam_min, lam_max))
        lam_m = float(np.clip(lam - ck, lam_min, lam_max))

        w_p = w_fun_1d(lam_p)
        w_m = w_fun_1d(lam_m)

        E_p = qaoa_energy_p(n, edges, Z, w_p, theta, shots=None, rng=None)
        E_m = qaoa_energy_p(n, edges, Z, w_m, theta, shots=None, rng=None)
        J_p, J_m = -E_p, -E_m

        # Geschätzter Gradient
        g_lambda = (J_p - J_m) / (2.0 * ck) if ck > 0 else 0.0
        grad_norm_lambda_hist.append(abs(g_lambda))
        
        # Standard Update (Gradient Ascent)
        lam = float(np.clip(lam + ak * g_lambda, lam_min, lam_max))
        lam_hist.append(lam)

        evals_cum += inner_eval_cost + 3
        evals_cum_hist.append(float(evals_cum))

        print(f"[QAOA-λ-SPSA (Standard) {t:02d}] J={J_val:.6f}  λ={lam:.6f}  |g_est|={abs(g_lambda):.2e}")

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "J_best_cut": np.array(J_best_cut_hist, dtype=float),
        "J_mode_cut": np.array(J_mode_cut_hist, dtype=float),
        "evals_cum": np.array(evals_cum_hist, dtype=float),
        "grad_norm_lambda": np.array(grad_norm_lambda_hist, dtype=float),
        "w_hist": np.array(w_hist, dtype=float)
    }
    return best_lam, best_theta, hist


def optimize_lambda_with_SPSA_1d_vqe(n, edges, Z, w_fun_1d,
                                     phi_init: np.ndarray, L_layers: int,
                                     lam0: float, lam_bounds: Tuple[float, float],
                                     outer_iters: int = 12,
                                     inner_spsa_iters: int = 100, inner_num_starts: int = 1,
                                     seed: int = 0, shots: Optional[int] = None,
                                     spsa_vqe_a: float = 0.25, spsa_vqe_c: float = 0.12, spsa_vqe_A: float = 60.0,
                                     spsa_vqe_alpha: float = 0.602, spsa_vqe_gamma: float = 0.101,
                                     idx_opt: Optional[int] = None,
                                     readout_shots_per_outer: int = 0):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))

    lam_min, lam_max = lam_bounds
    lam = float(np.clip(lam0, lam_min, lam_max))

    phi = phi_init.astype(float).copy()
    D = len(phi)
    bounds = [(-math.pi, math.pi)] * D

    # Standard SPSA Parameter
    Delta = lam_max - lam_min
    if Delta <= 0: Delta = 1.0
    lambda_a = 0.2 * Delta
    lambda_c = 0.05 * Delta
    lambda_A = 10.0
    lambda_alpha = 0.602
    lambda_gamma = 0.101

    best_J = -1e300
    best_lam = lam
    best_phi = phi.copy()

    lam_hist = []
    lam_hist_pre = []
    J_hist = []
    J_env_hist = []
    J_best_cut_hist = []
    J_mode_cut_hist = []

    best_cut_so_far = -1e300
    inner_eval_cost = inner_num_starts * inner_spsa_iters * 3
    evals_cum = 0.0
    evals_cum_hist = []
    
    # Stats
    grad_norm_lambda_hist = []
    w_hist = []

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)

        # 1) Innerer SPSA in φ
        w_cur = w_fun_1d(lam)
        w_hist.append(w_cur.copy())

        def energy_fun_phi(params):
            return vqe_energy(n, edges, Z, w_cur, params, L_layers, shots=None, rng=None)

        phi, _, _ = spsa_minimize(
            energy_fun_phi, phi,
            bounds=bounds,
            iters=inner_spsa_iters,
            a=spsa_vqe_a, c=spsa_vqe_c, A=spsa_vqe_A,
            alpha=spsa_vqe_alpha, gamma=spsa_vqe_gamma,
            num_starts=inner_num_starts,
            seed=seed + 2000 * t,
        )

        J_val, psi, zexp = vqe_expectation_and_state(
            n, edges, Z, w_cur, phi, L_layers, shots=shots, rng=rng_expect
        )
        J_hist.append(float(J_val))

        J_env, idx_active = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        if readout_shots_per_outer and readout_shots_per_outer > 0:
            J_best_samp_t, mode_idx_t, mode_cut_t = sample_readout_statistics(
                psi, w_cur, edges, Z, readout_shots_per_outer, rng_readout
            )
            best_cut_so_far = max(best_cut_so_far, J_best_samp_t)
            J_best_cut_hist.append(float(J_best_samp_t))
            J_mode_cut_hist.append(float(mode_cut_t))

        if J_val > best_J + 1e-12:
            best_J = J_val
            best_lam = lam
            best_phi = phi.copy()

        # 3) SPSA-Schritt in λ (STANDARD SPSA - KEIN ADAM)
        ck = lambda_c / (t ** lambda_gamma)
        ak = lambda_a / ((t + lambda_A) ** lambda_alpha)

        lam_p = float(np.clip(lam + ck, lam_min, lam_max))
        lam_m = float(np.clip(lam - ck, lam_min, lam_max))

        w_p = w_fun_1d(lam_p)
        w_m = w_fun_1d(lam_m)

        E_p = vqe_energy(n, edges, Z, w_p, phi, L_layers, shots=None, rng=None)
        E_m = vqe_energy(n, edges, Z, w_m, phi, L_layers, shots=None, rng=None)
        J_p, J_m = -E_p, -E_m

        # Geschätzter Gradient
        g_lambda = (J_p - J_m) / (2.0 * ck) if ck > 0 else 0.0
        grad_norm_lambda_hist.append(abs(g_lambda))
        
        # Standard Update (Gradient Ascent)
        lam = float(np.clip(lam + ak * g_lambda, lam_min, lam_max))
        lam_hist.append(lam)

        evals_cum += inner_eval_cost + 3
        evals_cum_hist.append(float(evals_cum))

        print(f"[VQE-λ-SPSA (Standard)  {t:02d}] J={J_val:.6f}  λ={lam:.6f}  |g_est|={abs(g_lambda):.2e}")

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "J_best_cut": np.array(J_best_cut_hist, dtype=float),
        "J_mode_cut": np.array(J_mode_cut_hist, dtype=float),
        "evals_cum": np.array(evals_cum_hist, dtype=float),
        "grad_norm_lambda": np.array(grad_norm_lambda_hist, dtype=float),
        "w_hist": np.array(w_hist, dtype=float)
    }
    return best_lam, best_phi, hist

# ======================================================================
# Metrics
# ======================================================================

def compute_algo_metrics(name: str,
                         hist: Dict[str, np.ndarray],
                         psi_final: Optional[np.ndarray],
                         lam_true: float,
                         J_true: float,
                         idx_opt: int) -> Dict[str, float]:
    J_series = hist.get("J", np.array([]))
    if J_series.size:
        J_final = float(J_series[-1])
        J_best = float(J_series.max())
    else:
        J_final = float("nan")
        J_best = float("nan")

    J_best_cut_series = hist.get("J_best_cut", np.array([]))
    lam_pre = hist.get("lam_pre", hist.get("lam", np.array([])))

    if J_best_cut_series.size and lam_pre.size >= J_best_cut_series.size:
        k = int(J_best_cut_series.argmax())
        J_best_cut = float(J_best_cut_series[k])
        lam_best_cut = float(lam_pre[k])
        delta_lam = abs(lam_best_cut - lam_true)
    else:
        J_best_cut = float("nan")
        lam_best_cut = float("nan")
        delta_lam = float("nan")

    # p_opt_final
    if psi_final is None:
        p_opt_final = float("nan")
    else:
        probs = (psi_final.conj() * psi_final).real
        s = probs.sum()
        if (not np.isfinite(s)) or (s <= 0):
            p_opt_final = 0.0
        else:
            probs = probs / s
            p_opt_final = float(probs[idx_opt]) if idx_opt < probs.size else 0.0

    if J_true > 0 and np.isfinite(J_true):
        J_best_ratio = float(J_best) / J_true if np.isfinite(J_best) else float("nan")
        J_best_cut_ratio = float(J_best_cut) / J_true if np.isfinite(J_best_cut) else float("nan")
    else:
        J_best_ratio = float("nan")
        J_best_cut_ratio = float("nan")

    evals_cum = hist.get("evals_cum", np.array([]))
    num_evals = int(round(float(evals_cum[-1]))) if evals_cum.size else 0

    return dict(
        name=name,
        J_final=J_final,
        J_best=J_best,
        J_best_cut=J_best_cut,
        lam_best_cut=lam_best_cut,
        delta_lam=delta_lam,
        p_opt_final=p_opt_final,
        J_best_ratio=J_best_ratio,
        J_best_cut_ratio=J_best_cut_ratio,
        num_evals=num_evals,
    )

# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="ID(QAOA) & ID(VQE) + SPSA-λ-Baselines für Max-Cut (k=1, kein Cluster)"
    )
    # Problem
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.5)
    p.add_argument("--lam_bounds", type=float, nargs=2, default=[-5.0, 5.0])
    p.add_argument("--lam0", type=float, default=1.0)
    p.add_argument("--resp_kind", type=str, default="periodic",
                   choices=["linear", "quadratic", "periodic"])
    # Optimierung
    p.add_argument("--outer_iters", type=int, default=10)
    p.add_argument("--eta0", type=float, default=0.1)
    
    # FIXED: Höhere Iterationen (100 statt 10) damit SPSA konvergiert
    p.add_argument("--inner_spsa_iters", type=int, default=10) 
    
    p.add_argument("--inner_num_starts", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    
    # FIXED: Kleineres p (3 statt 10), damit SPSA eine Chance hat
    p.add_argument("--p_main", type=int, default=3,
                   help="QAOA-Layerzahl p")
                   
    p.add_argument("--shots", type=int, default=None,
                   help="Shots für Erwartungswerte (None = exakt)")
    # VQE-Layer: "fair" ist jetzt implizit "budget"
    p.add_argument("--vqe_layers", type=int, default=None, 
                   help="Manuelle Anzahl VQE Layers. Falls None, gleich wie QAOA p.")
    
    # QAOA Init: "mode" ist jetzt implizit "ramp"
    p.add_argument("--qaoa_gamma_star", type=float, default=0.8)
    p.add_argument("--qaoa_beta_star", type=float, default=0.3)
    
    # SPSA Hyperparameter (inner)
    p.add_argument("--spsa_theta_a", type=float, default=0.25)
    p.add_argument("--spsa_theta_c", type=float, default=0.12)
    p.add_argument("--spsa_theta_A", type=float, default=60.0)
    p.add_argument("--spsa_theta_alpha", type=float, default=0.602)
    p.add_argument("--spsa_theta_gamma", type=float, default=0.101)
    p.add_argument("--spsa_vqe_a", type=float, default=0.25)
    p.add_argument("--spsa_vqe_c", type=float, default=0.12)
    p.add_argument("--spsa_vqe_A", type=float, default=60.0)
    p.add_argument("--spsa_vqe_alpha", type=float, default=0.602)
    p.add_argument("--spsa_vqe_gamma", type=float, default=0.101)
    # Hülle/Plots/CSV
    p.add_argument("--grid_points_exact", type=int, default=201)
    p.add_argument("--save_prefix", type=str, default="figures/raw/out")
    # Readout-Budget
    p.add_argument("--readout_shots_per_outer", type=int, default=128,
                   help="Shots pro Outer-Iteration für best-cut & Mode-Bitstring (0 = deaktiviert)")
    return p.parse_args()

# ======================================================================
# Main
# ======================================================================

def main(args):
    rng = np.random.default_rng(to_uint_seed(args.seed))

    # Problem
    edges, A = generate_random_graph(args.n, args.p_edge, rng)
    Z = precompute_z_patterns(args.n)
    resp = make_response_params(edges, rng, resp_kind=args.resp_kind, lam_bounds=tuple(args.lam_bounds))
    w_fun_1d, gradJ_1d = make_w_and_grad_1d(resp)

    # Ground Truth
    lam_true, J_true = ground_truth_lambda_1d(
        args.n, edges, Z, w_fun_1d, tuple(args.lam_bounds),
        grid_points=args.grid_points_exact
    )
    w_true = w_fun_1d(lam_true)
    cut_vals_true = compute_cut_vals_for_w_on_the_fly(w_true, edges, Z)
    idx_opt = int(np.argmax(cut_vals_true))
    assignment_opt = index_to_bitstring(idx_opt, args.n)

    # QAOA Init (Ramp)
    p_layers = args.p_main
    theta0 = build_qaoa_theta_init(
        p=p_layers,
        gamma_star=args.qaoa_gamma_star,
        beta_star=args.qaoa_beta_star
    )
    lam0 = float(args.lam0)

    # VQE-Layer (Budget Fairness)
    if args.vqe_layers is not None:
        L_vqe = args.vqe_layers
    else:
        # Default: "Budget" fairness -> Same depth as QAOA
        L_vqe = max(2, args.p_main)
        
    D_vqe = 2 * args.n * L_vqe
    phi0 = np.zeros(D_vqe, dtype=float)

    # ===== ID(QAOA) =====
    lam_q_id, theta_q_id, hist_q_id = optimize_lambda_with_ID_1d_qaoa(
        args.n, edges, Z, w_fun_1d, gradJ_1d,
        theta_init=theta0, p_layers=p_layers,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters, eta0=args.eta0,
        inner_spsa_iters=args.inner_spsa_iters, inner_num_starts=args.inner_num_starts,
        seed=args.seed, shots=args.shots,
        spsa_theta_a=args.spsa_theta_a, spsa_theta_c=args.spsa_theta_c, spsa_theta_A=args.spsa_theta_A,
        spsa_theta_alpha=args.spsa_theta_alpha, spsa_theta_gamma=args.spsa_theta_gamma,
        idx_opt=idx_opt,
        readout_shots_per_outer=args.readout_shots_per_outer
    )
    J_q_id, psi_q_id, _ = qaoa_expectation_and_state_p(
        args.n, edges, Z, w_fun_1d(lam_q_id),
        theta_q_id[:p_layers], theta_q_id[p_layers:], shots=args.shots, rng=rng
    )

    # ===== ID(VQE) =====
    lam_v_id, phi_v_id, hist_v_id = optimize_lambda_with_ID_1d_vqe(
        args.n, edges, Z, w_fun_1d, gradJ_1d,
        phi_init=phi0, L_layers=L_vqe,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters, eta0=args.eta0,
        inner_spsa_iters=args.inner_spsa_iters, inner_num_starts=1,
        seed=args.seed + 777, shots=args.shots,
        spsa_vqe_a=args.spsa_vqe_a, spsa_vqe_c=args.spsa_vqe_c, spsa_vqe_A=args.spsa_vqe_A,
        spsa_vqe_alpha=args.spsa_vqe_alpha, spsa_vqe_gamma=args.spsa_vqe_gamma,
        idx_opt=idx_opt,
        readout_shots_per_outer=args.readout_shots_per_outer
    )
    J_v_id, psi_v_id, _ = vqe_expectation_and_state(
        args.n, edges, Z, w_fun_1d(lam_v_id),
        phi_v_id, L_vqe, shots=args.shots, rng=rng
    )

    # ===== SPSA-λ-Baseline (QAOA, Envelope) =====
    lam_q_spsa, theta_q_spsa, hist_q_spsa = optimize_lambda_with_SPSA_1d_qaoa(
        args.n, edges, Z, w_fun_1d,
        theta_init=theta0, p_layers=p_layers,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters,
        inner_spsa_iters=args.inner_spsa_iters,
        inner_num_starts=args.inner_num_starts,
        seed=args.seed + 4242, shots=args.shots,
        spsa_theta_a=args.spsa_theta_a, spsa_theta_c=args.spsa_theta_c, spsa_theta_A=args.spsa_theta_A,
        spsa_theta_alpha=args.spsa_theta_alpha, spsa_theta_gamma=args.spsa_theta_gamma,
        idx_opt=idx_opt,
        readout_shots_per_outer=args.readout_shots_per_outer
    )
    J_q_spsa, psi_q_spsa, _ = qaoa_expectation_and_state_p(
        args.n, edges, Z, w_fun_1d(lam_q_spsa),
        theta_q_spsa[:p_layers], theta_q_spsa[p_layers:], shots=args.shots, rng=rng
    )

    # ===== SPSA-λ-Baseline (VQE, Envelope) =====
    lam_v_spsa, phi_v_spsa, hist_v_spsa = optimize_lambda_with_SPSA_1d_vqe(
        args.n, edges, Z, w_fun_1d,
        phi_init=phi0, L_layers=L_vqe,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters,
        inner_spsa_iters=args.inner_spsa_iters,
        inner_num_starts=args.inner_num_starts,
        seed=args.seed + 4242 + 999, shots=args.shots,
        spsa_vqe_a=args.spsa_vqe_a, spsa_vqe_c=args.spsa_vqe_c, spsa_vqe_A=args.spsa_vqe_A,
        spsa_vqe_alpha=args.spsa_vqe_alpha, spsa_vqe_gamma=args.spsa_vqe_gamma,
        idx_opt=idx_opt,
        readout_shots_per_outer=args.readout_shots_per_outer
    )
    J_v_spsa, psi_v_spsa, _ = vqe_expectation_and_state(
        args.n, edges, Z, w_fun_1d(lam_v_spsa),
        phi_v_spsa, L_vqe, shots=args.shots, rng=rng
    )

    # ===== Max-Cut-Lösungen (wahrscheinlichster Bitstring) =====
    def most_likely_assignment(psi: np.ndarray) -> Tuple[np.ndarray, int]:
        probs = (psi.conj() * psi).real
        idx = int(np.argmax(probs))
        return index_to_bitstring(idx, args.n), idx

    assignment_q_id, idx_q_id = most_likely_assignment(psi_q_id)
    assignment_v_id, idx_v_id = most_likely_assignment(psi_v_id)
    assignment_q_spsa, idx_q_spsa = most_likely_assignment(psi_q_spsa)
    assignment_v_spsa, idx_v_spsa = most_likely_assignment(psi_v_spsa)

    # ===== Hülle =====
    lam_grid, J_star, s_star = compute_upper_envelope(
        edges, Z, w_fun_1d, tuple(args.lam_bounds),
        grid_points=args.grid_points_exact
    )

    # ===== PLOTS via plotting_utils =====
    
    # Standard A-E
    pu.plot_envelope_and_expectations(
        save_path=f"{args.save_prefix}_envelope_expectations.png",
        lam_grid=lam_grid, J_star=J_star, hist_q=hist_q_id, hist_v=hist_v_id
    )
    if args.readout_shots_per_outer and args.readout_shots_per_outer > 0:
        pu.plot_envelope_and_best_cuts(
            save_path=f"{args.save_prefix}_envelope_bestcuts.png",
            lam_grid=lam_grid, J_star=J_star, hist_q=hist_q_id, hist_v=hist_v_id
        )
        pu.plot_envelope_and_mode_cuts(
            save_path=f"{args.save_prefix}_envelope_modecuts.png",
            lam_grid=lam_grid, J_star=J_star, hist_q=hist_q_id, hist_v=hist_v_id
        )

    pu.plot_bestJ_vs_eval_cost(
        save_path=f"{args.save_prefix}_bestJ_vs_evals.png",
        hist_q_id=hist_q_id, hist_v_id=hist_v_id,
        hist_q_spsa=hist_q_spsa, hist_v_spsa=hist_v_spsa,
        J_true=J_true
    )
    pu.plot_total_eval_costs_bar(
        save_path=f"{args.save_prefix}_evals_total_bar.png",
        hist_q_id=hist_q_id, hist_v_id=hist_v_id,
        hist_q_spsa=hist_q_spsa, hist_v_spsa=hist_v_spsa
    )

    # Neu: Plot J (Lambda Trajektorien)
    pu.plot_lambda_trajectories(
        save_path=f"{args.save_prefix}_lambda_convergence.png",
        hist_q_id=hist_q_id,
        hist_v_id=hist_v_id,
        hist_q_spsa=hist_q_spsa,
        hist_v_spsa=hist_v_spsa,
        lam_true=lam_true,
        lam_bounds=tuple(args.lam_bounds)
    )

    # Neu F-I
    pu.plot_gradient_norms(f"{args.save_prefix}_grad_norms_qaoa.png", hist_q_id, "ID(QAOA)")
    pu.plot_gradient_norms(f"{args.save_prefix}_grad_norms_vqe.png", hist_v_id, "ID(VQE)")
    pu.plot_weight_heatmap(f"{args.save_prefix}_weights_qaoa.png", hist_q_id, "ID(QAOA)")
    pu.plot_weight_heatmap(f"{args.save_prefix}_weights_vqe.png", hist_v_id, "ID(VQE)")
    pu.plot_param_trajectories(f"{args.save_prefix}_params_qaoa.png", hist_q_id, p_layers)
    pu.plot_final_distribution(f"{args.save_prefix}_probs_final.png", psi_q_id, psi_v_id, args.n)

    # ===== NEU: Plot der Spektral-Struktur (Röntgenblick) =====
    # Berechne das volle Spektrum (2^n Linien) auf dem Grid
    all_J_spec, J_star_spec, active_ids_spec, switch_lams_spec, switch_vals_spec = compute_full_spectrum(
        args.n, edges, w_fun_1d, lam_grid
    )
    
    pu.plot_full_envelope_structure(
        save_path=f"{args.save_prefix}_envelope_structure_full.png",
        lams=lam_grid,
        all_J_lines=all_J_spec,
        J_star=J_star_spec,
        active_ids=active_ids_spec,
        switch_lams=switch_lams_spec,
        switch_vals=switch_vals_spec,
        n_nodes=args.n
    )

    # CSV
    pu.save_active_instances_csv(
        save_path=f"{args.save_prefix}_active_qaoa.csv",
        algo_name="ID-QAOA", lams=hist_q_id["lam"], edges=edges, Z=Z, w_fun_1d=w_fun_1d,
        compute_cut_vals_func=compute_cut_vals_for_w_on_the_fly
    )
    pu.save_active_instances_csv(
        save_path=f"{args.save_prefix}_active_vqe.csv",
        algo_name="ID-VQE", lams=hist_v_id["lam"], edges=edges, Z=Z, w_fun_1d=w_fun_1d,
        compute_cut_vals_func=compute_cut_vals_for_w_on_the_fly
    )

    # Tabelle
    metrics_id_qaoa = compute_algo_metrics("ID-QAOA", hist_q_id, psi_q_id, lam_true, J_true, idx_opt)
    metrics_spsa_qaoa = compute_algo_metrics("SPSA-QAOA", hist_q_spsa, psi_q_spsa, lam_true, J_true, idx_opt)
    metrics_id_vqe = compute_algo_metrics("ID-VQE", hist_v_id, psi_v_id, lam_true, J_true, idx_opt)
    metrics_spsa_vqe = compute_algo_metrics("SPSA-VQE", hist_v_spsa, psi_v_spsa, lam_true, J_true, idx_opt)
    metrics_list = [metrics_id_qaoa, metrics_spsa_qaoa, metrics_id_vqe, metrics_spsa_vqe]
    
    pu.print_summary_table(metrics_list)

    print("\nOptimaler Bitstring (Ground Truth):", assignment_opt)
    print("QAOA (ID)       wahrscheinlichster Bitstring:", assignment_q_id)
    print("VQE (ID)        wahrscheinlichster Bitstring:", assignment_v_id)
    print(f"\nPlots & CSV gespeichert unter Prefix: {args.save_prefix}_*")


if __name__ == "__main__":
    main(parse_args())