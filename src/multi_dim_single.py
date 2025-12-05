# -- coding: utf-8 --
"""
COMPLETE Multidimensional Input Design (m-dim ID) for Max-Cut
============================================================
Algorithms: ID(QAOA), ID(VQE), SPSA-λ(QAOA), SPSA-λ(VQE)

Features:
- Exact Max-Cut Solver for Ground Truth calculation.
- Detailed Performance Table (J_final, p_opt, ratios, etc.).
- Bitstring comparison.
- Visualization: 'Real Trajectory Landscape' (Actual optimization path).
- Additional plots:
  * Convergence curves J(t), J_best_cut(t)
  * J vs. cumulative energy evaluations
  * Lambda norms and movement vs. iteration
  * Lambda components over iterations
  * Weight profiles w_e(lambda_t)
  * Gradient norms for ID methods
  * PCA trajectory of lambda
  * Hamming distances to ground truth
  * Metric barplots (J_final, ratios, p_opt)
  * Weight statistics and boxplots vs. GT-cut edges

Dependencies: numpy, matplotlib, networkx
"""

import math
import argparse
import os
from typing import Tuple, List, Dict, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ======================================================================
# Constants & Utils
# ======================================================================
SAFE_W_MAX = 100.0

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

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)

# ======================================================================
# Graph, Z-Pattern & Exact Solver
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

def index_to_bitstring(idx: int, n: int) -> str:
    # Little Endian (LSB first) for consistency with QAOA ordering usually
    # or just standard binary repr. Let's use array style [0 1 0 ...]
    bits = [(idx >> i) & 1 for i in range(n)]
    return str(bits).replace(',', '')

def solve_exact_maxcut(n: int, edges: List[Tuple[int, int]]) -> Tuple[float, int, List[int]]:
    """
    Brute Force Solver for Ground Truth.
    Returns: (MaxCutValue, OptimalIndex, OptimalBitStringList)
    """
    best_cut = -1.0
    best_idx = -1
    
    # Iterate all 2^n
    for k in range(1 << n):
        # Convert k to spins +/- 1. x_i = (k >> i) & 1. s_i = 1 - 2*x_i
        cut_val = 0.0
        for (u, v) in edges:
            s_u = 1 - 2 * ((k >> u) & 1)
            s_v = 1 - 2 * ((k >> v) & 1)
            cut_val += 0.5 * (1 - s_u * s_v)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_idx = k
    
    # All optimal indices (there might be degeneracy, we pick best_idx)
    return best_cut, best_idx, [(best_idx >> i) & 1 for i in range(n)]

# ======================================================================
# Multidimensional Response Function w(vec_λ) and Grad J'(vec_λ)
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
        "m": m
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

def make_w_and_grad_md(resp: Dict) -> Tuple[Callable[[np.ndarray], np.ndarray],
                                            Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    kind = resp["kind"]
    w0 = resp["w_init"]      # shape (m,)
    lam0 = resp["lambda0"]   # shape (m,)
    
    if kind == "periodic":
        A = resp["A"]
        kappa = resp["kappa"]
        w_floor = resp.get("w_floor", None)
    elif kind == "linear":
        b = resp["b"]
    elif kind == "quadratic":
        c = resp["c"]
    
    def w_fun(lam_vec: np.ndarray) -> np.ndarray:
        diff = lam_vec - lam0
        if kind == "linear":
            w = w0 + b * diff
        elif kind == "quadratic":
            w = w0 + c * (diff ** 2)
        else: # periodic
            w = w0 + A * np.cos(kappa * diff)
            if w_floor is not None:
                w = np.maximum(w, w_floor)
        return np.clip(w, -SAFE_W_MAX, SAFE_W_MAX)

    def gradJ(lam_vec: np.ndarray, p_cut: np.ndarray) -> np.ndarray:
        diff = lam_vec - lam0
        if kind == "linear":
            dw = b
        elif kind == "quadratic":
            dw = 2.0 * c * diff
        else: # periodic
            raw_dw = -A * kappa * np.sin(kappa * diff)
            if resp.get("w_floor", None) is not None:
                w_curr = w0 + A * np.cos(kappa * diff)
                mask = (w_curr <= resp["w_floor"])
                raw_dw[mask] = 0.0
            dw = raw_dw
            
        return dw * p_cut

    return w_fun, gradJ

# ======================================================================
# Quantum Simulation: Cuts & Expectation
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
# QAOA Utils
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
                  seed: int = 0) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(to_uint_seed(seed))
    D = len(params0)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    best_params = None
    best_E = None

    for s in range(num_starts):
        params = params0.astype(float).copy() if s == 0 else rng.uniform(lo, hi, size=D)
        best_E_run = None
        best_params_run = params.copy()

        for k in range(1, iters + 1):
            ak = a / ((k + A) ** alpha)
            ck = c / (k ** gamma)
            
            delta = rng.choice([-1.0, 1.0], size=D)

            pp = np.clip(params + ck * delta, lo, hi)
            pm = np.clip(params - ck * delta, lo, hi)

            Ep = energy_fun(pp)
            Em = energy_fun(pm)

            gk = (Ep - Em) / (2.0 * ck) * delta if ck > 0 else np.zeros_like(params)
            
            params = np.clip(params - ak * gk, lo, hi)

            E = energy_fun(params)
            if (best_E_run is None) or (E < best_E_run):
                best_E_run = E
                best_params_run = params.copy()

        if (best_E is None) or (best_E_run < best_E):
            best_E = best_E_run
            best_params = best_params_run

    return best_params, float(best_E)

# ======================================================================
# Init & Helper
# ======================================================================
def choose_vqe_layers(fair_mode: str, p_main: int, n: int, m_edges: int, min_layers: int = 1) -> int:
    fair_mode = (fair_mode or "budget").lower()
    if fair_mode == "param":
        return max(min_layers, max(1, round(max(1, p_main / max(1, n)))))
    if fair_mode == "hardware":
        return max(min_layers, max(1, round((m_edges / max(1, n)) * p_main)))
    return max(2, p_main)

def build_qaoa_theta_init(p: int, mode: str, gamma_star: float, beta_star: float) -> np.ndarray:
    mode = (mode or "legacy").lower()
    if mode == "legacy":
        gammas0 = np.full(p, 0.8, dtype=float)
        betas0 = np.full(p, 0.3, dtype=float)
    elif mode == "stack":
        gammas0 = np.full(p, gamma_star, dtype=float)
        betas0 = np.full(p, beta_star, dtype=float)
    elif mode == "ramp":
        l = np.arange(1, p + 1, dtype=float)
        gammas0 = (l / float(p)) * gamma_star
        betas0 = (1.0 - l / float(p + 1.0)) * beta_star
    else:
        gammas0 = np.full(p, 0.8, dtype=float)
        betas0 = np.full(p, 0.3, dtype=float)
    return np.concatenate([gammas0, betas0])

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
# Special Plot: Real Trajectory
# ======================================================================
def plot_real_trajectory_landscape(save_path: str,
                                   n: int,
                                   edges: List[Tuple[int, int]],
                                   Z: List[np.ndarray],
                                   w_fun: Callable,
                                   lam_history: List[np.ndarray],
                                   theta_final: np.ndarray,
                                   n_random_dirs: int = 50):
    """
    Plot I: 'Real Trajectory vs Random Rays'.
    """
    _ensure_dir(save_path)
    
    lam_start = lam_history[0]
    
    # 1. Berechne Distanz und Energie für jeden Schritt der History
    real_dists = []
    real_energies = []
    
    def get_energy(lam_curr):
        w_curr = w_fun(lam_curr)
        # Wir nutzen -energy für J (Maximierung)
        return -qaoa_energy_p(n, edges, Z, w_curr, theta_final, shots=None, rng=None)
    
    for lam in lam_history:
        dist = np.linalg.norm(lam - lam_start)
        real_dists.append(dist)
        real_energies.append(get_energy(lam))
        
    real_dists = np.array(real_dists)
    real_energies = np.array(real_energies)
    max_dist = real_dists.max()
    if max_dist == 0: max_dist = 1.0

    # 2. Zufällige Rays generieren
    rng = np.random.default_rng(42)
    alpha_steps = np.linspace(0.0, 1.0, 30)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    x_ray_plot = alpha_steps * max_dist
    
    rand_mean_curve = np.zeros_like(x_ray_plot)
    
    for _ in range(n_random_dirs):
        d = rng.normal(size=len(lam_start))
        d /= (np.linalg.norm(d) + 1e-9)
        d *= max_dist 
        
        y_ray = []
        for alpha in alpha_steps:
            lam_rand = lam_start + alpha * d
            y_ray.append(get_energy(lam_rand))
        
        ax.plot(x_ray_plot, y_ray, color='gray', alpha=0.15, lw=1.0)
        rand_mean_curve += np.array(y_ray)
        
    rand_mean_curve /= n_random_dirs
    
    # Plot Mean Random
    ax.plot(x_ray_plot, rand_mean_curve, color='gray', linestyle='--', alpha=0.8, lw=1.5, label='Mean Random Ray')
    
    # Plot Real Trajectory
    ax.plot(real_dists, real_energies, color='#D32F2F', lw=2.5, marker='.', markersize=4, label='Real ID Trajectory')
    
    ax.scatter([real_dists[0]], [real_energies[0]], color='black', zorder=10, s=60, label='Start')
    ax.scatter([real_dists[-1]], [real_energies[-1]], color='#D32F2F', zorder=10, s=100, marker='*', label='End (Optimum)')

    ax.set_xlabel(r"Distance from Start $||\vec{\lambda}_t - \vec{\lambda}_{0}||$")
    ax.set_ylabel(r"Expected Cut Value $J(\vec{\lambda}_t)$ (on final landscape)")
    ax.set_title(r"Real Optimization Trajectory vs Random Directions")
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# ======================================================================
# Optimization Loops (With History)
# ======================================================================
def optimize_lambda_with_ID_md_qaoa(n, edges, Z, w_fun, gradJ,
                                    theta_init: np.ndarray, p_layers: int,
                                    lam0_vec: np.ndarray, lam_bounds: Tuple[float, float],
                                    outer_iters: int, eta0: float,
                                    inner_spsa_iters: int, inner_num_starts: int,
                                    seed: int, shots: Optional[int],
                                    spsa_params: Dict, readout_shots_per_outer: int):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))
    
    lam_min, lam_max = lam_bounds
    lam = lam0_vec.astype(float).copy()
    theta = theta_init.astype(float).copy()
    p = p_layers
    theta_bounds = [(-math.pi, math.pi)] * p + [(-math.pi / 2, math.pi / 2)] * p

    m_adam = np.zeros_like(lam)
    v_adam = np.zeros_like(lam)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    best_J = -1e300
    best_lam, best_theta = lam.copy(), theta.copy()
    J_hist, J_best_cut_hist, evals_cum_hist = [], [], []
    grad_norm_hist = []
    lam_history = [lam.copy()] 
    current_evals = 0

    for t in range(1, outer_iters + 1):
        w = w_fun(lam)
        def energy_fun(params): return qaoa_energy_p(n, edges, Z, w, params, shots=None, rng=None)

        theta, _ = spsa_minimize(energy_fun, theta, theta_bounds, 
                                 iters=inner_spsa_iters, num_starts=inner_num_starts, 
                                 seed=seed+10*t, **spsa_params)
        
        current_evals += inner_num_starts * inner_spsa_iters * 2

        J_val, psi, zexp = qaoa_expectation_and_state_p(n, edges, Z, w, theta[:p], theta[p:], shots=shots, rng=rng_expect)
        current_evals += 1
        J_hist.append(float(J_val))

        if readout_shots_per_outer > 0:
            J_best_samp_t, _, _ = sample_readout_statistics(psi, w, edges, Z, readout_shots_per_outer, rng_readout)
            J_best_cut_hist.append(float(J_best_samp_t))

        if J_val > best_J: best_J, best_lam, best_theta = J_val, lam.copy(), theta.copy()

        p_cut = 0.5 * (1.0 - zexp) 
        g = gradJ(lam, p_cut)
        grad_norm_hist.append(float(np.linalg.norm(g)))

        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        m_hat = m_adam / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        step = eta0 * m_hat / (np.sqrt(v_hat) + eps)

        lam = np.clip(lam + step, lam_min, lam_max)
        lam_history.append(lam.copy())
        
        evals_cum_hist.append(current_evals)
        print(f"[ID(QAOA) {t:02d}] J={J_val:.4f} | best={best_J:.4f}")

    return best_lam, best_theta, {"J": np.array(J_hist), 
                                  "J_best_cut": np.array(J_best_cut_hist), 
                                  "evals_cum": np.array(evals_cum_hist),
                                  "lam_history": lam_history,
                                  "grad_norm": np.array(grad_norm_hist)}

def optimize_lambda_with_ID_md_vqe(n, edges, Z, w_fun, gradJ,
                                   phi_init: np.ndarray, L_layers: int,
                                   lam0_vec: np.ndarray, lam_bounds: Tuple[float, float],
                                   outer_iters: int, eta0: float,
                                   inner_spsa_iters: int, inner_num_starts: int,
                                   seed: int, shots: Optional[int],
                                   spsa_params: Dict, readout_shots_per_outer: int):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))
    
    lam_min, lam_max = lam_bounds
    lam = lam0_vec.astype(float).copy()
    phi = phi_init.astype(float).copy()
    bounds = [(-math.pi, math.pi)] * len(phi)

    m_adam = np.zeros_like(lam)
    v_adam = np.zeros_like(lam)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    best_J = -1e300
    best_lam, best_phi = lam.copy(), phi.copy()
    J_hist, J_best_cut_hist, evals_cum_hist = [], [], []
    grad_norm_hist = []
    lam_history = [lam.copy()]
    current_evals = 0

    for t in range(1, outer_iters + 1):
        w = w_fun(lam)
        def energy_fun(params): return vqe_energy(n, edges, Z, w, params, L_layers, shots=None, rng=None)

        phi, _ = spsa_minimize(energy_fun, phi, bounds, 
                               iters=inner_spsa_iters, num_starts=inner_num_starts, 
                               seed=seed+10*t, **spsa_params)
        
        current_evals += inner_num_starts * inner_spsa_iters * 2

        J_val, psi, zexp = vqe_expectation_and_state(n, edges, Z, w, phi, L_layers, shots=shots, rng=rng_expect)
        current_evals += 1
        J_hist.append(float(J_val))

        if readout_shots_per_outer > 0:
            J_best_samp_t, _, _ = sample_readout_statistics(psi, w, edges, Z, readout_shots_per_outer, rng_readout)
            J_best_cut_hist.append(float(J_best_samp_t))

        if J_val > best_J: best_J, best_lam, best_phi = J_val, lam.copy(), phi.copy()

        p_cut = 0.5 * (1.0 - zexp)
        g = gradJ(lam, p_cut)
        grad_norm_hist.append(float(np.linalg.norm(g)))

        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        m_hat = m_adam / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        step = eta0 * m_hat / (np.sqrt(v_hat) + eps)

        lam = np.clip(lam + step, lam_min, lam_max)
        lam_history.append(lam.copy())
        
        evals_cum_hist.append(current_evals)
        print(f"[ID(VQE)  {t:02d}] J={J_val:.4f} | best={best_J:.4f}")

    return best_lam, best_phi, {"J": np.array(J_hist), 
                                "J_best_cut": np.array(J_best_cut_hist), 
                                "evals_cum": np.array(evals_cum_hist),
                                "lam_history": lam_history,
                                "grad_norm": np.array(grad_norm_hist)}

def optimize_lambda_with_SPSA_md_qaoa(n, edges, Z, w_fun,
                                      theta_init: np.ndarray, p_layers: int,
                                      lam0_vec: np.ndarray, lam_bounds: Tuple[float, float],
                                      outer_iters: int,
                                      inner_spsa_iters: int, inner_num_starts: int,
                                      seed: int, shots: Optional[int],
                                      spsa_params_theta: Dict, readout_shots_per_outer: int):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))
    rng_spsa_lam = np.random.default_rng(to_uint_seed(seed + 999))

    lam_min, lam_max = lam_bounds
    lam = np.clip(lam0_vec.astype(float).copy(), lam_min, lam_max)
    theta = theta_init.astype(float).copy()
    theta_bounds = [(-math.pi, math.pi)] * p_layers + [(-math.pi / 2, math.pi / 2)] * p_layers

    Delta = lam_max - lam_min if (lam_max - lam_min) > 0 else 1.0
    lambda_a, lambda_c = 0.2 * Delta, 0.05 * Delta
    
    best_J = -1e300
    best_lam, best_theta = lam.copy(), theta.copy()
    J_hist, J_best_cut_hist, evals_cum_hist = [], [], []
    lam_history = [lam.copy()]
    current_evals = 0

    for t in range(1, outer_iters + 1):
        w_cur = w_fun(lam)
        def energy_fun_theta(params): return qaoa_energy_p(n, edges, Z, w_cur, params, shots=None, rng=None)

        theta, _ = spsa_minimize(energy_fun_theta, theta, theta_bounds, 
                                 iters=inner_spsa_iters, num_starts=inner_num_starts, 
                                 seed=seed+1000*t, **spsa_params_theta)
        
        current_evals += inner_num_starts * inner_spsa_iters * 2

        J_val, psi, _ = qaoa_expectation_and_state_p(n, edges, Z, w_cur, theta[:p_layers], theta[p_layers:], shots=shots, rng=rng_expect)
        current_evals += 1
        J_hist.append(float(J_val))

        if readout_shots_per_outer > 0:
            J_best_samp_t, _, _ = sample_readout_statistics(psi, w_cur, edges, Z, readout_shots_per_outer, rng_readout)
            J_best_cut_hist.append(float(J_best_samp_t))

        if J_val > best_J: best_J, best_lam, best_theta = J_val, lam.copy(), theta.copy()

        # SPSA Step for Lambda
        ck = lambda_c / (t ** 0.101)
        ak = lambda_a / ((t + 10.0) ** 0.602)
        
        delta_vec = rng_spsa_lam.choice([-1.0, 1.0], size=len(lam))
        lam_p = np.clip(lam + ck * delta_vec, lam_min, lam_max)
        lam_m = np.clip(lam - ck * delta_vec, lam_min, lam_max)

        Ep = qaoa_energy_p(n, edges, Z, w_fun(lam_p), theta)
        Em = qaoa_energy_p(n, edges, Z, w_fun(lam_m), theta)
        current_evals += 2
        
        g_lambda = ((-Ep - (-Em)) / (2.0 * ck)) * delta_vec
        lam = np.clip(lam + ak * g_lambda, lam_min, lam_max)
        lam_history.append(lam.copy())
        evals_cum_hist.append(current_evals)
        print(f"[QAOA-SPSA {t:02d}] J={J_val:.4f} | best={best_J:.4f}")

    return best_lam, best_theta, {"J": np.array(J_hist), "J_best_cut": np.array(J_best_cut_hist), "evals_cum": np.array(evals_cum_hist), "lam_history": lam_history}

def optimize_lambda_with_SPSA_md_vqe(n, edges, Z, w_fun,
                                     phi_init: np.ndarray, L_layers: int,
                                     lam0_vec: np.ndarray, lam_bounds: Tuple[float, float],
                                     outer_iters: int,
                                     inner_spsa_iters: int, inner_num_starts: int,
                                     seed: int, shots: Optional[int],
                                     spsa_params_vqe: Dict, readout_shots_per_outer: int):
    rng_expect = np.random.default_rng(to_uint_seed(seed))
    rng_readout = np.random.default_rng(to_uint_seed(seed + 1234567))
    rng_spsa_lam = np.random.default_rng(to_uint_seed(seed + 2999))

    lam_min, lam_max = lam_bounds
    lam = np.clip(lam0_vec.astype(float).copy(), lam_min, lam_max)
    phi = phi_init.astype(float).copy()
    bounds = [(-math.pi, math.pi)] * len(phi)

    Delta = lam_max - lam_min if (lam_max - lam_min) > 0 else 1.0
    lambda_a, lambda_c = 0.2 * Delta, 0.05 * Delta
    
    best_J = -1e300
    best_lam, best_phi = lam.copy(), phi.copy()
    J_hist, J_best_cut_hist, evals_cum_hist = [], [], []
    lam_history = [lam.copy()]
    current_evals = 0

    for t in range(1, outer_iters + 1):
        w_cur = w_fun(lam)
        def energy_fun_phi(params): return vqe_energy(n, edges, Z, w_cur, params, L_layers, shots=None, rng=None)

        phi, _ = spsa_minimize(energy_fun_phi, phi, bounds, 
                               iters=inner_spsa_iters, num_starts=inner_num_starts, 
                               seed=seed+2000*t, **spsa_params_vqe)
        
        current_evals += inner_num_starts * inner_spsa_iters * 2

        J_val, psi, _ = vqe_expectation_and_state(n, edges, Z, w_cur, phi, L_layers, shots=shots, rng=rng_expect)
        current_evals += 1
        J_hist.append(float(J_val))

        if readout_shots_per_outer > 0:
            J_best_samp_t, _, _ = sample_readout_statistics(psi, w_cur, edges, Z, readout_shots_per_outer, rng_readout)
            J_best_cut_hist.append(float(J_best_samp_t))

        if J_val > best_J: best_J, best_lam, best_phi = J_val, lam.copy(), phi.copy()

        ck = lambda_c / (t ** 0.101)
        ak = lambda_a / ((t + 10.0) ** 0.602)
        delta_vec = rng_spsa_lam.choice([-1.0, 1.0], size=len(lam))
        
        lam_p = np.clip(lam + ck * delta_vec, lam_min, lam_max)
        lam_m = np.clip(lam - ck * delta_vec, lam_min, lam_max)

        Ep = vqe_energy(n, edges, Z, w_fun(lam_p), phi, L_layers)
        Em = vqe_energy(n, edges, Z, w_fun(lam_m), phi, L_layers)
        current_evals += 2
        
        g_lambda = ((-Ep - (-Em)) / (2.0 * ck)) * delta_vec
        lam = np.clip(lam + ak * g_lambda, lam_min, lam_max)
        lam_history.append(lam.copy())
        evals_cum_hist.append(current_evals)
        print(f"[VQE-SPSA  {t:02d}] J={J_val:.4f} | best={best_J:.4f}")

    return best_lam, best_phi, {"J": np.array(J_hist), "J_best_cut": np.array(J_best_cut_hist), "evals_cum": np.array(evals_cum_hist), "lam_history": lam_history}

# ======================================================================
# Additional Analysis & Plotting
# ======================================================================

def _safe_array_from_hist(hist: Dict, key: str) -> np.ndarray:
    value = hist.get(key, None)
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, list):
        return np.array(value, dtype=float)
    return value

def plot_convergence_curves(save_prefix: str,
                            hist_q_id: Dict,
                            hist_q_spsa: Dict,
                            hist_v_id: Dict,
                            hist_v_spsa: Dict):
    # Plot J and J_best_cut vs outer iteration for all algorithms
    save_path = f"{save_prefix}_convergence_J_vs_outer.png"
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 6))

    algo_hists = [
        ("ID-QAOA", hist_q_id),
        ("SPSA-QAOA", hist_q_spsa),
        ("ID-VQE", hist_v_id),
        ("SPSA-VQE", hist_v_spsa),
    ]
    for name, hist in algo_hists:
        J = _safe_array_from_hist(hist, "J")
        J_best_cut = _safe_array_from_hist(hist, "J_best_cut")
        if J.size == 0:
            continue
        iters = np.arange(1, J.size + 1)
        ax.plot(iters, J, linewidth=2.0, marker="o", markersize=4, label=f"{name} J")
        if J_best_cut.size == J.size and J_best_cut.size > 0:
            ax.plot(iters, J_best_cut, linestyle="--", linewidth=1.5, marker="x", markersize=4, label=f"{name} J_best_cut")

    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("Expected / sampled cut value")
    ax.set_title("Convergence of J and J_best_cut per outer iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_J_vs_evals(save_prefix: str,
                    hist_q_id: Dict,
                    hist_q_spsa: Dict,
                    hist_v_id: Dict,
                    hist_v_spsa: Dict):
    # Plot J vs cumulative #evals for all algorithms
    save_path = f"{save_prefix}_J_vs_evals.png"
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 6))

    algo_hists = [
        ("ID-QAOA", hist_q_id),
        ("SPSA-QAOA", hist_q_spsa),
        ("ID-VQE", hist_v_id),
        ("SPSA-VQE", hist_v_spsa),
    ]
    for name, hist in algo_hists:
        J = _safe_array_from_hist(hist, "J")
        evals = _safe_array_from_hist(hist, "evals_cum")
        if J.size == 0 or evals.size == 0 or J.size != evals.size:
            continue
        ax.plot(evals, J, linewidth=2.0, marker="o", markersize=4, label=name)

    ax.set_xlabel("Cumulative number of energy evaluations")
    ax.set_ylabel("Expected cut value J")
    ax.set_title("J vs. cumulative energy evaluations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_lambda_norms(save_prefix: str,
                      lam0_vec: np.ndarray,
                      hist_q_id: Dict,
                      hist_q_spsa: Dict,
                      hist_v_id: Dict,
                      hist_v_spsa: Dict):
    # Plot ||lambda_t|| and ||lambda_t - lambda0|| vs t for all algorithms
    save_path = f"{save_prefix}_lambda_norms.png"
    _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    algo_hists = [
        ("ID-QAOA", hist_q_id),
        ("SPSA-QAOA", hist_q_spsa),
        ("ID-VQE", hist_v_id),
        ("SPSA-VQE", hist_v_spsa),
    ]

    for name, hist in algo_hists:
        lam_hist = hist.get("lam_history", None)
        if lam_hist is None or len(lam_hist) == 0:
            continue
        lam_arr = np.array(lam_hist, dtype=float)
        iters = np.arange(lam_arr.shape[0])
        norms = np.linalg.norm(lam_arr, axis=1)
        deltas = np.linalg.norm(lam_arr - lam0_vec.reshape(1, -1), axis=1)
        axes[0].plot(iters, norms, marker="o", markersize=3, linewidth=2.0, label=name)
        axes[1].plot(iters, deltas, marker="o", markersize=3, linewidth=2.0, label=name)

    axes[0].set_ylabel(r"$||\lambda_t||_2$")
    axes[0].set_title("Norm of lambda over outer iterations")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration index t (including t=0)")
    axes[1].set_ylabel(r"$||\lambda_t - \lambda_0||_2$")
    axes[1].set_title("Distance of lambda to initial lambda_0")
    axes[1].grid(True, alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_lambda_components_for_algo(save_path: str,
                                    lam_history: List[np.ndarray],
                                    max_edges_to_plot: int,
                                    algo_label: str):
    _ensure_dir(save_path)
    lam_arr = np.array(lam_history, dtype=float)
    T_plus_1, m_edges = lam_arr.shape
    num_edges_to_plot = min(max_edges_to_plot, m_edges)
    fig, ax = plt.subplots(figsize=(9, 6))
    iters = np.arange(T_plus_1)
    for e in range(num_edges_to_plot):
        ax.plot(iters, lam_arr[:, e], linewidth=2.0, marker="o", markersize=3, label=f"edge {e}")
    ax.set_xlabel("Iteration index t (including t=0)")
    ax.set_ylabel(r"$\lambda_{e}(t)$")
    ax.set_title(f"Lambda components over iterations ({algo_label})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_w_profiles_for_algo(save_path: str,
                             lam_history: List[np.ndarray],
                             w_fun: Callable,
                             max_edges_to_plot: int,
                             algo_label: str):
    _ensure_dir(save_path)
    lam_arr = np.array(lam_history, dtype=float)
    T_plus_1, m_edges = lam_arr.shape
    num_edges_to_plot = min(max_edges_to_plot, m_edges)
    w_all = []
    for lam in lam_arr:
        w_all.append(w_fun(lam))
    w_arr = np.array(w_all, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))
    iters = np.arange(T_plus_1)
    for e in range(num_edges_to_plot):
        ax.plot(iters, w_arr[:, e], linewidth=2.0, marker="o", markersize=3, label=f"edge {e}")
    ax.set_xlabel("Iteration index t (including t=0)")
    ax.set_ylabel(r"$w_e(\lambda_t)$")
    ax.set_title(f"Weight profiles over iterations ({algo_label})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_grad_norms(save_prefix: str,
                    hist_q_id: Dict,
                    hist_v_id: Dict):
    save_path = f"{save_prefix}_grad_norms.png"
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    algo_hists = [
        ("ID-QAOA", hist_q_id),
        ("ID-VQE", hist_v_id),
    ]
    for name, hist in algo_hists:
        grad_norm = _safe_array_from_hist(hist, "grad_norm")
        if grad_norm.size == 0:
            continue
        iters = np.arange(1, grad_norm.size + 1)
        ax.plot(iters, grad_norm, linewidth=2.0, marker="o", markersize=4, label=name)
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel(r"$||\nabla_{\lambda} J||_2$")
    ax.set_title("Gradient norm of lambda-update (ID methods)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def _bitstring_to_list(bitstring: str) -> List[int]:
    return [int(ch) for ch in bitstring if ch in ("0", "1")]

def compute_hamming_distance_bitstrings(bitstring_a: str, bitstring_b: str) -> int:
    bits_a = _bitstring_to_list(bitstring_a)
    bits_b = _bitstring_to_list(bitstring_b)
    length = min(len(bits_a), len(bits_b))
    if length == 0:
        return 0
    distance = 0
    for i in range(length):
        if bits_a[i] != bits_b[i]:
            distance += 1
    return distance

def plot_hamming_distances(save_prefix: str,
                           gt_bitstring: str,
                           results: List[Dict[str, object]]):
    save_path = f"{save_prefix}_hamming_distances.png"
    _ensure_dir(save_path)
    algo_names = []
    distances = []
    for r in results:
        name = str(r.get("name", ""))
        best_str = str(r.get("best_str", ""))
        d = compute_hamming_distance_bitstrings(gt_bitstring, best_str)
        algo_names.append(name)
        distances.append(d)
    x = np.arange(len(algo_names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, distances)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha="right")
    ax.set_ylabel("Hamming distance to ground truth bitstring")
    ax.set_title("Hamming distance of most probable bitstring")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_metric_bars(save_prefix: str,
                     results: List[Dict[str, object]]):
    save_path = f"{save_prefix}_metrics_barplot.png"
    _ensure_dir(save_path)
    algo_names = [str(r.get("name", "")) for r in results]
    J_final = np.array([float(r.get("J_final", 0.0)) for r in results], dtype=float)
    ratio_exp = np.array([float(r.get("ratio_exp", 0.0)) for r in results], dtype=float)
    ratio_cut = np.array([float(r.get("ratio_cut", 0.0)) for r in results], dtype=float)
    p_opt = np.array([float(r.get("p_opt", 0.0)) for r in results], dtype=float)
    num_algos = len(algo_names)
    x = np.arange(num_algos)
    width = 0.2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].bar(x - width, J_final, width=width, label="J_final")
    axes[0].bar(x, ratio_exp, width=width, label="J_best / J*")
    axes[0].bar(x + width, ratio_cut, width=width, label="J_best_cut / J*")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algo_names, rotation=30, ha="right")
    axes[0].set_ylabel("Values")
    axes[0].set_title("Final and best ratios vs. ground truth")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x, p_opt, width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algo_names, rotation=30, ha="right")
    axes[1].set_ylabel("p_opt")
    axes[1].set_title("Probability of ground truth bitstring (final state)")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_weight_statistics(save_prefix: str,
                           w_fun: Callable,
                           lam0_vec: np.ndarray,
                           lam_q_id: np.ndarray,
                           lam_q_spsa: np.ndarray,
                           lam_v_id: np.ndarray,
                           lam_v_spsa: np.ndarray):
    save_path = f"{save_prefix}_weight_statistics.png"
    _ensure_dir(save_path)
    algo_names = ["ID-QAOA", "SPSA-QAOA", "ID-VQE", "SPSA-VQE"]
    lam_finals = [lam_q_id, lam_q_spsa, lam_v_id, lam_v_spsa]

    sum_w0 = float(np.sum(w_fun(lam0_vec)))
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(algo_names))
    sums_final = []
    stds_final = []
    for lam in lam_finals:
        w_val = w_fun(lam)
        sums_final.append(float(np.sum(w_val)))
        stds_final.append(float(np.std(w_val)))
    width = 0.35
    ax.bar(x - width / 2.0, sums_final, width=width, label="sum w(lambda_final)")
    ax.axhline(sum_w0, linestyle="--", linewidth=1.5, label="sum w(lambda0)")
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, rotation=30, ha="right")
    ax.set_ylabel("Sum of weights")
    ax.set_title("Sum of weights at lambda_final vs lambda0")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

    save_path_std = f"{save_prefix}_weight_std.png"
    _ensure_dir(save_path_std)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.bar(x, stds_final)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algo_names, rotation=30, ha="right")
    ax2.set_ylabel("Standard deviation of weights")
    ax2.set_title("Spread of edge weights at lambda_final")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(save_path_std, dpi=160)
    plt.close(fig2)
    print(f"[saved] {save_path_std}")

def plot_w_boxplots_gt_cut(save_prefix: str,
                           w_fun: Callable,
                           edges: List[Tuple[int, int]],
                           gt_bits: List[int],
                           lam_q_id: np.ndarray,
                           lam_q_spsa: np.ndarray,
                           lam_v_id: np.ndarray,
                           lam_v_spsa: np.ndarray):
    algo_names = ["ID-QAOA", "SPSA-QAOA", "ID-VQE", "SPSA-VQE"]
    lam_finals = [lam_q_id, lam_q_spsa, lam_v_id, lam_v_spsa]
    for algo_name, lam in zip(algo_names, lam_finals):
        w_val = w_fun(lam)
        cut_weights = []
        noncut_weights = []
        for idx_edge, (u, v) in enumerate(edges):
            is_cut = (gt_bits[u] != gt_bits[v])
            if is_cut:
                cut_weights.append(float(w_val[idx_edge]))
            else:
                noncut_weights.append(float(w_val[idx_edge]))
        if len(cut_weights) == 0 or len(noncut_weights) == 0:
            continue
        save_path = f"{save_prefix}_w_boxplot_{algo_name.replace('-', '_')}.png"
        _ensure_dir(save_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([cut_weights, noncut_weights], labels=["GT-cut edges", "non GT-cut edges"])
        ax.set_ylabel("w_e(lambda_final)")
        ax.set_title(f"Distribution of weights for GT vs non-GT edges ({algo_name})")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        print(f"[saved] {save_path}")

def perform_pca_2d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Center data
    mean = np.mean(data, axis=0)
    centered = data - mean
    # Compute covariance and eigen decomposition via SVD
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    transformed = centered @ components.T
    return transformed, components, mean

def plot_lambda_pca_trajectory(save_prefix: str,
                               lam_history: List[np.ndarray],
                               J_hist: np.ndarray,
                               algo_label: str):
    if lam_history is None or len(lam_history) == 0:
        return
    lam_arr = np.array(lam_history, dtype=float)
    transformed, components, mean_vec = perform_pca_2d(lam_arr)
    save_path = f"{save_prefix}_lambda_pca_{algo_label.replace('-', '_')}.png"
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = transformed[:, 0]
    y = transformed[:, 1]
    if J_hist is not None and (J_hist.size == (lam_arr.shape[0] - 1) or J_hist.size == lam_arr.shape[0]):
        if J_hist.size == lam_arr.shape[0] - 1:
            J_for_color = np.concatenate([[J_hist[0]], J_hist])
        else:
            J_for_color = J_hist
        scatter = ax.scatter(x, y, c=J_for_color, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="J value")
    else:
        ax.scatter(x, y)
    ax.plot(x, y, linewidth=1.5, alpha=0.7)
    ax.scatter([x[0]], [y[0]], color="red", s=60, label="start")
    ax.scatter([x[-1]], [y[-1]], color="green", s=80, label="end")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Lambda trajectory in PCA space ({algo_label})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def generate_additional_plots(save_prefix: str,
                              args,
                              edges: List[Tuple[int, int]],
                              Z: List[np.ndarray],
                              w_fun: Callable,
                              lam0_vec: np.ndarray,
                              gt_cut: float,
                              gt_idx: int,
                              gt_bits: List[int],
                              gt_bitstring: str,
                              results: List[Dict[str, object]],
                              lam_q_id: np.ndarray,
                              lam_q_spsa: np.ndarray,
                              lam_v_id: np.ndarray,
                              lam_v_spsa: np.ndarray,
                              hist_q_id: Dict,
                              hist_q_spsa: Dict,
                              hist_v_id: Dict,
                              hist_v_spsa: Dict):
    print("Generating additional plots...")
    plot_convergence_curves(save_prefix, hist_q_id, hist_q_spsa, hist_v_id, hist_v_spsa)
    plot_J_vs_evals(save_prefix, hist_q_id, hist_q_spsa, hist_v_id, hist_v_spsa)
    plot_lambda_norms(save_prefix, lam0_vec, hist_q_id, hist_q_spsa, hist_v_id, hist_v_spsa)
    lam_hist_q_id = hist_q_id.get("lam_history", [])
    lam_hist_q_spsa = hist_q_spsa.get("lam_history", [])
    lam_hist_v_id = hist_v_id.get("lam_history", [])
    lam_hist_v_spsa = hist_v_spsa.get("lam_history", [])
    if len(lam_hist_q_id) > 0:
        plot_lambda_components_for_algo(f"{save_prefix}_lambda_components_ID_QAOA.png", lam_hist_q_id, max_edges_to_plot=5, algo_label="ID-QAOA")
        plot_w_profiles_for_algo(f"{save_prefix}_w_profiles_ID_QAOA.png", lam_hist_q_id, w_fun, max_edges_to_plot=5, algo_label="ID-QAOA")
        J_q_id = _safe_array_from_hist(hist_q_id, "J")
        plot_lambda_pca_trajectory(save_prefix, lam_hist_q_id, J_q_id, algo_label="ID-QAOA")
    if len(lam_hist_q_spsa) > 0:
        plot_lambda_components_for_algo(f"{save_prefix}_lambda_components_SPSA_QAOA.png", lam_hist_q_spsa, max_edges_to_plot=5, algo_label="SPSA-QAOA")
        plot_w_profiles_for_algo(f"{save_prefix}_w_profiles_SPSA_QAOA.png", lam_hist_q_spsa, w_fun, max_edges_to_plot=5, algo_label="SPSA-QAOA")
        J_q_spsa = _safe_array_from_hist(hist_q_spsa, "J")
        plot_lambda_pca_trajectory(save_prefix, lam_hist_q_spsa, J_q_spsa, algo_label="SPSA-QAOA")
    if len(lam_hist_v_id) > 0:
        plot_lambda_components_for_algo(f"{save_prefix}_lambda_components_ID_VQE.png", lam_hist_v_id, max_edges_to_plot=5, algo_label="ID-VQE")
        plot_w_profiles_for_algo(f"{save_prefix}_w_profiles_ID_VQE.png", lam_hist_v_id, w_fun, max_edges_to_plot=5, algo_label="ID-VQE")
        J_v_id = _safe_array_from_hist(hist_v_id, "J")
        plot_lambda_pca_trajectory(save_prefix, lam_hist_v_id, J_v_id, algo_label="ID-VQE")
    if len(lam_hist_v_spsa) > 0:
        plot_lambda_components_for_algo(f"{save_prefix}_lambda_components_SPSA_VQE.png", lam_hist_v_spsa, max_edges_to_plot=5, algo_label="SPSA-VQE")
        plot_w_profiles_for_algo(f"{save_prefix}_w_profiles_SPSA_VQE.png", lam_hist_v_spsa, w_fun, max_edges_to_plot=5, algo_label="SPSA-VQE")
        J_v_spsa = _safe_array_from_hist(hist_v_spsa, "J")
        plot_lambda_pca_trajectory(save_prefix, lam_hist_v_spsa, J_v_spsa, algo_label="SPSA-VQE")
    plot_grad_norms(save_prefix, hist_q_id, hist_v_id)
    plot_hamming_distances(save_prefix, gt_bitstring, results)
    plot_metric_bars(save_prefix, results)
    plot_weight_statistics(save_prefix, w_fun, lam0_vec, lam_q_id, lam_q_spsa, lam_v_id, lam_v_spsa)
    plot_w_boxplots_gt_cut(save_prefix, w_fun, edges, gt_bits, lam_q_id, lam_q_spsa, lam_v_id, lam_v_spsa)
    print("Additional plots generated.")

# ======================================================================
# Main & CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Multidimensional ID with Final Table & Real Trajectory Plot")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.5)
    p.add_argument("--lam_bounds", type=float, nargs=2, default=[-5.0, 5.0])
    p.add_argument("--lam0", type=float, default=0.0)
    p.add_argument("--resp_kind", type=str, default="periodic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--outer_iters", type=int, default=200)
    p.add_argument("--eta0", type=float, default=0.2)
    p.add_argument("--inner_spsa_iters", type=int, default=40)
    p.add_argument("--inner_num_starts", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p_main", type=int, default=4)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--vqe_layers", type=int, default=2)
    p.add_argument("--vqe_fair", type=str, default="budget")
    p.add_argument("--readout_shots_per_outer", type=int, default=128)
    p.add_argument("--save_prefix", type=str, default="out_md/res")
    return p.parse_args()

def main(args):
    rng = np.random.default_rng(to_uint_seed(args.seed))
    
    # 1. Graph & Params
    edges, A = generate_random_graph(args.n, args.p_edge, rng)
    Z = precompute_z_patterns(args.n)
    m_edges = len(edges)
    print(f"Graph n={args.n}, m={m_edges} edges")
    
    # 2. Ground Truth (Max Cut)
    print("Computing exact Max-Cut (Ground Truth)...")
    gt_cut, gt_idx, gt_bits = solve_exact_maxcut(args.n, edges)
    gt_bitstring = index_to_bitstring(gt_idx, args.n)
    print(f"Ground Truth J*: {gt_cut}, Bits: {gt_bitstring}")

    resp = make_response_params(edges, rng, resp_kind=args.resp_kind, lam_bounds=tuple(args.lam_bounds))
    w_fun, gradJ = make_w_and_grad_md(resp)
    
    lam0_vec = np.full(m_edges, args.lam0, dtype=float) + rng.uniform(-0.1, 0.1, size=m_edges)
    theta0 = build_qaoa_theta_init(args.p_main, "legacy", 0.8, 0.3)
    
    L_vqe = args.vqe_layers if args.vqe_layers else choose_vqe_layers(args.vqe_fair, args.p_main, args.n, m_edges)
    phi0 = np.zeros(2 * args.n * L_vqe, dtype=float)
    spsa_params = {"a": 0.25, "c": 0.12}

    # 3. Optimization Loops
    print("\n--- ID(QAOA) ---")
    lam_q_id, theta_q_id, hist_q_id = optimize_lambda_with_ID_md_qaoa(
        args.n, edges, Z, w_fun, gradJ, theta0, args.p_main,
        lam0_vec, tuple(args.lam_bounds),
        args.outer_iters, args.eta0, args.inner_spsa_iters, args.inner_num_starts,
        args.seed, args.shots, spsa_params, args.readout_shots_per_outer
    )
    
    print("\n--- ID(VQE) ---")
    lam_v_id, phi_v_id, hist_v_id = optimize_lambda_with_ID_md_vqe(
        args.n, edges, Z, w_fun, gradJ, phi0, L_vqe,
        lam0_vec, tuple(args.lam_bounds),
        args.outer_iters, args.eta0, args.inner_spsa_iters, args.inner_num_starts,
        args.seed + 100, args.shots, spsa_params, args.readout_shots_per_outer
    )

    print("\n--- SPSA(QAOA) ---")
    lam_q_spsa, theta_q_spsa, hist_q_spsa = optimize_lambda_with_SPSA_md_qaoa(
        args.n, edges, Z, w_fun, theta0, args.p_main,
        lam0_vec, tuple(args.lam_bounds),
        args.outer_iters, args.inner_spsa_iters, args.inner_num_starts,
        args.seed + 200, args.shots, spsa_params, args.readout_shots_per_outer
    )
    
    print("\n--- SPSA(VQE) ---")
    lam_v_spsa, phi_v_spsa, hist_v_spsa = optimize_lambda_with_SPSA_md_vqe(
        args.n, edges, Z, w_fun, phi0, L_vqe,
        lam0_vec, tuple(args.lam_bounds),
        args.outer_iters, args.inner_spsa_iters, args.inner_num_starts,
        args.seed + 300, args.shots, spsa_params, args.readout_shots_per_outer
    )

    # 4. Final Analysis & Table Generation
    print("\nGenerating Final Table...")

    def analyze_algo(name, lam_final, params, hist, is_qaoa=True, is_vqe=False):
        # Metrics from History
        J_series = hist["J"]
        J_best_series = hist["J_best_cut"] if hist["J_best_cut"].size > 0 else hist["J"]
        evals = hist["evals_cum"]
        
        J_final = J_series[-1]
        J_best_exp = J_series.max()
        J_best_cut = J_best_series.max()
        num_evals = int(evals[-1])
        
        # Lambda stats
        norm_lam_final = np.linalg.norm(lam_final)
        delta_lam = np.linalg.norm(lam_final - lam0_vec)
        
        # Prob of optimal solution (p_opt)
        # We need to re-evaluate state with final params
        w_final = w_fun(lam_final)
        psi_final = None
        if is_qaoa:
            _, psi_final, _ = qaoa_expectation_and_state_p(args.n, edges, Z, w_final, params[:args.p_main], params[args.p_main:])
        elif is_vqe:
            _, psi_final, _ = vqe_expectation_and_state(args.n, edges, Z, w_final, params, L_vqe)
        
        # Prob of GT
        p_opt = 0.0
        best_str = ""
        if psi_final is not None:
            probs = (psi_final.conj() * psi_final).real
            p_opt = probs[gt_idx]
            best_idx_algo = int(np.argmax(probs))
            best_str = index_to_bitstring(best_idx_algo, args.n)

        # Ratios
        ratio_exp = J_best_exp / gt_cut if gt_cut > 1e-9 else 0.0
        ratio_cut = J_best_cut / gt_cut if gt_cut > 1e-9 else 0.0
        
        return {
            "name": name,
            "J_final": J_final,
            "J_best": J_best_exp,
            "J_best_cut": J_best_cut,
            "lam_norm": norm_lam_final,
            "delta_lam": delta_lam,
            "p_opt": p_opt,
            "ratio_exp": ratio_exp,
            "ratio_cut": ratio_cut,
            "evals": num_evals,
            "best_str": best_str
        }

    results = []
    results.append(analyze_algo("ID-QAOA", lam_q_id, theta_q_id, hist_q_id, is_qaoa=True))
    results.append(analyze_algo("SPSA-QAOA", lam_q_spsa, theta_q_spsa, hist_q_spsa, is_qaoa=True))
    results.append(analyze_algo("ID-VQE", lam_v_id, phi_v_id, hist_v_id, is_qaoa=False, is_vqe=True))
    results.append(analyze_algo("SPSA-VQE", lam_v_spsa, phi_v_spsa, hist_v_spsa, is_qaoa=False, is_vqe=True))

    # Print Table
    header = "{:<12} | {:>9} | {:>9} | {:>11} | {:>9} | {:>8} | {:>9} | {:>9} | {:>10} | {:>8}".format(
        "Algo", "J_final", "J_best", "J_best-cut", "||λ||", "|Δλ|", "p_opt", "J_b/J*", "J_cut/J*", "#evals"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in results:
        print("{name:<12} | {J_final:9.4f} | {J_best:9.4f} | {J_best_cut:11.4f} | {lam_norm:9.4f} | {delta_lam:8.4f} | {p_opt:9.4f} | {ratio_exp:9.4f} | {ratio_cut:10.4f} | {evals:8d}".format(**r))
    print("-" * len(header))

    print(f"\nOptimaler Bitstring (Ground Truth): {gt_bitstring}")
    for r in results:
        print(f"{r['name']:<15} wahrscheinlichster Bitstring: {r['best_str']}")

    # 5. Plot: Real Trajectory
    if "lam_history" in hist_q_id:
        print("\nGenerating Real Trajectory Plot...")
        plot_real_trajectory_landscape(
            f"{args.save_prefix}_real_trajectory.png",
            args.n, edges, Z, w_fun,
            hist_q_id["lam_history"],
            theta_q_id,
            n_random_dirs=50
        )

    # 6. Additional plots
    generate_additional_plots(
        args.save_prefix,
        args,
        edges,
        Z,
        w_fun,
        lam0_vec,
        gt_cut,
        gt_idx,
        gt_bits,
        gt_bitstring,
        results,
        lam_q_id,
        lam_q_spsa,
        lam_v_id,
        lam_v_spsa,
        hist_q_id,
        hist_q_spsa,
        hist_v_id,
        hist_v_spsa
    )
    
    print(f"\nPlots & CSV gespeichert unter Prefix: {args.save_prefix}_*")

if __name__ == "__main__":
    main(parse_args())
