# -*- coding: utf-8 -*-
"""
Core-Funktionen für ID(QAOA) & ID(VQE) für Max-Cut (k=1, kein Cluster).

Enthält:
  - Utils
  - Graph & Z-Pattern
  - Antwortfunktion w(λ) & Grad
  - Cut-Berechnung & Erwartungswerte
  - QAOA- und VQE-Energieauswertung
  - SPSA-Optimierer für QAOA/VQE-Parameter
  - QAOA-Initialisierung & VQE-Layer-Fairness
  - Ground-Truth & Hüllen-Berechnungen
  - ID-Outer-Loops in λ (QAOA/VQE)
  - SPSA-λ-Baselines (QAOA/VQE)
  - Obere Hülle & Top-K-Cuts

Kein Plotting, kein File-I/O, kein argparse.
"""

import math
from typing import Tuple, List, Dict, Optional, Callable

import numpy as np

GRID_POINTS_EXACT = 201
SAFE_W_MAX = 100.0

# ================== Utils ==================
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

# ================== Graph & Z-Pattern ==================
def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator) -> Tuple[List[Tuple[int, int]], np.ndarray]:
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

def index_to_bitstring(idx: int, n: int) -> np.ndarray:
    """Decode basis index -> bitstring (LSB-first, konsistent mit precompute_z_patterns)."""
    bits = [(idx >> i) & 1 for i in range(n)]
    return np.array(bits, dtype=int)

def idx_to_bitstring_str(idx: int, n: int) -> str:
    """Index -> Bitstring-String (MSB-first für Labels)."""
    bits = index_to_bitstring(idx, n)
    return "".join(str(int(b)) for b in bits[::-1])

# ================== Antwortklasse w(λ) & Grad (k=1) ==================
def make_response_params(edges: List[Tuple[int, int]],
                         rng: np.random.Generator,
                         resp_kind: str = "periodic",
                         lam_bounds: Tuple[float, float] = (-2.0, 2.0)) -> Dict[str, np.ndarray]:
    m = len(edges)
    par = {
        "kind": resp_kind,
        "w_init": rng.uniform(0.6, 1.4, size=m).astype(float),
        "lambda0": rng.uniform(-1.0, 1.0, size=m).astype(float)
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

def _w_of_lambda(kind: str, lam: float, w0: float, lam0: float, p: Dict[str, float]) -> float:
    if kind == "linear":
        return w0 + p["b"] * (lam - lam0)
    if kind == "quadratic":
        return w0 + p["c"] * ((lam - lam0) ** 2)
    val = w0 + p["A"] * math.cos(p["kappa"] * (lam - lam0))   # periodic
    wf = p.get("w_floor", None)
    return max(val, float(wf)) if (wf is not None) else val

def _dw_dlambda(kind: str, lam: float, lam0: float, p: Dict[str, float]) -> float:
    if kind == "linear":
        return p["b"]
    if kind == "quadratic":
        return 2.0 * p["c"] * (lam - lam0)
    return -p["A"] * p["kappa"] * math.sin(p["kappa"] * (lam - lam0))  # periodic

def make_w_and_grad_1d(resp: Dict[str, np.ndarray]) -> Tuple[Callable[[float], np.ndarray],
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

# ================== Cut/Erwartungen ==================
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

# ================== QAOA ==================
def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(psi, target, 0)
    block = psi.reshape(2, -1).astype(np.complex128, copy=False)
    np.nan_to_num(block, copy=False)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        two_by_M = gate @ block
    psi = two_by_M.reshape(2, *psi.shape[1:])
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
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
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

def qaoa_energy_p(n: int, edges: List[Tuple[int, int]], Z: List[np.ndarray],
                  w: np.ndarray, theta_vec: np.ndarray,
                  shots: Optional[int] = None,
                  rng: Optional[np.random.Generator] = None) -> float:
    p = len(theta_vec) // 2
    gammas = np.array(theta_vec[:p], dtype=float)
    betas = np.array(theta_vec[p:], dtype=float)
    J, _, _ = qaoa_expectation_and_state_p(n, edges, Z, w, gammas, betas, shots=shots, rng=rng)
    return -J

# ================== VQE (RY–RZ + Ring-CNOT) ==================
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
    probs = (probs / s) if (np.isfinite(s) and s > 0) else np.full_like(probs, 1.0 / K)
    if (shots is None) or (shots <= 0):
        zexp = z_expect_from_probs(probs, edges, Z)
    else:
        rng = rng or np.random.default_rng()
        idx = rng.choice(np.arange(K, dtype=np.int64), size=shots, replace=True, p=probs)
        zexp = z_expect_from_samples(idx, edges, Z)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    return J, psi, zexp

def vqe_energy(n: int, edges: List[Tuple[int, int]], Z: List[np.ndarray],
               w: np.ndarray, params: np.ndarray, L: int,
               shots: Optional[int] = None,
               rng: Optional[np.random.Generator] = None) -> float:
    J, _, _ = vqe_expectation_and_state(n, edges, Z, w, params, L, shots=shots, rng=rng)
    return -J

# ================== SPSA (θ/φ) mit Logging & Eval-Zähler ==================
def spsa_optimize_theta_p(n, edges, Z, w, theta0: np.ndarray,
                          iters: int = 120, a: float = 0.25, c: float = 0.12, A: float = 60.0,
                          alpha: float = 0.602, gamma: float = 0.101,
                          theta_bounds: Optional[List[Tuple[float, float]]] = None,
                          num_starts: int = 1, seed: int = 0, shots: Optional[int] = None,
                          log_trace: bool = False,
                          lambda_val: Optional[float] = None,
                          lambda_log: Optional[List[float]] = None,
                          J_log: Optional[List[float]] = None,
                          grad_k_all: Optional[List[float]] = None,
                          eval_counter: Optional[List[int]] = None):
    """
    SPSA für QAOA-Parameter θ (innerer Loop).

    eval_counter[0] zählt (optional) Anzahl der qaoa_energy_p-Auswertungen.
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    D = len(theta0)
    p = D // 2
    if theta_bounds is None:
        theta_bounds = [(-math.pi, math.pi)] * p + [(-math.pi / 2, math.pi / 2)] * p

    def eval_energy(theta_vec: np.ndarray) -> float:
        E = qaoa_energy_p(n, edges, Z, w, theta_vec, shots=shots, rng=rng)
        if eval_counter is not None:
            eval_counter[0] += 1
        return E

    def one_run(theta_start: np.ndarray, do_log: bool = False):
        theta = theta_start.astype(float).copy()
        best_E_run = None
        best_th_run = theta.copy()
        theta_trace: List[np.ndarray] = []
        J_trace: List[float] = []
        grad_trace_local: List[float] = []

        for k in range(1, iters + 1):
            ak = a / ((k + A) ** alpha)
            ck = c / (k ** gamma)
            delta = rng.choice([-1.0, 1.0], size=D)

            tp = np.clip(theta + ck * delta,
                         [b[0] for b in theta_bounds],
                         [b[1] for b in theta_bounds])
            tm = np.clip(theta - ck * delta,
                         [b[0] for b in theta_bounds],
                         [b[1] for b in theta_bounds])

            Ep = eval_energy(tp)
            Em = eval_energy(tm)

            diff = Ep - Em
            g_scalar = abs(diff / (2.0 * ck))  # Skalarer SPSA-Gradientfaktor
            if grad_k_all is not None:
                grad_k_all.append(float(g_scalar))
            if do_log:
                grad_trace_local.append(float(g_scalar))

            theta -= ak * diff / (2.0 * ck) * delta
            for i, (lo, hi) in enumerate(theta_bounds):
                theta[i] = float(np.clip(theta[i], lo, hi))

            E = eval_energy(theta)
            J_val = -E

            if (lambda_log is not None) and (J_log is not None) and (lambda_val is not None):
                lambda_log.append(float(lambda_val))
                J_log.append(float(J_val))

            if do_log:
                theta_trace.append(theta.copy())
                J_trace.append(float(J_val))

            if (best_E_run is None) or (E < best_E_run):
                best_E_run, best_th_run = E, theta.copy()

        return best_th_run, best_E_run, theta_trace, J_trace, grad_trace_local

    best_theta, best_E = None, None
    best_trace_theta: List[np.ndarray] = []
    best_trace_J: List[float] = []
    best_grad_trace: List[float] = []
    best_Es_starts: List[float] = []

    for s in range(num_starts):
        do_log = log_trace and (s == 0)
        if s == 0:
            th, E_run, theta_trace, J_trace, grad_trace_local = one_run(theta0.copy(), do_log=do_log)
        else:
            gamma0 = rng.uniform(-1.0, 1.0, size=p)
            beta0 = rng.uniform(0.0, 0.6, size=p)
            theta_start = np.concatenate([gamma0, beta0])
            th, E_run, theta_trace, J_trace, grad_trace_local = one_run(theta_start, do_log=False)

        best_Es_starts.append(E_run)

        if do_log:
            best_trace_theta = theta_trace
            best_trace_J = J_trace
            best_grad_trace = grad_trace_local

        if (best_E is None) or (E_run < best_E):
            best_theta, best_E = th, E_run

    theta_trace_arr = np.array(best_trace_theta, dtype=float) if best_trace_theta else np.empty((0, D), dtype=float)
    J_trace_arr = np.array(best_trace_J, dtype=float) if best_trace_J else np.empty(0, dtype=float)
    grad_trace_arr = np.array(best_grad_trace, dtype=float) if best_grad_trace else np.empty(0, dtype=float)
    best_Es_arr = np.array(best_Es_starts, dtype=float)

    return best_theta, float(best_E), theta_trace_arr, J_trace_arr, grad_trace_arr, best_Es_arr

def spsa_optimize_vqe_params(n, edges, Z, w, params0: np.ndarray, L_layers: int,
                             iters: int = 120, a: float = 0.25, c: float = 0.12, A: float = 60.0,
                             alpha: float = 0.602, gamma: float = 0.101,
                             param_bounds: Optional[List[Tuple[float, float]]] = None,
                             num_starts: int = 1, seed: int = 0, shots: Optional[int] = None,
                             log_trace: bool = False,
                             lambda_val: Optional[float] = None,
                             lambda_log: Optional[List[float]] = None,
                             J_log: Optional[List[float]] = None,
                             grad_k_all: Optional[List[float]] = None,
                             eval_counter: Optional[List[int]] = None):
    """
    SPSA für VQE-Parameter φ (innerer Loop).
    eval_counter[0] zählt (optional) vqe_energy-Auswertungen.
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    D = len(params0)
    bounds = param_bounds or [(-math.pi, math.pi)] * D

    def eval_energy(phi_vec: np.ndarray) -> float:
        E = vqe_energy(n, edges, Z, w, phi_vec, L_layers, shots=shots, rng=rng)
        if eval_counter is not None:
            eval_counter[0] += 1
        return E

    def one_run(phi_start: np.ndarray, do_log: bool = False):
        phi = phi_start.astype(float).copy()
        best_E_run = None
        best_phi_run = phi.copy()
        phi_trace: List[np.ndarray] = []
        J_trace: List[float] = []
        grad_trace_local: List[float] = []

        for k in range(1, iters + 1):
            ak = a / ((k + A) ** alpha)
            ck = c / (k ** gamma)
            delta = rng.choice([-1.0, 1.0], size=D)

            pp = np.clip(phi + ck * delta,
                         [b[0] for b in bounds],
                         [b[1] for b in bounds])
            pm = np.clip(phi - ck * delta,
                         [b[0] for b in bounds],
                         [b[1] for b in bounds])

            Ep = eval_energy(pp)
            Em = eval_energy(pm)

            diff = Ep - Em
            g_scalar = abs(diff / (2.0 * ck))
            if grad_k_all is not None:
                grad_k_all.append(float(g_scalar))
            if do_log:
                grad_trace_local.append(float(g_scalar))

            phi -= ak * diff / (2.0 * ck) * delta
            for i, (lo, hi) in enumerate(bounds):
                phi[i] = float(np.clip(phi[i], lo, hi))

            E = eval_energy(phi)
            J_val = -E

            if (lambda_log is not None) and (J_log is not None) and (lambda_val is not None):
                lambda_log.append(float(lambda_val))
                J_log.append(float(J_val))

            if do_log:
                phi_trace.append(phi.copy())
                J_trace.append(float(J_val))

            if (best_E_run is None) or (E < best_E_run):
                best_E_run, best_phi_run = E, phi.copy()

        return best_phi_run, best_E_run, phi_trace, J_trace, grad_trace_local

    best_phi, best_E = None, None
    best_trace_phi: List[np.ndarray] = []
    best_trace_J: List[float] = []
    best_grad_trace: List[float] = []
    best_Es_starts: List[float] = []

    for s in range(num_starts):
        do_log = log_trace and (s == 0)
        if s == 0:
            ph_start = params0.copy()
            ph, E_run, phi_trace, J_trace, grad_trace_local = one_run(ph_start, do_log=do_log)
        else:
            ph_start = np.random.default_rng(to_uint_seed(seed + 999 + s)).uniform(-1.0, 1.0, size=D)
            ph, E_run, phi_trace, J_trace, grad_trace_local = one_run(ph_start, do_log=False)

        best_Es_starts.append(E_run)

        if do_log:
            best_trace_phi = phi_trace
            best_trace_J = J_trace
            best_grad_trace = grad_trace_local

        if (best_E is None) or (E_run < best_E):
            best_phi, best_E = ph, E_run

    phi_trace_arr = np.array(best_trace_phi, dtype=float) if best_trace_phi else np.empty((0, D), dtype=float)
    J_trace_arr = np.array(best_trace_J, dtype=float) if best_trace_J else np.empty(0, dtype=float)
    grad_trace_arr = np.array(best_grad_trace, dtype=float) if best_grad_trace else np.empty(0, dtype=float)
    best_Es_arr = np.array(best_Es_starts, dtype=float)

    return best_phi, float(best_E), phi_trace_arr, J_trace_arr, grad_trace_arr, best_Es_arr

# ================== VQE-Layer-Wahl (Fairness) ==================
def choose_vqe_layers(fair_mode: str, p_main: int, n: int, m_edges: int, min_layers: int = 1) -> int:
    fair_mode = (fair_mode or "budget").lower()
    if fair_mode == "param":
        return max(min_layers, max(1, round(max(1, p_main / max(1, n)))))
    if fair_mode == "hardware":
        return max(min_layers, max(1, round((m_edges / max(1, n)) * p_main)))
    return max(2, p_main)  # budget (robust)

# ================== QAOA-Init aus (γ*, β*) ==================
def build_qaoa_theta_init(p: int,
                          mode: str,
                          gamma_star: float,
                          beta_star: float) -> np.ndarray:
    """
    QAOA-Initialisierung für p-Layer aus (γ*, β*).

    mode in {"legacy", "stack", "ramp"}:
      - legacy: alle Layer γ_l=0.8, β_l=0.3
      - stack:  alle Layer γ_l=γ*,  β_l=β*
      - ramp:   γ_l=(l/p)*γ*, β_l=(1 - l/(p+1))*β*, l=1..p
    """
    mode = (mode or "legacy").lower()
    if mode == "legacy":
        gammas0 = np.full(p, 0.8, dtype=float)
        betas0  = np.full(p, 0.3, dtype=float)
    elif mode == "stack":
        gammas0 = np.full(p, gamma_star, dtype=float)
        betas0  = np.full(p, beta_star, dtype=float)
    elif mode == "ramp":
        l = np.arange(1, p + 1, dtype=float)
        gammas0 = (l / float(p)) * gamma_star
        betas0  = (1.0 - l / float(p + 1.0)) * beta_star
    else:
        gammas0 = np.full(p, 0.8, dtype=float)
        betas0  = np.full(p, 0.3, dtype=float)
    return np.concatenate([gammas0, betas0])

# ================== Ground Truth & Hülle ==================
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
                           lam_bounds: Tuple[float, float], grid_points: int = GRID_POINTS_EXACT) -> Tuple[float, float]:
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

# ================== Hülle & Top-K (für Plots) ==================
def compute_upper_envelope_and_topk(edges, Z, w_fun_1d,
                                    lam_bounds: Tuple[float, float],
                                    grid_points: int = GRID_POINTS_EXACT,
                                    topk: int = 40, union_factor: int = 4):
    lam_grid = np.linspace(lam_bounds[0], lam_bounds[1], grid_points, dtype=float)
    L = len(lam_grid)
    J_star = np.empty(L, dtype=float)
    s_star = np.empty(L, dtype=np.int64)
    candidate_ids = set()
    for i, lam in enumerate(lam_grid):
        v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(float(lam)), edges, Z)
        J_star[i] = float(v.max())
        s_star[i] = int(v.argmax())
        if topk > 0:
            idx = np.argpartition(v, -min(topk, v.size))[-min(topk, v.size):]
            candidate_ids.update(int(j) for j in idx)
    switches = [i for i in range(1, L) if s_star[i] != s_star[i - 1]]
    switch_lams = lam_grid[switches]
    switch_Js = J_star[switches]

    cand_list = list(candidate_ids)
    if len(cand_list) > topk * union_factor:
        max_vals = {cid: -1e300 for cid in cand_list}
        for lam in lam_grid:
            v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(float(lam)), edges, Z)
            for cid in cand_list:
                val = float(v[cid])
                if val > max_vals[cid]:
                    max_vals[cid] = val
        cand_list = sorted(cand_list, key=lambda c: max_vals[c], reverse=True)[:topk * union_factor]

    if len(cand_list) > topk:
        max_vals = {cid: -1e300 for cid in cand_list}
        for lam in lam_grid:
            v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(float(lam)), edges, Z)
            for cid in cand_list:
                val = float(v[cid])
                if val > max_vals[cid]:
                    max_vals[cid] = val
        cand_list = sorted(cand_list, key=lambda c: max_vals[c], reverse=True)[:topk]

    top_ids = cand_list
    top_curves = np.zeros((len(top_ids), L), dtype=float)
    for j, lam in enumerate(lam_grid):
        v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(float(lam)), edges, Z)
        for r, cid in enumerate(top_ids):
            top_curves[r, j] = float(v[cid])

    return lam_grid, J_star, s_star, switch_lams, switch_Js, top_ids, top_curves

# ================== ID-Outer: best-so-far + Historie (Adam auf λ) ==================
def optimize_lambda_with_ID_1d_qaoa(n, edges, Z, w_fun_1d, gradJ_1d,
                                    theta_init: np.ndarray, p_layers: int,
                                    lam0: float, lam_bounds: Tuple[float, float],
                                    outer_iters: int = 12, eta0: float = 0.25,
                                    inner_spsa_iters: int = 100, inner_num_starts: int = 2,
                                    seed: int = 0, shots: Optional[int] = None,
                                    spsa_theta_a: float = 0.25, spsa_theta_c: float = 0.12, spsa_theta_A: float = 60.0,
                                    spsa_theta_alpha: float = 0.602, spsa_theta_gamma: float = 0.101,
                                    idx_opt: Optional[int] = None):
    """
    ID(QAOA) mit SPSA innen und Adam-Update für λ außen.

    Gibt best_lam, best_theta, hist zurück.
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    lam = float(lam0)
    theta = theta_init.astype(float).copy()
    p = p_layers
    theta_bounds = [(-math.pi, math.pi)] * p + [(-math.pi / 2, math.pi / 2)] * p
    a, b = lam_bounds

    # Adam-States für λ
    m = 0.0
    v = 0.0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    eta = eta0

    best_J = -1e300
    best_lam = lam
    best_theta = theta.copy()

    lam_hist: List[float] = []
    lam_hist_pre: List[float] = []
    grad_hist: List[float] = []
    grad_true_hist: List[float] = []
    J_hist: List[float] = []
    J_env_hist: List[float] = []

    lambda_log: List[float] = []
    J_log: List[float] = []
    inner_theta_last: Optional[np.ndarray] = None
    inner_J_last: Optional[np.ndarray] = None
    inner_spsa_grad_last: Optional[np.ndarray] = None
    p_opt_hist: List[float] = []

    spsa_grad_all: List[float] = []
    bestJ_per_start: List[float] = []

    eval_counter = [0]

    for t in range(1, outer_iters + 1):
        # λ vor Update loggen
        lam_hist_pre.append(lam)

        w = w_fun_1d(lam)
        log_trace = (t == outer_iters)

        theta, _, theta_trace, J_trace, grad_trace, bestEs_arr = spsa_optimize_theta_p(
            n, edges, Z, w, theta0=theta,
            iters=inner_spsa_iters,
            a=spsa_theta_a, c=spsa_theta_c, A=spsa_theta_A,
            alpha=spsa_theta_alpha, gamma=spsa_theta_gamma,
            theta_bounds=theta_bounds, num_starts=inner_num_starts,
            seed=seed + 10 * t, shots=shots,
            log_trace=log_trace,
            lambda_val=lam,
            lambda_log=lambda_log,
            J_log=J_log,
            grad_k_all=spsa_grad_all,
            eval_counter=eval_counter,
        )
        if log_trace:
            inner_theta_last = theta_trace
            inner_J_last = J_trace
            inner_spsa_grad_last = grad_trace

        bestJ_per_start.extend(list(-bestEs_arr))

        J_val, psi, zexp = qaoa_expectation_and_state_p(
            n, edges, Z, w,
            theta[:p], theta[p:], shots=shots, rng=rng
        )
        eval_counter[0] += 1
        J_hist.append(float(J_val))

        J_env, idx_active = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        p_cut_true = np.empty(len(edges), dtype=float)
        for e, (i, j) in enumerate(edges):
            z_i = Z[i][idx_active]
            z_j = Z[j][idx_active]
            p_cut_true[e] = 0.5 * (1.0 - float(z_i) * float(z_j))
        g_true = gradJ_1d(lam, p_cut_true)
        grad_true_hist.append(float(g_true))

        if idx_opt is not None:
            probs = (psi.conj() * psi).real.astype(float)
            p_opt_hist.append(float(probs[idx_opt]))

        if J_val > best_J + 1e-12:
            best_J, best_lam, best_theta = J_val, lam, theta.copy()

        g = gradJ_1d(lam, 0.5 * (1.0 - zexp))
        grad_hist.append(g)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        step = eta * m_hat / (math.sqrt(v_hat) + eps)
        lam = float(np.clip(lam + step, a, b))

        lam_hist.append(lam)
        print(f"[ID(QAOA-Adam) {t:02d}] J={J_val:.6f}  λ={lam:.6f}  | best={best_J:.6f} @ λ={best_lam:.6f}")

    if inner_theta_last is None:
        inner_theta_last = np.empty((0, len(theta_init)), dtype=float)
    if inner_J_last is None:
        inner_J_last = np.empty(0, dtype=float)
    if inner_spsa_grad_last is None:
        inner_spsa_grad_last = np.empty(0, dtype=float)

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "grad": np.array(grad_hist, dtype=float),
        "grad_true": np.array(grad_true_hist, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "lambda_J_lam": np.array(lambda_log, dtype=float),
        "lambda_J_J": np.array(J_log, dtype=float),
        "inner_theta": inner_theta_last,
        "inner_J": inner_J_last,
        "inner_spsa_grad": inner_spsa_grad_last,
        "p_opt": np.array(p_opt_hist, dtype=float),
        "spsa_grad_all": np.array(spsa_grad_all, dtype=float),
        "spsa_bestJ_per_start": np.array(bestJ_per_start, dtype=float),
        "evals": int(eval_counter[0]),
    }
    return best_lam, best_theta, hist

def optimize_lambda_with_ID_1d_vqe(n, edges, Z, w_fun_1d, gradJ_1d,
                                   phi_init: np.ndarray, L_layers: int,
                                   lam0: float, lam_bounds: Tuple[float, float],
                                   outer_iters: int = 12, eta0: float = 0.25,
                                   inner_spsa_iters: int = 100, inner_num_starts: int = 1,
                                   seed: int = 0, shots: Optional[int] = None,
                                   spsa_vqe_a: float = 0.25, spsa_vqe_c: float = 0.12, spsa_vqe_A: float = 60.0,
                                   spsa_vqe_alpha: float = 0.602, spsa_vqe_gamma: float = 0.101,
                                   idx_opt: Optional[int] = None):
    """
    ID(VQE) mit SPSA innen und Adam-Update für λ außen.
    Logging analog zu ID(QAOA).
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    lam = float(lam0)
    phi = phi_init.astype(float).copy()
    D = len(phi)
    bounds = [(-math.pi, math.pi)] * D
    a, b = lam_bounds

    m = 0.0
    v = 0.0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    eta = eta0

    best_J = -1e300
    best_lam = lam
    best_phi = phi.copy()

    lam_hist: List[float] = []
    lam_hist_pre: List[float] = []
    grad_hist: List[float] = []
    grad_true_hist: List[float] = []
    J_hist: List[float] = []
    J_env_hist: List[float] = []

    lambda_log: List[float] = []
    J_log: List[float] = []
    inner_phi_last: Optional[np.ndarray] = None
    inner_J_last: Optional[np.ndarray] = None
    inner_spsa_grad_last: Optional[np.ndarray] = None
    p_opt_hist: List[float] = []

    spsa_grad_all: List[float] = []
    bestJ_per_start: List[float] = []

    eval_counter = [0]

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)

        w = w_fun_1d(lam)
        log_trace = (t == outer_iters)

        phi, _, phi_trace, J_trace, grad_trace, bestEs_arr = spsa_optimize_vqe_params(
            n, edges, Z, w, params0=phi, L_layers=L_layers,
            iters=inner_spsa_iters, param_bounds=bounds,
            a=spsa_vqe_a, c=spsa_vqe_c, A=spsa_vqe_A,
            alpha=spsa_vqe_alpha, gamma=spsa_vqe_gamma,
            num_starts=inner_num_starts, seed=seed + 10 * t, shots=shots,
            log_trace=log_trace,
            lambda_val=lam,
            lambda_log=lambda_log,
            J_log=J_log,
            grad_k_all=spsa_grad_all,
            eval_counter=eval_counter,
        )
        if log_trace:
            inner_phi_last = phi_trace
            inner_J_last = J_trace
            inner_spsa_grad_last = grad_trace

        bestJ_per_start.extend(list(-bestEs_arr))

        J_val, psi, zexp = vqe_expectation_and_state(
            n, edges, Z, w, phi, L_layers, shots=shots, rng=rng
        )
        eval_counter[0] += 1
        J_hist.append(float(J_val))

        J_env, idx_active = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        p_cut_true = np.empty(len(edges), dtype=float)
        for e, (i, j) in enumerate(edges):
            z_i = Z[i][idx_active]
            z_j = Z[j][idx_active]
            p_cut_true[e] = 0.5 * (1.0 - float(z_i) * float(z_j))
        g_true = gradJ_1d(lam, p_cut_true)
        grad_true_hist.append(float(g_true))

        if idx_opt is not None:
            probs = (psi.conj() * psi).real.astype(float)
            p_opt_hist.append(float(probs[idx_opt]))

        if J_val > best_J + 1e-12:
            best_J, best_lam, best_phi = J_val, lam, phi.copy()

        g = gradJ_1d(lam, 0.5 * (1.0 - zexp))
        grad_hist.append(g)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        step = eta * m_hat / (math.sqrt(v_hat) + eps)
        lam = float(np.clip(lam + step, a, b))

        lam_hist.append(lam)
        print(f"[ID(VQE-Adam)  {t:02d}] J={J_val:.6f}  λ={lam:.6f}  | best={best_J:.6f} @ λ={best_lam:.6f}")

    if inner_phi_last is None:
        inner_phi_last = np.empty((0, D), dtype=float)
    if inner_J_last is None:
        inner_J_last = np.empty(0, dtype=float)
    if inner_spsa_grad_last is None:
        inner_spsa_grad_last = np.empty(0, dtype=float)

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "grad": np.array(grad_hist, dtype=float),
        "grad_true": np.array(grad_true_hist, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "lambda_J_lam": np.array(lambda_log, dtype=float),
        "lambda_J_J": np.array(J_log, dtype=float),
        "inner_theta": inner_phi_last,
        "inner_J": inner_J_last,
        "inner_spsa_grad": inner_spsa_grad_last,
        "p_opt": np.array(p_opt_hist, dtype=float),
        "spsa_grad_all": np.array(spsa_grad_all, dtype=float),
        "spsa_bestJ_per_start": np.array(bestJ_per_start, dtype=float),
        "evals": int(eval_counter[0]),
    }
    return best_lam, best_phi, hist

# ================== SPSA-λ-Baselines (QAOA & VQE) ==================
def optimize_lambda_with_SPSA_1d_qaoa(
        n, edges, Z, w_fun_1d,
        theta_init: np.ndarray, p_layers: int,
        lam0: float, lam_bounds: Tuple[float, float],
        outer_iters: int = 12,
        inner_spsa_iters: int = 100, inner_num_starts: int = 2,
        seed: int = 0, shots: Optional[int] = None,
        spsa_theta_a: float = 0.25, spsa_theta_c: float = 0.12, spsa_theta_A: float = 60.0,
        spsa_theta_alpha: float = 0.602, spsa_theta_gamma: float = 0.101,
        idx_opt: Optional[int] = None):
    """
    SPSA-Baseline für λ bei QAOA:
      - Innerer Loop: SPSA auf θ (wie ID)
      - Äußerer Loop: SPSA auf λ mit endlichem Differenzenquotienten von J(λ)
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    lam = float(lam0)
    theta = theta_init.astype(float).copy()
    p = p_layers
    theta_bounds = [(-math.pi, math.pi)] * p + [(-math.pi / 2, math.pi / 2)] * p
    a, b = lam_bounds

    a_lam = spsa_theta_a
    c_lam = spsa_theta_c
    A_lam = spsa_theta_A
    alpha_lam = spsa_theta_alpha
    gamma_lam = spsa_theta_gamma

    best_J = -1e300
    best_lam = lam
    best_theta = theta.copy()

    lam_hist: List[float] = []
    lam_hist_pre: List[float] = []
    J_hist: List[float] = []
    J_env_hist: List[float] = []
    p_opt_hist: List[float] = []

    lambda_log: List[float] = []
    J_log: List[float] = []
    spsa_grad_all: List[float] = []
    bestJ_per_start: List[float] = []

    inner_theta_last: Optional[np.ndarray] = None
    inner_J_last: Optional[np.ndarray] = None
    inner_spsa_grad_last: Optional[np.ndarray] = None

    eval_counter = [0]
    spsa_lambda_grad_abs: List[float] = []

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)
        w = w_fun_1d(lam)
        log_trace = (t == outer_iters)

        theta, _, theta_trace, J_trace, grad_trace, bestEs_arr = spsa_optimize_theta_p(
            n, edges, Z, w, theta0=theta,
            iters=inner_spsa_iters,
            a=spsa_theta_a, c=spsa_theta_c, A=spsa_theta_A,
            alpha=spsa_theta_alpha, gamma=spsa_theta_gamma,
            theta_bounds=theta_bounds, num_starts=inner_num_starts,
            seed=seed + 10 * t, shots=shots,
            log_trace=log_trace,
            lambda_val=lam,
            lambda_log=lambda_log,
            J_log=J_log,
            grad_k_all=spsa_grad_all,
            eval_counter=eval_counter,
        )
        if log_trace:
            inner_theta_last = theta_trace
            inner_J_last = J_trace
            inner_spsa_grad_last = grad_trace

        bestJ_per_start.extend(list(-bestEs_arr))

        J_val, psi, _ = qaoa_expectation_and_state_p(
            n, edges, Z, w,
            theta[:p], theta[p:], shots=shots, rng=rng
        )
        eval_counter[0] += 1
        J_hist.append(float(J_val))

        J_env, _ = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        if idx_opt is not None:
            probs = (psi.conj() * psi).real.astype(float)
            p_opt_hist.append(float(probs[idx_opt]))

        if J_val > best_J + 1e-12:
            best_J, best_lam, best_theta = J_val, lam, theta.copy()

        ak = a_lam / ((t + A_lam) ** alpha_lam)
        ck = c_lam / (t ** gamma_lam)
        delta = rng.choice([-1.0, 1.0])

        lam_p = float(np.clip(lam + ck * delta, a, b))
        lam_m = float(np.clip(lam - ck * delta, a, b))

        Jp, _, _ = qaoa_expectation_and_state_p(
            n, edges, Z, w_fun_1d(lam_p),
            theta[:p], theta[p:], shots=shots, rng=rng
        )
        Jm, _, _ = qaoa_expectation_and_state_p(
            n, edges, Z, w_fun_1d(lam_m),
            theta[:p], theta[p:], shots=shots, rng=rng
        )
        eval_counter[0] += 2

        g_lam = (Jp - Jm) / (2.0 * ck * delta)
        spsa_lambda_grad_abs.append(abs(float(g_lam)))

        lam = float(np.clip(lam + ak * g_lam, a, b))

        lam_hist.append(lam)
        print(f"[SPSA-λ(QAOA) {t:02d}] J={J_val:.6f}  λ={lam:.6f}  | best={best_J:.6f} @ λ={best_lam:.6f}")

    if inner_theta_last is None:
        inner_theta_last = np.empty((0, len(theta_init)), dtype=float)
    if inner_J_last is None:
        inner_J_last = np.empty(0, dtype=float)
    if inner_spsa_grad_last is None:
        inner_spsa_grad_last = np.empty(0, dtype=float)

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "grad": np.empty(0, dtype=float),
        "grad_true": np.empty(0, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "lambda_J_lam": np.array(lambda_log, dtype=float),
        "lambda_J_J": np.array(J_log, dtype=float),
        "inner_theta": inner_theta_last,
        "inner_J": inner_J_last,
        "inner_spsa_grad": inner_spsa_grad_last,
        "p_opt": np.array(p_opt_hist, dtype=float),
        "spsa_grad_all": np.array(spsa_grad_all, dtype=float),
        "spsa_bestJ_per_start": np.array(bestJ_per_start, dtype=float),
        "spsa_lambda_grad_abs": np.array(spsa_lambda_grad_abs, dtype=float),
        "evals": int(eval_counter[0]),
    }
    return best_lam, best_theta, hist

def optimize_lambda_with_SPSA_1d_vqe(
        n, edges, Z, w_fun_1d,
        phi_init: np.ndarray, L_layers: int,
        lam0: float, lam_bounds: Tuple[float, float],
        outer_iters: int = 12,
        inner_spsa_iters: int = 100, inner_num_starts: int = 1,
        seed: int = 0, shots: Optional[int] = None,
        spsa_vqe_a: float = 0.25, spsa_vqe_c: float = 0.12, spsa_vqe_A: float = 60.0,
        spsa_vqe_alpha: float = 0.602, spsa_vqe_gamma: float = 0.101,
        idx_opt: Optional[int] = None):
    """
    SPSA-Baseline für λ bei VQE:
      - Innerer Loop: SPSA auf φ
      - Äußerer Loop: SPSA auf λ mit endlichem Differenzenquotienten von J(λ)
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    lam = float(lam0)
    phi = phi_init.astype(float).copy()
    D = len(phi)
    bounds = [(-math.pi, math.pi)] * D
    a, b = lam_bounds

    a_lam = spsa_vqe_a
    c_lam = spsa_vqe_c
    A_lam = spsa_vqe_A
    alpha_lam = spsa_vqe_alpha
    gamma_lam = spsa_vqe_gamma

    best_J = -1e300
    best_lam = lam
    best_phi = phi.copy()

    lam_hist: List[float] = []
    lam_hist_pre: List[float] = []
    J_hist: List[float] = []
    J_env_hist: List[float] = []
    p_opt_hist: List[float] = []

    lambda_log: List[float] = []
    J_log: List[float] = []
    spsa_grad_all: List[float] = []
    bestJ_per_start: List[float] = []

    inner_phi_last: Optional[np.ndarray] = None
    inner_J_last: Optional[np.ndarray] = None
    inner_spsa_grad_last: Optional[np.ndarray] = None

    eval_counter = [0]
    spsa_lambda_grad_abs: List[float] = []

    for t in range(1, outer_iters + 1):
        lam_hist_pre.append(lam)
        w = w_fun_1d(lam)
        log_trace = (t == outer_iters)

        phi, _, phi_trace, J_trace, grad_trace, bestEs_arr = spsa_optimize_vqe_params(
            n, edges, Z, w, params0=phi, L_layers=L_layers,
            iters=inner_spsa_iters, param_bounds=bounds,
            a=a_lam, c=c_lam, A=A_lam,
            alpha=alpha_lam, gamma=gamma_lam,
            num_starts=inner_num_starts, seed=seed + 10 * t, shots=shots,
            log_trace=log_trace,
            lambda_val=lam,
            lambda_log=lambda_log,
            J_log=J_log,
            grad_k_all=spsa_grad_all,
            eval_counter=eval_counter,
        )
        if log_trace:
            inner_phi_last = phi_trace
            inner_J_last = J_trace
            inner_spsa_grad_last = grad_trace

        bestJ_per_start.extend(list(-bestEs_arr))

        J_val, psi, _ = vqe_expectation_and_state(
            n, edges, Z, w, phi, L_layers, shots=shots, rng=rng
        )
        eval_counter[0] += 1
        J_hist.append(float(J_val))

        J_env, _ = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
        J_env_hist.append(float(J_env))

        if idx_opt is not None:
            probs = (psi.conj() * psi).real.astype(float)
            p_opt_hist.append(float(probs[idx_opt]))

        if J_val > best_J + 1e-12:
            best_J, best_lam, best_phi = J_val, lam, phi.copy()

        ak = a_lam / ((t + A_lam) ** alpha_lam)
        ck = c_lam / (t ** gamma_lam)
        delta = rng.choice([-1.0, 1.0])

        lam_p = float(np.clip(lam + ck * delta, a, b))
        lam_m = float(np.clip(lam - ck * delta, a, b))

        Jp, _, _ = vqe_expectation_and_state(
            n, edges, Z, w_fun_1d(lam_p), phi, L_layers, shots=shots, rng=rng
        )
        Jm, _, _ = vqe_expectation_and_state(
            n, edges, Z, w_fun_1d(lam_m), phi, L_layers, shots=shots, rng=rng
        )
        eval_counter[0] += 2

        g_lam = (Jp - Jm) / (2.0 * ck * delta)
        spsa_lambda_grad_abs.append(abs(float(g_lam)))

        lam = float(np.clip(lam + ak * g_lam, a, b))

        lam_hist.append(lam)
        print(f"[SPSA-λ(VQE)  {t:02d}] J={J_val:.6f}  λ={lam:.6f}  | best={best_J:.6f} @ λ={best_lam:.6f}")

    if inner_phi_last is None:
        inner_phi_last = np.empty((0, D), dtype=float)
    if inner_J_last is None:
        inner_J_last = np.empty(0, dtype=float)
    if inner_spsa_grad_last is None:
        inner_spsa_grad_last = np.empty(0, dtype=float)

    hist = {
        "lam": np.array(lam_hist, dtype=float),
        "lam_pre": np.array(lam_hist_pre, dtype=float),
        "grad": np.empty(0, dtype=float),
        "grad_true": np.empty(0, dtype=float),
        "J": np.array(J_hist, dtype=float),
        "J_env": np.array(J_env_hist, dtype=float),
        "lambda_J_lam": np.array(lambda_log, dtype=float),
        "lambda_J_J": np.array(J_log, dtype=float),
        "inner_theta": inner_phi_last,
        "inner_J": inner_J_last,
        "inner_spsa_grad": inner_spsa_grad_last,
        "p_opt": np.array(p_opt_hist, dtype=float),
        "spsa_grad_all": np.array(spsa_grad_all, dtype=float),
        "spsa_bestJ_per_start": np.array(bestJ_per_start, dtype=float),
        "spsa_lambda_grad_abs": np.array(spsa_lambda_grad_abs, dtype=float),
        "evals": int(eval_counter[0]),
    }
    return best_lam, best_phi, hist
