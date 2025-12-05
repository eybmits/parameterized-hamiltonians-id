# -*- coding: utf-8 -*-
"""
plotting_utils.py
Enthält alle Plotting-Funktionen, CSV-Export und Tabellenausgaben.
Wird von main_simulation.py importiert.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Dict, List, Tuple, Optional

# --- Helper für Plotting ---

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)

def index_to_bitstring(idx: int, n: int) -> np.ndarray:
    """Basisindex -> Bitstring (LSB-first). Wird für Labels benötigt."""
    bits = [(idx >> i) & 1 for i in range(n)]
    return np.array(bits, dtype=int)

def envelope_value_and_active_id(lam: float, edges, Z, w_fun_1d, compute_cut_vals_func) -> Tuple[float, int]:
    """Hilfsfunktion für CSV-Export, um J_star neu zu berechnen."""
    v = compute_cut_vals_func(w_fun_1d(lam), edges, Z)
    idx = int(np.argmax(v))
    return float(v[idx]), idx

# --- CSV Export ---

def save_active_instances_csv(save_path: str, algo_name: str, lams: np.ndarray,
                              edges, Z, w_fun_1d, compute_cut_vals_func):
    _ensure_dir(save_path)
    prev = None
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "iter", "lambda", "active_id", "J_star", "switched"])
        for i, lam in enumerate(lams, start=1):
            J_star, s_star = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d, compute_cut_vals_func)
            switched = int(prev is not None and s_star != prev)
            w.writerow([algo_name, i, f"{lam:.12f}", s_star, f"{J_star:.12f}", switched])
            prev = s_star
    print(f"[saved] {save_path}")

# --- Plots A-E (Standard) ---

def plot_envelope_and_expectations(save_path: str,
                                   lam_grid: np.ndarray,
                                   J_star: np.ndarray,
                                   hist_q: Dict[str, np.ndarray],
                                   hist_v: Dict[str, np.ndarray]):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.fill_between(lam_grid, np.zeros_like(J_star), J_star, color="lightgray", alpha=0.4, label=r"Obere Hülle $J^*(\lambda)$")
    ax.plot(lam_grid, J_star, color="black", lw=2)

    def plot_hist(hist, color, marker, label):
        lam = hist.get("lam_pre", np.array([]))
        J = hist.get("J", np.array([]))
        if lam.size and J.size:
            L = min(lam.size, J.size)
            ax.plot(lam[:L], J[:L], "-" + marker, color=color, lw=1.8, markersize=4, label=label)

    plot_hist(hist_q, "red", "o", "ID(QAOA): Erwartungswerte")
    plot_hist(hist_v, "green", "s", "ID(VQE): Erwartungswerte")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J$")
    ax.set_title(r"Obere Hülle und Erwartungswerte $J_t$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_envelope_and_best_cuts(save_path: str,
                                lam_grid: np.ndarray,
                                J_star: np.ndarray,
                                hist_q: Dict[str, np.ndarray],
                                hist_v: Dict[str, np.ndarray]):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.fill_between(lam_grid, np.zeros_like(J_star), J_star, color="lightgray", alpha=0.4, label=r"Obere Hülle $J^*(\lambda)$")
    ax.plot(lam_grid, J_star, color="black", lw=2)

    def plot_hist(hist, color, marker, label):
        lam = hist.get("lam_pre", np.array([]))
        Jb = hist.get("J_best_cut", np.array([]))
        if lam.size and Jb.size:
            L = min(lam.size, Jb.size)
            ax.plot(lam[:L], Jb[:L], "-" + marker, color=color, lw=1.8, markersize=4, label=label)

    plot_hist(hist_q, "red", "o", "ID(QAOA): bester Cut (pro λ)")
    plot_hist(hist_v, "green", "s", "ID(VQE): bester Cut (pro λ)")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J$ (bester beobachteter Cut)")
    ax.set_title(r"Obere Hülle und bester beobachteter Cut pro $\lambda$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_envelope_and_mode_cuts(save_path: str,
                                lam_grid: np.ndarray,
                                J_star: np.ndarray,
                                hist_q: Dict[str, np.ndarray],
                                hist_v: Dict[str, np.ndarray]):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.fill_between(lam_grid, np.zeros_like(J_star), J_star, color="lightgray", alpha=0.4, label=r"Obere Hülle $J^*(\lambda)$")
    ax.plot(lam_grid, J_star, color="black", lw=2)

    def plot_hist(hist, color, marker, label):
        lam = hist.get("lam_pre", np.array([]))
        Jm = hist.get("J_mode_cut", np.array([]))
        if lam.size and Jm.size:
            L = min(lam.size, Jm.size)
            ax.plot(lam[:L], Jm[:L], "-" + marker, color=color, lw=1.8, markersize=4, label=label)

    plot_hist(hist_q, "red", "o", "ID(QAOA): Cut des Mode-Bitstrings")
    plot_hist(hist_v, "green", "s", "ID(VQE): Cut des Mode-Bitstrings")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J$ (Mode-Bitstring)")
    ax.set_title(r"Obere Hülle und Cut des häufigsten Bitstrings")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_bestJ_vs_eval_cost(save_path: str,
                            hist_q_id: Dict[str, np.ndarray],
                            hist_v_id: Dict[str, np.ndarray],
                            hist_q_spsa: Dict[str, np.ndarray],
                            hist_v_spsa: Dict[str, np.ndarray],
                            J_true: float):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    def add_curve(hist, color, marker, label):
        evals = hist.get("evals_cum", None)
        J = hist.get("J", None)
        if evals is None or J is None or not evals.size or not J.size:
            return
        L = min(evals.size, J.size)
        evals = evals[:L]
        J = J[:L]
        bestJ = np.maximum.accumulate(J)
        ax.plot(evals, bestJ, "-" + marker, color=color, lw=1.8, markersize=4, label=label)

    add_curve(hist_q_id, "red", "o", "ID(QAOA)")
    add_curve(hist_v_id, "green", "s", "ID(VQE)")
    add_curve(hist_q_spsa, "orange", "x", "SPSA-λ(QAOA)")
    add_curve(hist_v_spsa, "blue", "^", "SPSA-λ(VQE)")

    ax.axhline(J_true, color="black", lw=1.5, ls="--", label=r"$J^*$ (Ground Truth)")
    ax.set_xlabel("Kumulative Auswertungskosten (Energie-Evaluationen)")
    ax.set_ylabel(r"Bester beobachteter Cut-Wert $J$")
    ax.set_title(r"Best-so-far $J$ vs. Auswertungskosten")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_total_eval_costs_bar(save_path: str,
                              hist_q_id: Dict[str, np.ndarray],
                              hist_v_id: Dict[str, np.ndarray],
                              hist_q_spsa: Dict[str, np.ndarray],
                              hist_v_spsa: Dict[str, np.ndarray]):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = ["ID(QAOA)", "ID(VQE)", "SPSA-λ(QAOA)", "SPSA-λ(VQE)"]
    hists = [hist_q_id, hist_v_id, hist_q_spsa, hist_v_spsa]
    totals = []
    for h in hists:
        if h is None:
            totals.append(0.0)
        else:
            evals = h.get("evals_cum", None)
            totals.append(float(evals[-1]) if (evals is not None and evals.size) else 0.0)

    x = np.arange(len(labels))
    ax.bar(x, totals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Gesamte Auswertungskosten")
    ax.set_title("Vergleich der Gesamtkosten (Energie-Evaluationen)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# --- Plots F-I (Analysen) ---

def plot_gradient_norms(save_path: str, hist: Dict[str, np.ndarray], algo_label: str):
    _ensure_dir(save_path)
    outer_grads = hist.get("grad_norm_lambda", np.array([]))
    inner_grads = hist.get("grad_norm_theta", np.array([]))
    iters = np.arange(1, len(outer_grads) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    if outer_grads.size > 0:
        ax1.plot(iters, outer_grads, 'o-', color='purple', label=r"Outer $\|\nabla_\lambda J\|$")
        ax1.set_yscale('log')
        ax1.set_ylabel(r"Gradient Norm $\lambda$")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        ax1.set_title(f"{algo_label}: Herzschlag des Lernens (Gradient Norms)")

    if inner_grads.size > 0:
        ax2.plot(iters, inner_grads, 's-', color='teal', label=r"Inner $\|\nabla_\theta E\|$ (avg)")
        ax2.set_yscale('log')
        ax2.set_ylabel(r"Gradient Norm $\theta$")
        ax2.set_xlabel("Outer Iteration")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_weight_heatmap(save_path: str, hist: Dict[str, np.ndarray], algo_label: str):
    _ensure_dir(save_path)
    w_hist = hist.get("w_hist", None)
    if w_hist is None or w_hist.size == 0:
        return

    w_matrix = w_hist.T
    m_edges, n_iters = w_matrix.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    
    vmin, vmax = w_matrix.min(), w_matrix.max()
    if vmin >= 0: vmin = -1e-6
    if vmax <= 0: vmax = 1e-6
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    
    im = ax.imshow(w_matrix, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest',
                   extent=[0.5, n_iters + 0.5, m_edges - 0.5, -0.5])
    
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Edge ID")
    ax.set_title(f"{algo_label}: Graph Equalizer (Weight Evolution)")
    ax.set_yticks(np.arange(m_edges))
    fig.colorbar(im, ax=ax, label=r"Weight $w_e(\lambda)$")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_param_trajectories(save_path: str, hist: Dict[str, np.ndarray], p_layers: int):
    _ensure_dir(save_path)
    theta_hist = hist.get("param_hist", None)
    if theta_hist is None or theta_hist.size == 0:
        return

    theta_hist = np.array(theta_hist)
    iters = np.arange(1, theta_hist.shape[0] + 1)
    
    gammas = theta_hist[:, :p_layers]
    betas = theta_hist[:, p_layers:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for i in range(p_layers):
        ax1.plot(iters, gammas[:, i], marker='o', markersize=3, label=rf"$\gamma_{{{i+1}}}$")
    ax1.set_ylabel(r"$\gamma$ Value")
    ax1.set_title(r"QAOA Parameter Drift ($\gamma$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    for i in range(p_layers):
        ax2.plot(iters, betas[:, i], marker='s', markersize=3, ls='--', label=rf"$\beta_{{{i+1}}}$")
    ax2.set_ylabel(r"$\beta$ Value")
    ax2.set_xlabel("Outer Iteration")
    ax2.set_title(r"QAOA Parameter Drift ($\beta$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_final_distribution(save_path: str, psi_q: np.ndarray, psi_v: np.ndarray, n: int, top_k: int = 20):
    _ensure_dir(save_path)
    
    def get_probs(psi):
        p = (psi.conj() * psi).real
        return p / p.sum()

    prob_q = get_probs(psi_q)
    prob_v = get_probs(psi_v)
    
    idx_q_sorted = np.argsort(prob_q)[::-1]
    idx_v_sorted = np.argsort(prob_v)[::-1]
    
    top_indices = set(idx_q_sorted[:top_k]) | set(idx_v_sorted[:top_k])
    top_indices = sorted(list(top_indices))
    
    top_indices = sorted(top_indices, key=lambda x: prob_q[x], reverse=True)
    if len(top_indices) > top_k + 5:
        top_indices = top_indices[:top_k]

    labels = ["".join(map(str, index_to_bitstring(idx, n)[::-1])) for idx in top_indices]
    vals_q = prob_q[top_indices]
    vals_v = prob_v[top_indices]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, vals_q, width, label='ID(QAOA)', color='red', alpha=0.7)
    rects2 = ax.bar(x + width/2, vals_v, width, label='ID(VQE)', color='green', alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title(f'Final State Probability Distribution (Top Bitstrings)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# --- NEU: Plot der Spektral-Struktur (Röntgenblick) ---

def plot_full_envelope_structure(save_path: str,
                                 lams: np.ndarray,
                                 all_J_lines: np.ndarray,
                                 J_star: np.ndarray,
                                 active_ids: np.ndarray,
                                 switch_lams: np.ndarray,
                                 switch_vals: np.ndarray,
                                 n_nodes: int):
    """
    Plottet das gesamte Spektrum (alle möglichen Lösungen) und hebt die
    aktiven Instanzen sowie die Hülle hervor.
    
    all_J_lines: shape (n_patterns, n_lambda_points)
    """
    _ensure_dir(save_path)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_patterns = all_J_lines.shape[0]
    
    # Identifiziere, welche Instanzen jemals "aktiv" (Gewinner) waren
    unique_active_ids = np.unique(active_ids)
    is_active_mask = np.zeros(n_patterns, dtype=bool)
    is_active_mask[unique_active_ids] = True
    
    # A) HINTERGRUND: Die "Verlierer" (Grau, Nebel)
    # Da dies sehr viele Linien sein können, plotten wir sie sehr transparent.
    ax.plot(lams, all_J_lines[~is_active_mask].T, color='grey', alpha=0.05, linewidth=0.5)

    # B) MITTELGRUND: Die "Gewinner" (Aktive Instanzen, Blau)
    ax.plot(lams, all_J_lines[is_active_mask].T, color='dodgerblue', alpha=0.6, linewidth=1.5)
    
    # Fake-Lines für Legende
    ax.plot([], [], color='grey', alpha=0.3, label='Inaktive Instanzen')
    ax.plot([], [], color='dodgerblue', alpha=0.8, linewidth=1.5, label='Aktive Instanzen')

    # C) VORDERGRUND: Obere Hülle J*(λ) (Gestrichelt)
    # Gestrichelt, damit man die blaue Linie darunter durchsieht (kein "Ghosting")
    ax.plot(lams, J_star, color='black', linewidth=3.0, linestyle='--', dashes=(4, 2), alpha=0.85, 
            label=r'Obere Hülle $J^*(\lambda)$')
    
    # D) Switch Points (Rot)
    if switch_lams.size > 0:
        ax.scatter(switch_lams, switch_vals, color='red', s=60, zorder=10, 
                   edgecolor='white', linewidth=1.5, label='Switch Point')
    
    ax.set_title(f"Input Design Landscape (Full Spectrum)\nGraph: n={n_nodes}", fontsize=14)
    ax.set_xlabel(r"Steuerparameter $\lambda$", fontsize=12)
    ax.set_ylabel(r"Max-Cut Wert $J(\lambda)$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {save_path}")

# --- Tabellenausgabe ---

def print_summary_table(metrics_list: List[Dict[str, float]]):
    header = (
        "Algo         |   J_final |    J_best |  J_best-cut |  λ_best-cut |"
        "      |Δλ| | p_opt_final | J_best/J* | J_best-cut/J* |   #evals"
    )
    print("\n" + header)
    print("-" * len(header))
    row_fmt = (
        "{name:<12} | {J_final:9.4f} | {J_best:9.4f} | {J_best_cut:11.4f} |"
        " {lam_best_cut:10.4f} | {delta_lam:9.4f} | {p_opt_final:12.4f} |"
        " {J_best_ratio:10.4f} | {J_best_cut_ratio:14.4f} | {num_evals:8d}"
    )
    for m in metrics_list:
        print(row_fmt.format(**m))

# --- Plot J: Lambda Trajektorien vs. Global Optimum ---

def plot_lambda_trajectories(save_path: str,
                             hist_q_id: Dict[str, np.ndarray],
                             hist_v_id: Dict[str, np.ndarray],
                             hist_q_spsa: Dict[str, np.ndarray],
                             hist_v_spsa: Dict[str, np.ndarray],
                             lam_true: float,
                             lam_bounds: Tuple[float, float]):
    """
    Plot J: Zeigt, wie sich lambda über die Iterationen entwickelt und ob es
    gegen das globale Optimum (schwarz gestrichelt) konvergiert.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ground Truth Linie
    ax.axhline(lam_true, color='black', linestyle='--', linewidth=2.0, label=r'Global Optimal $\lambda^*$')

    # Hilfsfunktion zum Plotten
    def add_trace(hist, color, marker, label):
        if hist is None: return
        lam = hist.get("lam", np.array([]))
        if lam.size == 0: return
        iters = np.arange(1, lam.size + 1)
        ax.plot(iters, lam, marker=marker, color=color, linewidth=1.5, markersize=4, alpha=0.8, label=label)

    # Plotten der Pfade
    add_trace(hist_q_id, "red", "o", "ID(QAOA)")
    add_trace(hist_v_id, "green", "s", "ID(VQE)")
    add_trace(hist_q_spsa, "orange", "x", "SPSA-λ(QAOA)")
    add_trace(hist_v_spsa, "blue", "^", "SPSA-λ(VQE)")

    # Styling
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(r"Steuerparameter $\lambda$")
    ax.set_title(r"Konvergenz der Steuerparameter $\lambda_t$ zum Optimum")
    ax.set_ylim(lam_bounds) # Grenzen setzen, damit man den Suchraum sieht
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {save_path}")