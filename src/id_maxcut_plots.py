# -*- coding: utf-8 -*-
"""
Plot- und CSV-Utilities + Metriken für ID(QAOA)/ID(VQE) Max-Cut.

- CSV-Export der aktiven Instanzen
- Plots 1–20 + Diagnoseplots
- Metriktabelle + Metrik-Plots
- Best-Cut-Plot auf der Hülle
- Plot: Hülle + Erwartungswerte
"""

import os
import csv
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .id_maxcut_core import (
    compute_cut_vals_for_w_on_the_fly,
    envelope_value_and_active_id,
    idx_to_bitstring_str,
)

# ================== CSV & Verzeichnis-Helfer ==================
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)

def save_active_instances_csv(save_path: str, algo_name: str, lams: np.ndarray,
                              edges, Z, w_fun_1d):
    _ensure_dir(save_path)
    prev = None
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "iter", "lambda", "active_id", "J_star", "switched"])
        for i, lam in enumerate(lams, start=1):
            J_star, s_star = envelope_value_and_active_id(float(lam), edges, Z, w_fun_1d)
            switched = int(prev is not None and s_star != prev)
            w.writerow([algo_name, i, f"{lam:.12f}", s_star, f"{J_star:.12f}", switched])
            prev = s_star
    print(f"[saved] {save_path}")

# ================== Plotter Basis (1–4) ==================
def plot_landscape_and_trajectories(save_path: str,
                                    lam_grid, J_star,
                                    lam_true, lam_q_best, lam_v_best,
                                    lam_hist_q, lam_hist_v,
                                    w_fun_1d, edges, Z):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(lam_grid, J_star, lw=3, label=r"Obere Hülle $J^*(\lambda)$")
    if lam_hist_q is not None and len(lam_hist_q) > 0:
        J_path_q = [envelope_value_and_active_id(float(l), edges, Z, w_fun_1d)[0] for l in lam_hist_q]
        ax.plot(lam_hist_q, J_path_q, marker='o', lw=1.5, color='red', label="QAOA Trajektorie")
    if lam_hist_v is not None and len(lam_hist_v) > 0:
        J_path_v = [envelope_value_and_active_id(float(l), edges, Z, w_fun_1d)[0] for l in lam_hist_v]
        ax.plot(lam_hist_v, J_path_v, marker='s', lw=1.5, color='green', label="VQE Trajektorie")
    ax.axvline(lam_true, ls='--', lw=2, color='black', label=r"$\lambda_{\rm true}$")
    ax.axvline(lam_q_best, ls=':', lw=2, color='red', label=r"$\lambda_{\rm QAOA}$")
    ax.axvline(lam_v_best, ls='-.', lw=2, color='green', label=r"$\lambda_{\rm VQE}$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J(\lambda)$")
    ax.set_title("Plot 1: Lösungslandschaft & Optimierungspfade")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_upper_envelope_with_topk(save_path: str, lam_grid, J_star, switch_lams, switch_Js,
                                  top_ids, top_curves):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for r in range(len(top_ids)):
        ax.plot(lam_grid, top_curves[r], color='gray', lw=1, alpha=0.5, zorder=1)
    ax.plot(lam_grid, J_star, lw=3, zorder=3, label=r"Obere Hülle $J^*(\lambda)$")
    if switch_lams is not None and len(switch_lams) > 0:
        ax.scatter(switch_lams, switch_Js, s=30, zorder=4, color='red', label="Wechsel aktiver Cut")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J_s(\lambda)$")
    ax.set_title(f"Plot 2: Obere Hülle $J^*(\\lambda)$ (+ Top-{len(top_ids)} Cuts)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_switch_points_of_algorithms(save_path: str, lam_grid, J_star,
                                     hist_q, hist_v, edges, Z, w_fun_1d):
    _ensure_dir(save_path)

    def active_ids_along(lams):
        return [envelope_value_and_active_id(float(l), edges, Z, w_fun_1d)[1] for l in lams]

    lam_q = hist_q["lam"] if hist_q and "lam" in hist_q else np.array([])
    lam_v = hist_v["lam"] if hist_v and "lam" in hist_v else np.array([])
    ids_q = active_ids_along(lam_q) if lam_q.size else []
    ids_v = active_ids_along(lam_v) if lam_v.size else []

    def switches(lams, ids):
        sw_idx = [i for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
        sw_lam = [float(lams[i]) for i in sw_idx]
        sw_J = [envelope_value_and_active_id(float(lams[i]), edges, Z, w_fun_1d)[0] for i in sw_idx]
        return np.array(sw_lam, dtype=float), np.array(sw_J, dtype=float)

    sw_lam_q, sw_J_q = switches(lam_q, ids_q) if len(ids_q) > 0 else (np.array([]), np.array([]))
    sw_lam_v, sw_J_v = switches(lam_v, ids_v) if len(ids_v) > 0 else (np.array([]), np.array([]))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(lam_grid, J_star, lw=3, label=r"Obere Hülle $J^*(\lambda)$", zorder=1)
    if sw_lam_q.size:
        ax.scatter(sw_lam_q, sw_J_q, s=45, marker='o', color='red',
                   label="QAOA: Wechsel aktive Instanz", zorder=3)
    if sw_lam_v.size:
        ax.scatter(sw_lam_v, sw_J_v, s=45, marker='s', color='green',
                   label="VQE: Wechsel aktive Instanz", zorder=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J(\lambda)$")
    ax.set_title("Plot 3: Wechselpunkte aktiver Instanzen entlang der Algorithmen")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def gather_active_points_on_curves(lams: np.ndarray,
                                   edges, Z, w_fun_1d,
                                   target_ids: List[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[int, Tuple[List[float], List[float]]] = {cid: ([], []) for cid in target_ids}
    for lam in lams:
        v = compute_cut_vals_for_w_on_the_fly(w_fun_1d(float(lam)), edges, Z)
        s_star = int(np.argmax(v))
        if s_star in out:
            out[s_star][0].append(float(lam))
            out[s_star][1].append(float(v[s_star]))
    return {cid: (np.array(xs, float), np.array(ys, float)) for cid, (xs, ys) in out.items()}

def plot_topk_with_active_instances(save_path: str, lam_grid, top_ids, top_curves,
                                    active_ids_q: List[int], active_ids_v: List[int],
                                    lams_q: np.ndarray, lams_v: np.ndarray,
                                    edges, Z, w_fun_1d):
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for r in range(len(top_ids)):
        ax.plot(lam_grid, top_curves[r], color='gray', lw=1, alpha=0.45, zorder=1)

    active_union = sorted(set(active_ids_q) | set(active_ids_v))
    for cid in active_union:
        if cid in top_ids:
            r = top_ids.index(cid)
            ax.plot(lam_grid, top_curves[r], lw=2.8, zorder=5, label=f"Instanz {cid} (aktiv)")

    q_pts = gather_active_points_on_curves(lams_q, edges, Z, w_fun_1d, active_union)
    v_pts = gather_active_points_on_curves(lams_v, edges, Z, w_fun_1d, active_union)

    for cid in active_union:
        if cid in q_pts and q_pts[cid][0].size:
            ax.scatter(q_pts[cid][0], q_pts[cid][1], s=36, marker='o',
                       edgecolors='white', linewidths=0.7, color='red', zorder=6, label=None)
        if cid in v_pts and v_pts[cid][0].size:
            ax.scatter(v_pts[cid][0], v_pts[cid][1], s=36, marker='s',
                       edgecolors='white', linewidths=0.7, color='green', zorder=6, label=None)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J_s(\lambda)$")
    ax.set_title(f"Plot 4: Top-{len(top_ids)} Instanzen (grau) + aktive Instanzen mit Pfad-Markern")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# ================== Additional Plots (5–10) ==================
def compute_circular_layout(n: int, radius: float = 1.0) -> Dict[int, Tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return {i: (radius * math.cos(a), radius * math.sin(a)) for i, a in enumerate(angles)}

def plot_optimal_maxcut_graph(save_path: str, n: int, edges, assignment_opt: np.ndarray):
    """Plot 5: Optimaler Max-Cut als Graph."""
    _ensure_dir(save_path)
    pos = compute_circular_layout(n)
    fig, ax = plt.subplots(figsize=(6, 6))

    for (i, j) in edges:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        if assignment_opt[i] != assignment_opt[j]:
            color = 'red'
            lw = 2.5
            zorder = 2
        else:
            color = 'lightgray'
            lw = 1.0
            zorder = 1
        ax.plot(x, y, color=color, lw=lw, zorder=zorder)

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    node_colors = ['tab:blue' if assignment_opt[i] == 0 else 'tab:orange' for i in range(n)]
    ax.scatter(xs, ys, s=80, c=node_colors, edgecolors='black', zorder=3)
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), fontsize=9,
                ha='center', va='center', color='black', zorder=4)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Plot 5: Optimaler Max-Cut (Graph-Darstellung)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_alg_vs_optimal_cut(save_path: str, n: int, edges,
                             assignment_opt: np.ndarray, assignment_alg: np.ndarray,
                             alg_name: str, alg_color: str):
    """Plot 6/7: Max-Cut-Lösung des jeweiligen Algorithmus (ohne Ground-Truth-Overlay)."""
    _ensure_dir(save_path)
    pos = compute_circular_layout(n)
    fig, ax = plt.subplots(figsize=(6, 6))

    for (i, j) in edges:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, color='lightgray', lw=0.8, zorder=1)

    for (i, j) in edges:
        if assignment_alg[i] != assignment_alg[j]:
            x = [pos[i][0], pos[j][0]]
            y = [pos[i][1], pos[j][1]]
            ax.plot(x, y, color=alg_color, lw=2.5, zorder=3)

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    node_colors = ['white' if assignment_alg[i] == 0 else alg_color for i in range(n)]
    ax.scatter(xs, ys, s=80, c=node_colors, edgecolors='black', zorder=4)
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), fontsize=9,
                ha='center', va='center', color='black', zorder=5)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{alg_name}: Max-Cut-Lösung")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_plain_graph(save_path: str, n: int, edges, p_edge: float):
    """Plot 8: Basisgraph mit n und p_edge."""
    _ensure_dir(save_path)
    pos = compute_circular_layout(n)
    fig, ax = plt.subplots(figsize=(6, 6))

    for (i, j) in edges:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, color='gray', lw=1.0, zorder=1)

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=80, c='tab:blue', edgecolors='black', zorder=2)
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), fontsize=9,
                ha='center', va='center', color='black', zorder=3)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Plot 8: Zufallsgraph (n={n}, p_edge={p_edge})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_gradients(save_path: str, hist_q: Dict[str, np.ndarray], hist_v: Dict[str, np.ndarray]):
    """Plot 9: Gradientenverlauf von QAOA und VQE (∂J/∂λ der ID-Methoden)."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if hist_q and "grad" in hist_q and hist_q["grad"].size:
        it_q = np.arange(1, hist_q["grad"].size + 1)
        ax.plot(it_q, hist_q["grad"], marker='o', lw=1.5, color='red', label="QAOA Gradient")
    if hist_v and "grad" in hist_v and hist_v["grad"].size:
        it_v = np.arange(1, hist_v["grad"].size + 1)
        ax.plot(it_v, hist_v["grad"], marker='s', lw=1.5, color='green', label="VQE Gradient")
    ax.axhline(0.0, color='black', lw=1.0, ls='--', label=r"$\partial J/\partial \lambda = 0$")
    ax.set_xlabel("Outer-Iteration t")
    ax.set_ylabel(r"Gradient $\partial J / \partial \lambda$")
    ax.set_title("Plot 9: Gradienten von ID(QAOA) und ID(VQE)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_lambda_vs_optimal(save_path: str, hist_q: Dict[str, np.ndarray],
                           hist_v: Dict[str, np.ndarray], lam_true: float):
    """Plot 10: Entwicklung von λ pro Algorithmus + horizontale Linie bei λ_true."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if hist_q and "lam" in hist_q and hist_q["lam"].size:
        it_q = np.arange(1, hist_q["lam"].size + 1)
        ax.plot(it_q, hist_q["lam"], marker='o', lw=1.5, color='red', label=r"QAOA $\lambda$")
    if hist_v and "lam" in hist_v and hist_v["lam"].size:
        it_v = np.arange(1, hist_v["lam"].size + 1)
        ax.plot(it_v, hist_v["lam"], marker='s', lw=1.5, color='green', label=r"VQE $\lambda$")
    ax.axhline(lam_true, color='black', lw=1.5, ls='--', label=r"$\lambda_{\rm true}$")
    ax.set_xlabel("Outer-Iteration t")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Plot 10: Entwicklung von $\lambda$ gegenüber $\lambda_{\rm true}$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# ================== Zusätzliche Diagnose-Plots (11–15) ==================
def plot_inner_energy_trace(save_path: str,
                            J_q: np.ndarray,
                            J_v: np.ndarray):
    """Plot 11: Innerer Energie-Verlauf J_k im SPSA (letzte Outer-Iteration)."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
    if J_q is not None and J_q.size:
        k_q = np.arange(1, J_q.size + 1)
        ax.plot(k_q, J_q, marker='o', lw=1.5, color='red',
                label="QAOA: innerer Cut-Wert J_k")
        plotted = True
    if J_v is not None and J_v.size:
        k_v = np.arange(1, J_v.size + 1)
        ax.plot(k_v, J_v, marker='s', lw=1.5, color='green',
                label="VQE: innerer Cut-Wert J_k")
        plotted = True
    if not plotted:
        plt.close(fig)
        print("[warn] Plot 11: keine inneren Energie-Traces vorhanden, Plot wird übersprungen.")
        return
    ax.set_xlabel("Innerer SPSA-Schritt k")
    ax.set_ylabel(r"Cut-Wert $J$")
    ax.set_title("Plot 11: Innerer Energie-Verlauf (letzte Outer-Iteration)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_qaoa_parameter_trajectory(save_path: str,
                                   theta_trace: np.ndarray,
                                   p_layers: int):
    """Plot 12: QAOA-Parametertrajektorien θ_k (γ und β) im Innerloop."""
    _ensure_dir(save_path)
    if theta_trace is None or theta_trace.size == 0:
        print("[warn] Plot 12: keine QAOA-Parametertrajektorie vorhanden, Plot wird übersprungen.")
        return
    steps, D = theta_trace.shape
    p = p_layers
    if D < 2 * p:
        print(f"[warn] Plot 12: inkonsistente Dimensionen (D={D}, 2p={2*p}), Plot wird übersprungen.")
        return
    gammas = theta_trace[:, :p]
    betas = theta_trace[:, p:2 * p]
    k = np.arange(1, steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for j in range(p):
        ax1.plot(k, gammas[:, j], lw=1.2, marker='o', markersize=3,
                 label=rf"$\gamma_{j+1}$")
        ax2.plot(k, betas[:, j], lw=1.2, marker='o', markersize=3,
                 label=rf"$\beta_{j+1}$")

    ax1.set_xlabel("Innerer SPSA-Schritt k")
    ax1.set_ylabel("Parameterwert")
    ax1.set_title("QAOA: γ-Parameter (letzte Outer-Iteration)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.set_xlabel("Innerer SPSA-Schritt k")
    ax2.set_ylabel("Parameterwert")
    ax2.set_title("QAOA: β-Parameter (letzte Outer-Iteration)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Plot 12: QAOA-Parametertrajektorien im Innerloop", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_topk_bitstring_distributions(save_path: str,
                                      probs_q: np.ndarray,
                                      probs_v: np.ndarray,
                                      n: int,
                                      idx_opt: Optional[int] = None,
                                      topk: int = 20):
    """Plot 13: Top-k Bitstring-Wahrscheinlichkeiten für QAOA und VQE."""
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    def make_plot(ax, probs, color, title, idx_opt_local):
        if probs is None or probs.size == 0:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center")
            ax.axis("off")
            return
        K = probs.size
        k = min(topk, K)
        idxs = np.argsort(probs)[-k:][::-1]
        vals = probs[idxs]
        labels = [idx_to_bitstring_str(int(i), n) for i in idxs]
        xs = np.arange(k)
        edgecolors = ["none"] * k
        if idx_opt_local is not None:
            for t_idx, idx in enumerate(idxs):
                if int(idx) == int(idx_opt_local):
                    edgecolors[t_idx] = "black"
        ax.bar(xs, vals, color=color, edgecolor=edgecolors)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Wahrscheinlichkeit")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    make_plot(ax1, probs_q, "red", "QAOA: Top-k Bitstrings", idx_opt)
    make_plot(ax2, probs_v, "green", "VQE: Top-k Bitstrings", idx_opt)

    fig.suptitle(f"Plot 13: Top-{topk} Bitstring-Wahrscheinlichkeiten", y=0.98)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_p_opt_history(save_path: str,
                       hist_q: Dict[str, np.ndarray],
                       hist_v: Dict[str, np.ndarray]):
    """Plot 14: Zeitliche Entwicklung von P(opt) für QAOA & VQE."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
    p_q = hist_q.get("p_opt", None)
    if p_q is not None and p_q.size:
        t_q = np.arange(1, p_q.size + 1)
        ax.plot(t_q, p_q, marker='o', lw=1.5, color='red', label="QAOA: P(opt)")
        plotted = True
    p_v = hist_v.get("p_opt", None)
    if p_v is not None and p_v.size:
        t_v = np.arange(1, p_v.size + 1)
        ax.plot(t_v, p_v, marker='s', lw=1.5, color='green', label="VQE: P(opt)")
        plotted = True
    if not plotted:
        plt.close(fig)
        print("[warn] Plot 14: keine p_opt-Daten vorhanden, Plot wird übersprungen.")
        return
    ax.set_xlabel("Outer-Iteration t")
    ax.set_ylabel("Wahrscheinlichkeit des optimalen Max-Cut-Bitstrings")
    ax.set_title("Plot 14: Entwicklung von P(opt) unter ID(QAOA)/ID(VQE)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_lambda_J_scatter(save_path: str,
                          hist_q: Dict[str, np.ndarray],
                          hist_v: Dict[str, np.ndarray]):
    """Plot 15: Joint-Plot über (λ, J) aus allen inneren Evaluierungen."""
    _ensure_dir(save_path)
    lam_q = hist_q.get("lambda_J_lam", None)
    J_q = hist_q.get("lambda_J_J", None)
    lam_v = hist_v.get("lambda_J_lam", None)
    J_v = hist_v.get("lambda_J_J", None)
    if (lam_q is None or lam_q.size == 0) and (lam_v is None or lam_v.size == 0):
        print("[warn] Plot 15: keine (λ,J)-Daten vorhanden, Plot wird übersprungen.")
        return
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if lam_q is not None and lam_q.size:
        ax.scatter(lam_q, J_q, s=18, marker='o', alpha=0.4, color='red',
                   label="QAOA: innere Evaluierungen")
    if lam_v is not None and lam_v.size:
        ax.scatter(lam_v, J_v, s=18, marker='s', alpha=0.4, color='green',
                   label="VQE: innere Evaluierungen")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J$")
    ax.set_title("Plot 15: Joint-Plot über (λ, J) aus allen inneren Evaluierungen")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# ================== Neue Plots (16–20): SPSA-Gradienten, num_starts, J/Gradient-Qualität ==================
def plot_spsa_gradient_trace(save_path: str,
                             gk_q: np.ndarray,
                             gk_v: np.ndarray):
    """Plot 16: Verlauf von |g_k| im Innerloop (letzte Outer-Iteration)."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False

    if gk_q is not None and gk_q.size:
        k_q = np.arange(1, gk_q.size + 1)
        ax.plot(k_q, gk_q, marker='o', lw=1.5, color='red', label="QAOA |g_k|")
        plotted = True
    if gk_v is not None and gk_v.size:
        k_v = np.arange(1, gk_v.size + 1)
        ax.plot(k_v, gk_v, marker='s', lw=1.5, color='green', label="VQE |g_k|")
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] Plot 16: keine SPSA-Gradient-Traces vorhanden, Plot wird übersprungen.")
        return

    ax.set_xlabel("Innerer SPSA-Schritt k")
    ax.set_ylabel(r"$|g_k| = \left|\frac{E_+ - E_-}{2 c_k}\right|$")
    ax.set_title("Plot 16: SPSA-Gradientenbetrag im Innerloop (letzte Outer-Iteration)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_spsa_gradient_hist(save_path: str,
                            gk_all_q: np.ndarray,
                            gk_all_v: np.ndarray):
    """Plot 17: Histogramm der SPSA-Gradienten |g_k| über alle Inner-Loops und Outer-Iterationen."""
    _ensure_dir(save_path)
    if ((gk_all_q is None or not gk_all_q.size) and
        (gk_all_v is None or not gk_all_v.size)):
        print("[warn] Plot 17: keine SPSA-Gradient-Daten vorhanden, Plot wird übersprungen.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = 40

    if gk_all_q is not None and gk_all_q.size:
        ax.hist(gk_all_q, bins=bins, alpha=0.5, density=True,
                color='red', label="QAOA |g_k|")
    if gk_all_v is not None and gk_all_v.size:
        ax.hist(gk_all_v, bins=bins, alpha=0.5, density=True,
                color='green', label="VQE |g_k|")

    ax.set_xlabel(r"$|g_k| = \left|\frac{E_+ - E_-}{2 c_k}\right|$")
    ax.set_ylabel("Dichte")
    ax.set_title("Plot 17: Verteilung der SPSA-Gradienten |g_k|")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_bestJ_per_start(save_path: str,
                         bestJ_q: np.ndarray,
                         bestJ_v: np.ndarray):
    """Plot 18: Verteilung der besten Cut-Werte J pro SPSA-Start (über alle Outer-Iterationen)."""
    _ensure_dir(save_path)
    have_q = bestJ_q is not None and bestJ_q.size
    have_v = bestJ_v is not None and bestJ_v.size
    if (not have_q) and (not have_v):
        print("[warn] Plot 18: keine bestJ_per_start-Daten vorhanden, Plot wird übersprungen.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    data = []
    labels = []
    if have_q:
        data.append(bestJ_q)
        labels.append("QAOA")
    if have_v:
        data.append(bestJ_v)
        labels.append("VQE")

    ax.boxplot(data, labels=labels, vert=True)
    ax.set_ylabel(r"Bester Cut-Wert $J$ pro Start")
    ax.set_title("Plot 18: Verteilung der besten J pro SPSA-Start")
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_J_history(save_path: str,
                   hist_q: Dict[str, np.ndarray],
                   hist_v: Dict[str, np.ndarray],
                   J_true: float):
    """Plot 19: Entwicklung der ID-Erwartungswerte J_t für QAOA/VQE gegenüber dem globalen Optimum."""
    _ensure_dir(save_path)
    J_q = hist_q.get("J", None)
    J_v = hist_v.get("J", None)
    have_q = J_q is not None and J_q.size
    have_v = J_v is not None and J_v.size
    if (not have_q) and (not have_v):
        print("[warn] Plot 19: keine J-Verläufe vorhanden, Plot wird übersprungen.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    if have_q:
        t_q = np.arange(1, J_q.size + 1)
        ax.plot(t_q, J_q, marker='o', lw=1.5, color='red', label="QAOA J_t")
    if have_v:
        t_v = np.arange(1, J_v.size + 1)
        ax.plot(t_v, J_v, marker='s', lw=1.5, color='green', label="VQE J_t")

    ax.axhline(J_true, color='black', lw=1.5, ls='--', label=r"$J^* = J(\lambda_{\rm true})$")
    ax.set_xlabel("Outer-Iteration t")
    ax.set_ylabel(r"Cut-Wert $J$")
    ax.set_title(r"Plot 19: Entwicklung der ID-Erwartungswerte $J_t$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_gradient_quality(save_path: str,
                          hist_q: Dict[str, np.ndarray],
                          hist_v: Dict[str, np.ndarray]):
    """Plot 20: Vergleich ∂J/∂λ (Algorithmus) vs. dJ*/dλ (Hülle) + Fehlerverlauf."""
    _ensure_dir(save_path)
    g_q = hist_q.get("grad", None)
    gt_q = hist_q.get("grad_true", None)
    g_v = hist_v.get("grad", None)
    gt_v = hist_v.get("grad_true", None)

    have_q = (g_q is not None and gt_q is not None and g_q.size and gt_q.size)
    have_v = (g_v is not None and gt_v is not None and g_v.size and gt_v.size)
    if (not have_q) and (not have_v):
        print("[warn] Plot 20: keine Gradienten-Daten vorhanden, Plot wird übersprungen.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)

    if have_q:
        t_q = np.arange(1, g_q.size + 1)
        ax1.plot(t_q, gt_q, marker='o', lw=1.5, color='tab:red', label="QAOA dJ*/dλ (Hülle)")
        ax1.plot(t_q, g_q, marker='x', lw=1.0, color='red', label="QAOA ∂J/∂λ (ID)")
        ax2.plot(t_q, np.abs(g_q - gt_q), marker='o', lw=1.5, color='red', label="QAOA |Δg|")

    if have_v:
        t_v = np.arange(1, g_v.size + 1)
        ax1.plot(t_v, gt_v, marker='s', lw=1.5, color='tab:green', label="VQE dJ*/dλ (Hülle)")
        ax1.plot(t_v, g_v, marker='x', lw=1.0, color='green', label="VQE ∂J/∂λ (ID)")
        ax2.plot(t_v, np.abs(g_v - gt_v), marker='s', lw=1.5, color='green', label="VQE |Δg|")

    ax1.set_xlabel("Outer-Iteration t")
    ax1.set_ylabel(r"Gradient")
    ax1.set_title(r"∂J/∂λ (ID) vs. dJ*/dλ (Hülle)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2.set_xlabel("Outer-Iteration t")
    ax2.set_ylabel(r"$|Δg|$")
    ax2.set_title(r"Fehler $|∂J/∂λ - dJ^*/dλ|$")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Plot 20: Qualität des λ-Gradienten von ID(QAOA)/ID(VQE)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")

# ================== METRIKEN & REPORTING ==================
def compute_algorithm_metrics(name: str,
                              hist: Dict[str, np.ndarray],
                              lam_true: float,
                              J_true: float) -> Dict[str, float]:
    """
    Extrahiert:
      - J_final, J_best, J_best_cut
      - λ_final, λ_best_exp (max J), λ_best_cut (max J_env)
      - Bias |λ_best_cut - λ_true|
      - p_opt_final
      - J_best / J*, J_best_cut / J*
      - evals (aus hist["evals"], falls vorhanden)
    """
    J_hist = hist.get("J", np.array([]))
    J_env_hist = hist.get("J_env", np.array([]))
    lam_pre = hist.get("lam_pre", hist.get("lam", np.array([])))
    p_opt_hist = hist.get("p_opt", np.array([]))
    evals = int(hist.get("evals", 0)) if "evals" in hist else 0

    metrics: Dict[str, float] = {"name": name}

    if J_hist.size:
        metrics["J_final"] = float(J_hist[-1])
        idx_best_J = int(np.argmax(J_hist))
        metrics["J_best"] = float(J_hist[idx_best_J])
        metrics["lam_best_exp"] = float(lam_pre[idx_best_J]) if lam_pre.size else float("nan")
    else:
        metrics["J_final"] = float("nan")
        metrics["J_best"] = float("nan")
        metrics["lam_best_exp"] = float("nan")

    metrics["lam_final"] = float(lam_pre[-1]) if lam_pre.size else float("nan")

    if J_env_hist.size and lam_pre.size:
        idx_best_cut = int(np.argmax(J_env_hist))
        metrics["J_best_cut"] = float(J_env_hist[idx_best_cut])
        metrics["lam_best_cut"] = float(lam_pre[idx_best_cut])
        metrics["lambda_bias"] = abs(metrics["lam_best_cut"] - lam_true)
    else:
        metrics["J_best_cut"] = float("nan")
        metrics["lam_best_cut"] = float("nan")
        metrics["lambda_bias"] = float("nan")

    if p_opt_hist is not None and getattr(p_opt_hist, "size", 0):
        metrics["p_opt_final"] = float(p_opt_hist[-1])
    else:
        metrics["p_opt_final"] = float("nan")

    if J_true != 0.0 and np.isfinite(J_true):
        metrics["ratio_J_best"] = metrics["J_best"] / J_true
        metrics["ratio_J_best_cut"] = metrics["J_best_cut"] / J_true
    else:
        metrics["ratio_J_best"] = float("nan")
        metrics["ratio_J_best_cut"] = float("nan")

    metrics["evals"] = evals
    return metrics

def print_metrics_table(metrics_list: List[Dict[str, float]],
                        lam_true: float,
                        J_true: float):
    print("\n=== Zusammenfassung Metriken (pro Algorithmus) ===")
    print(f"Ground Truth: λ_true = {lam_true:.6f},   J* = {J_true:.6f}\n")

    header = (
        f"{'Algo':12s} | {'J_final':>9s} | {'J_best':>9s} | {'J_best-cut':>11s} | "
        f"{'λ_best-cut':>11s} | {'|Δλ|':>9s} | {'p_opt_final':>11s} | "
        f"{'J_best/J*':>9s} | {'J_best-cut/J*':>13s} | {'#evals':>8s}"
    )
    print(header)
    print("-" * len(header))

    for m in metrics_list:
        print(
            f"{m['name']:12s} | "
            f"{m['J_final']:9.4f} | "
            f"{m['J_best']:9.4f} | "
            f"{m['J_best_cut']:11.4f} | "
            f"{m['lam_best_cut']:11.4f} | "
            f"{m['lambda_bias']:9.4f} | "
            f"{m['p_opt_final']:11.4f} | "
            f"{m['ratio_J_best']:9.4f} | "
            f"{m['ratio_J_best_cut']:13.4f} | "
            f"{int(m['evals']):8d}"
        )

def plot_metrics_J_and_popt(save_prefix: str,
                            metrics_list: List[Dict[str, float]]):
    _ensure_dir(os.path.dirname(save_prefix) if "." in save_prefix else save_prefix)
    names = [m["name"] for m in metrics_list]
    x = np.arange(len(names))

    ratios_J_best = np.array([m["ratio_J_best"] for m in metrics_list], dtype=float)
    ratios_J_best_cut = np.array([m["ratio_J_best_cut"] for m in metrics_list], dtype=float)
    p_opts = np.array([m["p_opt_final"] for m in metrics_list], dtype=float)

    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    width = 0.38
    ax1.bar(x - width / 2, ratios_J_best, width, label=r"$J_{\rm best} / J^*$")
    ax1.bar(x + width / 2, ratios_J_best_cut, width, label=r"$J_{\rm best-cut} / J^*$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Normierter Cut-Wert")
    ax1.set_title("Erwartungswert vs. Best-Cut relativ zu Ground Truth")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best")
    fig1.tight_layout()
    path1 = f"{save_prefix}_metrics_J.png"
    fig1.savefig(path1, dpi=160)
    plt.close(fig1)
    print(f"[saved] {path1}")

    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    ax2.bar(x, p_opts)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel(r"$p_{\rm opt}$ (letzte Outer-Iteration)")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Wahrscheinlichkeit des optimalen Max-Cut-Bitstrings")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    path2 = f"{save_prefix}_metrics_popt.png"
    fig2.savefig(path2, dpi=160)
    plt.close(fig2)
    print(f"[saved] {path2}")

def plot_eval_counts(save_path: str,
                     metrics_list: List[Dict[str, float]]):
    _ensure_dir(save_path)
    names = [m["name"] for m in metrics_list]
    evals = [int(m.get("evals", 0)) for m in metrics_list]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, evals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Anzahl der Energie-/Erwartungswert-Auswertungen")
    ax.set_title("Vergleich der Evaluationsanzahl (ID vs. SPSA-Baselines)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# ================== Best-Cut-Plot auf der Hülle ==================
def plot_bestcuts(save_path: str,
                  lam_grid: np.ndarray,
                  J_star: np.ndarray,
                  algo_hists: Dict[str, Dict[str, np.ndarray]]):
    """
    Best-Cut-Plot:
      - schwarze Linie: Obere Hülle J*(λ)
      - pro Algorithmus: Punkte (λ_t, J_env_t)
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    ax.plot(lam_grid, J_star, color="black", lw=2, label=r"Obere Hülle $J^*(\lambda)$")

    markers = ["o", "s", "D", "^"]
    for k, (name, hist) in enumerate(algo_hists.items()):
        lam = hist.get("lam_pre", hist.get("lam", np.array([])))
        J_env = hist.get("J_env", np.array([]))
        if lam is None or J_env is None or not getattr(lam, "size", 0) or not getattr(J_env, "size", 0):
            continue
        m = markers[k % len(markers)]
        ax.scatter(lam, J_env,
                   s=35,
                   marker=m,
                   alpha=0.85,
                   label=f"{name}: Best-Cut (Hülle)")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J^*(\lambda)$")
    ax.set_title("Best-so-far Cuts entlang der Hülle (pro Algorithmus)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")

# ================== Plot: Hülle + Erwartungswerte ==================
def plot_envelope_and_expectations(save_path: str,
                                   lam_grid: np.ndarray,
                                   J_star: np.ndarray,
                                   hist_q: Dict[str, np.ndarray],
                                   hist_v: Dict[str, np.ndarray]):
    """
    Plot:
      - Obere Hülle J*(λ) als graue Fläche + schwarze Kontur
      - Erwartungswerte J_t pro Outer-Iteration:
          * rot:  ID(QAOA)
          * grün: ID(VQE)
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    ax.fill_between(lam_grid,
                    np.zeros_like(J_star),
                    J_star,
                    color="lightgray",
                    alpha=0.4,
                    label=r"Obere Hülle $J^*(\lambda)$")
    ax.plot(lam_grid, J_star, color="black", lw=2)

    lam_q = hist_q.get("lam_pre", np.array([]))
    J_q = hist_q.get("J", np.array([]))
    if lam_q is not None and J_q is not None and len(lam_q) and len(J_q):
        ax.plot(lam_q, J_q,
                "-o",
                color="red",
                lw=1.8,
                markersize=4,
                label="ID(QAOA): Erwartungswerte")

    lam_v = hist_v.get("lam_pre", np.array([]))
    J_v = hist_v.get("J", np.array([]))
    if lam_v is not None and J_v is not None and len(lam_v) and len(J_v):
        ax.plot(lam_v, J_v,
                "-s",
                color="green",
                lw=1.8,
                markersize=4,
                label="ID(VQE): Erwartungswerte")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Cut-Wert $J$")
    ax.set_title(r"Obere Hülle und Erwartungswerte $J_t$ während der Optimierung")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {save_path}")
