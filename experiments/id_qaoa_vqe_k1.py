# -*- coding: utf-8 -*-
"""
ID(QAOA) & ID(VQE) für Max-Cut (k=1, kein Cluster).

Erzeugt:
  - Alle 20 ursprünglichen Plots (1–20)
  - Bestcuts-Plot
  - Metrik-Plots (J_best/J*, J_best-cut/J*, p_opt, Eval-Counts)
  - Vier Ansätze:
      * ID-QAOA
      * ID-VQE
      * SPSA-QAOA (λ-Baseline)
      * SPSA-VQE (λ-Baseline)

Und gibt im Terminal eine Metrik-Tabelle aus.
"""

import argparse
import numpy as np
import os
import sys

# Projekt-Root (Ordner mit src/ und experiments/) auf sys.path legen
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.id_maxcut_core import (
    to_uint_seed,
    generate_random_graph,
    precompute_z_patterns,
    make_response_params,
    make_w_and_grad_1d,
    ground_truth_lambda_1d,
    compute_cut_vals_for_w_on_the_fly,
    index_to_bitstring,
    build_qaoa_theta_init,
    qaoa_expectation_and_state_p,
    vqe_expectation_and_state,
    choose_vqe_layers,
    compute_upper_envelope_and_topk,
    optimize_lambda_with_ID_1d_qaoa,
    optimize_lambda_with_ID_1d_vqe,
    optimize_lambda_with_SPSA_1d_qaoa,
    optimize_lambda_with_SPSA_1d_vqe,
    envelope_value_and_active_id,   # <<< WICHTIGER FIX
)

from src.id_maxcut_plots import (
    save_active_instances_csv,
    plot_envelope_and_expectations,
    plot_bestcuts,
    plot_landscape_and_trajectories,
    plot_upper_envelope_with_topk,
    plot_switch_points_of_algorithms,
    plot_topk_with_active_instances,
    plot_optimal_maxcut_graph,
    plot_alg_vs_optimal_cut,
    plot_plain_graph,
    plot_gradients,
    plot_lambda_vs_optimal,
    plot_inner_energy_trace,
    plot_qaoa_parameter_trajectory,
    plot_topk_bitstring_distributions,
    plot_p_opt_history,
    plot_lambda_J_scatter,
    plot_spsa_gradient_trace,
    plot_spsa_gradient_hist,
    plot_bestJ_per_start,
    plot_J_history,
    plot_gradient_quality,
    compute_algorithm_metrics,
    print_metrics_table,
    plot_metrics_J_and_popt,
    plot_eval_counts,
)


def parse_args():
    p = argparse.ArgumentParser(description="ID(QAOA) & ID(VQE) – k=1, kein Cluster – Plots + CSV")
    # Problem
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--p_edge", type=float, default=0.5)
    p.add_argument("--lam_bounds", type=float, nargs=2, default=[-5.0, 5.0])
    p.add_argument("--lam0", type=float, default=1.0)
    p.add_argument("--resp_kind", type=str, default="periodic", choices=["linear", "quadratic", "periodic"])
    # Optimierung
    p.add_argument("--outer_iters", type=int, default=10)
    p.add_argument("--eta0", type=float, default=0.1)
    p.add_argument("--inner_spsa_iters", type=int, default=20)
    p.add_argument("--inner_num_starts", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--p_main", type=int, default=10)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--vqe_fair", type=str, default="budget", choices=["budget", "param", "hardware"])
    p.add_argument("--vqe_layers", type=int, default=None)
    # QAOA Init (Heuristik)
    p.add_argument("--qaoa_init_mode", type=str, default="legacy",
                   choices=["legacy", "stack", "ramp"],
                   help="QAOA-Init: legacy=(0.8,0.3) pro Layer; stack=(γ*,β*); ramp=Linear-Ramp aus (γ*,β*)")
    p.add_argument("--qaoa_gamma_star", type=float, default=0.8,
                   help="γ* für QAOA-Heuristik (nur stack/ramp)")
    p.add_argument("--qaoa_beta_star", type=float, default=0.3,
                   help="β* für QAOA-Heuristik (nur stack/ramp)")
    # SPSA Hyperparameter
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
    p.add_argument("--topk", type=int, default=100)
    # Output standardmäßig in figures/raw (von Git ignoriert)
    p.add_argument("--save_prefix", type=str, default="figures/raw/id_qaoa_vqe_k1")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(to_uint_seed(args.seed))

    # Problem
    edges, A = generate_random_graph(args.n, args.p_edge, rng)
    Z = precompute_z_patterns(args.n)
    resp = make_response_params(edges, rng, resp_kind=args.resp_kind, lam_bounds=tuple(args.lam_bounds))
    w_fun_1d, gradJ_1d = make_w_and_grad_1d(resp)

    # Ground Truth (λ_true + optimaler Max-Cut-Bitstring)
    lam_true, J_true = ground_truth_lambda_1d(
        args.n, edges, Z, w_fun_1d, tuple(args.lam_bounds),
        grid_points=args.grid_points_exact
    )
    w_true = w_fun_1d(lam_true)
    cut_vals_true = compute_cut_vals_for_w_on_the_fly(w_true, edges, Z)
    idx_opt = int(np.argmax(cut_vals_true))
    assignment_opt = index_to_bitstring(idx_opt, args.n)

    # Startwerte QAOA (Heuristik aus γ*,β* bzw. legacy)
    p_layers = args.p_main
    theta0 = build_qaoa_theta_init(
        p=p_layers,
        mode=args.qaoa_init_mode,
        gamma_star=args.qaoa_gamma_star,
        beta_star=args.qaoa_beta_star
    )
    lam0 = float(args.lam0)

    # ===== ID(QAOA) – best-so-far + Historie =====
    lam_q_best, theta_q_best, hist_q = optimize_lambda_with_ID_1d_qaoa(
        args.n, edges, Z, w_fun_1d, gradJ_1d,
        theta_init=theta0, p_layers=p_layers,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters, eta0=args.eta0,
        inner_spsa_iters=args.inner_spsa_iters, inner_num_starts=args.inner_num_starts,
        seed=args.seed, shots=args.shots,
        spsa_theta_a=args.spsa_theta_a, spsa_theta_c=args.spsa_theta_c, spsa_theta_A=args.spsa_theta_A,
        spsa_theta_alpha=args.spsa_theta_alpha, spsa_theta_gamma=args.spsa_theta_gamma,
        idx_opt=idx_opt
    )
    J_q_best, psi_q_best, _ = qaoa_expectation_and_state_p(
        args.n, edges, Z, w_fun_1d(lam_q_best),
        theta_q_best[:p_layers], theta_q_best[p_layers:], shots=args.shots, rng=rng
    )

    # ===== ID(VQE) – best-so-far + Historie =====
    if args.vqe_layers is not None:
        L_vqe = args.vqe_layers
    else:
        L_vqe = choose_vqe_layers(args.vqe_fair, p_main=p_layers, n=args.n, m_edges=len(edges), min_layers=1)
    D_vqe = 2 * args.n * L_vqe
    phi0 = np.zeros(D_vqe, dtype=float)
    lam_v_best, phi_v_best, hist_v = optimize_lambda_with_ID_1d_vqe(
        args.n, edges, Z, w_fun_1d, gradJ_1d,
        phi_init=phi0, L_layers=L_vqe,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters, eta0=args.eta0,
        inner_spsa_iters=args.inner_spsa_iters, inner_num_starts=1,
        seed=args.seed + 777, shots=args.shots,
        spsa_vqe_a=args.spsa_vqe_a, spsa_vqe_c=args.spsa_vqe_c, spsa_vqe_A=args.spsa_vqe_A,
        spsa_vqe_alpha=args.spsa_vqe_alpha, spsa_vqe_gamma=args.spsa_vqe_gamma,
        idx_opt=idx_opt
    )
    J_v_best, psi_v_best, _ = vqe_expectation_and_state(
        args.n, edges, Z, w_fun_1d(lam_v_best),
        phi_v_best, L_vqe, shots=args.shots, rng=rng
    )

    # ===== Max-Cut-Lösungen (optimal + Algorithmen) =====
    probs_q = (psi_q_best.conj() * psi_q_best).real
    idx_q_best = int(np.argmax(probs_q))
    assignment_q = index_to_bitstring(idx_q_best, args.n)

    probs_v = (psi_v_best.conj() * psi_v_best).real
    idx_v_best = int(np.argmax(probs_v))
    assignment_v = index_to_bitstring(idx_v_best, args.n)

    # ===== SPSA-Baselines für λ (QAOA & VQE) =====
    lam_q_spsa, theta_q_spsa, hist_q_spsa = optimize_lambda_with_SPSA_1d_qaoa(
        args.n, edges, Z, w_fun_1d,
        theta_init=theta0, p_layers=p_layers,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters,
        inner_spsa_iters=args.inner_spsa_iters, inner_num_starts=args.inner_num_starts,
        seed=args.seed + 2000, shots=args.shots,
        spsa_theta_a=args.spsa_theta_a, spsa_theta_c=args.spsa_theta_c, spsa_theta_A=args.spsa_theta_A,
        spsa_theta_alpha=args.spsa_theta_alpha, spsa_theta_gamma=args.spsa_theta_gamma,
        idx_opt=idx_opt,
    )

    lam_v_spsa, phi_v_spsa, hist_v_spsa = optimize_lambda_with_SPSA_1d_vqe(
        args.n, edges, Z, w_fun_1d,
        phi_init=phi0, L_layers=L_vqe,
        lam0=lam0, lam_bounds=tuple(args.lam_bounds),
        outer_iters=args.outer_iters,
        inner_spsa_iters=args.inner_spsa_iters, inner_num_starts=1,
        seed=args.seed + 2777, shots=args.shots,
        spsa_vqe_a=args.spsa_vqe_a, spsa_vqe_c=args.spsa_vqe_c, spsa_vqe_A=args.spsa_vqe_A,
        spsa_vqe_alpha=args.spsa_vqe_alpha, spsa_vqe_gamma=args.spsa_vqe_gamma,
        idx_opt=idx_opt,
    )

    # ===== Hülle & Top-K Kurven =====
    lam_grid, J_star, s_star, switch_lams, switch_Js, top_ids, top_curves = \
        compute_upper_envelope_and_topk(
            edges, Z, w_fun_1d, tuple(args.lam_bounds),
            grid_points=args.grid_points_exact, topk=args.topk
        )

    # ===== Plot: Hülle + Erwartungswerte (ID) =====
    plot_envelope_and_expectations(
        save_path=f"{args.save_prefix}_envelope_expectations.png",
        lam_grid=lam_grid,
        J_star=J_star,
        hist_q=hist_q,
        hist_v=hist_v
    )

    # ===== Best-Cut-Plot =====
    algo_hists_for_bestcuts = {
        "ID-QAOA": hist_q,
        "SPSA-QAOA": hist_q_spsa,
        "ID-VQE": hist_v,
        "SPSA-VQE": hist_v_spsa,
    }
    plot_bestcuts(
        save_path=f"{args.save_prefix}_bestcuts.png",
        lam_grid=lam_grid,
        J_star=J_star,
        algo_hists=algo_hists_for_bestcuts
    )

    # ===== Plots 1–4 =====
    plot_landscape_and_trajectories(
        save_path=f"{args.save_prefix}_plot1_landscape_trajectories.png",
        lam_grid=lam_grid,
        J_star=J_star,
        lam_true=lam_true,
        lam_q_best=lam_q_best,
        lam_v_best=lam_v_best,
        lam_hist_q=hist_q.get("lam", np.array([])),
        lam_hist_v=hist_v.get("lam", np.array([])),
        w_fun_1d=w_fun_1d,
        edges=edges,
        Z=Z
    )

    plot_upper_envelope_with_topk(
        save_path=f"{args.save_prefix}_plot2_envelope_topk.png",
        lam_grid=lam_grid,
        J_star=J_star,
        switch_lams=switch_lams,
        switch_Js=switch_Js,
        top_ids=top_ids,
        top_curves=top_curves
    )

    plot_switch_points_of_algorithms(
        save_path=f"{args.save_prefix}_plot3_switch_points.png",
        lam_grid=lam_grid,
        J_star=J_star,
        hist_q=hist_q,
        hist_v=hist_v,
        edges=edges,
        Z=Z,
        w_fun_1d=w_fun_1d
    )

    lam_q_path = hist_q.get("lam", np.array([]))
    lam_v_path = hist_v.get("lam", np.array([]))
    active_ids_q = [envelope_value_and_active_id(float(l), edges, Z, w_fun_1d)[1] for l in lam_q_path] if lam_q_path.size else []
    active_ids_v = [envelope_value_and_active_id(float(l), edges, Z, w_fun_1d)[1] for l in lam_v_path] if lam_v_path.size else []

    plot_topk_with_active_instances(
        save_path=f"{args.save_prefix}_plot4_topk_active.png",
        lam_grid=lam_grid,
        top_ids=top_ids,
        top_curves=top_curves,
        active_ids_q=active_ids_q,
        active_ids_v=active_ids_v,
        lams_q=lam_q_path,
        lams_v=lam_v_path,
        edges=edges,
        Z=Z,
        w_fun_1d=w_fun_1d
    )

    # ===== Plots 5–8 =====
    plot_optimal_maxcut_graph(
        save_path=f"{args.save_prefix}_plot5_optimal_graph.png",
        n=args.n,
        edges=edges,
        assignment_opt=assignment_opt
    )

    plot_alg_vs_optimal_cut(
        save_path=f"{args.save_prefix}_plot6_qaoa_graph.png",
        n=args.n,
        edges=edges,
        assignment_opt=assignment_opt,
        assignment_alg=assignment_q,
        alg_name="ID-QAOA",
        alg_color="red"
    )

    plot_alg_vs_optimal_cut(
        save_path=f"{args.save_prefix}_plot7_vqe_graph.png",
        n=args.n,
        edges=edges,
        assignment_opt=assignment_opt,
        assignment_alg=assignment_v,
        alg_name="ID-VQE",
        alg_color="green"
    )

    plot_plain_graph(
        save_path=f"{args.save_prefix}_plot8_plain_graph.png",
        n=args.n,
        edges=edges,
        p_edge=args.p_edge
    )

    # ===== Plots 9–15 =====
    plot_gradients(
        save_path=f"{args.save_prefix}_plot9_gradients.png",
        hist_q=hist_q,
        hist_v=hist_v
    )

    plot_lambda_vs_optimal(
        save_path=f"{args.save_prefix}_plot10_lambda_vs_optimal.png",
        hist_q=hist_q,
        hist_v=hist_v,
        lam_true=lam_true
    )

    plot_inner_energy_trace(
        save_path=f"{args.save_prefix}_plot11_inner_energy.png",
        J_q=hist_q.get("inner_J", np.array([])),
        J_v=hist_v.get("inner_J", np.array([]))
    )

    plot_qaoa_parameter_trajectory(
        save_path=f"{args.save_prefix}_plot12_qaoa_params.png",
        theta_trace=hist_q.get("inner_theta", np.empty((0, 2 * p_layers))),
        p_layers=p_layers
    )

    plot_topk_bitstring_distributions(
        save_path=f"{args.save_prefix}_plot13_bitstrings.png",
        probs_q=probs_q,
        probs_v=probs_v,
        n=args.n,
        idx_opt=idx_opt,
        topk=20
    )

    plot_p_opt_history(
        save_path=f"{args.save_prefix}_plot14_popt_history.png",
        hist_q=hist_q,
        hist_v=hist_v
    )

    plot_lambda_J_scatter(
        save_path=f"{args.save_prefix}_plot15_lambda_J_scatter.png",
        hist_q=hist_q,
        hist_v=hist_v
    )

    # ===== Plots 16–20 =====
    plot_spsa_gradient_trace(
        save_path=f"{args.save_prefix}_plot16_spsa_grad_trace.png",
        gk_q=hist_q.get("inner_spsa_grad", np.array([])),
        gk_v=hist_v.get("inner_spsa_grad", np.array([]))
    )

    plot_spsa_gradient_hist(
        save_path=f"{args.save_prefix}_plot17_spsa_grad_hist.png",
        gk_all_q=hist_q.get("spsa_grad_all", np.array([])),
        gk_all_v=hist_v.get("spsa_grad_all", np.array([]))
    )

    plot_bestJ_per_start(
        save_path=f"{args.save_prefix}_plot18_bestJ_per_start.png",
        bestJ_q=hist_q.get("spsa_bestJ_per_start", np.array([])),
        bestJ_v=hist_v.get("spsa_bestJ_per_start", np.array([]))
    )

    plot_J_history(
        save_path=f"{args.save_prefix}_plot19_J_history.png",
        hist_q=hist_q,
        hist_v=hist_v,
        J_true=J_true
    )

    plot_gradient_quality(
        save_path=f"{args.save_prefix}_plot20_grad_quality.png",
        hist_q=hist_q,
        hist_v=hist_v
    )

    # ===== CSV: aktive Instanzen (ID) =====
    save_active_instances_csv(
        save_path=f"{args.save_prefix}_active_qaoa.csv",
        algo_name="ID-QAOA",
        lams=hist_q["lam"],
        edges=edges, Z=Z, w_fun_1d=w_fun_1d
    )
    save_active_instances_csv(
        save_path=f"{args.save_prefix}_active_vqe.csv",
        algo_name="ID-VQE",
        lams=hist_v["lam"],
        edges=edges, Z=Z, w_fun_1d=w_fun_1d
    )

    # ===== Metriken (4 Algorithmen) =====
    metrics_id_qaoa = compute_algorithm_metrics("ID-QAOA", hist_q, lam_true, J_true)
    metrics_id_vqe = compute_algorithm_metrics("ID-VQE", hist_v, lam_true, J_true)
    metrics_spsa_qaoa = compute_algorithm_metrics("SPSA-QAOA", hist_q_spsa, lam_true, J_true)
    metrics_spsa_vqe = compute_algorithm_metrics("SPSA-VQE", hist_v_spsa, lam_true, J_true)

    all_metrics = [
        metrics_id_qaoa,
        metrics_spsa_qaoa,
        metrics_id_vqe,
        metrics_spsa_vqe,
    ]

    print_metrics_table(all_metrics, lam_true, J_true)
    plot_metrics_J_and_popt(args.save_prefix, all_metrics)
    plot_eval_counts(f"{args.save_prefix}_eval_counts.png", all_metrics)

    # ===== Abschließender Report =====
    f6 = lambda x: f"{x:.6f}"
    print("\n=== Ergebnis (einzelne Instanz) — k=1, kein Cluster, best-so-far ===")
    print(f"n={args.n}, m={len(edges)}, resp={args.resp_kind}, p={args.p_main}, L_vqe={L_vqe}")
    print(f"QAOA-Init: mode={args.qaoa_init_mode}, gamma_star={args.qaoa_gamma_star}, beta_star={args.qaoa_beta_star}")
    print(f"Ground Truth: λ_true={f6(lam_true)}   J*={f6(J_true)}")
    print(f"ID(QAOA-Adam): λ_best={f6(lam_q_best)}  J_best={f6(J_q_best)}")
    print(f"ID(VQE-Adam):  λ_best={f6(lam_v_best)}  J_best={f6(J_v_best)}")
    print(f"SPSA(QAOA-λ): λ_best={f6(lam_q_spsa)}")
    print(f"SPSA(VQE-λ):  λ_best={f6(lam_v_spsa)}")
    print(f"Plots & CSV gespeichert unter Prefix: {args.save_prefix}_*")


if __name__ == "__main__":
    main()
