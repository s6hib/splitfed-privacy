#!/usr/bin/env python3
"""
make_step3_figure.py -- plot the step 3 label-leakage sweep vs cut depth.

reads results/labelleak_<method>_cut<C>_seed<N>.csv, averages the two attack
metrics across seeds, plots them against cut depth:
  left  : unsupervised norm-leak AUC (0.5 line = no leak)
  right : supervised probe accuracy  (chance line = 10%)

    python scripts/make_step3_figure.py
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)


def aggregate(results_dir: str, method: str, cuts: list) -> list:
    by_cut = defaultdict(list)
    for cut in cuts:
        for path in sorted(glob.glob(os.path.join(
                results_dir, f"labelleak_{method}_cut{cut}_seed*.csv"))):
            with open(path, newline="") as f:
                by_cut[cut].append(next(csv.DictReader(f)))
    summary = []
    for cut in sorted(by_cut):
        rows = by_cut[cut]
        if not rows:
            continue
        auc = np.array([float(r["norm_leak_auc"]) for r in rows])
        probe = np.array([float(r["probe_label_acc"]) for r in rows])
        std = lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0
        summary.append({
            "cut": cut, "n": len(rows),
            "auc_mean": float(auc.mean()), "auc_std": std(auc),
            "probe_mean": float(probe.mean()), "probe_std": std(probe),
            "chance": float(rows[0]["chance_acc"]),
        })
    return summary


def plot_figure(summary: list, out_path: str, method: str) -> None:
    cuts = [s["cut"] for s in summary]
    n = summary[0]["n"] if summary else 0
    fig, (ax_auc, ax_probe) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax_auc.errorbar(cuts, [s["auc_mean"] for s in summary],
                    yerr=[s["auc_std"] for s in summary], marker="o", capsize=4,
                    color="#b45309", linewidth=2, markersize=7)
    ax_auc.axhline(0.5, ls="--", color="gray", lw=1, label="no leak (0.5)")
    ax_auc.set_ylabel("norm-leak AUC  ↑ = more leakage")
    ax_auc.set_ylim(0.45, 1.0)
    ax_auc.legend()

    ax_probe.errorbar(cuts, [s["probe_mean"] for s in summary],
                      yerr=[s["probe_std"] for s in summary], marker="o", capsize=4,
                      color="#7c3aed", linewidth=2, markersize=7)
    ax_probe.axhline(summary[0]["chance"], ls="--", color="gray", lw=1,
                     label=f"chance ({summary[0]['chance']:.0f}%)")
    ax_probe.set_ylabel("probe label accuracy (%)  ↑ = more leakage")
    ax_probe.legend()

    for ax in (ax_auc, ax_probe):
        ax.set_xlabel("cut layer (split depth →)")
        ax.set_xticks(cuts)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Step 3 — gradient label leakage vs cut depth "
                 f"({method}, {n} seed{'s' if n != 1 else ''})", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=os.path.join(Q2_DIR, "results"))
    p.add_argument("--figures_dir", default=os.path.join(Q2_DIR, "figures"))
    p.add_argument("--method", default="splitnn")
    p.add_argument("--cuts", type=int, nargs="+", default=[1, 2, 3, 4])
    args = p.parse_args()

    summary = aggregate(args.results_dir, args.method, args.cuts)
    if not summary:
        raise SystemExit(f"No labelleak_{args.method}_cut*_seed*.csv in {args.results_dir}")

    print(f"\nStep 3 label-leakage summary — {args.method}")
    print(f"{'cut':<5}{'n':<4}{'norm-AUC':<18}{'probe acc %':<16}")
    for s in summary:
        print(f"{s['cut']:<5}{s['n']:<4}"
              + f"{s['auc_mean']:.3f} ± {s['auc_std']:.3f}".ljust(18)
              + f"{s['probe_mean']:.2f} ± {s['probe_std']:.2f}".ljust(16))

    out_path = os.path.join(args.figures_dir, f"step3_label_leakage_vs_cut_{args.method}.png")
    plot_figure(summary, out_path, args.method)
    print(f"\nFigure -> {out_path}")


if __name__ == "__main__":
    main()
