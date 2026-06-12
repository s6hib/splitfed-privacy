#!/usr/bin/env python3
"""
run_label_leakage.py -- gradient label-leakage sweep across cut layers (step 3).

for each (cut, seed): train a plain split model, capture the cut-layer
gradients, run the norm-leak (Li et al.) + supervised-probe attacks. the
companion sweep to step 2.

no checkpoints needed -- each run trains its own split model from scratch
(the Li et al. two-party setting is plain split learning, not federated).

  python scripts/run_label_leakage.py --seeds 1234 --train_epochs 15
  python scripts/run_label_leakage.py --seeds 1234 2026 42 --train_epochs 15
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)
ATTACK_SCRIPT = os.path.join(Q2_DIR, "src", "attacks", "label_leakage.py")


def run_one(cut, seed, args):
    cmd = [
        sys.executable, ATTACK_SCRIPT,
        "--cut_layer", str(cut),
        "--seed", str(seed),
        "--base_channels", str(args.base_channels),
        "--train_epochs", str(args.train_epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--attack_samples", str(args.attack_samples),
        "--device", args.device,
        "--method", args.method,
        "--data_dir", args.data_dir,
        "--output_dir", args.results_dir,
    ]
    tag = f"label-leak cut={cut} seed={seed}"
    print("\n" + "=" * 70)
    print(f"  {tag}")
    print("=" * 70)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=Q2_DIR, check=False).returncode
    elapsed = (time.time() - t0) / 60
    status = "OK" if rc == 0 else f"FAIL ({rc})"
    print(f"  {tag} -> {status} in {elapsed:.1f} min")
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut_layers", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 2026, 42])
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--train_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--attack_samples", type=int, default=4000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--method", type=str, default="splitnn")
    p.add_argument("--data_dir", type=str, default=os.path.join(Q2_DIR, "data"))
    p.add_argument("--results_dir", type=str, default=os.path.join(Q2_DIR, "results"))
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Q2 dir:      {Q2_DIR}")
    print(f"Cut layers:  {args.cut_layers}")
    print(f"Seeds:       {args.seeds}")
    print(f"Train epochs:{args.train_epochs}  base_channels={args.base_channels}")

    failures = []
    for cut in args.cut_layers:
        for seed in args.seeds:
            if run_one(cut, seed, args) != 0:
                failures.append((cut, seed))

    print("\n" + "=" * 70)
    if failures:
        print(f"  DONE with {len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("  ALL LABEL-LEAKAGE RUNS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
