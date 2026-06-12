#!/usr/bin/env python3
"""
run_defenses.py -- privacy-utility frontier sweep under smashed-data noise (step 4).

for each (cut, sigma, seed): train a noise-defended split model, then run the
inversion attack against it and record (accuracy, PSNR/SSIM). sweeping sigma
traces the frontier. sigma=0 is the undefended reference point.

  # frontier at the standard cut (cut 2), one seed:
  python scripts/run_defenses.py --cut_layers 2 --seeds 1234 \
      --sigmas 0 0.1 0.25 0.5 1.0

no checkpoints needed. outputs land as
results/defense_<pixeldp|none>_cut<C>_sigma<S>_seed<N>.csv (sigma=0 gets written as
defense_none), aggregated by make_step4_figure.py.
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q2_DIR = os.path.dirname(SCRIPT_DIR)
DEFENSE_SCRIPT = os.path.join(Q2_DIR, "src", "defenses", "defense_frontier.py")


def run_one(cut, sigma, seed, args):
    cmd = [
        sys.executable, DEFENSE_SCRIPT,
        "--cut_layer", str(cut),
        "--sigma", str(sigma),
        "--seed", str(seed),
        "--base_channels", str(args.base_channels),
        "--train_epochs", str(args.train_epochs),
        "--decoder_epochs", str(args.decoder_epochs),
        "--batch_size", str(args.batch_size),
        "--width", str(args.width),
        "--device", args.device,
        "--defense", "none" if float(sigma) == 0 else "pixeldp",
        "--data_dir", args.data_dir,
        "--output_dir", args.results_dir,
    ]
    tag = f"defense cut={cut} sigma={sigma} seed={seed}"
    print("\n" + "=" * 70)
    print(f"  {tag}")
    print("=" * 70)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=Q2_DIR, check=False).returncode
    elapsed = (time.time() - t0) / 60
    print(f"  {tag} -> {'OK' if rc == 0 else f'FAIL ({rc})'} in {elapsed:.1f} min")
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut_layers", type=int, nargs="+", default=[2])
    p.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1234])
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--train_epochs", type=int, default=25)
    p.add_argument("--decoder_epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data_dir", type=str, default=os.path.join(Q2_DIR, "data"))
    p.add_argument("--results_dir", type=str, default=os.path.join(Q2_DIR, "results"))
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Cut layers: {args.cut_layers}")
    print(f"Sigmas:     {args.sigmas}")
    print(f"Seeds:      {args.seeds}")

    failures = []
    for cut in args.cut_layers:
        for sigma in args.sigmas:
            for seed in args.seeds:
                if run_one(cut, sigma, seed, args) != 0:
                    failures.append((cut, sigma, seed))

    print("\n" + "=" * 70)
    if failures:
        print(f"  DONE with {len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("  ALL DEFENSE RUNS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
