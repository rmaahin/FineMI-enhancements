# Local 3-pair test of Slim Conformer B, in one command. Run from Slim_conformer_fine/:
#
#   python -m scripts.local_test --mode smoke        # ~minutes: proves everything runs
#   python -m scripts.local_test --mode full         # the real 3-pair test vs Conformer A / B
#
# Steps (stops at the first failure):
#   1. preflight     packages, device, all 18 data files, exactly 25,752 parameters at
#                    every window, determinism, and (smoke) a timing projection for the
#                    full-mode run of the same pairs on this machine
#   2. training      scripts.run_selected_pairs on the selected pairs, one after another
#   3. check         scripts.check_results: every expected file, subject and fold present
#   4. comparison    scripts.compare_to_reference: Slim vs Conformer A and B from
#                    ../Conformer_decimated_CWT/results (skipped if those are missing)
#
# Options:
#   --pairs WAA_SAA HOC_WFE EPS_SPS   (default: the 3 pilot pairs)
#   --dataset-root PATH               (default: <Fine MI>/FineMI_0.5_3hz)
#   --no-overwrite                    smoke re-trains by default; full never re-trains a
#                                     finished pair, so an interrupted full run resumes
#   --skip-preflight                  only when you already passed it on this machine

import argparse
import os
import subprocess
import sys
import time

from src.config import DATASET_ROOT_DEFAULT, DEFAULT_PAIRS, PROJECT_ROOT, parse_pairs


def step(title, args, env):
    print("\n" + "#" * 72, flush=True)
    print(f"# {title}", flush=True)
    print("#   python " + " ".join(args), flush=True)
    print("#" * 72, flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable] + args, cwd=PROJECT_ROOT, env=env)
    print(f"# {title}: {'ok' if rc == 0 else f'FAILED (exit {rc})'} in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description="Local 3-pair test of Slim Conformer B.")
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    ap.add_argument('--pairs', nargs='+', default=list(DEFAULT_PAIRS))
    ap.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    ap.add_argument('--no-overwrite', action='store_true',
                    help="in smoke mode, keep existing smoke results instead of re-training")
    ap.add_argument('--skip-preflight', action='store_true')
    args = ap.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        ap.error(str(e))
    pair_args = list(dict.fromkeys(args.pairs))
    if not os.path.isdir(args.dataset_root):
        ap.error(f"--dataset-root does not exist: {args.dataset_root}")

    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    t_start = time.time()

    if not args.skip_preflight:
        pre = ['-m', 'scripts.preflight', '--allow-cpu', '--mode', args.mode,
               '--dataset-root', args.dataset_root, '--pairs', *pair_args,
               '--project-pairs', str(len(pairs)), '--n-gpus', '1']
        if args.mode == 'full':
            pre.append('--skip-timing')     # the smoke run already reported the projection
        if step("1/4 preflight", pre, env) != 0:
            print("\nLOCAL TEST STOPPED: the preflight failed. Its summary above names the check.")
            return 1

    train = ['-m', 'scripts.run_selected_pairs', '--mode', args.mode,
             '--dataset-root', args.dataset_root, '--pairs', *pair_args]
    if args.mode == 'smoke' and not args.no_overwrite:
        train.append('--overwrite')
    if step("2/4 training", train, env) != 0:
        print("\nLOCAL TEST STOPPED: training failed. The traceback is above and in "
              f"results/{args.mode}/logs/.")
        return 1

    check = ['-m', 'scripts.check_results', '--mode', args.mode,
             '--dataset-root', args.dataset_root, '--pairs', *pair_args]
    if step("3/4 output check", check, env) != 0:
        print("\nLOCAL TEST STOPPED: some outputs are missing or wrong (listed above).")
        return 1

    comp = ['-m', 'scripts.compare_to_reference', '--mode', args.mode, '--pairs', *pair_args]
    rc = step("4/4 comparison with Conformer A and B", comp, env)

    print("\n" + "=" * 72)
    if rc != 0:
        print("Training and checks passed, but the comparison could not run (see above). "
              "Usually the Conformer A / B results are missing from "
              "../Conformer_decimated_CWT/results.")
    print(f"LOCAL {args.mode.upper()} TEST PASSED in {(time.time() - t_start) / 60:.1f} min")
    print(f"  results:    results/{args.mode}/conformer_b_slim/")
    print(f"  comparison: results/{args.mode}/comparison/vs_reference.md")
    if args.mode == 'smoke':
        print("  next:       python -m scripts.local_test --mode full")
    print("=" * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
