"""Run from any working directory: python reproduce.py MODE."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "r6", "r7", "r8", "figures", "full", "external"))
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("NPROC", "4")

    def py(name, folder="code", extra=()):
        subprocess.run([sys.executable, name, *extra], cwd=ROOT / folder, env=env, check=True)

    groups = {
        "verify": ("verify_r5_campaign.py", "verify_r6_campaign.py", "verify_common_recovery.py", "verify_resource_sensitivity.py"),
        "r6": ("paired_resource_campaign.py", "verify_r6_campaign.py"),
        "r7": ("common_recovery_campaign.py", "verify_common_recovery.py"),
        "r8": ("resource_sensitivity.py", "verify_resource_sensitivity.py"),
    }
    if args.mode in groups:
        for name in groups[args.mode]:
            py(name)
    elif args.mode == "figures":
        py("make_figures.py", extra=("--architecture-only",))
        py("make_tables.py")
    elif args.mode == "full":
        subprocess.run(["bash", "run_pipeline.sh"], cwd=ROOT / "code", env=env, check=True)
    elif args.mode == "external":
        for name in ("observed_comparison.py", "observed_acc_comparison.py", "jb_compare.py"):
            py(name, "external_validation")
        py("jb_compare.py", "external_validation", ("--acc",))
        for name in ("verify_jb_driver.py", "observed_block_bootstrap.py", "observed_causal_forecast.py", "reference_altitude_sensitivity.py", "observed_residual_replay.py"):
            py(name, "external_validation")


if __name__ == "__main__":
    main()
