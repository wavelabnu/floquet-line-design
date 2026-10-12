"""Run the self-pumped analytical/direct hpeesofsim/ADS OA comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tvtl_ads_builder_v2.circuit_spec import SelfPumpedCircuitSpec
from tvtl_ads_builder_v2.validation.compare_self_pumped import run_self_pumped_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-ghz", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "validation_self_pumped")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--draw-oa", action="store_true")
    parser.add_argument(
        "--oa-workspace",
        type=Path,
        default=REPO_ROOT / "ads_projects" / "validation_self_pumped_oa",
    )
    args = parser.parse_args()

    report = run_self_pumped_validation(
        SelfPumpedCircuitSpec(fs_hz=args.freq_ghz * 1.0e9),
        args.out,
        simulate=args.simulate,
        draw_oa=args.draw_oa,
        oa_workspace_dir=args.oa_workspace,
    )
    print(f"Comparison report: {args.out / 'comparison_report.json'}")
    return 0 if report["topology_comparison"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
