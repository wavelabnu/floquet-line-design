"""ADS-enabled smoke check for the OA self-pumped schematic writer.

Run with ADS Python:

    python tests/ads_oa_smoke.py --simulate
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tvtl_ads_builder_v2.ads_oa_writer import build_self_pumped_minimal_schematic
from tvtl_ads_builder_v2.circuit_spec import SelfPumpedCircuitSpec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=str(REPO_ROOT / "ads_projects" / "validation_self_pumped_oa"),
    )
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    result = build_self_pumped_minimal_schematic(
        args.workspace,
        spec=SelfPumpedCircuitSpec(),
        simulate=args.simulate,
    )
    netlist = Path(result.netlist_path).read_text(encoding="utf-8")

    assert result.instance_cells["PORT_SIG"] == "P_1Tone"
    assert result.instance_cells["PORT_PUMP"] == "P_1Tone"
    assert result.instance_cells["TL_RF"] == "TLIN"
    assert "RF_MAIN" in result.net_names
    assert "Port:PORT_SIG  RF_MAIN 0" in netlist
    assert "Port:PORT_PUMP  RF_MAIN 0" in netlist
    assert "Freq[1]=fs" in netlist
    assert "Freq[1]=2*fs" in netlist
    assert not any(name in result.net_names for name in ["PUMP", "CONTROL", "BIAS"])
    if args.simulate:
        assert result.output_dir is not None
        assert any(name.endswith(".ds") for name in result.simulation_output_files)

    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
