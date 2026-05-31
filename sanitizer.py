"""
MAXXKI MLOps Produktionsdaten-Sanitizer
=========================================
Lokal, DSGVO-konform, EU AI Act ready.
Usage:
    python sanitizer.py --input data/sample.csv --output out/ --report
"""

import argparse
import sys
import json
from pathlib import Path

from core.profiler     import SchemaProfiler
from core.anonymizer   import PIIAnonymizer
from core.synthesizer  import SyntheticGenerator
from validators.leakage   import LeakageValidator
from validators.fidelity  import FidelityValidator
from validators.utility   import UtilityValidator
from exporters.report  import ComplianceReporter
from exporters.dataset import DatasetExporter


def run_pipeline(input_path: str, output_dir: str, synth_rows: int,
                 epsilon: float, generate_report: bool) -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 1. INGEST & PROFILE ──────────────────────────────────────────────────
    print("\n[1/6] Profiling schema...")
    profiler = SchemaProfiler()
    df_raw, profile = profiler.profile(input_path)
    results["profile"] = profile
    print(f"      Rows: {len(df_raw)} | Cols: {len(df_raw.columns)}")
    print(f"      Sensitive columns detected: {profile['sensitive_cols']}")

    # ── 2. PII ANONYMIZATION ─────────────────────────────────────────────────
    print("\n[2/6] Anonymizing PII...")
    anon = PIIAnonymizer(strategy="pseudonymize")
    df_anon, anon_log = anon.transform(df_raw, profile["sensitive_cols"])
    results["anonymization"] = anon_log
    print(f"      Replaced {anon_log['total_replacements']} values across "
          f"{len(anon_log['columns_affected'])} columns")

    # ── 3. SYNTHETIC DATA GENERATION ─────────────────────────────────────────
    print(f"\n[3/6] Generating {synth_rows} synthetic rows (ε={epsilon})...")
    gen = SyntheticGenerator(epsilon=epsilon)
    df_synth = gen.fit_sample(df_anon, n_rows=synth_rows)
    results["synthesis"] = {"rows_generated": len(df_synth), "epsilon": epsilon}
    print(f"      Done – synthetic dataset shape: {df_synth.shape}")

    # ── 4. VALIDATION ────────────────────────────────────────────────────────
    print("\n[4/6] Running validation suite...")

    leak_val  = LeakageValidator()
    leak_result = leak_val.check(df_synth, profile["sensitive_cols"])
    results["leakage"] = leak_result
    status_icon = "✓" if leak_result["passed"] else "✗"
    print(f"      [{status_icon}] PII Leakage:    score={leak_result['score']:.3f}  "
          f"leaks={leak_result['leaks_found']}")

    fid_val  = FidelityValidator()
    fid_result = fid_val.check(df_anon, df_synth)
    results["fidelity"] = fid_result
    status_icon = "✓" if fid_result["passed"] else "✗"
    print(f"      [{status_icon}] Stat. Fidelity: score={fid_result['score']:.3f}  "
          f"threshold={fid_result['threshold']}")

    util_val = UtilityValidator()
    util_result = util_val.check(df_anon, df_synth, profile)
    results["utility"] = util_result
    status_icon = "✓" if util_result["passed"] else "✗"
    print(f"      [{status_icon}] Utility Score:  score={util_result['score']:.3f}  "
          f"(ML trainability)")

    # ── 5. COMPLIANCE GATE ───────────────────────────────────────────────────
    print("\n[5/6] Compliance Gate check...")
    all_passed = all([
        leak_result["passed"],
        fid_result["passed"],
        util_result["passed"],
    ])
    results["compliance_gate"] = {
        "passed": all_passed,
        "gdpr_art25": True,   # privacy by design – local processing
        "eu_ai_act_annex_iv": True,
        "checks": {
            "pii_leakage": leak_result["passed"],
            "fidelity":    fid_result["passed"],
            "utility":     util_result["passed"],
        }
    }
    if all_passed:
        print("      ✓ All checks passed – data cleared for export")
    else:
        print("      ✗ Compliance Gate FAILED – review validation results")
        print("        Aborting export to protect data integrity.")

    # ── 6. EXPORT ────────────────────────────────────────────────────────────
    if all_passed:
        print("\n[6/6] Exporting...")
        exporter = DatasetExporter(output_dir)
        export_paths = exporter.export(df_synth, profile)
        results["export"] = export_paths
        print(f"      CSV  → {export_paths['csv']}")
        print(f"      JSON → {export_paths['metadata']}")

        if generate_report:
            reporter = ComplianceReporter(output_dir)
            report_path = reporter.generate(results, input_path.name)
            results["report"] = str(report_path)
            print(f"      PDF-Report → {report_path}")
    else:
        results["export"] = None

    # Summary
    print("\n" + "═"*55)
    print("  MAXXKI Sanitizer – Pipeline complete")
    print("═"*55)
    print(f"  Input:    {input_path.name}  ({len(df_raw)} rows)")
    print(f"  Output:   {output_dir}/")
    print(f"  Gate:     {'✓ PASSED' if all_passed else '✗ FAILED'}")
    print("═"*55 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MAXXKI MLOps Produktionsdaten-Sanitizer"
    )
    parser.add_argument("--input",   required=True,  help="Input file (CSV/JSON)")
    parser.add_argument("--output",  default="./sanitized_output", help="Output directory")
    parser.add_argument("--rows",    type=int, default=1000, help="Synthetic rows to generate")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Differential privacy ε")
    parser.add_argument("--report",  action="store_true", help="Generate compliance PDF report")
    parser.add_argument("--json-out", action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    results = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        synth_rows=args.rows,
        epsilon=args.epsilon,
        generate_report=args.report,
    )

    if args.json_out:
        print(json.dumps(results, indent=2, default=str))

    sys.exit(0 if results.get("compliance_gate", {}).get("passed") else 1)


if __name__ == "__main__":
    main()
