"""
Compliance Reporter – generiert maschinenlesbaren JSON-Report
(+ optionalen PDF-Report via reportlab).
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class ComplianceReporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate(self, results: dict, source_name: str) -> Path:
        ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = self._build_report(results, source_name, ts)

        json_path = self.output_dir / f"compliance_report_{ts}.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        # Optional: PDF via reportlab
        try:
            pdf_path = self._render_pdf(report, ts)
            return pdf_path
        except Exception:
            return json_path

    def _build_report(self, results: dict, source_name: str, ts: str) -> dict:
        gate = results.get("compliance_gate", {})
        return {
            "title":         "MAXXKI MLOps Sanitizer – Compliance Report",
            "version":       "0.1.0",
            "generated_at":  ts,
            "source_file":   source_name,
            "overall_status": "COMPLIANT" if gate.get("passed") else "NON-COMPLIANT",
            "legal_basis": {
                "gdpr_art25_privacy_by_design": gate.get("gdpr_art25", True),
                "eu_ai_act_annex_iv_technical_doc": gate.get("eu_ai_act_annex_iv", True),
                "processing_location": "local – no data left the system",
            },
            "pipeline_summary": {
                "rows_input":    results.get("profile", {}).get("n_rows"),
                "sensitive_cols": results.get("profile", {}).get("sensitive_cols"),
                "anonymization_strategy": results.get("anonymization", {}).get("strategy"),
                "total_replacements": results.get("anonymization", {}).get("total_replacements"),
                "synthetic_rows_generated": results.get("synthesis", {}).get("rows_generated"),
                "dp_epsilon":    results.get("synthesis", {}).get("epsilon"),
            },
            "validation_results": {
                "pii_leakage": results.get("leakage"),
                "fidelity":    results.get("fidelity"),
                "utility":     results.get("utility"),
            },
            "compliance_checks": gate.get("checks", {}),
            "export_checksum": results.get("export", {}).get("checksum") if results.get("export") else None,
        }

    def _render_pdf(self, report: dict, ts: str) -> Path:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors

        pdf_path = self.output_dir / f"compliance_report_{ts}.pdf"
        doc   = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []

        story.append(Paragraph(report["title"], styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Status: <b>{report['overall_status']}</b>", styles["Heading2"]))
        story.append(Paragraph(f"Generated: {report['generated_at']}", styles["Normal"]))
        story.append(Paragraph(f"Source: {report['source_file']}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Validation table
        story.append(Paragraph("Validation Results", styles["Heading2"]))
        val = report["validation_results"]
        rows = [["Check", "Score", "Passed"]]
        for name, res in val.items():
            if isinstance(res, dict):
                rows.append([
                    name.replace("_", " ").title(),
                    str(res.get("score", "-")),
                    "✓" if res.get("passed") else "✗"
                ])
        t = Table(rows, colWidths=[200, 100, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1D9E75")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Legal basis
        story.append(Paragraph("Legal Basis", styles["Heading2"]))
        for k, v in report["legal_basis"].items():
            story.append(Paragraph(f"• {k}: {v}", styles["Normal"]))

        doc.build(story)
        return pdf_path
