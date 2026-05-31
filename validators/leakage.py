"""Leakage Validator – prüft ob PII durch die Synthese durchgesickert ist."""

import re
import pandas as pd


EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+\d{1,3}[\s-])?\(?\d{3,5}\)?[\s.-]\d{3,5}[\s.-]\d{2,5}")
IBAN_RE  = re.compile(r"\b[A-Z]{2}\d{2}[\s\d]{10,}\b")
IP_RE    = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")

PII_PATTERNS = [EMAIL_RE, PHONE_RE, IBAN_RE, IP_RE]

THRESHOLD = 0.02   # max 2% verdächtige Werte → pass


class LeakageValidator:
    """
    Scannt den synthetischen Datensatz auf verbleibende PII-Muster.
    Score = Anteil gefundener PII-Treffer an allen Zellwerten.
    """

    def check(self, df_synth: pd.DataFrame, sensitive_cols: list) -> dict:
        total_cells  = df_synth.size
        leaks        = []

        # Only scan columns that were NOT explicitly anonymized
        # (those already contain fake values by design)
        scan_cols = [c for c in df_synth.columns if c not in sensitive_cols]

        for col in scan_cols:
            col_leaks = self._scan_column(df_synth[col])
            if col_leaks:
                leaks.extend([(col, v) for v in col_leaks])

        score  = len(leaks) / max(total_cells, 1)
        passed = score <= THRESHOLD

        return {
            "passed":        passed,
            "score":         round(score, 5),
            "leaks_found":   len(leaks),
            "threshold":     THRESHOLD,
            "leak_samples":  [f"{c}: {v}" for c, v in leaks[:5]],
        }

    def _scan_column(self, series: pd.Series) -> list:
        found = []
        for val in series.dropna().astype(str).head(500):
            for pat in PII_PATTERNS:
                if pat.search(val):
                    found.append(val)
                    break
        return found
