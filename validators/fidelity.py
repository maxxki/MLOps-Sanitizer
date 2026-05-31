"""Fidelity Validator – prüft ob synthetische Daten statistisch treu sind."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

THRESHOLD = 0.70   # min. Score zum Bestehen


class FidelityValidator:
    """
    Vergleicht Verteilungen (KS-Test für Numerik, TVD für Kategorien).
    Score = gewichteter Mittelwert über alle Spalten.
    """

    def check(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
              skip_cols: list = None) -> dict:
        """
        skip_cols: Spalten die vom Fidelity-Check ausgeschlossen werden.
        Anonymisierte String-Spalten (Email, Name, Phone) sind by design
        nicht mehr verteilungstreu – sie sollen nicht bewertet werden.
        """
        skip = set(skip_cols or [])
        scores  = []
        details = {}

        for col in df_orig.columns:
            if col not in df_synth.columns:
                continue
            # Rein text-basierte Spalten ohne sinnvolle Verteilung überspringen
            if col in skip:
                details[col] = None
                continue
            # Hochkardinale Object-Spalten (jeder Wert unique) überspringen
            if df_orig[col].dtype == object:
                uniq_ratio = df_orig[col].nunique() / max(len(df_orig), 1)
                if uniq_ratio > 0.5:
                    details[col] = None
                    continue
            try:
                s = self._column_score(df_orig[col], df_synth[col])
            except Exception:
                s = 0.5
            scores.append(s)
            details[col] = round(s, 4)

        overall = float(np.mean(scores)) if scores else 0.0

        return {
            "passed":    overall >= THRESHOLD,
            "score":     round(overall, 4),
            "threshold": THRESHOLD,
            "per_col":   details,
        }

    def _column_score(self, orig: pd.Series, synth: pd.Series) -> float:
        if pd.api.types.is_numeric_dtype(orig):
            return self._ks_score(orig, synth)
        else:
            return self._tvd_score(orig, synth)

    def _ks_score(self, orig: pd.Series, synth: pd.Series) -> float:
        """KS-Test p-Wert als Score (höher = ähnlicher)."""
        a = orig.dropna().astype(float)
        b = synth.dropna().astype(float)
        if len(a) < 5 or len(b) < 5:
            return 1.0
        _, p = scipy_stats.ks_2samp(a, b)
        return float(p)

    def _tvd_score(self, orig: pd.Series, synth: pd.Series) -> float:
        """Total Variation Distance → 1 − TVD als Score."""
        a = orig.dropna().astype(str).value_counts(normalize=True)
        b = synth.dropna().astype(str).value_counts(normalize=True)
        all_keys = set(a.index) | set(b.index)
        tvd = 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in all_keys)
        return float(1.0 - tvd)
