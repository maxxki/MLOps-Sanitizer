"""
Synthetischer Datengenerator.
Primär: SDV GaussianCopulaSynthesizer (structure-preserving)
Fallback: statistik-basierte Neugenerierung ohne externe Bibliothek
"""

import numpy as np
import pandas as pd
from typing import Optional


class SyntheticGenerator:
    """
    Generiert synthetische Daten die statistische Eigenschaften des Originals
    erhalten (Verteilungen, Korrelationen, Datentypen).

    epsilon: Differential-Privacy-Budget (nur mit SDV-DP-Modul aktiv)
             Niedrig (0.1) = mehr Privacy, weniger Fidelity
             Hoch  (10.0) = mehr Fidelity, weniger Privacy
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon   = epsilon
        self._sdv_ok   = self._check_sdv()
        self._model    = None

    def fit_sample(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        if self._sdv_ok:
            return self._sdv_generate(df, n_rows)
        print("      [!] SDV not found – using stats-based fallback generator")
        return self._stats_generate(df, n_rows)

    # ── SDV path ─────────────────────────────────────────────────────────────

    def _check_sdv(self) -> bool:
        try:
            import sdv  # noqa: F401
            return True
        except ImportError:
            return False

    def _sdv_generate(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        synth = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
        synth.fit(df)
        self._model = synth
        return synth.sample(num_rows=n_rows)

    # ── Stats-based fallback ─────────────────────────────────────────────────

    def _stats_generate(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        """
        Spalte für Spalte: Verteilung schätzen, neue Werte samplen.
        Differential Privacy: Gauss-Rauschen auf numerische Werte proportional zu 1/epsilon.
        """
        rng     = np.random.default_rng(42)
        result  = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                result[col] = [None] * n_rows
                continue

            dtype = str(df[col].dtype)

            if "int" in dtype or "float" in dtype:
                result[col] = self._sample_numeric(series, n_rows, rng)
            elif df[col].nunique() / max(len(df), 1) < 0.05:
                result[col] = self._sample_categorical(series, n_rows, rng)
            else:
                result[col] = self._sample_text_ids(series, n_rows, rng)

        synth_df = pd.DataFrame(result)

        # Differential Privacy noise auf numerische Spalten
        if self.epsilon < 10.0:
            synth_df = self._apply_dp_noise(df, synth_df, rng)

        return synth_df

    def _sample_numeric(self, series: pd.Series, n: int, rng) -> np.ndarray:
        mu, sigma = series.mean(), series.std()
        sigma = sigma if sigma > 0 else abs(mu) * 0.1 + 1e-6
        samples = rng.normal(mu, sigma, n)
        lo, hi  = series.min(), series.max()
        samples = np.clip(samples, lo, hi)
        if "int" in str(series.dtype):
            samples = np.round(samples).astype(int)
        return samples

    def _sample_categorical(self, series: pd.Series, n: int, rng) -> np.ndarray:
        vals   = series.values
        counts = series.value_counts(normalize=True)
        return rng.choice(counts.index.tolist(), size=n, p=counts.values)

    def _sample_text_ids(self, series: pd.Series, n: int, rng) -> list:
        # Für Freitext / IDs: resample aus vorhandenen Werten (bereits anonymisiert)
        return series.sample(n=n, replace=True, random_state=42).tolist()

    def _apply_dp_noise(self, df_orig: pd.DataFrame,
                        df_synth: pd.DataFrame, rng) -> pd.DataFrame:
        """Gauss-Mechanismus: σ = sensitivity / epsilon"""
        df_out = df_synth.copy()
        for col in df_orig.columns:
            if pd.api.types.is_numeric_dtype(df_orig[col]):
                sensitivity = float(df_orig[col].max() - df_orig[col].min())
                sigma = sensitivity / (self.epsilon + 1e-9)
                noise = rng.normal(0, sigma * 0.05, len(df_out))
                df_out[col] = df_out[col].astype(float) + noise
                # Re-clamp to original range
                df_out[col] = df_out[col].clip(
                    df_orig[col].min(), df_orig[col].max()
                )
                if "int" in str(df_orig[col].dtype):
                    df_out[col] = df_out[col].round().astype(int)
        return df_out
