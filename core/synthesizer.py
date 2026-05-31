"""
Synthetischer Datengenerator.
Primär: SDV GaussianCopulaSynthesizer (structure-preserving)
Fallback: statistik-basierte Neugenerierung ohne externe Bibliothek
"""

import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Spaltennamen → Faker-Methode (de_DE)
_NAME_MAP = [
    (r"(^|[_\s\-])first[_\s\-]?name($|[_\s\-])",  "first_name"),
    (r"(^|[_\s\-])vorname($|[_\s\-])",              "first_name"),
    (r"(^|[_\s\-])last[_\s\-]?name($|[_\s\-])",    "last_name"),
    (r"(^|[_\s\-])nachname($|[_\s\-])",             "last_name"),
    (r"(^|[_\s\-])familienname($|[_\s\-])",         "last_name"),
    (r"(^|[_\s\-])full[_\s\-]?name($|[_\s\-])",    "name"),
    (r"^name$",                                      "name"),
    (r"(^|[_\s\-])email($|[_\s\-])",                "email"),
    (r"(^|[_\s\-])e[_\s]?mail($|[_\s\-])",         "email"),
    (r"(^|[_\s\-])phone($|[_\s\-])",                "phone_number"),
    (r"(^|[_\s\-])telefon($|[_\s\-])",              "phone_number"),
    (r"(^|[_\s\-])handy($|[_\s\-])",                "phone_number"),
    (r"(^|[_\s\-])mobile($|[_\s\-])",               "phone_number"),
    (r"(^|[_\s\-])city($|[_\s\-])",                 "city"),
    (r"(^|[_\s\-])stadt($|[_\s\-])",                "city"),
    (r"(^|[_\s\-])ort($|[_\s\-])",                  "city"),
    (r"(^|[_\s\-])plz($|[_\s\-])",                  "postcode"),
    (r"(^|[_\s\-])zip($|[_\s\-])",                  "postcode"),
    (r"(^|[_\s\-])postcode($|[_\s\-])",             "postcode"),
    (r"(^|[_\s\-])street($|[_\s\-])",               "street_address"),
    (r"(^|[_\s\-])adresse($|[_\s\-])",              "street_address"),
    (r"(^|[_\s\-])iban($|[_\s\-])",                 "iban"),
    (r"(^|[_\s\-])ip[_\s\-]?addr($|[_\s\-])",      "ipv4"),
]


def _get_faker_method(col_name: str):
    """Gibt den passenden Faker-Methodennamen für eine Spalte zurück, oder None."""
    for pattern, method in _NAME_MAP:
        if re.search(pattern, col_name, re.IGNORECASE):
            return method
    return None


def _make_faker():
    try:
        from faker import Faker
        return Faker("de_DE")
    except ImportError:
        return None


class SyntheticGenerator:
    """
    Generiert synthetische Daten die statistische Eigenschaften des Originals
    erhalten (Verteilungen, Korrelationen, Datentypen).

    epsilon: Differential-Privacy-Budget
             Niedrig (0.1) = mehr Privacy, weniger Fidelity
             Hoch  (10.0) = mehr Fidelity, weniger Privacy
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon  = epsilon
        self._sdv_ok  = self._check_sdv()
        self._faker   = _make_faker()
        self._model   = None

    def fit_sample(self, df: pd.DataFrame, n_rows: int,
                   sensitive_cols: list = None) -> pd.DataFrame:
        if self._sdv_ok:
            result = self._sdv_generate(df, n_rows)
        else:
            print("      [!] SDV not found – using stats-based fallback generator")
            result = self._stats_generate(df, n_rows)

        # SDV generiert englische Namen/Städte – nach Synthese mit de_DE Faker überschreiben
        if sensitive_cols and self._faker:
            result = self._reapply_de_faker(result, sensitive_cols)

        return result

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

    # ── de_DE Faker post-processing ──────────────────────────────────────────

    def _reapply_de_faker(self, df: pd.DataFrame, sensitive_cols: list) -> pd.DataFrame:
        """
        Überschreibt NUR Personen-identifizierende Spalten (Namen, Email, Phone)
        mit deutschen Faker-Werten. Geo-Spalten (city, plz, street) werden
        NICHT überschrieben – SDV hat deren Verteilung korrekt gelernt.
        """
        # Nur diese Faker-Methoden dürfen SDV-Output ersetzen
        PERSON_METHODS = {"first_name", "last_name", "name", "email",
                          "phone_number", "iban", "ipv4"}

        f   = self._faker
        out = df.copy()
        n   = len(out)

        for col in sensitive_cols:
            if col not in out.columns:
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                continue
            method = _get_faker_method(col)
            if method not in PERSON_METHODS:
                continue
            faker_fn = getattr(f, method, None)
            if faker_fn is None:
                continue
            out[col] = [faker_fn() for _ in range(n)]

        return out

    # ── Stats-based fallback ─────────────────────────────────────────────────

    def _stats_generate(self, df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        rng    = np.random.default_rng(42)
        result = {}

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
        if self.epsilon < 10.0:
            synth_df = self._apply_dp_noise(df, synth_df, rng)
        return synth_df

    def _sample_numeric(self, series: pd.Series, n: int, rng) -> np.ndarray:
        mu, sigma = series.mean(), series.std()
        sigma = sigma if sigma > 0 else abs(mu) * 0.1 + 1e-6
        samples = np.clip(rng.normal(mu, sigma, n), series.min(), series.max())
        if "int" in str(series.dtype):
            samples = np.round(samples).astype(int)
        return samples

    def _sample_categorical(self, series: pd.Series, n: int, rng) -> np.ndarray:
        counts = series.value_counts(normalize=True)
        return rng.choice(counts.index.tolist(), size=n, p=counts.values)

    def _sample_text_ids(self, series: pd.Series, n: int, rng) -> list:
        return series.sample(n=n, replace=True, random_state=42).tolist()

    def _apply_dp_noise(self, df_orig: pd.DataFrame,
                        df_synth: pd.DataFrame, rng) -> pd.DataFrame:
        df_out = df_synth.copy()
        for col in df_orig.columns:
            if pd.api.types.is_numeric_dtype(df_orig[col]):
                sensitivity = float(df_orig[col].max() - df_orig[col].min())
                sigma = sensitivity / (self.epsilon + 1e-9)
                noise = rng.normal(0, sigma * 0.05, len(df_out))
                df_out[col] = df_out[col].astype(float) + noise
                df_out[col] = df_out[col].clip(df_orig[col].min(), df_orig[col].max())
                if "int" in str(df_orig[col].dtype):
                    df_out[col] = df_out[col].round().astype(int)
        return df_out
