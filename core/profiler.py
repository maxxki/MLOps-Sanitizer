"""Schema Profiler – erkennt Spaltentypen und sensitive Felder automatisch."""

import re
import pandas as pd
from pathlib import Path
from typing import Tuple

# Regex patterns für sensitive Spalten (erweiterbar)
SENSITIVE_PATTERNS = [
    # Namen – Underscore-safe (first_name, last_name, full_name etc.)
    r"(^|[_\s\-])(first|last|full|sur|given|middle)[_\s\-]?name",
    r"(^|[_\s\-])(firstname|lastname|fullname|forename)($|[_\s\-])",
    r"(^|[_\s\-])(vorname|nachname|familienname|rufname)($|[_\s\-])",
    r"^name$",
    # Kontakt
    r"(^|[_\s\-])(email|e[_\s]?mail|mail)($|[_\s\-])",
    r"(^|[_\s\-])(phone|telefon|tel|mobile|handy|mobil|fax|rufnummer)($|[_\s\-])",
    # Adresse
    r"(^|[_\s\-])(address|adresse|street|strasse|stra[ß]e)($|[_\s\-])",
    r"(^|[_\s\-])(city|stadt|ort|wohnort)($|[_\s\-])",
    r"(^|[_\s\-])(plz|zip|postcode|postleitzahl)($|[_\s\-])",
    # IDs
    r"(^|[_\s\-])(iban|kontonummer|bic|swift)($|[_\s\-])",
    r"(^|[_\s\-])(ssn|sozialversicherung|svnr|steuer[_\s\-]?id|tax[_\s\-]?id)($|[_\s\-])",
    r"(^|[_\s\-])(customer[_\s\-]?id|kunden[_\s\-]?(nr|id|nummer))($|[_\s\-])",
    r"(^|[_\s\-])(user[_\s\-]?id|userid|person[_\s\-]?id)($|[_\s\-])",
    # Sonstiges
    r"(^|[_\s\-])(dob|birth[_\s\-]?date|geburtsdatum|geburtstag)($|[_\s\-])",
    r"(^|[_\s\-])(ip[_\s\-]?addr(ess)?)($|[_\s\-])",
    r"(^|[_\s\-])(password|passwort|passwd|secret|token)($|[_\s\-])",
    r"(^|[_\s\-])(gender|geschlecht)($|[_\s\-])",
    r"(^|[_\s\-])(nationalit[yäa]t?|country[_\s\-]?of[_\s\-]?birth)($|[_\s\-])",
]

SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


class SchemaProfiler:
    """Liest CSV/JSON, profiliert Spaltentypen, erkennt sensitive Felder."""

    def profile(self, path: Path) -> Tuple[pd.DataFrame, dict]:
        df = self._load(path)
        col_types   = self._classify_columns(df)
        sensitive   = self._detect_sensitive(df)
        stats       = self._compute_stats(df)

        profile = {
            "source":        str(path),
            "n_rows":        len(df),
            "n_cols":        len(df.columns),
            "col_types":     col_types,
            "sensitive_cols": sensitive,
            "stats":         stats,
            "target_col":    self._guess_target(df),
        }
        return df, profile

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".json", ".jsonl"):
            return pd.read_json(path, lines=(suffix == ".jsonl"))
        raise ValueError(f"Unsupported file type: {suffix}")

    def _classify_columns(self, df: pd.DataFrame) -> dict:
        types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "int" in dtype or "float" in dtype:
                types[col] = "numeric"
            elif "datetime" in dtype:
                types[col] = "datetime"
            elif df[col].nunique() / max(len(df), 1) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "text"
        return types

    def _detect_sensitive(self, df: pd.DataFrame) -> list:
        sensitive = []
        for col in df.columns:
            if SENSITIVE_RE.search(col):
                sensitive.append(col)
                continue
            # Heuristic: check sample values for email/IBAN patterns
            sample = df[col].dropna().astype(str).head(20)
            if sample.str.contains(r"@[\w.-]+\.\w+", regex=True).any():
                sensitive.append(col)
            elif sample.str.contains(r"\b[A-Z]{2}\d{2}[\s\d]{10,}", regex=True).any():
                sensitive.append(col)
        return list(set(sensitive))

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        stats = {}
        for col in df.columns:
            s = df[col]
            stats[col] = {
                "null_rate": float(s.isna().mean()),
                "unique":    int(s.nunique()),
            }
            if pd.api.types.is_numeric_dtype(s):
                stats[col].update({
                    "mean": float(s.mean()),
                    "std":  float(s.std()),
                    "min":  float(s.min()),
                    "max":  float(s.max()),
                })
        return stats

    def _guess_target(self, df: pd.DataFrame) -> str | None:
        """Einfache Heuristik: letzte binäre Spalte als Zielvariable."""
        for col in reversed(df.columns.tolist()):
            if df[col].nunique() == 2:
                return col
        return None
