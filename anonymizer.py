"""PII Anonymizer – Pseudonymisierung und Hashing sensitiver Felder."""

import hashlib
import re
import uuid
import pandas as pd
from typing import Tuple


# ── Faker-basierte Ersetzungsgeneratoren ─────────────────────────────────────

def _make_faker():
    try:
        from faker import Faker
        return Faker("de_DE")
    except ImportError:
        return None


class PIIAnonymizer:
    """
    Strategien:
        pseudonymize  – deterministisches Fake-Replacement (Faker, seed-basiert)
        hash          – SHA-256 (nicht umkehrbar, referentielle Integrität erhalten)
        mask          – Wert auf fixe Länge maskieren
        redact        – Wert vollständig entfernen (leer / NaN)
    """

    def __init__(self, strategy: str = "pseudonymize"):
        assert strategy in {"pseudonymize", "hash", "mask", "redact"}
        self.strategy = strategy
        self._faker   = _make_faker()
        self._id_map: dict = {}   # Konsistenz: gleicher Wert → gleicher Ersatz

    def transform(self, df: pd.DataFrame, sensitive_cols: list) -> Tuple[pd.DataFrame, dict]:
        df_out = df.copy()
        log = {"columns_affected": [], "total_replacements": 0, "strategy": self.strategy}

        for col in sensitive_cols:
            if col not in df_out.columns:
                continue
            original_count = df_out[col].notna().sum()
            df_out[col] = df_out[col].apply(
                lambda v: self._replace(v, col) if pd.notna(v) else v
            )
            log["columns_affected"].append(col)
            log["total_replacements"] += int(original_count)

        return df_out, log

    # ── private ──────────────────────────────────────────────────────────────

    def _replace(self, value, col_name: str):
        key = (col_name, str(value))
        if key in self._id_map:
            return self._id_map[key]

        result = self._generate(value, col_name)
        self._id_map[key] = result
        return result

    def _generate(self, value, col_name: str):
        name_lower = col_name.lower()

        if self.strategy == "hash":
            return hashlib.sha256(str(value).encode()).hexdigest()[:16]

        if self.strategy == "mask":
            s = str(value)
            return s[:2] + "*" * max(0, len(s) - 2)

        if self.strategy == "redact":
            return ""

        # pseudonymize ────────────────────────────────────────────────────────
        if self._faker:
            f = self._faker
            if re.search(r"(email|mail)", name_lower):
                return f.email()
            if re.search(r"(phone|tel|mobile|handy)", name_lower):
                return f.phone_number()
            if re.search(r"(first|vorname)", name_lower):
                return f.first_name()
            if re.search(r"(last|nach|surname)", name_lower):
                return f.last_name()
            if re.search(r"(name)", name_lower):
                return f.name()
            if re.search(r"(address|adresse|street|strasse)", name_lower):
                return f.street_address()
            if re.search(r"(city|stadt)", name_lower):
                return f.city()
            if re.search(r"(zip|plz|postcode)", name_lower):
                return f.postcode()
            if re.search(r"(iban|konto)", name_lower):
                return f.iban()
            if re.search(r"(ip)", name_lower):
                return f.ipv4()
            if re.search(r"(user.?id|customer.?id|kunden)", name_lower):
                return f"USR-{uuid.uuid4().hex[:8].upper()}"
            # fallback: UUID-basierter Pseudonym
            return f"ANON-{uuid.uuid4().hex[:10].upper()}"

        # Faker nicht verfügbar – Hash-Fallback
        return f"ANON-{hashlib.md5(str(value).encode()).hexdigest()[:10].upper()}"
