"""Dataset Exporter – exportiert sanitisierte Daten + Metadaten."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


class DatasetExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, df: pd.DataFrame, profile: dict) -> dict:
        ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"sanitized_{ts}"

        csv_path = self.output_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)

        checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()

        meta = {
            "maxxki_sanitizer_version": "0.1.0",
            "created_at":   datetime.now(timezone.utc).isoformat(),
            "source":       profile.get("source", "unknown"),
            "n_rows":       len(df),
            "n_cols":       len(df.columns),
            "columns":      df.columns.tolist(),
            "checksum_sha256": checksum,
            "gdpr_art25":   True,
            "eu_ai_act_annex_iv": True,
            "processing":   "fully_local",
        }

        meta_path = self.output_dir / f"{stem}_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        return {"csv": str(csv_path), "metadata": str(meta_path), "checksum": checksum}
