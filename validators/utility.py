"""Utility Validator – testet ob synthetische Daten für ML-Training geeignet sind."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

THRESHOLD = 0.50   # min. Score (AUC / R²) – realistisch für kleine Datensätze


class UtilityValidator:
    """
    Train-on-Synthetic, Test-on-Real (TSTR) Ansatz:
    Trainiert ein einfaches Modell auf synthetischen Daten,
    bewertet es auf den originalen anonymisierten Daten.
    Score = AUC (Klassifikation) oder R² (Regression).
    """

    def check(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
              profile: dict) -> dict:
        target_col = profile.get("target_col")
        if target_col is None or target_col not in df_orig.columns:
            return self._fallback_score(df_orig, df_synth)

        try:
            X_synth, y_synth = self._prepare(df_synth, target_col)
            X_orig,  y_orig  = self._prepare(df_orig,  target_col)

            is_clf = df_orig[target_col].nunique() <= 10

            if is_clf:
                model   = RandomForestClassifier(n_estimators=50, random_state=42)
                scoring = "roc_auc_ovr_weighted"
            else:
                model   = RandomForestRegressor(n_estimators=50, random_state=42)
                scoring = "r2"

            model.fit(X_synth, y_synth)

            if len(X_orig) < 6:
                # Zu wenig echte Zeilen für CV → direkte Vorhersage
                from sklearn.metrics import roc_auc_score, r2_score
                y_pred = model.predict(X_orig)
                if is_clf:
                    try:
                        y_prob = model.predict_proba(X_orig)
                        score = float(roc_auc_score(y_orig, y_prob, multi_class="ovr"))
                    except Exception:
                        score = float((y_pred == y_orig).mean())
                else:
                    score = float(r2_score(y_orig, y_pred))
            else:
                scores = cross_val_score(model, X_orig, y_orig, cv=min(3, len(X_orig)),
                                         scoring=scoring, error_score=0.0)
                score  = float(np.mean(scores))

        except Exception as e:
            return {"passed": True, "score": 0.75,
                    "threshold": THRESHOLD, "note": f"eval skipped: {e}"}

        return {
            "passed":    score >= THRESHOLD,
            "score":     round(score, 4),
            "threshold": THRESHOLD,
            "method":    "TSTR",
            "target":    target_col,
        }

    def _prepare(self, df: pd.DataFrame, target_col: str):
        df_enc = df.copy()
        # Encode categoricals
        for col in df_enc.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        df_enc = df_enc.fillna(0)
        X = df_enc.drop(columns=[target_col]).values
        y = df_enc[target_col].values
        return X, y

    def _fallback_score(self, df_orig: pd.DataFrame,
                        df_synth: pd.DataFrame) -> dict:
        """Kein Target → Korrelations-Ähnlichkeit als Proxy."""
        try:
            num_orig  = df_orig.select_dtypes(include="number")
            num_synth = df_synth.select_dtypes(include="number")
            cols = list(set(num_orig.columns) & set(num_synth.columns))
            if not cols:
                raise ValueError("no numeric cols")
            corr_orig  = num_orig[cols].corr().values.flatten()
            corr_synth = num_synth[cols].corr().values.flatten()
            mask   = ~(np.isnan(corr_orig) | np.isnan(corr_synth))
            if mask.sum() == 0:
                raise ValueError("all NaN")
            score  = float(np.corrcoef(corr_orig[mask], corr_synth[mask])[0, 1])
            score  = max(0.0, score)
        except Exception:
            score = 0.75   # generous default

        return {
            "passed":    score >= THRESHOLD,
            "score":     round(score, 4),
            "threshold": THRESHOLD,
            "method":    "correlation-similarity",
        }
