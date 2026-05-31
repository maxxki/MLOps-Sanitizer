# MAXXKI MLOps Produktionsdaten-Sanitizer

**Lokal. DSGVO-konform. EU AI Act ready.**

Unternehmen wollen echte Produktionsdaten zum Fine-Tuning von ML-Modellen nutzen –
dürfen es aber nicht. Dieser Sanitizer löst genau dieses Problem: er transformiert
sensible Produktionsdaten in statistisch treue, synthetische Trainingsdaten,
die nachweislich keine personenbezogenen Informationen mehr enthalten.

```
Input: Produktionsdaten (CSV/JSON)
         ↓
   [Schema Profiler]  →  Erkennt sensitive Spalten automatisch
         ↓
   [PII Anonymizer]   →  Pseudonymisierung / Hashing / Redaction
         ↓
   [Synth. Generator] →  Struktur-erhaltende Neugenerierung (SDV / stats)
         ↓
   [Validation Suite] →  Leakage · Fidelity · Utility
         ↓
   [Compliance Gate]  →  DSGVO Art. 25 + EU AI Act Annex IV
         ↓
Output: Sanitisiertes Dataset + Compliance Report (PDF/JSON)
```

---

## Schnellstart

```bash
pip install -r requirements.txt
python sanitizer.py --input data/sample.csv --output ./out --report
```

### Alle Optionen

```
--input     Eingabedatei (CSV oder JSON/JSONL)         [required]
--output    Ausgabeverzeichnis                          [default: ./sanitized_output]
--rows      Anzahl synthetischer Zeilen                [default: 1000]
--epsilon   Differential Privacy Budget ε              [default: 1.0]
--report    Compliance PDF-Report generieren           [flag]
--json-out  Ergebnis-JSON auf stdout ausgeben          [flag]
```

### Exit Codes
- `0` – Compliance Gate bestanden, Daten exportiert
- `1` – Validation fehlgeschlagen, kein Export

---

## Architektur

### 1. Schema Profiler (`core/profiler.py`)
- Liest CSV und JSON/JSONL
- Klassifiziert Spaltentypen (numeric, categorical, text, datetime)
- Erkennt sensitive Felder via Regex-Pattern gegen Spaltennamen **und** Stichproben-Werte
- Schätzt Zielvariable für ML-Utility-Test

### 2. PII Anonymizer (`core/anonymizer.py`)
Vier Strategien (via `--strategy`):

| Strategie | Beschreibung | Referentielle Integrität |
|-----------|-------------|--------------------------|
| `pseudonymize` | Faker-basierte Ersetzung (DE locale) | ✓ |
| `hash` | SHA-256 (16 Zeichen) | ✓ |
| `mask` | Erste 2 Zeichen + `***` | ✗ |
| `redact` | Leerstring | ✗ |

### 3. Synthetic Generator (`core/synthesizer.py`)
- **Primär:** SDV `GaussianCopulaSynthesizer` (korrelationserhaltend)
- **Fallback:** Statistik-basierte Generierung ohne externe Abhängigkeit
- **Differential Privacy:** Gauss-Mechanismus mit konfigurierbarem ε

### 4. Validation Suite

| Validator | Methode | Schwellwert |
|-----------|---------|-------------|
| `LeakageValidator` | Regex (Email, Phone, IBAN, IP) auf non-sensitiven Spalten | ≤ 2% |
| `FidelityValidator` | KS-Test (numerisch) + TVD (kategorisch) | ≥ 0.70 |
| `UtilityValidator` | TSTR: Train-on-Synth, Test-on-Real (RF) | ≥ 0.50 |

### 5. Compliance Gate
Alle drei Validatoren müssen bestehen. Bei Fehlschlag: kein Export, keine Datei.

---

## EU AI Act Mapping

| Anforderung | Umsetzung |
|-------------|-----------|
| Art. 10 – Data Governance | Automatische Profilerstellung, Dokumentation sensitiver Felder |
| Art. 11 – Technische Dokumentation | Compliance Report mit vollständigem Pipeline-Audit-Trail |
| Art. 25 DSGVO – Privacy by Design | Vollständig lokale Verarbeitung, kein Cloud-Aufruf |
| Annex IV – Technische Dokumentation | JSON/PDF Report, maschinenlesbar, signiert (SHA-256) |

---

## Abhängigkeiten

```
pandas, numpy, scipy, scikit-learn   # Core (required)
faker                                 # Pseudonymisierung
sdv                                  # Synthetische Daten (empfohlen, Fallback aktiv)
reportlab                            # PDF Report (optional)
diffprivlib                          # Erweiterte DP-Mechanismen (optional)
```

SDV ist **optional** – ohne SDV greift ein interner statistik-basierter
Generator ein, der ohne externe Abhängigkeiten auskommt.

---

## Erweiterungen (Roadmap)

- [ ] n8n Community Node (wie `Anonymization-Suite`)
- [ ] Claude Code Webhook Hook (wie `Anonymization-Claude-Webhook`)
- [ ] MLflow Integration: sanitisierte Datasets als MLflow Artifact
- [ ] DVC Pipeline Stage
- [ ] HuggingFace Dataset Push (lokal, mit `datasets` lib)
- [ ] Image-Anonymisierung (Gesichter, Kennzeichen) via `AMPVisio`-Integration
- [ ] Web-UI (Gradio / Streamlit) für nicht-technische Nutzer

---

## Lizenz

MIT – siehe LICENSE

---

*MAXXKI – Industrial-grade local AI infrastructure*
