# QuantumMed ADNI/PPMI Benchmark Suite

## Tool Overview
A 100% local, offline hybrid quantum-classical ML benchmarking pipeline for ADNI and PPMI datasets.  
Compliant with ADNI/PPMI Data Use Agreements — no external APIs, no network calls.

## Files
| File | Purpose |
|------|---------|
| `quantum_pipeline.py` | Core engine: DataProcessor, PennyLaneVQC, evaluate_classifier, run_benchmark |
| `app.py` | Streamlit UI (recommended entry point) |
| `test_pipeline.py` | Headless CLI test with synthetic ADNI-style data |

## How to Run

### Streamlit App (recommended)
```
cd C:\Q_exten
streamlit run app.py
```

### CLI headless test
```
cd C:\Q_exten
python test_pipeline.py
```

## Package Versions Required
```
pennylane==0.35.1
pennylane-lightning==0.35.1
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
streamlit>=1.37.1
matplotlib>=3.8
```

## Architecture

```
CSV / folder
     │
     ▼
DataProcessor
 ├─ SimpleImputer (mean/median/most_frequent)
 ├─ StandardScaler
 └─ PCA (n_components = n_qubits, configurable 2–8)
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
Classical Models                   PennyLane VQC
 ├─ RandomForestClassifier          ├─ AngleEmbedding (X rotation)
 └─ LogisticRegression              ├─ BasicEntanglerLayers
                                    └─ Nesterov Momentum Optimizer
     │                                      │
     └───────────── evaluate_classifier ────┘
                         │
                  Benchmark Report
                  (Accuracy, Sensitivity, Specificity,
                   Precision, F1, ROC-AUC, PR-AUC,
                   Training Time)
```

## VQC Architecture (PennyLane)
- **Embedding**: `AngleEmbedding` (rotation='X') — maps PCA features to qubit rotation angles
- **Ansatz**: `BasicEntanglerLayers` (configurable depth)
- **Measurement**: `PauliZ` expectation on qubit 0 (binary), or multiple qubits (multi-class)
- **Optimizer**: `NesterovMomentumOptimizer`
- **Loss**: Mean squared error (MSE) against ±1 targets (binary) or one-hot (multi-class)

## Output Files (saved to results/ by default)
- `benchmark_metrics.csv` — Full metric table
- `pca_loadings.csv` — PCA component loadings on original features
- `rf_feature_importance.csv` — RF importance per PCA component
- `benchmark_comparison.png` — Bar chart comparing all models
- `pca_loadings_heatmap.png` — Heatmap of PCA loadings
- `run_summary.json` — Config snapshot for reproducibility

## Compliance Notes
- Zero outbound network calls
- No telemetry, no auto-updates
- Works fully offline once packages are installed
- All models, metrics, and plots saved locally only

streamlit run app.py