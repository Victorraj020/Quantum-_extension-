import os
import numpy as np
import pandas as pd
from quantum_pipeline import run_benchmark

def generate_synthetic_adni_ppmi_data(n_samples=200, n_features=12, multi_class=False):
    np.random.seed(42)
    # Generate realistic fake biomarker names (neuroimaging + CSF/blood + clinical assessments)
    feature_names = [
        "Hippocampus_Volume", "Entorhinal_Thickness", "FDG_PET_SUVr", "AV45_Amyloid_PET",
        "CSF_Abeta42", "CSF_pTau181", "CSF_tTau", "MoCA_Score", "MMSE_Score",
        "UPDRS_Part3_Motor", "DatSCAN_SBR_Striatum", "Age"
    ][:n_features]

    X = np.random.randn(n_samples, n_features)
    # Inject missing values to test imputer handling (e.g. 5% missing)
    mask = np.random.rand(*X.shape) < 0.05
    X[mask] = np.nan

    if not multi_class:
        # Binary: Healthy Control (CN) vs Disease (AD or PD)
        # Linear combination + noise for label generation
        logits = 0.8 * X[:, 0] - 0.7 * X[:, 1] + 0.9 * X[:, 2] - 0.6 * X[:, 4] + np.random.randn(n_samples) * 0.5
        labels = np.where(logits > 0, "AD", "CN")
    else:
        # Multi-class: CN vs MCI vs AD
        logits1 = 0.5 * X[:, 0] - 0.5 * X[:, 1]
        logits2 = 0.8 * X[:, 2] - 0.8 * X[:, 4]
        labels = []
        for i in range(n_samples):
            if logits1[i] > 0.3:
                labels.append("CN")
            elif logits2[i] > 0.2:
                labels.append("MCI")
            else:
                labels.append("AD")
        labels = np.array(labels)

    df = pd.DataFrame(X, columns=feature_names)
    df["Diagnosis_Group"] = labels
    return df

def test_pipeline_binary():
    print("=== Testing Binary Classification Pipeline ===")
    df = generate_synthetic_adni_ppmi_data(n_samples=100, n_features=8, multi_class=False)
    feature_cols = [c for c in df.columns if c != "Diagnosis_Group"]
    target_col = "Diagnosis_Group"

    config = {
        "imputer_strategy": "mean",
        "n_components": 4,
        "vqc_n_layers": 2,
        "vqc_max_steps": 30,
        "test_size": 0.2,
        "random_state": 42
    }

    out_dir = "test_output_binary"
    benchmark_df, loadings_df, rf_importance, output_dir = run_benchmark(
        df, feature_cols, target_col, config, out_dir
    )
    print("\nBenchmark Results (Binary):")
    print(benchmark_df)
    print("\nPCA Loadings (First 2 PC preview):")
    print(loadings_df.head(2))
    print(f"\nSuccessfully generated outputs in: {output_dir}")

def test_pipeline_multiclass():
    print("\n=== Testing Multi-Class Classification Pipeline ===")
    df = generate_synthetic_adni_ppmi_data(n_samples=100, n_features=8, multi_class=True)
    feature_cols = [c for c in df.columns if c != "Diagnosis_Group"]
    target_col = "Diagnosis_Group"

    config = {
        "imputer_strategy": "median",
        "n_components": 4,
        "vqc_n_layers": 2,
        "vqc_max_steps": 30,
        "test_size": 0.2,
        "random_state": 42
    }

    out_dir = "test_output_multiclass"
    benchmark_df, loadings_df, rf_importance, output_dir = run_benchmark(
        df, feature_cols, target_col, config, out_dir
    )
    print("\nBenchmark Results (Multi-Class):")
    print(benchmark_df)
    print(f"\nSuccessfully generated outputs in: {output_dir}")

if __name__ == "__main__":
    test_pipeline_binary()
    test_pipeline_multiclass()
