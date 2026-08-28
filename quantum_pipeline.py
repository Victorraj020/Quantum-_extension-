import os
import json
import time
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

class DataProcessor:
    def __init__(self, imputer_strategy='mean', n_components=4):
        self.imputer_strategy = imputer_strategy
        self.n_components = n_components
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.target_name = ""
        self.classes_ = []

    def fit_transform(self, df, feature_cols, target_col):
        self.feature_names = list(feature_cols)
        self.target_name = target_col

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle missing target rows if any
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Fit LabelEncoder
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = [str(c) for c in self.label_encoder.classes_]

        # Impute missing values in features
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)

        return X_pca, y_encoded, self.classes_

    def transform(self, df):
        X = df[self.feature_names].copy()
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def get_pca_loadings(self):
        loadings = pd.DataFrame(
            self.pca.components_,
            columns=self.feature_names,
            index=[f"PC{i+1}" for i in range(self.n_components)]
        )
        return loadings

class PennyLaneVQC:
    def __init__(self, n_qubits=4, n_layers=2, n_classes=2, max_steps=30, lr=0.1, stepsize=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.max_steps = max_steps
        self.lr = lr
        self.stepsize = stepsize
        self.weights = None
        self.bias = None
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, features):
            qml.AngleEmbedding(features, wires=range(self.n_qubits), rotation='X')
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            if self.n_classes == 2:
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(i)) for i in range(min(self.n_classes, self.n_qubits))]
        return circuit

    def fit(self, X, y):
        # Normalize features to [-pi, pi] for AngleEmbedding
        min_vals = np.min(X, axis=0)
        ptp_vals = np.ptp(X, axis=0) + 1e-8
        X_norm = np.pi * (X - min_vals) / ptp_vals - (np.pi / 2)
        circuit = self._build_circuit()

        np.random.seed(42)
        shape = qml.BasicEntanglerLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        init_weights = pnp.array(np.random.randn(*shape), requires_grad=True)

        if self.n_classes == 2:
            init_bias = pnp.array(0.0, requires_grad=True)
            y_targets = pnp.array(np.where(y == 0, -1.0, 1.0))
            
            def cost_fn(w, b):
                preds = pnp.stack([circuit(w, x) + b for x in X_norm])
                return pnp.mean((preds - y_targets) ** 2)

            opt = qml.NesterovMomentumOptimizer(stepsize=self.stepsize)
            weights = init_weights
            bias = init_bias

            for step in range(self.max_steps):
                (weights, bias), cost = opt.step_and_cost(cost_fn, weights, bias)

            self.weights = weights
            self.bias = bias
        else:
            init_bias = pnp.array(np.zeros(self.n_classes), requires_grad=True)
            y_onehot = np.zeros((len(y), self.n_classes))
            for i, val in enumerate(y):
                y_onehot[i, val] = 1.0

            def cost_fn(w, b):
                preds_list = []
                for x in X_norm:
                    expvals = circuit(w, x)
                    if isinstance(expvals, (tuple, list)):
                        exp_arr = pnp.array(expvals)
                    else:
                        exp_arr = pnp.array([expvals])
                    if len(exp_arr) < self.n_classes:
                        pad = pnp.zeros(self.n_classes - len(exp_arr))
                        exp_arr = pnp.concatenate([exp_arr, pad])
                    preds_list.append(exp_arr[:self.n_classes] + b)
                preds = pnp.stack(preds_list)
                return pnp.mean((preds - pnp.array(y_onehot)) ** 2)

            opt = qml.NesterovMomentumOptimizer(stepsize=self.stepsize)
            weights = init_weights
            bias = init_bias

            for step in range(self.max_steps):
                (weights, bias), cost = opt.step_and_cost(cost_fn, weights, bias)

            self.weights = weights
            self.bias = bias

    def predict_proba(self, X):
        min_vals = np.min(X, axis=0)
        ptp_vals = np.ptp(X, axis=0) + 1e-8
        X_norm = np.pi * (X - min_vals) / ptp_vals - (np.pi / 2)
        circuit = self._build_circuit()

        if self.n_classes == 2:
            raw_out = np.array([float(circuit(self.weights, x)) + float(self.bias) for x in X_norm])
            p1 = 1.0 / (1.0 + np.exp(-raw_out))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        else:
            raw_out = []
            for x in X_norm:
                expvals = circuit(self.weights, x)
                if isinstance(expvals, (tuple, list)):
                    exp_arr = [float(v) for v in expvals]
                else:
                    exp_arr = [float(expvals)]
                if len(exp_arr) < self.n_classes:
                    exp_arr = exp_arr + [0.0] * (self.n_classes - len(exp_arr))
                raw_out.append(np.array(exp_arr[:self.n_classes]) + np.array(self.bias))
            raw_out = np.array(raw_out)
            exp_s = np.exp(raw_out - np.max(raw_out, axis=1, keepdims=True))
            return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def evaluate_classifier(y_true, y_pred, y_proba, classes):
    n_classes = len(classes)
    acc = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        pos_label = 1
        prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            roc_auc = 0.0

        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = auc(r_curve, p_curve)
        except Exception:
            pr_auc = 0.0

    else:
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        specs = []
        for i in range(n_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        spec = float(np.mean(specs))

        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            roc_auc = 0.0

        pr_aucs = []
        for i in range(n_classes):
            y_binary = (y_true == i).astype(int)
            try:
                p_curve, r_curve, _ = precision_recall_curve(y_binary, y_proba[:, i])
                pr_aucs.append(auc(r_curve, p_curve))
            except Exception:
                pr_aucs.append(0.0)
        pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    return {
        "Accuracy": float(acc),
        "Sensitivity": float(rec),
        "Specificity": float(spec),
        "Precision": float(prec),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc)
    }

def run_benchmark(df, feature_cols, target_col, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    processor = DataProcessor(
        imputer_strategy=config.get("imputer_strategy", "mean"),
        n_components=config.get("n_components", 4)
    )
    X_pca, y_encoded, classes = processor.fit_transform(df, feature_cols, target_col)

    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    n_classes = len(classes)
    results = {}
    models_fitted = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=config.get("rf_n_estimators", 100),
        random_state=random_state
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_metrics = evaluate_classifier(y_test, rf_pred, rf_proba, classes)
    rf_metrics["Training Time (s)"] = float(rf_time)
    results["Random Forest"] = rf_metrics
    models_fitted["Random Forest"] = rf

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=random_state)
    t0 = time.time()
    lr.fit(X_train, y_train)
    lr_time = time.time() - t0

    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)
    lr_metrics = evaluate_classifier(y_test, lr_pred, lr_proba, classes)
    lr_metrics["Training Time (s)"] = float(lr_time)
    results["Logistic Regression"] = lr_metrics
    models_fitted["Logistic Regression"] = lr

    # PennyLane VQC
    n_qubits = config.get("n_components", 4)
    n_layers = config.get("vqc_n_layers", 2)
    max_steps = config.get("vqc_max_steps", 20)
    vqc = PennyLaneVQC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        max_steps=max_steps
    )
    t0 = time.time()
    vqc.fit(X_train, y_train)
    vqc_time = time.time() - t0

    vqc_pred = vqc.predict(X_test)
    vqc_proba = vqc.predict_proba(X_test)
    vqc_metrics = evaluate_classifier(y_test, vqc_pred, vqc_proba, classes)
    vqc_metrics["Training Time (s)"] = float(vqc_time)
    results["Variational Quantum Classifier (VQC)"] = vqc_metrics
    models_fitted["VQC"] = vqc

    benchmark_df = pd.DataFrame(results).T
    benchmark_csv_path = os.path.join(output_dir, "benchmark_metrics.csv")
    benchmark_df.to_csv(benchmark_csv_path)

    loadings_df = processor.get_pca_loadings()
    loadings_csv_path = os.path.join(output_dir, "pca_loadings.csv")
    loadings_df.to_csv(loadings_csv_path)

    rf_importance = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(processor.n_components)],
        "RF_Importance": rf.feature_importances_
    }).set_index("Component")
    importance_csv_path = os.path.join(output_dir, "rf_feature_importance.csv")
    rf_importance.to_csv(importance_csv_path)

    # Plotting
    metrics_to_plot = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "ROC-AUC", "PR-AUC"]
    plot_df = benchmark_df[metrics_to_plot]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.T.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title("Model Comparison Benchmark")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(loadings_df.values, cmap='coolwarm', aspect='auto')
    fig.colorbar(cax, label='Loading Coefficient')
    ax.set_yticks(range(processor.n_components))
    ax.set_yticklabels(loadings_df.index)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax.set_title("PCA Component Loadings vs Original Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_loadings_heatmap.png"), dpi=300)
    plt.close()

    config_summary = {
        "target_column": target_col,
        "classes": classes,
        "n_samples": len(df),
        "n_features_input": len(feature_cols),
        "n_components_pca": processor.n_components,
        "vqc_layers": n_layers,
        "vqc_steps": max_steps,
        "output_dir": output_dir
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(config_summary, f, indent=4)

    return benchmark_df, loadings_df, rf_importance, output_dir
