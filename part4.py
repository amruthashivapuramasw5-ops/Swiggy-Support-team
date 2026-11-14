import csv
import os
import numpy as np


def load_records(filepath):
    """Load all rows from data/records.csv preserving file order."""
    records = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -250.0, 250.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_gini_impurity(y_subset):
    """Gini Impurity: G = 1 - (p0^2 + p1^2)."""
    n = len(y_subset)
    if n == 0:
        return 0.0
    p1 = float(np.sum(y_subset == 1) / n)
    p0 = 1.0 - p1
    return 1.0 - (p0 * p0 + p1 * p1)


def evaluate_candidate_splits(X, y, feature_cols):
    """Evaluate candidate splits across all features to maximize Gini Gain."""
    n, d = X.shape
    base_gini = compute_gini_impurity(y)
    split_records = []

    for f_idx, feat in enumerate(feature_cols):
        vals = X[:, f_idx]
        percentiles = np.percentile(vals, np.linspace(10, 90, 17))
        thresholds = np.unique(percentiles)

        for th in thresholds:
            left_mask = vals <= th
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            g_left = compute_gini_impurity(y_left)
            g_right = compute_gini_impurity(y_right)

            w_gini = (len(y_left) / n) * g_left + (len(y_right) / n) * g_right
            gain = base_gini - w_gini

            split_records.append({
                "feature": feat,
                "threshold": float(th),
                "samples_left": int(len(y_left)),
                "samples_right": int(len(y_right)),
                "gini_left": g_left,
                "gini_right": g_right,
                "weighted_gini": w_gini,
                "gini_gain": gain,
            })

    split_records.sort(key=lambda s: s["gini_gain"], reverse=True)
    return split_records, base_gini




class DecisionStump:
    """1-level decision tree for binary classification with labels in {-1, +1}."""
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # 1: x <= th -> +1; -1: x <= th -> -1
        self.alpha = 0.0

    def predict(self, X):
        n = X.shape[0]
        preds = np.ones(n)
        col = X[:, self.feature_idx]
        if self.polarity == 1:
            preds[col > self.threshold] = -1
        else:
            preds[col <= self.threshold] = -1
        return preds


class AdaBoostClassifierFromScratch:
    """AdaBoost with Decision Stumps as weak learners."""
    def __init__(self, n_estimators=15):
        self.n_estimators = n_estimators
        self.estimators = []
        self.iteration_log = []

    def fit(self, X_train, y_train, X_test, y_test, feature_cols):
        n_train, n_features = X_train.shape
        y_tr = np.where(y_train == 1, 1, -1)
        y_te = np.where(y_test == 1, 1, -1)

        weights = np.full(n_train, 1.0 / n_train, dtype=float)
        self.estimators = []
        self.iteration_log = []

        for t in range(self.n_estimators):
            stump = DecisionStump()
            min_err = float("inf")
            best_preds = None

            for f_idx in range(n_features):
                vals = X_train[:, f_idx]
                thresholds = np.unique(np.percentile(vals, np.linspace(5, 95, 19)))

                for th in thresholds:
                    for pol in [1, -1]:
                        preds = np.ones(n_train)
                        if pol == 1:
                            preds[vals > th] = -1
                        else:
                            preds[vals <= th] = -1

                        mis = (preds != y_tr).astype(float)
                        err = float(np.sum(weights * mis))

                        if err < min_err:
                            min_err = err
                            stump.feature_idx = f_idx
                            stump.threshold = th
                            stump.polarity = pol
                            best_preds = preds

            eps = 1e-10
            min_err = np.clip(min_err, eps, 1.0 - eps)
            alpha = 0.5 * np.log((1.0 - min_err) / min_err)
            stump.alpha = alpha

            weights *= np.exp(-alpha * y_tr * best_preds)
            weights /= np.sum(weights)

            self.estimators.append(stump)

            # Cumulative ensemble predictions
            cum_tr_pred = self.predict_up_to(X_train, t + 1)
            cum_te_pred = self.predict_up_to(X_test, t + 1)

            cum_tr_acc = float(np.mean(cum_tr_pred == y_train))
            cum_te_acc = float(np.mean(cum_te_pred == y_test))

            self.iteration_log.append({
                "round": t + 1,
                "feature": feature_cols[stump.feature_idx],
                "threshold": stump.threshold,
                "polarity": stump.polarity,
                "weighted_error": min_err,
                "alpha": alpha,
                "cum_train_accuracy": cum_tr_acc,
                "cum_test_accuracy": cum_te_acc,
            })

    def predict_up_to(self, X, num_estimators):
        scores = np.zeros(len(X))
        for idx in range(num_estimators):
            est = self.estimators[idx]
            scores += est.alpha * est.predict(X)
        return (scores >= 0).astype(int)

    def predict(self, X):
        return self.predict_up_to(X, len(self.estimators))

    def predict_proba(self, X):
        scores = np.zeros(len(X))
        for est in self.estimators:
            scores += est.alpha * est.predict(X)
        return sigmoid(scores)


def train_simple_logistic(X, y, eta=0.1, epochs=500):
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        err = p - y
        w -= eta * (X.T @ err) / n
        b -= eta * float(np.sum(err) / n)
    return w, b


def knn_predict_and_proba(X_ref, y_ref, X_query, k=5):
    n_query = len(X_query)
    preds = np.zeros(n_query, dtype=int)
    probs = np.zeros(n_query, dtype=float)
    for i in range(n_query):
        diff = X_ref - X_query[i]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(dists, kind="stable")
        nearest_labels = y_ref[order[:k]]
        p1 = float(np.mean(nearest_labels == 1))
        probs[i] = p1
        preds[i] = 1 if p1 >= 0.5 else 0
    return preds, probs


def run_voting_ensemble(X_train, y_train, X_test, y_test, candidate_splits):
    # Model 1: Logistic Regression
    w_lr, b_lr = train_simple_logistic(X_train, y_train, eta=0.1, epochs=500)
    p_lr = sigmoid(X_test @ w_lr + b_lr)
    y_lr = (p_lr >= 0.5).astype(int)

    # Model 2: KNN (k=5)
    y_knn, p_knn = knn_predict_and_proba(X_train, y_train, X_test, k=5)

    # Model 3: Best Decision Stump from candidate splits
    best_split = candidate_splits[0]
    best_f_idx = ["customer_tenure_months", "earlier_ticket_count_90d", "sla_consumed_ratio",
                  "account_value_inr_thousands", "satisfaction_score", "reopened_flag",
                  "self_service_first_flag"].index(best_split["feature"])
    y_dt = (X_test[:, best_f_idx] > best_split["threshold"]).astype(int)
    p_dt = np.where(y_dt == 1, 0.70, 0.30)

    # Ensemble 1: Hard Voting (Majority rule: sum >= 2)
    stacked = np.vstack([y_lr, y_knn, y_dt])
    y_hard = (np.sum(stacked, axis=0) >= 2).astype(int)
    p_hard = np.mean(stacked, axis=0)

    # Ensemble 2: Soft Voting (Average probabilities)
    p_soft = (p_lr + p_knn + p_dt) / 3.0
    y_soft = (p_soft >= 0.5).astype(int)

    models = [
        ("Logistic Regression", y_lr, p_lr),
        ("KNN (k=5)", y_knn, p_knn),
        ("Decision Stump", y_dt, p_dt),
        ("Hard Voting Ensemble", y_hard, p_hard),
        ("Soft Voting Ensemble", y_soft, p_soft),
    ]

    voting_results = []
    for name, pred, prob in models:
        tp = int(np.sum((y_test == 1) & (pred == 1)))
        fp = int(np.sum((y_test == 0) & (pred == 1)))
        tn = int(np.sum((y_test == 0) & (pred == 0)))
        fn = int(np.sum((y_test == 1) & (pred == 0)))

        acc = (tp + tn) / len(y_test)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        brier = float(np.mean((prob - y_test) ** 2))

        voting_results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "brier_score": brier,
        })

    return voting_results


class MultiLayerPerceptronFromScratch:
    """
    3-Layer Neural Network: Input(7) -> Hidden1(16, ReLU) -> Hidden2(8, ReLU) -> Output(1, Sigmoid)
    Trained with forward and backward backpropagation using NumPy only.
    """
    def __init__(self, input_dim=7, h1=16, h2=8, out=1, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, h1))

        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))

        self.W3 = np.random.randn(h2, out) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, out))

        self.loss_history = []

    def relu(self, Z):
        return np.maximum(0.0, Z)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        return self.A3

    def compute_loss(self, y, p):
        eps = 1e-15
        p_c = np.clip(p, eps, 1.0 - eps)
        y_col = y.reshape(-1, 1)
        return float(-np.mean(y_col * np.log(p_c) + (1.0 - y_col) * np.log(1.0 - p_c)))

    def backward(self, X, y, lr=0.05):
        m = X.shape[0]
        y_col = y.reshape(-1, 1)

        dZ3 = self.A3 - y_col
        dW3 = (self.A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (self.Z2 > 0).astype(float)
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0).astype(float)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X_train, y_train, epochs=300, lr=0.05):
        self.loss_history = []
        for _ in range(epochs):
            A3 = self.forward(X_train)
            loss = self.compute_loss(y_train, A3)
            self.loss_history.append(loss)
            self.backward(X_train, y_train, lr=lr)

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict(self, X, tau=0.5):
        return (self.predict_proba(X) >= tau).astype(int)


def generate_architecture_specs(input_dim=7):
    h1 = 16
    h2 = 8
    out = 1

    layers = [
        {"layer_id": 1, "layer_name": "Input Layer", "layer_type": "Input", "input_dim": input_dim, "output_dim": input_dim, "activation": "None", "weight_params": 0, "bias_params": 0, "total_params": 0},
        {"layer_id": 2, "layer_name": "Dense_Hidden_1", "layer_type": "Dense", "input_dim": input_dim, "output_dim": h1, "activation": "ReLU", "weight_params": input_dim * h1, "bias_params": h1, "total_params": input_dim * h1 + h1},
        {"layer_id": 3, "layer_name": "Dense_Hidden_2", "layer_type": "Dense", "input_dim": h1, "output_dim": h2, "activation": "ReLU", "weight_params": h1 * h2, "bias_params": h2, "total_params": h1 * h2 + h2},
        {"layer_id": 4, "layer_name": "Output_Classification", "layer_type": "Dense", "input_dim": h2, "output_dim": out, "activation": "Sigmoid", "weight_params": h2 * out, "bias_params": out, "total_params": h2 * out + out},
    ]

    total_w = sum(l["weight_params"] for l in layers)
    total_b = sum(l["bias_params"] for l in layers)
    total_p = sum(l["total_params"] for l in layers)

    summary = {
        "layer_id": "TOTAL",
        "layer_name": "Full Multi-Layer Perceptron",
        "layer_type": "Feedforward Network",
        "input_dim": input_dim,
        "output_dim": out,
        "activation": "ReLU -> Sigmoid",
        "weight_params": total_w,
        "bias_params": total_b,
        "total_params": total_p,
    }
    return layers, summary



def main():
    data_path = os.path.join("data", "records.csv")
    rows = load_records(data_path)

    feature_cols = [
        "customer_tenure_months",       # X1
        "earlier_ticket_count_90d",     # X2
        "sla_consumed_ratio",           # X3
        "account_value_inr_thousands",  # X4
        "satisfaction_score",           # X5
        "reopened_flag",                # X6
        "self_service_first_flag",      # X7
    ]
    target_col = "escalated"

    # Stratified 70/30 split consistent with Part 2
    idx0 = [i for i, r in enumerate(rows) if int(r[target_col]) == 0]
    idx1 = [i for i, r in enumerate(rows) if int(r[target_col]) == 1]

    n0_train = (7 * len(idx0)) // 10
    n1_train = (7 * len(idx1)) // 10

    train_idx = sorted(idx0[:n0_train] + idx1[:n1_train])
    test_idx = sorted(idx0[n0_train:] + idx1[n1_train:])

    X_train_raw = np.array([[float(rows[i][c]) for c in feature_cols] for i in train_idx], dtype=float)
    y_train = np.array([int(rows[i][target_col]) for i in train_idx], dtype=int)
    X_test_raw = np.array([[float(rows[i][c]) for c in feature_cols] for i in test_idx], dtype=float)
    y_test = np.array([int(rows[i][target_col]) for i in test_idx], dtype=int)

    # Standardisation with sample SD (ddof=1)
    mean_tr = np.mean(X_train_raw, axis=0)
    sd_tr = np.std(X_train_raw, axis=0, ddof=1)
    sd_tr = np.where(sd_tr == 0, 1.0, sd_tr)

    X_train = (X_train_raw - mean_tr) / sd_tr
    X_test = (X_test_raw - mean_tr) / sd_tr

    # 1. Evaluate Decision Tree Candidate Splits
    candidate_splits, base_gini = evaluate_candidate_splits(X_train, y_train, feature_cols)

    os.makedirs("outputs", exist_ok=True)
    splits_path = os.path.join("outputs", "part4_splits.csv")
    with open(splits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature", "threshold", "samples_left", "samples_right", "gini_left", "gini_right", "weighted_gini", "gini_gain"])
        for idx, sp in enumerate(candidate_splits[:25], 1):
            writer.writerow([
                idx,
                sp["feature"],
                f"{sp['threshold']:.6f}",
                sp["samples_left"],
                sp["samples_right"],
                f"{sp['gini_left']:.6f}",
                f"{sp['gini_right']:.6f}",
                f"{sp['weighted_gini']:.6f}",
                f"{sp['gini_gain']:.6f}",
            ])
    print("Successfully wrote", splits_path)

    # 2. AdaBoost Classifier (15 rounds)
    adaboost = AdaBoostClassifierFromScratch(n_estimators=15)
    adaboost.fit(X_train, y_train, X_test, y_test, feature_cols)

    adaboost_path = os.path.join("outputs", "part4_adaboost.csv")
    with open(adaboost_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "feature", "threshold", "polarity", "weighted_error", "alpha", "cum_train_accuracy", "cum_test_accuracy"])
        for row in adaboost.iteration_log:
            writer.writerow([
                row["round"],
                row["feature"],
                f"{row['threshold']:.6f}",
                row["polarity"],
                f"{row['weighted_error']:.6f}",
                f"{row['alpha']:.6f}",
                f"{row['cum_train_accuracy']:.6f}",
                f"{row['cum_test_accuracy']:.6f}",
            ])
    print("Successfully wrote", adaboost_path)

    # 3. Voting Ensemble
    voting_metrics = run_voting_ensemble(X_train, y_train, X_test, y_test, candidate_splits)

    voting_path = os.path.join("outputs", "part4_voting.csv")
    with open(voting_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "specificity", "f1_score", "brier_score"])
        for vm in voting_metrics:
            writer.writerow([
                vm["model"],
                f"{vm['accuracy']:.6f}",
                f"{vm['precision']:.6f}",
                f"{vm['recall']:.6f}",
                f"{vm['specificity']:.6f}",
                f"{vm['f1_score']:.6f}",
                f"{vm['brier_score']:.6f}",
            ])
    print("Successfully wrote", voting_path)

    # 4. Neural Network Architecture Specification & Training
    layers_spec, summary_spec = generate_architecture_specs(input_dim=len(feature_cols))

    arch_path = os.path.join("outputs", "part4_architecture.csv")
    with open(arch_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_id", "layer_name", "layer_type", "input_dim", "output_dim", "activation", "weight_params", "bias_params", "total_params"])
        for l in layers_spec:
            writer.writerow([l["layer_id"], l["layer_name"], l["layer_type"], l["input_dim"], l["output_dim"], l["activation"], l["weight_params"], l["bias_params"], l["total_params"]])
        writer.writerow([summary_spec["layer_id"], summary_spec["layer_name"], summary_spec["layer_type"], summary_spec["input_dim"], summary_spec["output_dim"], summary_spec["activation"], summary_spec["weight_params"], summary_spec["bias_params"], summary_spec["total_params"]])
    print("Successfully wrote", arch_path)

    # Train MLP
    mlp = MultiLayerPerceptronFromScratch(input_dim=7, h1=16, h2=8, out=1, seed=42)
    mlp.train(X_train, y_train, epochs=300, lr=0.05)

    mlp_test_preds = mlp.predict(X_test, tau=0.5)
    tp = int(np.sum((y_test == 1) & (mlp_test_preds == 1)))
    fp = int(np.sum((y_test == 0) & (mlp_test_preds == 1)))
    tn = int(np.sum((y_test == 0) & (mlp_test_preds == 0)))
    fn = int(np.sum((y_test == 1) & (mlp_test_preds == 0)))

    mlp_acc = (tp + tn) / len(y_test)
    mlp_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mlp_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mlp_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mlp_f1 = (2.0 * mlp_prec * mlp_rec / (mlp_prec + mlp_rec)) if (mlp_prec + mlp_rec) > 0 else 0.0

    # 5. Export outputs/part4_results.txt
    results_path = os.path.join("outputs", "part4_results.txt")
    lines = [
        f"train_samples: {len(y_train)}",
        f"test_samples: {len(y_test)}",
        f"base_gini_impurity: {base_gini:.6f}",
        f"best_split_feature: {candidate_splits[0]['feature']}",
        f"best_split_threshold: {candidate_splits[0]['threshold']:.6f}",
        f"best_split_gini_gain: {candidate_splits[0]['gini_gain']:.6f}",
        f"adaboost_rounds: {len(adaboost.iteration_log)}",
        f"adaboost_final_train_acc: {adaboost.iteration_log[-1]['cum_train_accuracy']:.6f}",
        f"adaboost_final_test_acc: {adaboost.iteration_log[-1]['cum_test_accuracy']:.6f}",
        f"voting_hard_accuracy: {voting_metrics[3]['accuracy']:.6f}",
        f"voting_hard_f1: {voting_metrics[3]['f1_score']:.6f}",
        f"voting_soft_accuracy: {voting_metrics[4]['accuracy']:.6f}",
        f"voting_soft_f1: {voting_metrics[4]['f1_score']:.6f}",
        f"mlp_total_params: {summary_spec['total_params']}",
        f"mlp_initial_loss: {mlp.loss_history[0]:.6f}",
        f"mlp_final_loss: {mlp.loss_history[-1]:.6f}",
        f"mlp_test_accuracy: {mlp_acc:.6f}",
        f"mlp_test_precision: {mlp_prec:.6f}",
        f"mlp_test_recall: {mlp_rec:.6f}",
        f"mlp_test_specificity: {mlp_spec:.6f}",
        f"mlp_test_f1: {mlp_f1:.6f}",
    ]

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Successfully wrote", results_path)


if __name__ == "__main__":
    main()
