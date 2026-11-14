import csv
import os
import numpy as np


def load_records(filepath):
    """Load all 800 rows from data/records.csv preserving file order."""
    records = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def sigmoid(z):
    """Logistic sigmoid activation: sigma(z) = 1 / (1 + e^-z)."""
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(X, y, w, b):
    """Binary cross-entropy loss: -(1/n) * sum(y * ln(y_hat) + (1 - y) * ln(1 - y_hat))."""
    y_hat = sigmoid(X @ w + b)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return float(-np.sum(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)) / len(y))


def train_logistic_regression(X, y, eta, epochs, record_epochs=()):
    """Batch gradient descent for logistic regression starting from w = 0 and b = 0."""
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    losses = {}
    if 0 in record_epochs:
        losses[0] = bce_loss(X, y, w, b)
    for epoch in range(1, epochs + 1):
        z = X @ w + b
        y_hat = sigmoid(z)
        error = y_hat - y
        grad_w = (X.T @ error) / n
        grad_b = float(np.sum(error) / n)
        w -= eta * grad_w
        b -= eta * grad_b
        if epoch in record_epochs:
            losses[epoch] = bce_loss(X, y, w, b)
    return w, b, losses


def compute_confusion_metrics(y_true, y_pred):
    """Compute tp, fp, tn, fn, accuracy, precision, recall, f1, specificity, tpr, fpr."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tpr": tpr,
        "fpr": fpr,
    }


def knn_predict(X_ref, y_ref, X_query, k):
    """K-nearest neighbours using Euclidean distance, stable sort tie-breaking, and majority vote >= (K + 1) / 2."""
    n_query = len(X_query)
    preds = np.zeros(n_query, dtype=int)
    min_positive_votes = (k + 1) // 2
    for i in range(n_query):
        diff = X_ref - X_query[i]
        euclidean_dist = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(euclidean_dist, kind="stable")
        k_nearest_labels = y_ref[order[:k]]
        preds[i] = 1 if int(np.sum(k_nearest_labels)) >= min_positive_votes else 0
    return preds


def main():
    data_path = os.path.join("data", "records.csv")
    rows = load_records(data_path)

    # Predictors X1 to X7 and binary outcome (escalated)
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

   
    idx_outcome_0 = [i for i, r in enumerate(rows) if int(r[target_col]) == 0]
    idx_outcome_1 = [i for i, r in enumerate(rows) if int(r[target_col]) == 1]

    n0_train = (7 * len(idx_outcome_0)) // 10
    n1_train = (7 * len(idx_outcome_1)) // 10

    train_idx = sorted(idx_outcome_0[:n0_train] + idx_outcome_1[:n1_train])
    test_idx = sorted(idx_outcome_0[n0_train:] + idx_outcome_1[n1_train:])

    X_train_raw = np.array([[float(rows[i][c]) for c in feature_cols] for i in train_idx], dtype=float)
    y_train = np.array([int(rows[i][target_col]) for i in train_idx], dtype=int)
    X_test_raw = np.array([[float(rows[i][c]) for c in feature_cols] for i in test_idx], dtype=float)
    y_test = np.array([int(rows[i][target_col]) for i in test_idx], dtype=int)

    train_rows = len(X_train_raw)
    train_positives = int(np.sum(y_train))
    test_rows = len(X_test_raw)
    test_positives = int(np.sum(y_test))

    # Standardise X1 to X7 using training mean and training sample SD (ddof=1)
    train_mean = np.sum(X_train_raw, axis=0) / train_rows
    train_sd = np.sqrt(np.sum((X_train_raw - train_mean) ** 2, axis=0) / (train_rows - 1))

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd
    
    w_final, b_final, losses_22 = train_logistic_regression(
        X_train, y_train, eta=0.1, epochs=1000, record_epochs=(0, 1, 10, 100, 1000)
    )
    loss_epoch_0 = losses_22[0]
    loss_epoch_1 = losses_22[1]
    loss_epoch_10 = losses_22[10]
    loss_epoch_100 = losses_22[100]
    loss_epoch_1000 = losses_22[1000]
    test_bce = bce_loss(X_test, y_test, w_final, b_final)

    _, _, l_001 = train_logistic_regression(X_train, y_train, eta=0.01, epochs=200, record_epochs=(200,))
    _, _, l_01 = train_logistic_regression(X_train, y_train, eta=0.1, epochs=200, record_epochs=(200,))
    _, _, l_10 = train_logistic_regression(X_train, y_train, eta=1.0, epochs=200, record_epochs=(200,))
    loss_epoch_200_eta_001 = l_001[200]
    loss_epoch_200_eta_01 = l_01[200]
    loss_epoch_200_eta_10 = l_10[200]


    odds_ratios = np.exp(w_final)

    p_test = sigmoid(X_test @ w_final + b_final)
    metrics_tau_05 = compute_confusion_metrics(y_test, (p_test >= 0.5).astype(int))
    metrics_tau_03 = compute_confusion_metrics(y_test, (p_test >= 0.3).astype(int))
    blind_accuracy = float(np.sum(y_test == 0) / test_rows)


    os.makedirs("outputs", exist_ok=True)
    thresholds_path = os.path.join("outputs", "part2_thresholds.csv")
    with open(thresholds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        for step in range(1, 10):
            tau = round(step * 0.1, 1)
            m = compute_confusion_metrics(y_test, (p_test >= tau).astype(int))
            writer.writerow([
                f"{tau:.1f}",
                m["tp"],
                m["fp"],
                m["tn"],
                m["fn"],
                f"{m['tpr']:.6f}",
                f"{m['fpr']:.6f}",
            ])

   
    k_values = list(range(1, 22, 2))  # Odd K in {1, 3, 5, ..., 21}
    folds = np.arange(train_rows) % 5  # 0-based position in training set modulo 5
    cv_results = []
    knn_best_k = None
    best_mean_cv_f1 = -1.0

    for k in k_values:
        fold_f1_scores = []
        for fold_id in range(5):
            val_mask = (folds == fold_id)
            ref_mask = ~val_mask
            val_preds = knn_predict(X_train[ref_mask], y_train[ref_mask], X_train[val_mask], k)
            fold_m = compute_confusion_metrics(y_train[val_mask], val_preds)
            fold_f1_scores.append(fold_m["f1"])
        mean_cv_f1 = float(np.sum(fold_f1_scores) / 5.0)
        cv_results.append((k, mean_cv_f1))
        # Strict inequality ensures ties go to the smaller K
        if mean_cv_f1 > best_mean_cv_f1:
            best_mean_cv_f1 = mean_cv_f1
            knn_best_k = k

    knn_cv_path = os.path.join("outputs", "part2_knn_cv.csv")
    with open(knn_cv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        for k, mean_cv_f1 in cv_results:
            writer.writerow([k, f"{mean_cv_f1:.6f}"])

    # Refit with all 560 training rows as reference and evaluate on test set
    knn_test_preds = knn_predict(X_train, y_train, X_test, knn_best_k)
    knn_metrics = compute_confusion_metrics(y_test, knn_test_preds)


    p_train = sigmoid(X_train @ w_final + b_final)
    cost_tau = None
    min_train_cost = float("inf")
    for step in range(1, 100):
        tau = round(step * 0.01, 2)
        pred_tr = (p_train >= tau).astype(int)
        fp_tr = int(np.sum((y_train == 0) & (pred_tr == 1)))
        fn_tr = int(np.sum((y_train == 1) & (pred_tr == 0)))
        cost_tr = 50 * fp_tr + 400 * fn_tr
        if cost_tr < min_train_cost:
            min_train_cost = cost_tr
            cost_tau = tau

    pred_test_at_tau = (p_test >= cost_tau).astype(int)
    fp_test_at_tau = int(np.sum((y_test == 0) & (pred_test_at_tau == 1)))
    fn_test_at_tau = int(np.sum((y_test == 1) & (pred_test_at_tau == 0)))
    cost_train_at_tau = min_train_cost
    cost_test_at_tau = 50 * fp_test_at_tau + 400 * fn_test_at_tau

    fp_test_05 = metrics_tau_05["fp"]
    fn_test_05 = metrics_tau_05["fn"]
    cost_test_at_05 = 50 * fp_test_05 + 400 * fn_test_05
    cost_test_flag_nobody = 400 * test_positives

    out_path = os.path.join("outputs", "part2_results.txt")
    lines = [
        f"train_rows: {train_rows}",
        f"train_positives: {train_positives}",
        f"test_rows: {test_rows}",
        f"test_positives: {test_positives}",
        f"loss_epoch_0: {loss_epoch_0:.6f}",
        f"loss_epoch_1: {loss_epoch_1:.6f}",
        f"loss_epoch_10: {loss_epoch_10:.6f}",
        f"loss_epoch_100: {loss_epoch_100:.6f}",
        f"loss_epoch_1000: {loss_epoch_1000:.6f}",
        f"weight_X1: {w_final[0]:.6f}",
        f"weight_X2: {w_final[1]:.6f}",
        f"weight_X3: {w_final[2]:.6f}",
        f"weight_X4: {w_final[3]:.6f}",
        f"weight_X5: {w_final[4]:.6f}",
        f"weight_X6: {w_final[5]:.6f}",
        f"weight_X7: {w_final[6]:.6f}",
        f"bias: {b_final:.6f}",
        f"test_bce: {test_bce:.6f}",
        f"loss_epoch_200_eta_0.01: {loss_epoch_200_eta_001:.6f}",
        f"loss_epoch_200_eta_0.1: {loss_epoch_200_eta_01:.6f}",
        f"loss_epoch_200_eta_1.0: {loss_epoch_200_eta_10:.6f}",
        f"odds_ratio_X1: {odds_ratios[0]:.6f}",
        f"odds_ratio_X2: {odds_ratios[1]:.6f}",
        f"odds_ratio_X3: {odds_ratios[2]:.6f}",
        f"odds_ratio_X4: {odds_ratios[3]:.6f}",
        f"odds_ratio_X5: {odds_ratios[4]:.6f}",
        f"odds_ratio_X6: {odds_ratios[5]:.6f}",
        f"odds_ratio_X7: {odds_ratios[6]:.6f}",
        f"tau_0.5_tp: {metrics_tau_05['tp']}",
        f"tau_0.5_fp: {metrics_tau_05['fp']}",
        f"tau_0.5_tn: {metrics_tau_05['tn']}",
        f"tau_0.5_fn: {metrics_tau_05['fn']}",
        f"tau_0.5_accuracy: {metrics_tau_05['accuracy']:.6f}",
        f"tau_0.5_precision: {metrics_tau_05['precision']:.6f}",
        f"tau_0.5_recall: {metrics_tau_05['recall']:.6f}",
        f"tau_0.5_f1: {metrics_tau_05['f1']:.6f}",
        f"tau_0.5_specificity: {metrics_tau_05['specificity']:.6f}",
        f"tau_0.3_tp: {metrics_tau_03['tp']}",
        f"tau_0.3_fp: {metrics_tau_03['fp']}",
        f"tau_0.3_tn: {metrics_tau_03['tn']}",
        f"tau_0.3_fn: {metrics_tau_03['fn']}",
        f"tau_0.3_accuracy: {metrics_tau_03['accuracy']:.6f}",
        f"tau_0.3_precision: {metrics_tau_03['precision']:.6f}",
        f"tau_0.3_recall: {metrics_tau_03['recall']:.6f}",
        f"tau_0.3_f1: {metrics_tau_03['f1']:.6f}",
        f"tau_0.3_specificity: {metrics_tau_03['specificity']:.6f}",
        f"blind_accuracy: {blind_accuracy:.6f}",
        f"knn_best_k: {knn_best_k}",
        f"knn_tp: {knn_metrics['tp']}",
        f"knn_fp: {knn_metrics['fp']}",
        f"knn_tn: {knn_metrics['tn']}",
        f"knn_fn: {knn_metrics['fn']}",
        f"knn_accuracy: {knn_metrics['accuracy']:.6f}",
        f"knn_precision: {knn_metrics['precision']:.6f}",
        f"knn_recall: {knn_metrics['recall']:.6f}",
        f"knn_f1: {knn_metrics['f1']:.6f}",
        f"cost_tau: {cost_tau}",
        f"cost_train_at_tau: {cost_train_at_tau}",
        f"cost_test_at_tau: {cost_test_at_tau}",
        f"cost_test_at_0.5: {cost_test_at_05}",
        f"cost_test_flag_nobody: {cost_test_flag_nobody}",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Successfully wrote", out_path)


if __name__ == "__main__":
    main()
