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


def pca_from_scratch(X_std):
    """
    Computing PCA via sample covariance matrix eigendecomposition:
    Sigma = (1 / (n - 1)) * X_std^T @ X_std
    """
    n, d = X_std.shape
    cov = (X_std.T @ X_std) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort in descending order
    order = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[order]
    sorted_evecs = eigenvectors[:, order]

    total_variance = np.sum(sorted_evals)
    explained_ratios = sorted_evals / total_variance
    cumulative_ratios = np.cumsum(explained_ratios)

    # Project to principal components
    projections = X_std @ sorted_evecs

    return sorted_evals, sorted_evecs, explained_ratios, cumulative_ratios, projections



def kmeans_fit(X_std, k, max_iter=100, seed=42):
    """
    K-Means clustering with deterministic spread initialization.
    Minimizes Within-Cluster Sum of Squares (WCSS).
    """
    np.random.seed(seed)
    n, d = X_std.shape

    # Deterministic initialization: spread across dataset indices
    init_indices = [int(i * (n // k)) for i in range(k)]
    centroids = X_std[init_indices].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Compute squared Euclidean distance to each centroid
        distances = np.zeros((n, k))
        for c in range(k):
            diff = X_std - centroids[c]
            distances[:, c] = np.sum(diff * diff, axis=1)

        new_labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = X_std[new_labels == c]
            if len(members) > 0:
                new_centroids[c] = np.mean(members, axis=0)
            else:
                new_centroids[c] = centroids[c]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            centroids = new_centroids
            labels = new_labels
            break

        centroids = new_centroids
        labels = new_labels

    # Calculate final WCSS (inertia)
    wcss = 0.0
    for c in range(k):
        members = X_std[labels == c]
        if len(members) > 0:
            diff = members - centroids[c]
            wcss += float(np.sum(diff * diff))

    return labels, centroids, wcss




def hierarchical_clustering_average_linkage(X_std, target_clusters=4):
    """
    Bottom-up agglomerative hierarchical clustering using Average Linkage.
    Returns full merge history and cluster labels at target_clusters.
    """
    n = len(X_std)

    # Initial pairwise Euclidean distance matrix
    sq_norms = np.sum(X_std ** 2, axis=1, keepdims=True)
    dist_sq = sq_norms - 2.0 * (X_std @ X_std.T) + sq_norms.T
    dist_sq = np.maximum(dist_sq, 0.0)
    dist_matrix = np.sqrt(dist_sq)
    np.fill_diagonal(dist_matrix, np.inf)

    clusters = {i: [i] for i in range(n)}
    active_ids = list(range(n))

    merge_history = []
    next_id = n
    flat_labels = np.zeros(n, dtype=int)
    total_merges = n - 1
    stop_step = n - target_clusters

    for step in range(total_merges):
        min_idx = int(np.argmin(dist_matrix))
        c_i, c_j = np.unravel_index(min_idx, dist_matrix.shape)

        if c_i > c_j:
            c_i, c_j = c_j, c_i

        id_a = active_ids[c_i]
        id_b = active_ids[c_j]
        merge_dist = float(dist_matrix[c_i, c_j])

        size_a = len(clusters[id_a])
        size_b = len(clusters[id_b])
        new_size = size_a + size_b

        merged_members = clusters[id_a] + clusters[id_b]
        new_cluster_id = next_id
        next_id += 1

        clusters[new_cluster_id] = merged_members
        del clusters[id_a]
        del clusters[id_b]

        merge_history.append({
            "step": step + 1,
            "cluster_1": id_a,
            "cluster_2": id_b,
            "distance": merge_dist,
            "cluster_size": new_size,
        })

        if (step + 1) == stop_step:
            active_now = [cid for idx, cid in enumerate(active_ids) if idx not in (c_i, c_j) and cid != -1]
            active_now.append(new_cluster_id)
            for cl_idx, cid in enumerate(active_now):
                for member in clusters[cid]:
                    flat_labels[member] = cl_idx

        # Update distance matrix (Average Linkage formula)
        dist_row_a = dist_matrix[c_i].copy()
        dist_row_b = dist_matrix[c_j].copy()
        new_dists = (size_a * dist_row_a + size_b * dist_row_b) / new_size

        dist_matrix[c_i, :] = new_dists
        dist_matrix[:, c_i] = new_dists
        dist_matrix[c_i, c_i] = np.inf

        dist_matrix[c_j, :] = np.inf
        dist_matrix[:, c_j] = np.inf

        active_ids[c_i] = new_cluster_id
        active_ids[c_j] = -1

    return merge_history, flat_labels





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

    n = len(rows)
    d = len(feature_cols)

    X_raw = np.array([[float(r[col]) for col in feature_cols] for r in rows], dtype=float)
    record_ids = [r["record_id"] for r in rows]
    escalated_arr = np.array([int(r["escalated"]) for r in rows], dtype=int)
    resolution_arr = np.array([float(r["resolution_minutes"]) for r in rows], dtype=float)

    # Standardisation with sample SD (ddof=1)
    mean_vec = np.mean(X_raw, axis=0)
    sd_vec = np.std(X_raw, axis=0, ddof=1)
    sd_vec = np.where(sd_vec == 0, 1.0, sd_vec)
    X_std = (X_raw - mean_vec) / sd_vec

    # 1. PCA from Scratch
    evals, evecs, exp_ratios, cum_ratios, pc_projections = pca_from_scratch(X_std)

    # 2. K-Means Across K in {2, 3, 4, 5, 6}
    k_values = [2, 3, 4, 5, 6]
    wcss_dict = {}
    models_dict = {}
    for k in k_values:
        lbls, cents, wcss = kmeans_fit(X_std, k=k, seed=42)
        wcss_dict[k] = wcss
        models_dict[k] = (lbls, cents)

    best_k = 4
    best_labels, best_centroids = models_dict[best_k]

    # 3. Export outputs/part3_segments.csv
    os.makedirs("outputs", exist_ok=True)
    segments_path = os.path.join("outputs", "part3_segments.csv")
    with open(segments_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "record_id",
            "cluster",
            "customer_tenure_months",
            "earlier_ticket_count_90d",
            "sla_consumed_ratio",
            "account_value_inr_thousands",
            "satisfaction_score",
            "reopened_flag",
            "self_service_first_flag",
            "escalated",
            "resolution_minutes",
            "pc1",
            "pc2",
        ]
        writer.writerow(header)
        for i in range(n):
            writer.writerow([
                record_ids[i],
                best_labels[i],
                f"{X_raw[i, 0]:.1f}",
                int(X_raw[i, 1]),
                f"{X_raw[i, 2]:.3f}",
                f"{X_raw[i, 3]:.1f}",
                f"{X_raw[i, 4]:.1f}",
                int(X_raw[i, 5]),
                int(X_raw[i, 6]),
                escalated_arr[i],
                f"{resolution_arr[i]:.2f}",
                f"{pc_projections[i, 0]:.6f}",
                f"{pc_projections[i, 1]:.6f}",
            ])
    print("Successfully wrote", segments_path)

    # 4. Hierarchical Clustering
    merge_history, hier_labels = hierarchical_clustering_average_linkage(X_std, target_clusters=4)

    # Export outputs/part3_hierarchical.csv
    hier_path = os.path.join("outputs", "part3_hierarchical.csv")
    with open(hier_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cluster_1", "cluster_2", "distance", "cluster_size"])
        for m in merge_history:
            writer.writerow([
                m["step"],
                m["cluster_1"],
                m["cluster_2"],
                f"{m['distance']:.6f}",
                m["cluster_size"],
            ])
    print("Successfully wrote", hier_path)

    # 5. Export outputs/part3_results.txt
    results_path = os.path.join("outputs", "part3_results.txt")
    lines = [
        f"n_samples: {n}",
        f"n_features: {d}",
    ]

    for idx in range(d):
        lines.append(f"pca_eigenvalue_{idx+1}: {evals[idx]:.6f}")

    for idx in range(d):
        lines.append(f"pca_explained_ratio_{idx+1}: {exp_ratios[idx]:.6f}")

    lines.append(f"pca_cumulative_ratio_2: {cum_ratios[1]:.6f}")

    for f_idx in range(d):
        lines.append(f"pc1_loading_X{f_idx+1}: {evecs[f_idx, 0]:.6f}")
    for f_idx in range(d):
        lines.append(f"pc2_loading_X{f_idx+1}: {evecs[f_idx, 1]:.6f}")

    for k in k_values:
        lines.append(f"kmeans_wcss_k{k}: {wcss_dict[k]:.6f}")

    lines.append(f"kmeans_best_k: {best_k}")

    for c in range(best_k):
        c_count = int(np.sum(best_labels == c))
        c_esc_rate = float(np.mean(escalated_arr[best_labels == c]))
        lines.append(f"cluster_{c}_count: {c_count}")
        lines.append(f"cluster_{c}_escalation_rate: {c_esc_rate:.6f}")

    lines.extend([
        f"hierarchical_initial_dist: {merge_history[0]['distance']:.6f}",
        f"hierarchical_final_dist: {merge_history[-1]['distance']:.6f}",
        f"hierarchical_merges: {len(merge_history)}",
    ])

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Successfully wrote", results_path)


if __name__ == "__main__":
    main()
