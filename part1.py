import csv
import math
import os
import numpy as np


def load_records(filepath):
    records = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def sample_mean(arr):
    return float(np.sum(arr) / len(arr))


def sample_var(arr, mean_val):
    return float(np.sum((arr - mean_val) ** 2) / (len(arr) - 1))


def sample_sd(arr, mean_val=None):
    if mean_val is None:
        mean_val = sample_mean(arr)
    return math.sqrt(sample_var(arr, mean_val))


def main():
    data_path = os.path.join("data", "records.csv")
    rows = load_records(data_path)

    x1 = np.array([float(r["customer_tenure_months"]) for r in rows])
    x2 = np.array([float(r["earlier_ticket_count_90d"]) for r in rows])
    x3 = np.array([float(r["sla_consumed_ratio"]) for r in rows])
    x4 = np.array([float(r["account_value_inr_thousands"]) for r in rows])
    x5 = np.array([float(r["satisfaction_score"]) for r in rows])
    x6 = np.array([float(r["reopened_flag"]) for r in rows])
    x7 = np.array([float(r["self_service_first_flag"]) for r in rows])
    y = np.array([float(r["resolution_minutes"]) for r in rows])

    n = len(y)

    # Task 1.1: Estimate with an interval
    mean_value = sample_mean(y)
    sd_value = sample_sd(y, mean_value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2: Test a planning claim
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # Task 1.3: Size a future sample
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    # Task 1.4: Simple regression by the closed form
    mean_x4 = sample_mean(x4)
    var_x4 = sample_var(x4, mean_x4)
    cov_x4_y = float(np.sum((x4 - mean_x4) * (y - mean_value)) / (n - 1))

    reg_slope = cov_x4_y / var_x4
    reg_intercept = mean_value - reg_slope * mean_x4
    fitted_raw = reg_intercept + reg_slope * x4

    tss = float(np.sum((y - mean_value) ** 2))
    ess = float(np.sum((fitted_raw - mean_value) ** 2))
    rss = float(np.sum((y - fitted_raw) ** 2))
    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = 1.0 - (rss / tss)
    rmse = math.sqrt(rss / n)
    mae = float(np.sum(np.abs(y - fitted_raw)) / n)

    # Task 1.5: Fit the same line by gradient descent
    sd_x4 = math.sqrt(var_x4)
    z = (x4 - mean_x4) / sd_x4
    m = 0.0
    c = 0.0
    eta = 0.1

    def compute_cost(m_val, c_val):
        err = y - (m_val * z + c_val)
        return float(np.sum(err ** 2) / n)

    gd_J0 = compute_cost(m, c)
    gd_costs = {}
    for step in range(1, 501):
        err = y - (m * z + c)
        dm = -(2.0 / n) * float(np.sum(z * err))
        dc = -(2.0 / n) * float(np.sum(err))
        m -= eta * dm
        c -= eta * dc
        if step in (1, 10, 100, 500):
            gd_costs[step] = compute_cost(m, c)

    gd_slope = m / sd_x4
    gd_intercept = c - m * mean_x4 / sd_x4
    gd_slope_gap = abs(gd_slope - reg_slope)

    # Task 1.6: Check for a funnel, then apply the log remedy
    order_raw = np.argsort(fitted_raw, kind="stable")
    resid_raw_sorted = (y - fitted_raw)[order_raw]
    lower_raw = resid_raw_sorted[:400]
    upper_raw = resid_raw_sorted[400:]

    resid_sd_lower_raw = sample_sd(lower_raw)
    resid_sd_upper_raw = sample_sd(upper_raw)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    ln_y = np.log(y)
    ln_x4 = np.log(x4)
    mean_ln_y = sample_mean(ln_y)
    mean_ln_x4 = sample_mean(ln_x4)
    var_ln_x4 = sample_var(ln_x4, mean_ln_x4)
    cov_ln = float(np.sum((ln_x4 - mean_ln_x4) * (ln_y - mean_ln_y)) / (n - 1))

    loglog_slope = cov_ln / var_ln_x4
    loglog_intercept = mean_ln_y - loglog_slope * mean_ln_x4
    fitted_log = loglog_intercept + loglog_slope * ln_x4

    tss_log = float(np.sum((ln_y - mean_ln_y) ** 2))
    rss_log = float(np.sum((ln_y - fitted_log) ** 2))
    loglog_r2 = 1.0 - (rss_log / tss_log)
    loglog_rmse = math.sqrt(rss_log / n)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    order_log = np.argsort(fitted_log, kind="stable")
    resid_log_sorted = (ln_y - fitted_log)[order_log]
    lower_log = resid_log_sorted[:400]
    upper_log = resid_log_sorted[400:]

    resid_sd_lower_log = sample_sd(lower_log)
    resid_sd_upper_log = sample_sd(upper_log)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # Task 1.7: Multiple regression by the normal equation
    x_design = np.column_stack([np.ones(n), ln_x4, x1, x2, x3, x5, x6, x7])
    xtx = x_design.T @ x_design
    xty = x_design.T @ ln_y
    betas = np.linalg.solve(xtx, xty)

    fitted_mr = x_design @ betas
    rss_mr = float(np.sum((ln_y - fitted_mr) ** 2))
    mr_r2 = 1.0 - (rss_mr / tss_log)
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)

    # Write results to outputs/part1_results.txt
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "part1_results.txt")

    lines = [
        f"n: {n}",
        f"mean_value: {mean_value:.6f}",
        f"sd_value: {sd_value:.6f}",
        f"se_value: {se_value:.6f}",
        f"ci95_low: {ci95_low:.6f}",
        f"ci95_high: {ci95_high:.6f}",
        f"z_stat: {z_stat:.6f}",
        f"reject_h0: {reject_h0}",
        f"n_required: {n_required}",
        f"reg_slope: {reg_slope:.6f}",
        f"reg_intercept: {reg_intercept:.6f}",
        f"tss: {tss:.6f}",
        f"ess: {ess:.6f}",
        f"rss: {rss:.6f}",
        f"tss_equals_ess_plus_rss: {tss_equals_ess_plus_rss}",
        f"r2: {r2:.6f}",
        f"rmse: {rmse:.6f}",
        f"mae: {mae:.6f}",
        f"gd_J0: {gd_J0:.6f}",
        f"gd_J1: {gd_costs[1]:.6f}",
        f"gd_J10: {gd_costs[10]:.6f}",
        f"gd_J100: {gd_costs[100]:.6f}",
        f"gd_J500: {gd_costs[500]:.6f}",
        f"gd_slope: {gd_slope:.6f}",
        f"gd_intercept: {gd_intercept:.6f}",
        f"gd_slope_gap: {gd_slope_gap:.12f}",
        f"resid_sd_lower_raw: {resid_sd_lower_raw:.6f}",
        f"resid_sd_upper_raw: {resid_sd_upper_raw:.6f}",
        f"resid_sd_ratio_raw: {resid_sd_ratio_raw:.6f}",
        f"loglog_intercept: {loglog_intercept:.6f}",
        f"loglog_slope: {loglog_slope:.6f}",
        f"loglog_r2: {loglog_r2:.6f}",
        f"loglog_rmse: {loglog_rmse:.6f}",
        f"resid_sd_lower_log: {resid_sd_lower_log:.6f}",
        f"resid_sd_upper_log: {resid_sd_upper_log:.6f}",
        f"resid_sd_ratio_log: {resid_sd_ratio_log:.6f}",
    ]

    for i in range(8):
        lines.append(f"mr_beta_{i}: {float(betas[i]):.6f}")

    lines.extend([
        f"mr_r2: {mr_r2:.6f}",
        f"mr_adj_r2: {mr_adj_r2:.6f}",
        f"loglog_adj_r2: {loglog_adj_r2:.6f}",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Successfully wrote", out_path)


if __name__ == "__main__":
    main()
