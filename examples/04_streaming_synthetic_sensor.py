"""
Mini-Project 04: Streaming Sensor Regression with RLS & Kalman Filter
=====================================================================
Demonstrates infinite-scale online learning and dynamic tracking of concept drift:
1. Simulates an IoT sensor stream generating observations (x_t, y_t).
2. Introduces an abrupt regime shift (concept drift) at t = 1,000.
3. Compares Recursive Least Squares (RLS with lambda=0.98) and Kalman Filter
   against static Batch OLS.
4. Validates tracking error and parameter convergence.
"""

import sys
from pathlib import Path
import numpy as np

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.streaming import RecursiveLeastSquares, KalmanFilterRegression
from src.solvers import ClosedFormLinearRegression


def run():
    print("=" * 70)
    print("MINI-PROJECT 04: STREAMING SENSOR REGRESSION WITH RLS & KALMAN")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 2000
    n_features = 3
    drift_point = 1000

    # True parameters: Regime 1 vs Regime 2
    w_true_1 = np.array([2.5, -1.8, 0.7])
    b_true_1 = 1.0

    w_true_2 = np.array([-1.0, 3.2, 0.0])  # abrupt shift in physics / calibration
    b_true_2 = -0.5

    print(f"[1/4] Generating synthetic streaming stream (N={n_samples} ticks)...")
    print(f"      Regime 1 (t < {drift_point}):  weights = {w_true_1}, bias = {b_true_1}")
    print(f"      Regime 2 (t >= {drift_point}): weights = {w_true_2}, bias = {b_true_2}")

    X_stream = np.random.randn(n_samples, n_features)
    y_stream = np.zeros(n_samples)

    for t in range(n_samples):
        noise = np.random.normal(0, 0.05)
        if t < drift_point:
            y_stream[t] = X_stream[t] @ w_true_1 + b_true_1 + noise
        else:
            y_stream[t] = X_stream[t] @ w_true_2 + b_true_2 + noise

    # 2. Online Estimators Initialization
    print("\n[2/4] Initializing streaming models...")
    rls_tracker = RecursiveLeastSquares(forgetting_factor=0.98)
    kalman_tracker = KalmanFilterRegression(q_cov=1e-3, r_cov=1e-2)

    # 3. Process Stream Observation by Observation
    print("\n[3/4] Streaming data point-by-point through RLS and Kalman Filter...")
    rls_errors = []
    kalman_errors = []

    for t in range(n_samples):
        x_t = X_stream[t]
        y_t = y_stream[t]

        # Prior prediction error before updating
        if t > 5:
            pred_rls = rls_tracker.predict(x_t.reshape(1, -1))[0]
            pred_kalman = kalman_tracker.predict(x_t.reshape(1, -1))[0]
            rls_errors.append((y_t - pred_rls) ** 2)
            kalman_errors.append((y_t - pred_kalman) ** 2)

        # Single observation rank-1 update
        rls_tracker.partial_fit(x_t, y_t)
        kalman_tracker.partial_fit(x_t, y_t)

    # 4. Fit Static Batch OLS for Contrast
    print("\n[4/4] Fitting static Batch OLS on all historical data...")
    static_ols = ClosedFormLinearRegression(method="svd")
    static_ols.fit(X_stream, y_stream)

    # Evaluation on the post-drift test window (last 200 ticks)
    X_test_post = X_stream[-200:]
    y_test_post = y_stream[-200:]

    mse_static = np.mean((y_test_post - static_ols.predict(X_test_post)) ** 2)
    mse_rls = np.mean((y_test_post - rls_tracker.predict(X_test_post)) ** 2)
    mse_kalman = np.mean((y_test_post - kalman_tracker.predict(X_test_post)) ** 2)

    print("\n" + "-" * 70)
    print("POST-DRIFT REGIME TRACKING PERFORMANCE (Last 200 Samples):")
    print(f"  - Static Batch OLS (blind to drift):   MSE = {mse_static:.5f}")
    print(f"  - RLS (forgetting factor lambda=0.98): MSE = {mse_rls:.5f}")
    print(f"  - Kalman Filter (adaptive Q/R):        MSE = {mse_kalman:.5f}")
    print("-" * 70)
    print("Recovered Weights after Drift:")
    print(f"  - Ground Truth: {w_true_2}, bias: {b_true_2}")
    print(f"  - RLS Recovered:    {np.round(rls_tracker.weights, 4)}, bias: {rls_tracker.bias:.4f}")
    print(
        f"  - Kalman Recovered: {np.round(kalman_tracker.weights, 4)}, bias: {kalman_tracker.bias:.4f}"
    )
    print(f"  - Static OLS:       {np.round(static_ols.weights, 4)}, bias: {static_ols.bias:.4f}")
    print("-" * 70)

    assert mse_rls < mse_static * 0.1, "RLS tracking should vastly outperform static OLS!"
    print("\n" + "=" * 70)
    print("MINI-PROJECT 04 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run()
