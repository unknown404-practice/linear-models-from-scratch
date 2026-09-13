"""
Smoke Test Runner for All Project Notebooks.
Headlessly executes notebooks using nbclient / nbformat and asserts zero execution errors.
"""

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent

NOTEBOOK_PATHS = [
    ROOT_DIR / "notebooks" / "01_linear_regression_from_scratch_real_data.ipynb",
    ROOT_DIR / "notebooks" / "02_optimizers_and_geometry.ipynb",
    ROOT_DIR / "notebooks" / "03_uncertainty_and_risk.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_01_linear_regression_basics.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_02_regularization_and_bias_variance.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_03_optimizers_and_schedulers.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_04_uncertainty_and_conformal_prediction.ipynb",
    ROOT_DIR / "docs" / "tutorials" / "tutorial_05_streaming_and_quantile_regression.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_house_prices_baseline.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_california_housing_pipeline.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_robust_regression_with_ransac.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_glm_logistic_poisson.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_bayesian_linear_regression.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_conformal_prediction_coverage.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_streaming_rls_kalman.ipynb",
    ROOT_DIR / "docs" / "examples" / "example_quantile_regression_risk.ipynb",
    ROOT_DIR / "examples" / "02_model_comparison_dashboard.ipynb",
    ROOT_DIR / "examples" / "03_uncertainty_aware_predictions.ipynb",
    ROOT_DIR / "competitions" / "house_prices_competition_baseline.ipynb",
]


def run_smoke_test():
    print("=" * 75)
    print(f"HEADLESS SMOKE TEST: EXECUTING {len(NOTEBOOK_PATHS)} NOTEBOOKS")
    print("=" * 75)

    passed = 0
    failed = 0
    start_total = time.perf_counter()

    for nb_path in NOTEBOOK_PATHS:
        rel_path = nb_path.relative_to(ROOT_DIR)
        print(f"Running: {rel_path} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            client = NotebookClient(nb, timeout=120, kernel_name="python3")
            # Execute within notebook directory context
            client.execute(cwd=str(nb_path.parent))

            elapsed = time.perf_counter() - t0
            print(f"PASSED ({elapsed:.2f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.perf_counter() - t0
            safe_err = str(e).encode("ascii", "replace").decode("ascii")
            print(f"   Error: {safe_err}")
            failed += 1

    total_elapsed = time.perf_counter() - start_total
    print("=" * 75)
    print(f"SMOKE TEST COMPLETE: {passed}/{len(NOTEBOOK_PATHS)} PASSED in {total_elapsed:.2f}s")
    if failed > 0:
        print(f"FAILED: {failed} notebook(s) encountered errors!")
        sys.exit(1)
    else:
        print("ALL NOTEBOOKS EXECUTED WITH 0 ERRORS!")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
