"""
Inject Key Takeaway and Deep Dive Callouts into scripts/build_notebook.py,
then regenerate notebooks/01_linear_regression_from_scratch_real_data.ipynb.
"""

from pathlib import Path


def enhance():
    script_path = Path("scripts/build_notebook.py")
    content = script_path.read_text(encoding="utf-8")

    # 1. Add Key Takeaway & Deep Dive after Section A
    target_a = '| **Hardware Fit** | High vectorization, high RAM | Fits into CPU L2/L3 cache, highly scalable |""")'
    replacement_a = (
        '| **Hardware Fit** | High vectorization, high RAM | Fits into CPU L2/L3 cache, highly scalable |""")\n\n'
        + '    add_md("""'
        + "> **Key Takeaways (Section A):**\\n"
        + "> 1. **Residual-Driven Optimization:** The gradient $\\\\nabla_w J = \\\\frac{2}{N} X^T e$ depends strictly on residual errors $e = \\\\hat{y} - y$.\\n"
        + "> 2. **Vectorized Hardware Cache Locality:** Expressing gradients as $X^T e$ leverages BLAS GEMV routines running at CPU cache speed.\\n\\n"
        + "<details>\\n"
        + "<summary><b>Deep Dive: Matrix Calculus of Quadratic Cost Functions</b></summary>\\n\\n"
        + "Expanding the Mean Squared Error:\\n"
        + "$$J(w) = \\\\frac{1}{N} (Xw - y)^T (Xw - y) = \\\\frac{1}{N} \\\\left( w^T X^T X w - 2 y^T X w + y^T y \\\\right)$$\\n"
        + "Differentiating with respect to $w$ yields $\\\\nabla_w J = \\\\frac{2}{N} X^T (Xw - y) = \\\\frac{2}{N} X^T e$.\\n"
        + "Setting $\\\\nabla_w J = 0$ yields the classic Normal Equations $(X^T X) w = X^T y$.\\n"
        + '</details>""")'
    )

    if target_a in content:
        content = content.replace(target_a, replacement_a, 1)
        print("Injected Section A takeaways and deep dive.")
    else:
        print("Target A not found.")

    # 2. Add Key Takeaways after Section D
    target_d = 'print(f"Mini-Batch Time: {time_mb:.3f}s | Final Test R^2: {model_mb.score(X_test_cal, y_test_cal):.4f}")""")'
    replacement_d = (
        'print(f"Mini-Batch Time: {time_mb:.3f}s | Final Test R^2: {model_mb.score(X_test_cal, y_test_cal):.4f}")""")\n\n'
        + '    add_md("""'
        + "> **Key Takeaways (Section D):**\\n"
        + "> 1. **Numerical Parity:** Pure-NumPy gradient descent matches Scikit-Learn closed-form weights to $< 10^{-5}$ relative tolerance.\\n"
        + "> 2. **Mini-Batch Scalability:** Mini-batch GD achieves competitive $R^2$ while processing small chunks fitting in CPU cache.\\n"
        + '""")'
    )

    if target_d in content:
        content = content.replace(target_d, replacement_d, 1)
        print("Injected Section D takeaways.")
    else:
        print("Target D not found.")

    # 3. Add Key Takeaways after Section N
    target_n = 'print("Phase 5 Pure-JSON Model Serialization & Reload Verified 100% Bit-for-Bit Identical!")""")'
    replacement_n = (
        'print("Phase 5 Pure-JSON Model Serialization & Reload Verified 100% Bit-for-Bit Identical!")""")\n\n'
        + '    add_md("""'
        + "> **Key Takeaways (Section N):**\\n"
        + "> 1. **Bayesian Uncertainty:** Epistemic uncertainty explodes in out-of-distribution space, turning standard regression into a reliable safety detector.\\n"
        + "> 2. **Conformal Safety:** Guarantees non-asymptotic coverage $P(y \\\\in \\\\mathcal{C}(x)) \\\\ge 1 - \\\\alpha$ under ANY arbitrary distribution with zero Gaussianity assumptions.\\n"
        + "> 3. **Streaming RLS:** Sherman-Morrison rank-1 updating achieves $\\\\mathcal{O}(D^2)$ time/memory online updates with forgetting factor tracking drift.\\n"
        + "> 4. **Quantile Corridors:** Asymmetric pinball loss models conditional percentiles ($\\\\tau \\\\in \\\\{0.10, 0.50, 0.90\\\\}$) directly capturing heteroscedastic spread.\\n\\n"
        + "<details>\\n"
        + "<summary><b>Deep Dive: The Sherman-Morrison Rank-1 Inversion Theorem</b></summary>\\n\\n"
        + "When a new sample $(x_t, y_t)$ arrives, the updated covariance is $A_t = \\\\lambda A_{t-1} + x_t x_t^T$.\\n"
        + "By the Sherman-Morrison formula:\\n"
        + "$$P_t = \\\\frac{1}{\\\\lambda} \\\\left( P_{t-1} - \\\\frac{P_{t-1} x_t x_t^T P_{t-1}}{\\\\lambda + x_t^T P_{t-1} x_t} \\\\right)$$\\n"
        + "This reduces matrix inversion from $\\\\mathcal{O}(D^3)$ to a matrix-vector product of complexity $\\\\mathcal{O}(D^2)$!\\n"
        + '</details>""")'
    )

    if target_n in content:
        content = content.replace(target_n, replacement_n, 1)
        print("Injected Section N takeaways and deep dive.")
    else:
        print("Target N not found.")

    script_path.write_text(content, encoding="utf-8")
    print("Successfully enhanced scripts/build_notebook.py!")


if __name__ == "__main__":
    enhance()
