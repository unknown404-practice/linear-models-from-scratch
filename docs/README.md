# Linear Models From Scratch: Documentation Hub

Welcome to the definitive reference and learning platform for **Linear Models built from pure mathematical principles in vectorized NumPy**.

> **Mission:** A from-scratch, vectorized NumPy implementation of linear models (OLS, regularized, robust, Bayesian, GLMs, streaming, conformal) with scikit-learn-compatible pipelines, full docs, and competition-grade examples.

### Quick Navigation
- 📦 [Installation Guide](../README.md#installation)
- 🎓 [Interactive Tutorials](tutorials/tutorial_01_linear_regression_basics.ipynb)
- 🚀 [Practical Examples](examples/example_california_housing_pipeline.ipynb)
- 📚 [API Reference](api/linear_regression.md)
- 🛠️ [Contributing Guide](../CONTRIBUTING.md)
- 📝 [Changelog](../CHANGELOG.md)
- ⚖️ [License (MIT)](../LICENSE)

---

## Documentation Roadmap

```
docs/
├── README.md               <-- You are here (Docs Hub)
├── faq.md                  <-- Common questions, debugging, design rationales
├── tutorials/              <-- Guided, step-by-step masterclasses with exercises
│   ├── tutorial_01_linear_regression_basics.ipynb
│   ├── tutorial_02_regularization_and_bias_variance.ipynb
│   ├── tutorial_03_optimizers_and_schedulers.ipynb
│   ├── tutorial_04_uncertainty_and_conformal_prediction.ipynb
│   └── tutorial_05_streaming_and_quantile_regression.ipynb
├── examples/               <-- Copy-pasteable, standalone notebooks
│   ├── example_house_prices_baseline.ipynb
│   ├── example_california_housing_pipeline.ipynb
│   ├── example_robust_regression_with_ransac.ipynb
│   ├── example_glm_logistic_poisson.ipynb
│   ├── example_bayesian_linear_regression.ipynb
│   ├── example_conformal_prediction_coverage.ipynb
│   ├── example_streaming_rls_kalman.ipynb
│   └── example_quantile_regression_risk.ipynb
├── api/                    <-- Exhaustive API references with minimal snippets
│   ├── linear_regression.md
│   ├── regularized_regression.md
│   ├── robust_regression.md
│   ├── bayesian_linear_regression.md
│   ├── conformal_prediction.md
│   ├── glms_irls.md
│   ├── streaming_rls_kalman.md
│   ├── quantile_regression.md
│   └── pipelines_and_column_transformers.md
└── design/                 <-- Technical architecture and numerical design notes
    ├── architecture.md
    ├── numerical_design.md
    ├── testing_strategy.md
    └── extensibility.md
```

---

## Recommended Learning Pathways

### 1. The Beginner Track (Concepts & Fundamentals)
If you are new to linear models or want to review the exact mechanics of gradient descent:
1. Read the [Overview and Foundations](../README.md#mathematical-foundations).
2. Work through [`tutorials/tutorial_01_linear_regression_basics.ipynb`](tutorials/tutorial_01_linear_regression_basics.ipynb).
3. Try [`examples/example_california_housing_pipeline.ipynb`](examples/example_california_housing_pipeline.ipynb).

### 2. The Intermediate Track (Optimization, Regularization & Pipelines)
If you know ML basics and want to master numerical optimization, ill-conditioned surfaces, and leak-free pipelines:
1. Complete [`tutorials/tutorial_02_regularization_and_bias_variance.ipynb`](tutorials/tutorial_02_regularization_and_bias_variance.ipynb).
2. Master first-order and second-order methods in [`tutorials/tutorial_03_optimizers_and_schedulers.ipynb`](tutorials/tutorial_03_optimizers_and_schedulers.ipynb).
3. Review [`api/pipelines_and_column_transformers.md`](api/pipelines_and_column_transformers.md) and [`examples/01_end_to_end_house_prices.py`](../examples/01_end_to_end_house_prices.py).

### 3. The Advanced Track (GLMs, Bayesian, Streaming & Conformal Safety)
If you are preparing for elite technical interviews or deploying mission-critical models requiring safety bounds:
1. Study epistemic uncertainty in [`tutorials/tutorial_04_uncertainty_and_conformal_prediction.ipynb`](tutorials/tutorial_04_uncertainty_and_conformal_prediction.ipynb).
2. Explore streaming updates and risk corridors in [`tutorials/tutorial_05_streaming_and_quantile_regression.ipynb`](tutorials/tutorial_05_streaming_and_quantile_regression.ipynb).
3. Read the in-depth architectural and numerical notes in [`design/numerical_design.md`](design/numerical_design.md) and [`design/architecture.md`](design/architecture.md).

---

## Verification & Trust
Every algorithm in this codebase is covered by an automated test suite of **85 unit tests** across 18 test files, ensuring 100% mathematical validity, zero data leakage, and exact scikit-learn statistical parity.

---

## About the Creator

| Creator | Affiliation | Contact & Socials |
| :--- | :--- | :--- |
| **Ranadeep Saha** | Member of Google Developer Group | [![GitHub](https://img.shields.io/badge/GitHub-unknown404--practice-181717?logo=github)](https://github.com/unknown404-practice) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Ranadeep%20Saha-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ranadeep-saha-a03296404/) [![Email](https://img.shields.io/badge/Email-ranadeep2021saha%40gmail.com-D14836?logo=gmail)](mailto:ranadeep2021saha@gmail.com) |

Ranadeep is an open-source machine learning engineer, educator, and member of Google Developer Group dedicated to making first-principles mathematical computing transparent, reliable, and accessible worldwide.