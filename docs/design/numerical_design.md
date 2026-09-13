# Numerical Design Choices & Stability Safeguards

Floating-point instability is the primary reason naive from-scratch ML implementations break. Our engine incorporates four core mathematical safeguards:

### 1. Zero-Variance Safeguard in Standardization
When standardizing constant features ($x_{:, j} = c$), standard division by $\sigma_j = 0$ results in `NaN`. `StandardScalerScratch` applies an $\epsilon$-floor:
$$\sigma_{\text{safe}} = \begin{cases} \sigma_j & \text{if } \sigma_j > 10^{-12} \\ 1.0 & \text{otherwise} \end{cases}$$

### 2. SVD Pseudo-Inverse for Rank-Deficient Systems
Direct matrix inversion $(X^T X)^{-1}$ fails if features are collinear (singular matrix). `ClosedFormLinearRegression(method="svd")` decomposes $X = U \Sigma V^T$ and computes:
$$w = V \Sigma^+ U^T y$$
where singular values $\sigma_i < 10^{-15}$ are truncated to zero.

### 3. Tikhonov Damping in Second-Order Optimization
In `NewtonLinearRegression`, the empirical Hessian $H = \frac{2}{N} X^T X$ is regularized via Levenberg-Marquardt damping:
$$H_{\text{damped}} = H + \lambda I$$
guaranteeing positive definiteness, strict condition number bounds, and invertible systems.

### 4. Sherman-Morrison Numerical Stability in RLS
In `RecursiveLeastSquares`, the inverse covariance matrix $P_t$ is updated via rank-1 subtraction:
$$P_t = \frac{1}{\lambda} \left( P_{t-1} - k_t x_t^T P_{t-1} \right)$$
To prevent positive-definiteness drift caused by accumulated roundoff errors over millions of iterations, symmetry is enforced:
$$P_t \leftarrow \frac{1}{2}(P_t + P_t^T)$$