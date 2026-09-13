# Contributing to Linear Models from Scratch

Thank you for your interest in contributing! This project bridges first-principles mathematical theory with production-grade engineering. We welcome contributions from researchers, students, educators, and machine learning engineers worldwide.

---

## Table of Contents
1. [Development Philosophy](#development-philosophy)
2. [Setting Up Your Local Development Environment](#setting-up-your-local-development-environment)
3. [Quality Assurance & Verification](#quality-assurance--verification)
   - [Running the Test Suite](#running-the-test-suite)
   - [Linting and Formatting](#linting-and-formatting)
   - [Type Checking with Mypy](#type-checking-with-mypy)
   - [Headless Notebook Execution](#headless-notebook-execution)
   - [Pre-Commit Hooks](#pre-commit-hooks)
4. [Adding New Features & Notebooks](#adding-new-features--notebooks)
5. [Pull Request Workflow & Template](#pull-request-workflow--template)
6. [Code of Conduct](#code-of-conduct)

---

## Development Philosophy
1. **Mathematical Rigor:** Every estimator and algorithm must be derived from first principles. Docstrings must cite relevant derivations, loss objectives, and gradient formulas.
2. **Zero ML Dependencies in Core Algorithms:** All estimators in `src/` must rely strictly on vectorized **NumPy** and **SciPy** linear algebra routines. External libraries like `scikit-learn` are permitted *only* for validation and benchmarking in unit tests and notebooks.
3. **Scikit-Learn Compatibility:** Any new estimator or transformer must implement standard `.fit()`, `.predict()`, `.transform()`, `.get_params()`, and `.set_params()` signatures where applicable.
4. **Zero Data Leakage:** Preprocessors and cross-validation utilities must strictly isolate training and evaluation sets.
5. **Deterministic Testing:** Random seeds must be parameterized with sensible defaults (`seed=42`) ensuring bit-for-bit reproducibility.

---

## Setting Up Your Local Development Environment

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/unknown404-practice/linear-models-from-scratch.git
cd linear-models-from-scratch

# Create isolated Python 3.11+ virtual environment
python -m venv .venv

# Activate environment:
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\.venv\Scripts\activate.bat
```

### 2. Install in Editable Mode with Dev Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -e .[dev,docs]
```

---

## Quality Assurance & Verification

Every contribution must satisfy five quality checks before being merged into `main`.

### 1. Running the Test Suite
Run the full pytest suite (85+ tests covering all 18 test modules):
```bash
pytest -v tests
```
To run a specific test file:
```bash
pytest -v tests/test_bayesian.py
```

### 2. Linting and Formatting
We enforce strict PEP 8 compliance using **Ruff**:
```bash
# Check code style and common lint errors
ruff check .

# Automatically apply safe lint fixes
ruff check --fix .

# Check code formatting without making changes
ruff format --check .

# Auto-format all Python files
ruff format .
```

### 3. Type Checking with Mypy
We maintain strict type hints across all library modules in `src/`:
```bash
mypy src/
```

### 4. Headless Notebook Execution
All 19 Jupyter tutorials, examples, and masterclass notebooks must execute without error:
```bash
python scripts/check_notebooks.py
```

### 5. Pre-Commit Hooks
Install pre-commit to automatically run linting and type checks before every `git commit`:
```bash
pre-commit install

# Test hooks against all files
pre-commit run --all-files
```

---

## Adding New Features & Notebooks

### Adding a New Model or Feature
1. Place implementation files in `src/` (e.g. `src/sparse_solvers.py`).
2. Add typed re-exports and scikit-learn aliases in `src/__init__.py`.
3. Re-export in `linear_models_from_scratch/`.
4. Create dedicated tests in `tests/test_<feature>.py`. Ensure tests verify:
   - Analytical correctness on noise-free synthetic data.
   - Resilience against ill-conditioned or collinear matrices.
   - Deterministic reproducibility under explicit random seeds.
   - Compatibility with `PipelineScratch` and `cross_val_score_scratch`.

### Adding a Tutorial or Example Notebook
1. Place tutorials in `docs/tutorials/` and examples in `docs/examples/`.
2. Include the standard robust root-path resolution at the top of cell 1:
   ```python
   import sys
   from pathlib import Path

   root_dir = Path.cwd().resolve()
   while not (root_dir / "src").exists() and root_dir != root_dir.parent:
       root_dir = root_dir.parent
   if str(root_dir) not in sys.path:
       sys.path.insert(0, str(root_dir))
   ```
3. Run `python scripts/check_notebooks.py` to ensure the notebook executes headlessly in under 60 seconds with 0 cell errors.

---

## Pull Request Workflow & Template

1. **Branch Naming:** Use descriptive branch names:
   - `feat/elasticnet-coordinate-descent`
   - `fix/conformal-quantile-edgecase`
   - `docs/irls-mathematical-derivation`
2. **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(glm): add negative binomial dispersion parameter`
   - `test(bayesian): add ARD sparsity verification test`
   - `docs(tutorials): expand tutorial 03 optimizer dynamics`
3. **Pull Request Description Template:**
   ```markdown
   ### Description
   <!-- Concise summary of changes and mathematical rationale -->

   ### Related Issue
   Closes #<!-- issue number -->

   ### Verification Checklist
   - [ ] `pytest -v tests` passes (85+ tests).
   - [ ] `ruff check .` passes with 0 errors.
   - [ ] `ruff format --check .` passes.
   - [ ] `mypy src/` passes with 0 issues.
   - [ ] `python scripts/check_notebooks.py` executes all notebooks with 0 errors.
   - [ ] Documentation / docstrings updated.
   - [ ] CHANGELOG.md updated under [Unreleased].
   ```

---

## Code of Conduct
We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please treat fellow maintainers and contributors with respect, intellectual honesty, and constructive feedback.

---

## Maintainer & Contact

- **Lead Maintainer:** Ranadeep Saha
- **Affiliation:** Member of Google Developer Group
- **Email:** [ranadeep2021saha@gmail.com](mailto:ranadeep2021saha@gmail.com)
- **GitHub:** [@unknown404-practice](https://github.com/unknown404-practice)
- **LinkedIn:** [Ranadeep Saha](https://www.linkedin.com/in/ranadeep-saha-a03296404/)
- **Repository:** [linear-models-from-scratch](https://github.com/unknown404-practice/linear-models-from-scratch.git)

We welcome bug reports, feature suggestions, and educational collaboration via GitHub Issues, email, or LinkedIn.