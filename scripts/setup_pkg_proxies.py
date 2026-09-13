from pathlib import Path

modules = [
    "data_loader",
    "linear_regression",
    "solvers",
    "optimizers",
    "schedulers",
    "features",
    "robust",
    "bayesian",
    "conformal",
    "glm",
    "streaming",
    "quantile",
    "pipeline",
    "model_selection",
    "statistics",
    "visualization",
    "serialization",
]

pkg_dir = Path(__file__).resolve().parent.parent / "linear_models_from_scratch"
pkg_dir.mkdir(parents=True, exist_ok=True)

for mod in modules:
    mod_file = pkg_dir / f"{mod}.py"
    content = f'"""Re-export of src.{mod} for package compatibility."""\n\nfrom src.{mod} import *  # noqa: F403\n'
    mod_file.write_text(content, encoding="utf-8")
    print(f"Created {mod_file}")
print("All proxy modules generated successfully.")
