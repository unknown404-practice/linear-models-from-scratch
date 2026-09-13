# API Reference: Pipelines & Preprocessing

Located in [`src/pipeline.py`](../../src/pipeline.py) and [`src/features.py`](../../src/features.py).

## Classes
- `StandardScalerScratch`: Z-score standardization ($z = (x - \mu)/\sigma$).
- `SimpleImputerScratch`: Missing value replacement (`mean`, `median`, `most_frequent`, `constant`).
- `OneHotEncoderScratch`: Categorical dummy encoding with `handle_unknown='ignore'`.
- `PolynomialFeaturesScratch`: Non-linear combinations and interaction terms.
- `ColumnTransformerScratch`: Selective column transformations.
- `PipelineScratch`: Sequential composition of transformers and terminal estimator.

### Minimal Usage
```python
from src.pipeline import (
    PipelineScratch,
    ColumnTransformerScratch,
    StandardScalerScratch,
    SimpleImputerScratch,
    OneHotEncoderScratch,
)
from src.solvers import ClosedFormLinearRegression

preprocessor = ColumnTransformerScratch(
    transformers=[
        (
            "num",
            PipelineScratch(
                [
                    ("imputer", SimpleImputerScratch(strategy="median")),
                    ("scaler", StandardScalerScratch()),
                ]
            ),
            ["MedInc", "HouseAge", "AveRooms"],
        ),
        (
            "cat",
            PipelineScratch(
                [
                    ("imputer", SimpleImputerScratch(strategy="most_frequent")),
                    ("ohe", OneHotEncoderScratch(handle_unknown="ignore")),
                ]
            ),
            ["OceanProximity"],
        ),
    ]
)

full_pipeline = PipelineScratch(
    [("prep", preprocessor), ("regressor", ClosedFormLinearRegression(method="svd"))]
)

full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```