# /// script
# dependencies = [
#   "xgboost-cpu",
#   "scikit-learn",
# ]
# ///

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    objective: str,
    num_class: int,
    output_dir: Path,
):
    n = X.shape[0]
    train_dataset = xgb.DMatrix(X[: n // 2], label=y[: n // 2])
    test_dataset = xgb.DMatrix(X[n // 2 :], label=y[n // 2 :])
    params = {
        "objective": objective,
        "num_class": num_class,
        "seed": 0,
    }
    model = xgb.train(params, train_dataset)
    y_pred = model.predict(test_dataset, output_margin=True)

    output_dir.mkdir(exist_ok=True)
    model.save_model(output_dir / "model.json")
    np.savetxt(output_dir / "X.csv", X[n // 2 :], delimiter=",")
    np.savetxt(output_dir / "y.csv", y_pred, delimiter=",")


if __name__ == "__main__":
    output_dir = Path("test_data/xgboost")
    n_samples = 100
    n_features = 5
    for target in ["regression", "binary_classification", "multiclass_classification"]:
        match target:
            case "regression":
                X, y = make_regression(n_samples=n_samples, n_features=n_features)
                objective = "reg:squarederror"
                num_class = 0
            case "binary_classification":
                X, y = make_classification(
                    n_samples=n_samples, n_features=n_features, n_classes=2
                )
                objective = "binary:logistic"
                num_class = 0
            case "multiclass_classification":
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=3,
                    n_informative=3,
                )
                objective = "multi:softprob"
                num_class = 3
        train_model(X, y, objective, num_class, output_dir / target)
