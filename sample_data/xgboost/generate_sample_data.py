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


def train_regression_model():
    # Load the diabetes dataset
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    dataset = xgb.DMatrix(X, label=y)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
    }
    model = xgb.train(params, dataset, num_boost_round=50)
    y_pred = model.predict(dataset, output_margin=True)

    output_dir = Path("regression")
    output_dir.mkdir(exist_ok=True)
    model.save_model(output_dir / "xgb_model.json")
    np.savetxt(output_dir / "X.csv", X, delimiter=",")
    np.savetxt(output_dir / "y.csv", y_pred, delimiter=",")


def train_binary_classification_model():
    # Load the breast cancer dataset
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, n_informative=5
    )
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    dataset = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    model = xgb.train(params, dataset, num_boost_round=50)
    y_pred = model.predict(dataset, output_margin=True)

    output_dir = Path("binary_classification")
    output_dir.mkdir(exist_ok=True)
    model.save_model(output_dir / "xgb_model.json")
    np.savetxt(output_dir / "X.csv", X, delimiter=",")
    np.savetxt(output_dir / "y.csv", y_pred, delimiter=",")


def train_multiclass_classification_model():
    # Load a synthetic multiclass classification dataset
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=3, n_informative=7
    )
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    dataset = xgb.DMatrix(X, label=y)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    }
    model = xgb.train(params, dataset, num_boost_round=50)
    y_pred = model.predict(dataset, output_margin=True)

    output_dir = Path("multiclass_classification")
    output_dir.mkdir(exist_ok=True)
    model.save_model(output_dir / "xgb_model.json")
    np.savetxt(output_dir / "X.csv", X, delimiter=",")
    np.savetxt(output_dir / "y.csv", y_pred, delimiter=",")


if __name__ == "__main__":
    train_regression_model()
    train_binary_classification_model()
    train_multiclass_classification_model()
