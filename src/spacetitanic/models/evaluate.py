import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import load
from sklearn import metrics


def main(repo_path: Path, non_feat_cols: List[str], y_col: str) -> None:
    """Generate predictions and metrics for the validation and test datasets.

    Parameters
    ----------
    repo_path : Path
        the path to the root of the project (that contains the data/ folder).
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    y_col : str
        the column names of the variable to be predicted.
    """
    print("Predicting using trained model\n--------------------------------")

    print("Loading test/val data and trained model...")
    test = pd.read_csv(repo_path / "data/processed/test_engineered.csv")
    val = pd.read_csv(repo_path / "data/processed/val_engineered.csv")
    clf = load(repo_path / "model/model.joblib")
    feat_cols = [c for c in test.columns if c not in non_feat_cols]

    print("Generating predictions and metrics for val...")
    y_val_pred = clf.predict(val.loc[:, feat_cols])

    val_metrics = get_metrics(val, y_col, y_val_pred)
    val_metrics_path = repo_path / "metrics/metrics.json"
    val_metrics_path.write_text(json.dumps(val_metrics))

    print("Generating predictions and submission for test...")
    y_test_pred = clf.predict(test.loc[:, feat_cols])
    submission = get_submission_df(test, y_test_pred)
    submission.to_csv(repo_path / "data/output/submission.csv", index=False)

    print("done!")


def get_metrics(df: pd.DataFrame, y_col: str, y_pred: str) -> dict:
    """Obtain a set of evaluation metrics for predictions.

    Parameters
    ----------
    df : pd.DataFrame
        contains the y column, here "transported".
    y_col : str
        the name of the column to be predicted.
    y_pred : str
        the predictions from the trained model.

    Returns
    -------
    dict
        contains the accuracy, f1_score, precision and recall metrics.
    """
    y = df.loc[:, y_col]

    eval_metrics = {
        "accuracy": metrics.accuracy_score(y, y_pred),
        "f1_score": metrics.f1_score(y, y_pred),
        "precision": metrics.precision_score(y, y_pred),
        "recall": metrics.recall_score(y, y_pred),
    }

    return eval_metrics


def get_submission_df(test: pd.DataFrame, y_test_pred: np.ndarray) -> pd.DataFrame:
    """Create a submission df for Kaggle.

    Parameters
    ----------
    test : pd.DataFrame
        contains the test data.
    y_test_pred : np.ndarray
        contains predicted y values for the test data.

    Returns
    -------
    pd.DataFrame
        contains the predicted y values formatted with the id ready to be submitted to
        Kaggle.
    """
    submission = pd.DataFrame(
        {"PassengerId": test["passengerid"], "Transported": y_test_pred}
    )
    submission["Transported"] = submission["Transported"].astype(bool)

    return submission


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.parent
    main(repo_path, ["passengerid", "transported", "dataset"], "transported")
