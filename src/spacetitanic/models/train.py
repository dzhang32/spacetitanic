from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def main(repo_path: Path, non_feat_cols: List[str], y_col: str) -> None:
    """Hold out a validation dataset, then train the model.

    Parameters
    ----------
    repo_path : Path
        the path to the root of the project (that contains the data/ folder).
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    y_col : str
        the column names of the variable to be predicted.
    """
    print("Training model\n--------------------------------")

    print("Loading train data...")
    train = pd.read_csv(repo_path / "data/processed/train_engineered.csv")

    print("Training model...")
    X_train, X_val, y_train, y_val = split_train_val(train, non_feat_cols, y_col)
    clf = train_model(X_train, y_train)

    print("Saving model...")
    dump(clf, repo_path / "model/model.joblib")

    print("Saving validation data...")
    X_val[y_col] = y_val
    X_val.to_csv(repo_path / "data/processed/val_engineered.csv", index=False)

    print("done!")


def split_train_val(
    train: pd.DataFrame, non_feat_cols: List[str], y_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,]:
    """Split the training data into training and validation datasets.

    Parameters
    ----------
    train : pd.DataFrame
        contains the train data.
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    y_col : str
        the column names of the variable to be predicted.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,]
        the X, y for the training and validation datasets.
    """
    feat_cols = [c for c in train.columns if c not in non_feat_cols]

    X = train.loc[:, feat_cols]
    y = train.loc[:, y_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=32
    )

    return X_train, X_val, y_train, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Callable:
    """Train a random forest model.

    Parameters
    ----------
    X_train : pd.DataFrame
        features for the training dataset.
    y_train : pd.Series


    Returns
    -------
    Callable
        a model trained on the training dataset.
    """
    clf = RandomForestClassifier()

    # create grid of hyperparameters for random search
    n_estimators = [int(x) for x in np.linspace(start=100, stop=600, num=6)]
    max_features = ["auto", "sqrt", 0.2, 0.3]
    criterion = ["gini", "entropy"]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

    clf_random = RandomizedSearchCV(
        estimator=clf,
        param_distributions=random_grid,
        n_iter=200,
        cv=5,
        verbose=2,
        random_state=32,
        n_jobs=5,
    )

    clf_random.fit(X_train, y_train)

    return clf_random


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.parent
    main(repo_path, ["passengerid", "transported", "dataset"], "transported")
