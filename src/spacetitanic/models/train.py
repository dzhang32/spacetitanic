from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    return clf


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.parent
    main(repo_path, ["passengerid", "transported", "dataset"], "transported")