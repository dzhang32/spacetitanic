from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer


def main(repo_path: Path, non_feat_cols: List[str]) -> None:
    """Perform feature engineering.

    Converts the categorical features into encoded integers. Imputes the missing values
    using a Knn imputation approach.

    Parameters
    ----------
    repo_path : Path
        the path to the root of the project (that contains the data/ folder).
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    """
    print("Engineering Features\n--------------------------------")

    print("Loading train/test data...")
    train_test = pd.read_csv(repo_path / "data/processed/train_test_preprocessed.csv")

    print("Encoding categorical features...")
    train_test = encode_cat_vars(train_test, non_feat_cols)

    print("Imputing missing values...")
    train_test = impute_missing(train_test, non_feat_cols)

    print("Split train/test data back into separate dfs...")
    train, test = split_train_test(train_test)

    print("Saving train/test data...")
    train.to_csv(
        repo_path / "data/processed/train_engineered.csv",
        index=False,
    )
    test.to_csv(
        repo_path / "data/processed/test_engineered.csv",
        index=False,
    )

    print("done!")


def encode_cat_vars(df: pd.DataFrame, non_feat_cols: List[str]) -> pd.DataFrame:
    """Encode categorical features as integers.

    Parameters
    ----------
    df : pd.DataFrame
        contains the train/test data.
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.

    Returns
    -------
    pd.DataFrame
        the input df with categorical features encoded as integers.
    """
    df = df.copy()

    le = preprocessing.LabelEncoder()
    feat_cols = [c for c in df.columns if c not in non_feat_cols]

    for feat in feat_cols:
        # if column is string or bool
        if df[feat].dtype == object:
            le = le.fit(df[feat])
            df[feat] = le.transform(df[feat])

    return df


def impute_missing(
    df: pd.DataFrame, non_feat_cols: List[str], imputer: str = "knn"
) -> pd.DataFrame:
    """Impute the missing values.

    Parameters
    ----------
    df : pd.DataFrame
        contains the train/test data.
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    imputer : str, optional
        the type of imputation method to use, by default "knn".

    Returns
    -------
    pd.DataFrame
        the input df with all missing values imputed.
    """
    df = df.copy()

    imputer = dispatch_imputer(imputer)
    feat_cols = [c for c in df.columns if c not in non_feat_cols]
    df.loc[:, feat_cols] = imputer.fit_transform(df.loc[:, feat_cols])

    return df


def dispatch_imputer(imputer: str) -> Callable:
    """Dispatch the method to perform imputation.

    Parameters
    ----------
    imputer : str
        the type of method to use for missing value imputation.

    Returns
    -------
    Callable
        a class provided by sklearn that can perform imputation of missing values.

    Raises
    ------
    ValueError
        when a method is entered that does not match "knn".
    """
    match imputer:
        case "knn":
            return KNNImputer(n_neighbors=2, weights="uniform")
        case _:
            raise ValueError("imputer must be 'knn' currently")


def split_train_test(train_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset back into separate training and test data.

    Parameters
    ----------
    train_test : pd.DataFrame
        merged dataset containing both train/test data. Can be generated using
        spacetitanic.features.preprocess.merge_train_test().

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        the train and the test data, in that order.
    """
    train = train_test[train_test["dataset"] == "train"].copy()
    test = train_test[train_test["dataset"] == "test"].copy()

    for data in [train, test]:
        data.reset_index(drop=True, inplace=True)
        data.drop(columns="dataset", inplace=True)

    return train, test


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.parent
    main(repo_path, ["passengerid", "transported", "dataset"])
