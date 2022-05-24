import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main(repo_path: Path, non_feat_cols: List[str], y_col: str) -> None:
    """Perform feature engineering.

    Converts the categorical features into encoded integers. Imputes the missing values
    using a Knn imputation approach.

    Parameters
    ----------
    repo_path : Path
        the path to the root of the project (that contains the data/ folder).
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    y_col : str
        the column names of the variable to be predicted.
    """
    print("Engineering Features\n--------------------------------")

    print("Loading train/test data...")
    train_test = pd.read_csv(repo_path / "data/processed/train_test_preprocessed.csv")

    print("Encoding categorical features...")
    train_test = encode_cat_vars(train_test, non_feat_cols, y_col)

    print("Imputing missing values...")
    train_test = impute_missing(train_test, non_feat_cols, imputer="knn")

    print("Performing feature selection...")
    train_test = select_features(train_test, non_feat_cols, y_col)

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


def encode_cat_vars(
    df: pd.DataFrame, non_feat_cols: List[str], y_col: str
) -> pd.DataFrame:
    """Encode categorical features as integers.

    Parameters
    ----------
    df : pd.DataFrame
        contains the train/test data.
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    y_col : str
        the column names of the variable to be predicted.

    Returns
    -------
    pd.DataFrame
        the input df with categorical features encoded as integers.
    """
    df = df.copy()

    df = pd.get_dummies(df, columns=["homeplanet", "destination", "deck"])

    le = preprocessing.LabelEncoder()
    feat_cols = [c for c in df.columns if c not in non_feat_cols]

    for feat in feat_cols + [y_col]:
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
        case "const_0":
            return SimpleImputer(strategy="constant", fill_value=0)
        case _:
            raise ValueError("imputer must be 'knn' currently")


def select_features(
    train_test: pd.DataFrame, non_feat_cols: List[str], y_col: str
) -> pd.DataFrame:
    """Perform feature selection.

    Parameters
    ----------
    train_test : pd.DataFrame
        merged dataset containing both train/test data. Can be generated using
        spacetitanic.features.preprocess.merge_train_test().
    non_feat_cols : List[str]
        the column names of the variables that are NOT features.
    y_col : str
        the column names of the variable to be predicted.

    Returns
    -------
    pd.DataFrame
        train_test data with features selected/removed.
    """
    # set up train and validation datasets
    train = train_test[train_test["dataset"] == "train"].copy()
    train.reset_index(drop=True, inplace=True)
    X = train.drop(columns=non_feat_cols)
    y = train[y_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=32
    )

    # fit an xgb classifier to obtain feature importances and baseline accuracy
    clf = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=32,
    )
    clf.fit(
        X_train,
        y_train,
        early_stopping_rounds=10,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    initial_accuracy = metrics.accuracy_score(y_val, y_pred)
    feat_imp = get_feat_imp(clf)

    # iteratively remove features, then fit a xgb classifer and store it's accuracy
    select_accuracy = {
        "n_feat": [0],
        "feat_dropped": [[]],
        "accuracy": [initial_accuracy],
    }

    for i in range(0, 20):
        feat_to_drop = list(feat_imp.loc[:i, "feat"])
        X_train_select = X_train.drop(columns=feat_to_drop)
        X_val_select = X_val.drop(columns=feat_to_drop)
        clf_select = XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=32,
        )
        clf_select.fit(
            X_train_select,
            y_train,
            early_stopping_rounds=10,
            eval_set=[(X_val_select, y_val)],
            verbose=0,
        )
        y_val_pred = clf_select.predict(X_val_select)
        accuracy = metrics.accuracy_score(y_val, y_val_pred)

        select_accuracy["n_feat"].append(X_train_select.shape[1])
        select_accuracy["feat_dropped"].append(feat_to_drop)
        select_accuracy["accuracy"].append(accuracy)

    select_accuracy = pd.DataFrame(select_accuracy)
    print(select_accuracy)

    # select the N features that obtain the highest accuracy
    # then drop any features from the train_test data
    feat_to_drop = select_accuracy.iloc[
        select_accuracy["accuracy"].idxmax(),
    ]
    feat_to_drop = feat_to_drop["feat_dropped"]
    if len(feat_to_drop) != 0:
        train_test = train_test.drop(columns=feat_to_drop)

    return train_test


def get_feat_imp(clf: Callable) -> pd.DataFrame:
    """Obtain feature importances from an XGboost classifier.

    Parameters
    ----------
    clf : Callable
        A trained XGboost classifier.

    Returns
    -------
    pd.DataFrame
        contains the feature names and their corresponding importances.
    """
    feat_imp = clf.get_booster().get_score(importance_type="weight")
    feat_imp = pd.DataFrame({"feat": feat_imp.keys(), "imp": feat_imp.values()})
    feat_imp.sort_values("imp", inplace=True)
    feat_imp.reset_index(drop=True, inplace=True)

    return feat_imp


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

    test.drop(columns="transported", inplace=True)

    return train, test


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.parent
    main(repo_path, ["passengerid", "transported", "dataset"], "transported")
