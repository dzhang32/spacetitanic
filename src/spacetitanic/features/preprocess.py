from pathlib import Path

import pandas as pd


def main(repo_path: Path):
    """Load, merge and preprocess the train and test data.

    Parameters
    ----------
    repo_path : Path
        the path to the root of the project (that contains the data/ folder).
    """
    print("Loading train/test data...")
    train = pd.read_csv(repo_path / "data/raw/train.csv")
    test = pd.read_csv(repo_path / "data/raw/test.csv")

    print("Preprocessing train/test data...")
    train_test = merge_train_test(train, test)
    train_test = tidy_colnames(train_test)
    train_test = tidy_passengerid(train_test)
    train_test = tidy_cabin(train_test)

    print("Saving preprocessed train/test data...")
    train_test.to_csv(repo_path / "data/processed/train_test_preprocessed.csv")

    print("done!")


def merge_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Merge train and test datasets into a single DataFrame.

    Parameters
    ----------
    train : pd.DataFrame
        Contains the train data.
    test : pd.DataFrame
        Contains the test data.

    Returns
    -------
    pd.DataFrame
        Contains both train and test data, merged row-wise.
    """
    train["dataset"] = "train"
    test["dataset"] = "test"
    train_test = pd.concat([train, test])
    train_test.reset_index(drop=True, inplace=True)

    return train_test


def tidy_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Make colnames lower case.

    Parameters
    ----------
    df : pd.DataFrame
        Contains your train/test data.

    Returns
    -------
    pd.DataFrame
        Equivalent object, with colnames shifted to lowercase.
    """
    # avoid modifying original df inside function
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


def tidy_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Split cabin into deck, cabin_num and side.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the cabin variable.

    Returns
    -------
    pd.DataFrame
        Contains the deck, cabin_num and side, with the original cabin variable
        dropped.
    """
    df = df.copy()
    df[["deck", "cabin_num", "side"]] = df["cabin"].str.split("/", 2, expand=True)
    df["cabin_num"] = df["cabin_num"].astype("Int32")
    df = df.drop(columns="cabin")

    return df


def tidy_passengerid(df: pd.DataFrame) -> pd.DataFrame:
    """Create a number in group variable from the passenger ID.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the passengerid variable.

    Returns
    -------
    pd.DataFrame
        Contains the num_in_group variable.
    """
    df = df.copy()
    df["num_in_group"] = df["passengerid"].str.replace(".*_", "", regex=True)

    return df


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.parent
    main(repo_path)
