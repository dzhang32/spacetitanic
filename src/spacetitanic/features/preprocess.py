import pandas as pd


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
    df[["deck", "cabin_num", "side"]] = df["Cabin"].str.split("/", 2, expand=True)
    df["cabin_num"] = df["cabin_num"].astype("Int32")
    df = df.drop(columns="Cabin")

    return df
