from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw semiconductor manufacturing data.

    Parameters
    ----------
    filename : str
        Name of the raw data file located in data/raw/

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    filepath = DATA_DIR / "raw" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    return df


def basic_sanity_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the dataset.
    """
    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    if df.isnull().all().any():
        raise ValueError("One or more columns contain only null values")