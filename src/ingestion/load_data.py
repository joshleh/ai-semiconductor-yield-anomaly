from pathlib import Path
import pandas as pd
import urllib.request



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
    def _download_secom_csv(dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # If you already have a stable URL you used to download SECOM, put it here.
        # Otherwise: keep deployment in "upload required" mode and skip download.
        url = "PUT_YOUR_PUBLIC_SECOM_CSV_URL_HERE"

        urllib.request.urlretrieve(url, dest_path)

    filepath = DATA_DIR / "raw" / filename

    if not filepath.exists():
        # Attempt download in hosted environments
        try:
            _download_secom_csv(filepath)
        except Exception as e:
            raise FileNotFoundError(
                f"Data file not found: {filepath}. "
                f"Upload dataset to data/raw/ or configure SECOM download. Original error: {e}"
            )


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