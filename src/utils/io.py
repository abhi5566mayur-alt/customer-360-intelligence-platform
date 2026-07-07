from pathlib import Path
import pandas as pd


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV safely and raise a clean error if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame as parquet and create parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    