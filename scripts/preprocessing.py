# src/preprocessing.py

from typing import Dict, List, Optional
import pandas as pd


class Preprocessor:

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor with a dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.df = df.copy()

    def remove_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> "Preprocessor":
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)

        print(f"Removed {before - after} duplicate rows")
        return self

    def correct_dtypes(
        self,
        dtype_map: Dict[str, str],
        date_columns: Optional[List[str]] = None
    ) -> "Preprocessor":
        # Convert specified dtypes
        for col, dtype in dtype_map.items():
            if col not in self.df.columns:
                print(f"⚠️ Column '{col}' not found, skipping")
                continue

            try:
                self.df[col] = self.df[col].astype(dtype)
            except Exception as e:
                print(f"❌ Failed to convert {col} to {dtype}: {e}")

        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col not in self.df.columns:
                    print(f"⚠️ Date column '{col}' not found, skipping")
                    continue

                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        return self

    def get_data(self) -> pd.DataFrame:

        return self.df
