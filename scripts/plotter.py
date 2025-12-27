# src/plotter.py

from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:

    def __init__(
        self,
        df: pd.DataFrame,
        style: str = "whitegrid",
        figsize: tuple = (8, 5)
    ):
    
        self.df = df
        self.figsize = figsize
        sns.set_style(style)

    # -----------------------
    # Internal validation
    # -----------------------
    def _validate_columns(self, columns: List[str]) -> None:
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")

    # -----------------------
    # Plot methods
    # -----------------------
    def barplot(
        self,
        x: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        palette: Optional[str] = None,
        rotation: int = 45,
        figsize: tuple | None = None
    ) -> None:
        # 1. Validate only X and HUE (since Y is calculated automatically)
        self._validate_columns([x] + ([hue] if hue else []))

        plt.figure(figsize=figsize or (12, 6))
        
        # 2. Use countplot to handle the counting logic
        sns.countplot(data=self.df, x=x, hue=hue, palette=palette)
        
        plt.xticks(rotation=rotation)
        plt.ylabel("Count")  # Y is now the frequency
        plt.title(title or f"Count of {x}")
        plt.tight_layout()
        plt.show()


    def histogram(
        self,
        column: str,
        bins: int = 30,
        title: Optional[str] = None
    ) -> None:
        self._validate_columns([column])

        plt.figure(figsize=(8, 4))  # updated to match your original code
        sns.histplot(self.df[column], bins=bins, kde=True)
        plt.xlabel(column)
        plt.title(title or f"Distribution of {column}")
        plt.tight_layout()
        plt.show()


    def boxplot(
        self,
        y: str,
        x: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        self._validate_columns([y] + ([x] if x else []))

        plt.figure(figsize=self.figsize)
        sns.boxplot(data=self.df, x=x, y=y)
        plt.title(title or f"Boxplot of {y}")
        plt.tight_layout()
        plt.show()


    def lineplot_time_series(
        self,
        column: str,
        title: Optional[str] = None,
        figsize: tuple = (12, 4)
    ) -> None:
        # 1. Ensure the column exists
        self._validate_columns([column])

        # 2. Group by date and calculate counts
        # Assumes column is already converted to datetime
        daily_counts = self.df.groupby(self.df[column].dt.date).size()

        # 3. Create the plot
        plt.figure(figsize=figsize)
        daily_counts.plot(kind="line")
        
        plt.title(title or f"Daily Trend: {column}")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


    def countplot(
        self,
        columns: list[str],
        target: str,
        rotation: int = 45,
        top_n: int | None = None
        ) -> None:
            
        self._validate_columns(columns + [target])

        for col in columns:
            # Print grouped counts (same as your notebook)
            print(f"\n{col.upper()} distribution by {target}:")
            print(self.df.groupby(target)[col].value_counts())

            # Order categories by frequency
            order = self.df[col].value_counts().index
            if top_n:
                order = order[:top_n]

            plt.figure(figsize=self.figsize)
            sns.countplot(
                data=self.df,
                x=col,
                hue=target,
                order=order
            )
            plt.title(f"{col} Distribution by {target}")
            plt.xticks(rotation=rotation)
            plt.tight_layout()
            plt.show()


    def boxplot_by_target(
        self,
        numeric_columns: list[str],
        target: str
    ) -> None:
        
        self._validate_columns(numeric_columns + [target])

        for col in numeric_columns:
            print(f"\n{col.upper()} distribution by {target}:")
            print(self.df.groupby(target)[col].value_counts().sort_index())

            plt.figure(figsize=self.figsize)
            sns.boxplot(
                data=self.df,
                x=target,
                y=col
            )
            plt.title(f"{col} Distribution by {target}")
            plt.tight_layout()
            plt.show()


    def lineplot_by_date(
        self,
        datetime_column: str,
        title: Optional[str] = None,
        xlabel: str = "Date",
        ylabel: str = "Count",
        figsize: tuple = (12, 4)
    ) -> None:
        
        self._validate_columns([datetime_column])

        # Aggregate counts by date
        daily_counts = self.df.groupby(self.df[datetime_column].dt.date).size()

        # Plot
        daily_counts.plot(figsize=figsize)
        plt.title(title or f"Daily Trend of {datetime_column}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    