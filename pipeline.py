#pip install matplotlib scipy numpy pandas
#usage  py .\pipeline.py --csv crypto.csv
#py .\pipeline.py --csv crypto.csv --date date --features 'open' --target volume

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

#config
@dataclass
class PipelineConfig:
    """Everything the pipeline needs to adapt to a new dataset."""
    target_col: str                              # the variable of interest / model output
    feature_cols: list[str]                      # numeric predictors
    date_col: Optional[str] = None               # set to enable the time-series stages
    seasonal_period: int = 7                      # STL cycle length (7=weekly daily data, 12=monthly)
     # anomaly detection
    lof_neighbors: int = 20
    lof_contamination: float = 0.01
    resid_sigma: float = 3.0                      # STL-residual flag threshold
    #ouput
    output_dir: str = "pipeline_output"
    drop_duplicates: bool = True
    impute_numeric: str = "median"                # "median" | "mean" | "drop" | "none"
    palette: dict = field(default_factory=lambda: {
        "blue": "#2563eb", "orange": "#ea580c",
        "red": "#dc2626", "green": "#16a34a", "grey": "#64748b",
    })

class DataAnalysisPipeline:
    """run the stages in order"""
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.out = Path(config.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
        self.results: dict = {}          # machine-readable findings per stage
        self.figures: list[Path] = []    # paths to generated PNGs
        self._setup_logging()
        self._setup_style()

    def run(self, data) -> dict:
        """Execute the full pipeline. `data` is a path or a DataFrame."""
        self._load(data)
        self._profile()
        self._clean()

    def _load(self, data) -> None:

        #check if data is an instance of dataframe,if it is load a copy of it so to not change it by mistake (deep copy),
        # if not, it's a csv and load dataframe from it
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            #use the column set in CLI as date column to parse the input data in the csv
            parse = [self.cfg.date_col] if self.cfg.date_col else None

            #the parse_dates argument can be None or an array of columns
            self.df = pd.read_csv(data, parse_dates=parse)

        #order rows of csv based on date (only if None)
        if self.cfg.date_col:
            self.df = self.df.sort_values(self.cfg.date_col).reset_index(drop=True)

        self.log.info("Loaded %d rows x %d cols", *self.df.shape)

    def _profile(self)->None:

        df,cfg = self.df, self.cfg
        num_cols = [cfg.target_col] + cfg.feature_cols
        prof = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "duplicates": int(df.duplicated().sum()), #the sum of duplicated rows
            "missing": df.isna().sum().to_dict(),     #sum of all misgginf values, by column {'colonna_A': 0, 'colonna_B': 5, 'colonna_C': 12}
            "describe": df[num_cols].describe().round(2).to_dict(), # describe params for each column round to 2 decimals
            "skew": {c: round(float(df[c].skew()), 3) for c in num_cols}, #calculate the skewness of each column
            "kurtosis": {c: round(float(df[c].kurt()), 3) for c in num_cols}, # Calculates the kurtosis for each column
        }

        #describe - {'describe': {'column_A': {'mean': 10.50, 'std': 2.15}}}.(count, mean, standard deviation, minimum, 25%, 50%, 75% percentiles, and maximum)
        #Skewness measures the asymmetry of the data distribution. A positive value means a tail stretching to the right; a negative value means a tail stretching to the left.
        # High kurtosis indicates the presence of heavy tails (more outliers), while low kurtosis indicates light tails (fewer outliers)
        self.log.info("Profiled: %d duplicates, %d cols with missing values",
                      prof["duplicates"], sum(v > 0 for v in prof["missing"].values()))

         # EDA figure: a histogram per numeric column + target scatter vs each feature
        ncols = len(num_cols)
        fig, ax = plt.subplots(2, max(ncols, len(cfg.feature_cols)),
                               figsize=(4.8 * max(ncols, 1), 8), squeeze=False)

        fig.suptitle("EDA — distributions & target relationships", fontsize=15, fontweight="bold")
        colors = list(cfg.palette.values())
        for i, c in enumerate(num_cols):
            a = ax[0, i]
            a.hist(df[c].dropna(), bins=40, color=colors[i % len(colors)],
                   alpha=0.8, edgecolor="white", linewidth=0.4)
            a.axvline(df[c].mean(), color="black", ls="--", lw=1)
            a.set_title(f"Distribution: {c}")
        for j in range(ncols, ax.shape[1]):
            ax[0, j].axis("off")
        for i, f in enumerate(cfg.feature_cols):
            a = ax[1, i]
            a.scatter(df[f], df[cfg.target_col], s=8, alpha=0.35,
                      color=cfg.palette["orange"])
            a.set_title(f"{cfg.target_col} vs {f}")
            a.set_xlabel(f); a.set_ylabel(cfg.target_col)
        for j in range(len(cfg.feature_cols), ax.shape[1]):
            ax[1, j].axis("off")
        plt.tight_layout()
        self._save_fig(fig, "01_eda.png")

    def _clean(self) -> None:
        df, cfg = self.df, self.cfg
        before = len(df)
        if cfg.drop_duplicates:
            df = df.drop_duplicates()
        num_cols = [cfg.target_col] + cfg.feature_cols

        # drop rows with missing values, fill them with the column's median, or fill them with the column's mean.

        #drop columns that are in the num_cols list, not all
        if cfg.impute_numeric == "drop":
            df = df.dropna(subset=num_cols)
        elif cfg.impute_numeric in ("median", "mean"):
            for c in num_cols:
                #It checks if the column actually contains any missing values 
                if df[c].isna().any():
                    #A ternary operator (conditional expression) that calculates either the median or the mean of that specific column.
                    fill = df[c].median() if cfg.impute_numeric == "median" else df[c].mean()
                    #It replaces all NaN values in that specific column with the calculated fill value.
                    df[c] = df[c].fillna(fill)

        self.df = df.reset_index(drop=True)
        self.results["clean"] = {"rows_before": before, "rows_after": len(self.df),
                                    "strategy": cfg.impute_numeric}
        self.log.info("Cleaned: %d -> %d rows", before, len(self.df))

    def _save_fig(self, fig, name: str) -> Path:
        path = self.out / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path)
        return path

    # -- infrastructure ----------------------------------------------------- #
    def _setup_logging(self) -> None:

        #creates a logger object called pipeline
        self.log = logging.getLogger("pipeline")

        #checks logger does not already have handlers
        if not self.log.handlers:

            #create streamhandler that sends logs directly do sys.stderror
            h = logging.StreamHandler()

            #Defines the exact visual layout and timestamp formatting for the log messages.
            #asctime timstamp  14:35:01
            #levelname = log severity level - restrict to 5 chars
            #messagge is the actual message we send to the log
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                                             datefmt="%H:%M:%S"))
            #attach handler to logging object
            self.log.addHandler(h)
        self.log.setLevel(logging.INFO)

    def _setup_style(self) -> None:

        #update matplotlib global dictionary
        mpl.rcParams.update({
            "figure.dpi": 110, "savefig.dpi": 110, "font.size": 11,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
            "figure.facecolor": "white", "axes.facecolor": "white",
        })


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the data-analysis pipeline on a CSV.")
    p.add_argument("--csv", required=True, help="path to input CSV")
    p.add_argument("--target", help="target / response column")
    p.add_argument("--features", nargs="+",  help="numeric feature columns")
    p.add_argument("--date", default=None, help="date column name example 'sale_date'(enables time-series stages)")
    p.add_argument("--period", type=int, default=7, help="STL seasonal period")
    p.add_argument("--contamination", type=float, default=0.01, help="expected outlier fraction")
    p.add_argument("--out", default="pipeline_output", help="output directory")
    return p.parse_args()

def main() -> None:
    a = _parse_args()
    cfg = PipelineConfig(
        target_col=a.target, feature_cols=a.features, date_col=a.date,
        seasonal_period=a.period, lof_contamination=a.contamination, output_dir=a.out,
    )
    DataAnalysisPipeline(cfg).run(a.csv)


if __name__ == "__main__":
    main()