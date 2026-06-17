#pip install matplotlib scipy numpy pandas

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