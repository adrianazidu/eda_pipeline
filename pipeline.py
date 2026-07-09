#pip install matplotlib scipy numpy pandas statsmodels pygam scikit-learn shap
#usage  py .\pipeline.py --csv crypto.csv that has colsumns : date,open,high,low,close,volume
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

#gam model
from pygam import LinearGAM, s as _spline

#seasonal decomposition to separate trend, seasonality, and residual
from statsmodels.tsa.seasonal import STL 

#shap
from sklearn.ensemble import GradientBoostingRegressor
import shap

#LOF and StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

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
    #shap
    shap_estimators: int = 200                    # trees in the gradient-boosting surrogate
    shap_max_depth: int = 3
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

        # ingestion
        self._load(data)
        self._profile()        #profiling, produces 01_eda.png
        self._clean()
        self._engineer_features()

        #time series
        if self.cfg.date_col:
            self._decompose()    #STL decomposition, produce 02_timeseries.png
            self._stationarity() #ADF  and KPSS
            self._arima()            # 14 — informed by the two stages above

        self._correlate()        #correlation spearman/pearson, produce 03_correlation.png
        self._model_gam()        # apply GAM additive model to compute non linear relationships, produces 04_gam.png
        self._explain_shap()     #apply shap gradient booster to compute mathematical importance of each feature, produces 06_shap.png
        self._detect_anomalies() # use LOF, produces 05_anomalies.png

        report_path = self._report()
        self.log.info("Done. Report -> %s", report_path)

        print(self.results)

    def _engineer_features(self) -> None:
        """Derive informative columns from raw OHLCV and register them as features.

        Runs right after _clean. Appends new columns to self.df AND their names to
        cfg.feature_cols, so every downstream stage picks them up automatically with
        no further changes. Assumes columns: open, high, low, close, volume.
        """
        df, cfg = self.df, self.cfg
        o, h, l, c, v = "open", "high", "low", "close", "volume"
        if not all(col in df.columns for col in [o, h, l, c, v]):
            self.results["feature_engineering"] = {"skipped": "missing OHLCV columns"}
            return

        # --- derived columns (each line = one new feature) ---
        df["ret_1d"]      = df[c].pct_change()                          # daily return (the key transform)
        df["ret_7d"]      = df[c].pct_change(7)                         # weekly momentum
        df["log_ret"]     = np.log(df[c]).diff()                       # log return (additive over time)
        df["volatility_14d"] = df["ret_1d"].rolling(14).std()          # recent turbulence
        df["range_pct"]   = (df[h] - df[l]) / df[c]                    # intraday swing size (high - low) / close, diff between end and start day divided by close, to normalize it
        df["ma_20"]       = df[c].rolling(20).mean()                   # 20-day trend level
        df["close_vs_ma20"] = df[c] / df["ma_20"] - 1                  # distance from trend (>0 = above)
        df["vol_change"]  = df[v].pct_change()                         # volume momentum
        df["vol_vs_avg"]  = df[v] / df[v].rolling(20).mean()           # today's volume vs its norm

        """
        explanation
        - pct-change = a pandas method that computes the percentage change from the previous row, formula (current − previous) / previous, 0.10 means it grew with 10%
        - pct_change(7) compares to the row 7 back instead of 1 back, computing weekly return
        - volatility_14d - apply on the column ret_1d that we have just created, then rolling(14) — creates a sliding window of 14 consecutive rows 
                 which means group each row with the 13 rows before it, then compute standard deviation to each window
                 So the value on any given row is the standard deviation of the last 14 daily returns
                 he std is low → calm market, the std is high → turbulent market (measureing volatility)
        """

        new_features = ["ret_1d", "ret_7d", "log_ret", "volatility_14d", "range_pct",
                        "close_vs_ma20", "vol_change", "vol_vs_avg"]

        # drop warm-up rows that the rolling/lag windows left as NaN  (X data must not contain Inf nor NaN)
        before = len(df)
        df = df.dropna(subset=new_features).reset_index(drop=True)

        # register the new columns so stages see them, avoiding duplicates
        for f in new_features:
            if f not in cfg.feature_cols:
                cfg.feature_cols.append(f)

        self.df = df
        self.results["feature_engineering"] = {
            "added": new_features,
            "rows_dropped_warmup": before - len(df),
            "feature_cols_now": cfg.feature_cols,
        }
        self.log.info("Engineered %d features; dropped %d warm-up rows",
                    len(new_features), before - len(df))

    def _report(self)-> Path:
        path = self.out/"results.json"
        path.write_text(json.dumps(self.results, indent = 2, default=str))
        return path

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

    #clean empty values
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

     #perform STL decomposition on time series
    def _decompose(self) -> None:
        df, cfg = self.df, self.cfg

        #set datraframe index to date column, enforce daily frequencey (D)
        s = (df.set_index(cfg.date_col)[cfg.target_col]).asfreq("D")

        #fill missing data gaps with linear interpolation
        s = s.interpolate()  # STL needs a gap-free series

        #run STL, use a robust loop to ignore existing extreme outliers during calculations.
        stl = STL(s, period=cfg.seasonal_period, robust=True).fit()
    
        """extract values from result
          - residual = what is left after you strip away the predictable patterns (trend and seasonality) = pure noise 
          - The Threshold is a statistical boundary used to separate "normal everyday noise" from "something unusual happened."
             Equals = multiplier (2 or 3) * standard deviation of all residuals
            Standard deviation measures how much your typical daily noise fluctuates. By multiplying it by a factor (usually 2 or 3),
             you create a strict mathematical boundary. Anything inside this boundary is considered acceptable random variation.
        - outliers = the anomalies, where the value breaks the threshold
        """

        #the residual component (random noise)
        resid = stl.resid                       

        #Calculates an outlier threshold by multiplying a user-defined multiplier (usually 2 or 3 standard deviations) by the standard deviation of the residuals
        thresh = cfg.resid_sigma * resid.std()
        
        ##the residual component (random noise)
        resid_outliers = resid[resid.abs() > thresh]

        #create a dict with the results
        #trend change - Calculates the overall net change in the underlying trend line from the very first day to the very last day, rounded to 2 decimal places
        #seasonal amplitude - Calculates the peak-to-trough range of the seasonal pattern, showing the maximum magnitude of the cyclical swings
        #resid outlier dates - Converts the datetime index of the flagged outliers into a clean list of string formatted dates (YYYY-MM-DD).
        self.results["decompose"] = {
            "period": cfg.seasonal_period,
            "trend_change": round(float(stl.trend.iloc[-1] - stl.trend.iloc[0]), 2),
            "seasonal_amplitude": round(float(stl.seasonal.max() - stl.seasonal.min()), 2),
            "resid_outlier_dates": [str(d.date()) for d in resid_outliers.index],
        }
        # store for later reuse by anomaly stage
        self._stl_resid = resid
        self.log.info("Decomposed: trend change %.0f, %d residual outliers",
                      self.results["decompose"]["trend_change"], len(resid_outliers))

        weekly = s.resample("W").mean()
        monthly = s.resample("ME").mean()
        fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Time series — bucket aggregation + STL decomposition",
                     fontsize=15, fontweight="bold")
        ax[0].plot(s.index, s.values, color=cfg.palette["blue"], lw=0.6, alpha=0.7, label="daily")
        ax[0].plot(weekly.index, weekly.values, color=cfg.palette["red"], lw=1.6, label="weekly mean")
        ax[0].plot(monthly.index, monthly.values, color="black", lw=2, label="monthly mean")
        ax[0].set_title("Raw + bucket aggregations"); ax[0].legend(fontsize=9)
        ax[1].plot(stl.observed.index, stl.observed, color=cfg.palette["blue"], lw=0.6); ax[1].set_title("Observed")
        ax[2].plot(stl.trend.index, stl.trend, color="black", lw=1.8); ax[2].set_title("Trend")
        ax[3].plot(stl.seasonal.index, stl.seasonal, color=cfg.palette["green"], lw=0.5); ax[3].set_title("Seasonal")
        ax[4].plot(resid.index, resid, color=cfg.palette["orange"], lw=0.4)
        ax[4].axhline(0, color="black", lw=0.8)
        ax[4].scatter(resid_outliers.index, resid_outliers.values, color=cfg.palette["red"], s=40, zorder=5)
        ax[4].set_title("Residual (flagged outliers in red)")
        plt.tight_layout()
        self._save_fig(fig, "02_timeseries.png")

    """
    run before arima time series model
    concept - a time series is stationary if its mean/variance don't change over time
    this function runs 2 complementary tests, their results are opposite, 
    Two tests rather than one because each is weak in different ways: ADF has low power, and KPSS can be fooled by certain trend shapes
    ADF - Augemented Dickey-Fuller) and KPSS
    """
    def _stationarity(self) -> None:
        from statsmodels.tsa.stattools import adfuller, kpss
        import warnings
        df, cfg = self.df, self.cfg

        #makes the date column the DataFrame's index instead of an ordinary column
        #[cfg.target_col] — selects that single column, so we now have a pandas Series (one column + a date index), not a DataFrame. adfuller and kpss want a 1-D sequence.
        # asfreq("D") — enforces a daily frequency. This is the important one.
        # If your data is missing 2024-03-05 entirely, asfreq inserts that date with a NaN value. Without this, pandas has 900 rows but no idea whether they're consecutive
        #interpolate where value is Nan
        #dropna() — a safety net. interpolate() can't fill NaNs at the very start of a series, since there's no earlier value to interpolate from
        s = df.set_index(cfg.date_col)[cfg.target_col].asfreq("D").interpolate().dropna()

        # the ADF test
        #adfuller(s, ...) returns a tuple, not a single number
        #autolag="AIC" — the ADF test needs to decide how many lagged terms to include (the "Augmented" part of Augmented Dickey-Fuller)
        #Rather than picking a number, this tells it to try a range and keep whichever minimizes the Akaike Information Criterion
        #[1] grabs element index 1 — the p-value. That's all we want
        #float(...) — converts numpy's float64 to a plain Python floa
        adf_p = float(adfuller(s, autolag="AIC")[1])

        #the KPSS test
        #catch warnings that KPSS emots a interpolationwarning when p value falls outside its lookup table
        #the with block supresses warnings only inside
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #regression="c" — tests stationarity around a constant. The alternative, regression="ct", tests around a constant and a linear trend
            #lags="auto" — KPSS needs a bandwidth for its variance estimator; "auto" uses a data-driven rule rather than a hardcoded guess.
            kpss_p = float(kpss(s, regression="c", nlags="auto")[1])

        #compute verdict using both results
        verdict = ("stationary" if adf_p < .05 and kpss_p > .05 else
                "non-stationary" if adf_p > .05 and kpss_p < .05 else "inconclusive")

        #store result to be interpreted by arima
        self._d_order = 0 if verdict == "stationary" else 1

        self.results["stationarity"] = {"adf_pvalue": round(adf_p, 4),   # <0.05 => stationary
                                        "kpss_pvalue": round(kpss_p, 4),  # <0.05 => NON-stationary
                                        "verdict": verdict}
        self.log.info("Stationarity verdict: %s", verdict)

    def _arima(self) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        df, cfg = self.df, self.cfg
        s = df.set_index(cfg.date_col)[cfg.target_col].asfreq("D").interpolate()
        train, test = s.iloc[:-30], s.iloc[-30:]

        d = getattr(self, "_d_order", 1)          # default to 1 if stationarity didn't run
        model = SARIMAX(train, order=(1, d, 1), seasonal_order=(1, 0, 1, cfg.seasonal_period),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

        fc = model.get_forecast(30); pred = fc.predicted_mean
        mae = float(np.mean(np.abs(pred.values - test.values)))
        self.results["arima"] = {"order": "(1,1,1)(1,0,1)", "test_mae": round(mae, 2),
                                "aic": round(float(model.aic), 1)}
        self.log.info("SARIMA: test MAE %.2f", mae)
        fig, ax = plt.subplots(figsize=(12, 4)); ci = fc.conf_int()
        ax.plot(train.index[-90:], train.values[-90:], color=cfg.palette["blue"], label="train")
        ax.plot(test.index, test.values, color="black", label="actual")
        ax.plot(pred.index, pred.values, color=cfg.palette["red"], label="forecast")
        ax.fill_between(pred.index, ci.iloc[:, 0], ci.iloc[:, 1], color=cfg.palette["red"], alpha=0.2)
        ax.set_title("SARIMA 30-day forecast"); ax.legend(fontsize=9)
        plt.tight_layout(); self._save_fig(fig, "14_arima.png")

    #Correlations tell you which variables move together
    #Pearson captures linear relationships; Spearman or Kendall capture monotonic ones
    def _correlate(self) -> None:
        df, cfg = self.df, self.cfg
        #compute rank correlation matrix between target variable and all other feature columns
        cols = [cfg.target_col] + cfg.feature_cols
        pear = df[cols].corr("pearson")   
        spear = df[cols].corr("spearman")

        # flag pairs where spearman notably exceeds pearson (monotonic nonlinearity)
        flags = []

        #iterate through every column
        #then iterate again to pair the current column with the next one (i+1)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):

                #extract the correlation values and check abs diff 
                p, sp = pear.iloc[i, j], spear.iloc[i, j]

                # If true, it means the two variables move together predictably, but the relationship is curved or non-linear.
                if abs(sp) - abs(p) > 0.1:
                    flags.append({"pair": [cols[i], cols[j]],
                                  "pearson": round(float(p), 3),
                                  "spearman": round(float(sp), 3)})

        self.results["correlation"] = {
            "pearson_with_target": {c: round(float(pear.loc[cfg.target_col, c]), 3)
                                    for c in cfg.feature_cols},
            "spearman_with_target": {c: round(float(spear.loc[cfg.target_col, c]), 3)
                                     for c in cfg.feature_cols},
            "nonlinear_flags": flags,
        }
        self.log.info("Correlated: %d pair(s) show monotonic nonlinearity", len(flags))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Correlation — Pearson vs Spearman", fontsize=14, fontweight="bold")
        for a, M, name in zip(ax, [pear, spear], ["Pearson (linear)", "Spearman (rank)"]):
            im = a.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
            a.set_xticks(range(len(cols))); a.set_yticks(range(len(cols)))
            a.set_xticklabels(cols, rotation=45, ha="right"); a.set_yticklabels(cols)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    v = M.iloc[i, j]
                    a.text(j, i, f"{v:.2f}", ha="center", va="center",
                           color="white" if abs(v) > 0.5 else "black", fontsize=9)
            a.set_title(name); plt.colorbar(im, ax=a, fraction=0.046)
        plt.tight_layout()
        self._save_fig(fig, "03_correlation.png")


    #trains a GAM (Generalized Additive Model) using the pygam library
    # capture non-linear relationships between your input variables (features) and the target variable, 
    #and then visualize the isolated effect of each feature using partial dependence plots.
    def _model_gam(self) -> None:

        df, cfg = self.df, self.cfg

        #Extracts the values of the feature columns as a 2D NumPy array (X)
        X = df[cfg.feature_cols].values
        #Extracts the values of the target column as a 1D NumPy array (y)
        y = df[cfg.target_col].values

        #Initializes the mathematical structure of the GAM by creating a spline (a flexible, smooth curve)  (on index 0 for first column)
        terms = _spline(0)

        #continue with a loop starting from index  1 , add a new spline for each feature 
        # gam is an additive model so feature functions are summed together
        for k in range(1, len(cfg.feature_cols)):
            terms = terms + _spline(k)

        #nstantiates a Linear GAM using the constructed spline terms and trains it (.fit(X, y)) to find the smooth curves that best fit the dataset
        gam = LinearGAM(terms).fit(X, y)

        #Extracts the Pseudo R² value based on explained deviance. This tells you the proportion of variance explained by this non-linear model 
        #(closer to 1.0 means a stronger fit)
        pr2 = float(gam.statistics_["pseudo_r2"]["explained_deviance"])

        #save the result
        self.results["gam"] = {"pseudo_r2": round(pr2, 3), "features": cfg.feature_cols}
        self.log.info("GAM fitted: pseudo R^2 = %.3f", pr2)

        """
        GAM Bridges the Gap Between Simple and Complex Models
        - Linear Regression: Easy to understand, but too dumb to capture complex curves.
        - Machine Learning (XGBoost / Random Forests): Highly accurate, but complete "black boxes" that are impossible to explain to a client
        """

        n = len(cfg.feature_cols)
        fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
        fig.suptitle("GAM — partial effect of each feature", fontsize=14, fontweight="bold")
        for k, f in enumerate(cfg.feature_cols):
            XX = gam.generate_X_grid(term=k)
            pdep, ci = gam.partial_dependence(term=k, X=XX, width=0.95)
            a = ax[0, k]
            a.plot(XX[:, k], pdep, color=cfg.palette["blue"], lw=2.5)
            a.fill_between(XX[:, k], ci[:, 0], ci[:, 1], color=cfg.palette["blue"], alpha=0.2)
            a.set_title(f"f({f})"); a.set_xlabel(f); a.set_ylabel("partial effect")
        plt.tight_layout()
        self._save_fig(fig, "04_gam.png")


    #trains a complex predictive model (Gradient Boosting) and then dissects it row by row using SHAP (SHapley Additive exPlanations) values
    def _explain_shap(self) -> None:
        """Fit a flexible gradient-boosting model and explain it with SHAP.

        The GAM stage gives interpretable *additive* partial effects; SHAP here
        explains a model that can also capture feature *interactions*, and
        attributes each prediction back to the features. Mean |SHAP| is a robust
        global importance ranking; the beeswarm shows direction and the
        dependence plot reveals nonlinear shape (e.g. the U in temperature).
        """
        df, cfg = self.df, self.cfg
        X = df[cfg.feature_cols]
        y = df[cfg.target_col].values
        
        """
        Instantiates and trains a Gradient Boosting Regressor.
         This model is an ensemble of decision trees that learns complex, non-linear patterns and deep interactions between variables.
         It uses the number of trees (n_estimators) and maximum depth (max_depth) defined in your configuration
        """
        model = GradientBoostingRegressor(
            n_estimators=cfg.shap_estimators, max_depth=cfg.shap_max_depth,
            random_state=0).fit(X, y)

        #Calculates the traditional R² (coefficient of determination) of the Gradient Boosting model on the training data to see how accurately it fits
        r2 = float(model.score(X, y))

        #This algorithm calculates the exact mathematical impact of each feature on every individual prediction incredibly fast
        explainer = shap.TreeExplainer(model)

        """
        Calculates the array of SHAP values (sv) for every single row in your dataset.
         For every row of data fed into it, SHAP assigns a positive or negative score to each feature, 
        quantifying how much that specific value pushed the prediction up or down relative to the global baseline average.
        """
        sv = explainer(X)

        """
        Calculates the overall global importance of each variable.
        It takes the absolute value (np.abs) of all SHAP scores to ignore whether the impact was positive or negative, 
        and averages them (.mean(axis=0)) across the entire dataset. 
        The feature with the highest average absolute SHAP value is the one that moves the model's needle the most.
        """
        mean_abs = np.abs(sv.values).mean(axis=0)

        #Creates a Python dictionary pairing each feature name with its average SHAP importance score, rounded to two decimal places
        importance = {f: round(float(v), 2) for f, v in zip(cfg.feature_cols, mean_abs)}

        #Sorts the importance dictionary in descending order (reverse=True), 
        #placing the most influential variable at the very top of the list and the least important one at the bottom.
        ranked = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        self.results["shap"] = {
            "surrogate_model": "GradientBoostingRegressor",
            "surrogate_r2": round(r2, 3),      #the accuracy R² of the Gradient Boosting model, rounded to 3 decimal places
            "mean_abs_shap": ranked,            #the complete, ordered ranking of feature importances
            "top_feature": next(iter(ranked)),  # the name of the absolute most important variable (the first key in the sorted ranked dictionary
        }
        self.log.info("SHAP: surrogate R^2 = %.3f, top feature = %s",
                      r2, next(iter(ranked)))

        # Figure: global importance (bar) + beeswarm + dependence of top feature
        top = next(iter(ranked))
        top_idx = cfg.feature_cols.index(top)
        fig = plt.figure(figsize=(16, 5))
        fig.suptitle("SHAP — explaining the gradient-boosting model",
                     fontsize=14, fontweight="bold")
        ax1 = fig.add_subplot(1, 3, 1)
        order = list(ranked.keys())
        ax1.barh(order[::-1], [ranked[f] for f in order[::-1]], color=cfg.palette["blue"])
        ax1.set_title("Global importance (mean |SHAP|)")
        ax1.set_xlabel("mean |SHAP value|")

        ax2 = fig.add_subplot(1, 3, 2)
        plt.sca(ax2)
        shap.summary_plot(sv.values, X, show=False, plot_size=None, color_bar=True)
        ax2.set_title("Beeswarm (impact + direction)")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(X[top], sv.values[:, top_idx], s=10, alpha=0.4,
                    color=cfg.palette["orange"])
        ax3.axhline(0, color="black", lw=0.8)
        ax3.set_title(f"Dependence: {top}")
        ax3.set_xlabel(top); ax3.set_ylabel(f"SHAP value for {top}")
        ax3.grid(alpha=0.25)
        plt.tight_layout()
        self._save_fig(fig, "06_shap.png")

    """
     LOF is a density-based anomaly detector. Its core idea: an outlier isn't simply a point far from the average 
     it's a point in a sparser neighborhood than its neighbors occupy
     LOF compares the local density around it to the local densities around its k nearest neighbors. 
    A score near 1 means "as crowded as my neighbors" (normal); a score well above 1 means "I sit in a much emptier region than the points near me" (anomalous).
    Scale your features (StandardScaler) before running LOF, because it relies on distances
    """
    def _detect_anomalies(self) -> None:
        df, cfg = self.df, self.cfg
        feats = [cfg.target_col] + cfg.feature_cols

        #scaling shifts all variables to a standardized space where the mean is 0 and the standard deviation is 1. 
        #This prevents massive columns from dominating the distance calculations
        Xs = StandardScaler().fit_transform(df[feats])

        #istantiate LOF algorithm
        #n_neighbors: The size of the local neighborhood cluster to evaluate against (e.g., 20 points)
        #contamination: The expected percentage of anomalies present in the dataset (e.g., 0.01 for 1%)
        lof = LocalOutlierFactor(n_neighbors=cfg.lof_neighbors,
                                 contamination=cfg.lof_contamination)

        #Executes the spatial proximity logic on the scaled matrix. It evaluates the density of each point against its closest neighbors
        # It outputs a simple array of flags: 1 for safe/normal rows, and -1 for anomalies
        labels = lof.fit_predict(Xs)

        #create dedicated copy to prevent warnings
        df = df.copy()

        """
        Extracts the raw local density scores. By default, scikit-learn records these as negative values where lower means more anomalous. 
        Inverting it with the negative sign (-) transforms it into a standard, clean score: a value near or below 1.0 is totally normal,
         while high scores (like 1.5, 2.0, or higher) indicate an extreme outlier
        """
        df["lof_score"] = -lof.negative_outlier_factor_

        #Creates a clear Boolean column (True or False) indicating whether LOF officially flagged that specific row as a high-density anomaly
        df["is_outlier"] = labels == -1

        #overwrite df with the new columns lof_score and is_outlier
        self.df = df

        #filter rows marked as true outliers, sorting them from the most extreme anomaly score down to the least extreme
        flagged = df[df.is_outlier].sort_values("lof_score", ascending=False)

        #isolate the columns we need
        record = flagged[feats + ["lof_score"]].round(2)
        if cfg.date_col:
            record.insert(0, cfg.date_col, flagged[cfg.date_col].dt.date.astype(str).values)
        self.results["anomalies"] = {
            "method": "LocalOutlierFactor",
            "n_flagged": int(df.is_outlier.sum()),            #the absolute count of how many rows were flagged as anomalies
            "top": record.head(15).to_dict(orient="records"), #takes the top 15 most severe anomalies and translates them into a clean list of row dictionaries
        }
        self.log.info("Anomaly detection: %d points flagged", int(df.is_outlier.sum()))

        if cfg.date_col:
            fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
            fig.suptitle("Anomalies — Local Outlier Factor", fontsize=14, fontweight="bold")
            ax[0].plot(df[cfg.date_col], df[cfg.target_col], color=cfg.palette["blue"], lw=0.6, alpha=0.7)
            ax[0].scatter(flagged[cfg.date_col], flagged[cfg.target_col],
                          color=cfg.palette["red"], s=70, zorder=5, label="outlier")
            ax[0].set_title("Outliers over time"); ax[0].legend(fontsize=9)
            f0 = cfg.feature_cols[0]
            sc = ax[1].scatter(df[f0], df[cfg.target_col], c=df.lof_score, cmap="viridis", s=14, alpha=0.6)
            ax[1].scatter(flagged[f0], flagged[cfg.target_col], facecolors="none",
                          edgecolors=cfg.palette["red"], s=140, linewidths=1.8)
            ax[1].set_title(f"Feature space ({f0} vs {cfg.target_col})")
            ax[1].set_xlabel(f0); ax[1].set_ylabel(cfg.target_col)
            plt.colorbar(sc, ax=ax[1], label="LOF score")
            plt.tight_layout()
            self._save_fig(fig, "05_anomalies.png")

        # persist the enriched dataset
        df.to_csv(self.out / "scored_data.csv", index=False)



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