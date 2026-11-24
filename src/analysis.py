"""Data cleaning, summaries and charts shared by the notebook and the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "uk_used_cars.csv"

# The listings were collected in 2020, so a car's age is measured from 2020.
DATA_YEAR = 2020
TARGET = "price"
MANUFACTURER_NAMES = {"hyundi": "Hyundai", "merc": "Mercedes", "ford": "Ford", "volkswagen": "Volkswagen",
                      "vauxhall": "Vauxhall", "toyota": "Toyota", "skoda": "Skoda", "BMW": "BMW", "Audi": "Audi"}
NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"

sns.set_theme(style="whitegrid")


def pounds(x, _=None):
    if abs(x) < 1000:
        return f"£{x:,.0f}"
    k = x / 1000
    return f"£{k:,.0f}k" if k == int(k) else f"£{k:,.1f}k"


# ---------------------------------------------------------------- loading and cleaning


def load_raw(path=RAW) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned data and a log of each cleaning step."""
    df = raw.copy()
    log = [("Raw rows", len(df))]

    df = df.dropna(subset=[TARGET])
    log.append(("After dropping rows with no price", len(df)))

    df = df.drop_duplicates()
    log.append(("After dropping exact duplicates", len(df)))

    df = df[df["year"] <= DATA_YEAR]
    log.append((f"After dropping cars registered after {DATA_YEAR}", len(df)))

    df["model"] = df["model"].str.strip()
    df["Manufacturer"] = df["Manufacturer"].map(MANUFACTURER_NAMES).fillna(df["Manufacturer"])

    # An engine size of 0 is impossible for a non-electric car, so it is treated as missing.
    zero_engine = (df["engineSize"] == 0) & (df["fuelType"] != "Electric")
    df.loc[zero_engine, "engineSize"] = np.nan

    df["age"] = DATA_YEAR - df["year"]
    log = pd.DataFrame(log, columns=["step", "rows"])
    log["removed"] = (-log["rows"].diff()).fillna(0).astype(int)
    return df.reset_index(drop=True), log


def load_clean(path=RAW) -> pd.DataFrame:
    return clean(load_raw(path))[0]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    return pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2)})[lambda t: t["missing"] > 0]


# ---------------------------------------------------------------- summaries


def price_by(df: pd.DataFrame, col: str, min_count: int = 1) -> pd.DataFrame:
    g = df.groupby(col)[TARGET]
    out = pd.DataFrame({"listings": g.size(), "median_price": g.median(), "mean_price": g.mean().round(0)})
    out = out[out["listings"] >= min_count]
    return out.sort_values("median_price", ascending=False).reset_index()


def depreciation_curve(df: pd.DataFrame, max_age: int = 10) -> pd.DataFrame:
    """Median price by age, and the value kept compared with cars under one year old."""
    d = df[df["age"] <= max_age].groupby("age")[TARGET].agg(listings="size", median_price="median").reset_index()
    d["value_kept_%"] = (d["median_price"] / d.loc[d["age"] == 0, "median_price"].iloc[0] * 100).round(1)
    return d


def depreciation_by_manufacturer(df: pd.DataFrame, ages=(0, 3, 5)) -> pd.DataFrame:
    pivot = (df[df["age"].isin(ages)].groupby(["Manufacturer", "age"])[TARGET].median().unstack())
    pivot.columns = [f"age_{a}" for a in pivot.columns]
    pivot["value_kept_after_3y_%"] = (pivot["age_3"] / pivot["age_0"] * 100).round(1)
    pivot["value_kept_after_5y_%"] = (pivot["age_5"] / pivot["age_0"] * 100).round(1)
    return pivot.sort_values("value_kept_after_3y_%", ascending=False).round(0).reset_index()


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], bins=60, color=BLUE, ax=axes[0])
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(pounds))
    axes[0].set_title(f"Price (skewness {df[TARGET].skew():.2f})", loc="left", fontweight="bold")
    sns.histplot(np.log(df[TARGET]), bins=60, color=NAVY, ax=axes[1])
    axes[1].set_title(f"Log of price (skewness {np.log(df[TARGET]).skew():.2f})", loc="left", fontweight="bold")
    axes[0].set_xlabel("Price")
    axes[1].set_xlabel("log(price)")
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_depreciation(df: pd.DataFrame, max_age: int = 10):
    d = depreciation_curve(df, max_age)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["age"], d["median_price"], marker="o", color=BLUE, linewidth=2.2)
    for _, r in d.iterrows():
        if r["age"] in (1, 3, 5, 7, 10):
            ax.annotate(f"{r['value_kept_%']:.0f}%", (r["age"], r["median_price"]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=10, color=NAVY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pounds))
    ax.set_xticks(range(0, max_age + 1))
    ax.set_xlabel("Age in 2020 (years)")
    ax.set_ylabel("Median price")
    return _finish(ax, "Depreciation: median price by age (labels show value kept vs new)")


def plot_mileage_vs_price(df: pd.DataFrame, sample: int = 6000):
    s = df.sample(min(sample, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.scatterplot(data=s, x="mileage", y=TARGET, hue="age", palette="viridis_r", s=10, alpha=0.6,
                    linewidth=0, ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pounds))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlim(0, df["mileage"].quantile(0.995))
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price (log scale)")
    ax.legend(title="Age", frameon=False, ncol=2, fontsize=8, title_fontsize=9)
    return _finish(ax, f"Mileage vs price ({len(s):,} random listings)")


def plot_price_by_manufacturer(df: pd.DataFrame):
    order = df.groupby("Manufacturer")[TARGET].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=df, x="Manufacturer", y=TARGET, order=order, color="#cfe2f3", fliersize=0, width=0.6, ax=ax)
    ax.set_ylim(0, df[TARGET].quantile(0.99))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pounds))
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    ax.tick_params(axis="x", rotation=30)
    return _finish(ax, "Price by manufacturer")


def plot_price_by_category(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["fuelType", "transmission", "engineSize"]):
        if col == "engineSize":
            d = df.assign(engine=pd.cut(df["engineSize"], [0, 1.0, 1.5, 2.0, 3.0, 7.0],
                                        labels=["up to 1.0", "1.1-1.5", "1.6-2.0", "2.1-3.0", "over 3.0"]))
            p = d.groupby("engine", observed=True)[TARGET].median()
            ax.set_xlabel("Engine size (litres)")
        else:
            p = price_by(df, col, min_count=50).set_index(col)["median_price"]
            ax.set_xlabel("")
        ax.bar(p.index.astype(str), p.values, color=BLUE)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(pounds))
        ax.set_title(f"Median price by {col.replace('Type', ' type').replace('Size', ' size').lower()}",
                     loc="left", fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_correlations(df: pd.DataFrame):
    cols = ["price", "age", "mileage", "engineSize", "mpg", "tax"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=1, ax=ax)
    ax.set_title("Correlation between numeric columns", fontsize=13, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig
