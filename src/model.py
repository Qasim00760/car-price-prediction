"""Used car price model.

The model is trained on the log of price, because prices are right-skewed and the
errors that matter are relative ones (being £1,000 off matters more on a £5,000 car
than on a £50,000 car). Predictions are converted back to pounds for evaluation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.analysis import BLUE, NAVY, RED, TARGET, pounds

CATEGORICAL = ["Manufacturer", "model", "transmission", "fuelType"]
NUMERIC = ["age", "mileage", "tax", "mpg", "engineSize"]
FEATURES = CATEGORICAL + NUMERIC
RANDOM_STATE = 42


def _one_hot_prep():
    numeric = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                OneHotEncoder(handle_unknown="ignore", min_frequency=20))
    return ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", numeric, NUMERIC)])


def _ordinal_prep():
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    return ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", "passthrough", NUMERIC)])


def _log_target(pipeline):
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def gradient_boosting():
    return _log_target(Pipeline([
        ("prep", _ordinal_prep()),
        ("model", HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=63,
                                                categorical_features=list(range(len(CATEGORICAL))),
                                                random_state=RANDOM_STATE)),
    ]))


def candidate_models() -> dict:
    return {
        "Baseline (median price)": DummyRegressor(strategy="median"),
        "Ridge Regression": _log_target(Pipeline([("prep", _one_hot_prep()), ("model", Ridge(alpha=1.0))])),
        "Random Forest": _log_target(Pipeline([
            ("prep", _one_hot_prep()),
            ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                            random_state=RANDOM_STATE)),
        ])),
        "Gradient Boosting": gradient_boosting(),
    }


def split(df: pd.DataFrame, features=FEATURES):
    return train_test_split(df[features], df[TARGET], test_size=0.2, random_state=RANDOM_STATE)


def scores(y_true, y_pred) -> dict:
    return {
        "MAE_gbp": mean_absolute_error(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = split(df)
    rows = []
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        rows.append({"model": name, **scores(y_test, model.predict(X_test))})
    return pd.DataFrame(rows).round({"MAE_gbp": 0, "MAPE_%": 1, "R2": 4})


def cross_validate_final(df: pd.DataFrame, folds: int = 5) -> pd.Series:
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    r2 = cross_val_score(gradient_boosting(), df[FEATURES], df[TARGET], cv=cv, scoring="r2")
    return pd.Series(r2, index=[f"fold {i + 1}" for i in range(folds)]).round(4)


def leakage_demo(df: pd.DataFrame) -> pd.DataFrame:
    """What the first version of this project did: a feature built from the price itself."""
    leaky = df.assign(price_per_mileage=df[TARGET] / (df["mileage"] + 1))
    X_train, X_test, y_train, y_test = split(leaky, FEATURES + ["price_per_mileage"])
    rows = []
    for name, feats, numeric in [("Without price_per_mileage", FEATURES, NUMERIC),
                                 ("With price_per_mileage (built from price)",
                                  FEATURES + ["price_per_mileage"], NUMERIC + ["price_per_mileage"])]:
        prep = ColumnTransformer([
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), CATEGORICAL),
            ("num", "passthrough", numeric),
        ])
        model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=60, n_jobs=-1,
                                                                          random_state=RANDOM_STATE))])
        model.fit(X_train[feats], y_train)
        rows.append({"features": name, **scores(y_test, model.predict(X_test[feats]))})
    return pd.DataFrame(rows).round({"MAE_gbp": 0, "MAPE_%": 2, "R2": 5})


def train_final_model(df: pd.DataFrame):
    return gradient_boosting().fit(df[FEATURES], df[TARGET])


def test_predictions(df: pd.DataFrame) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = split(df)
    model = gradient_boosting().fit(X_train, y_train)
    out = X_test.copy()
    out["actual"] = y_test
    out["predicted"] = model.predict(X_test).round(0)
    out["error"] = out["predicted"] - out["actual"]
    out["abs_pct_error"] = (out["error"].abs() / out["actual"] * 100).round(1)
    return out


def error_by(preds: pd.DataFrame, col: str) -> pd.DataFrame:
    g = preds.groupby(col)
    return (pd.DataFrame({"test_listings": g.size(),
                          "MAE_gbp": g["error"].apply(lambda e: e.abs().mean()).round(0),
                          "median_abs_pct_error": g["abs_pct_error"].median()})
            .sort_values("test_listings", ascending=False).reset_index())


def feature_importance(df: pd.DataFrame, sample: int = 5000) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = split(df)
    model = gradient_boosting().fit(X_train, y_train)
    Xs, ys = X_test.sample(sample, random_state=RANDOM_STATE), y_test.sample(sample, random_state=RANDOM_STATE)
    result = permutation_importance(model, Xs, ys, n_repeats=5, random_state=RANDOM_STATE,
                                    scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": FEATURES, "MAE_increase_gbp": result.importances_mean})
            .sort_values("MAE_increase_gbp", ascending=False).round(0).reset_index(drop=True))


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_model_comparison(comparison: pd.DataFrame):
    c = comparison.set_index("model")
    colors = [RED if "Baseline" in m else BLUE for m in c.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(c.index, c["MAE_gbp"], color=colors)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(pounds))
    axes[0].set_xlabel("Mean absolute error (lower is better)")
    axes[1].barh(c.index, c["R2"].clip(lower=0), color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("R2 on test set (higher is better)")
    axes[1].set_yticklabels([])
    for ax, col, fmt in [(axes[0], "MAE_gbp", "£{:,.0f}"), (axes[1], "R2", "{:.3f}")]:
        for i, v in enumerate(c[col]):
            ax.text(max(v, 0) * 1.01 + 0.01 * (col == "R2"), i, fmt.format(v), va="center", fontsize=10)
        ax.invert_yaxis()
        sns.despine(ax=ax)
    fig.suptitle("Model comparison on the same 20% test set", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(preds: pd.DataFrame, sample: int = 5000):
    s = preds.sample(min(sample, len(preds)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(s["actual"], s["predicted"], s=6, alpha=0.4, color=BLUE)
    lim = [s[["actual", "predicted"]].min().min() * 0.9, s[["actual", "predicted"]].max().max() * 1.1]
    ax.plot(lim, lim, color="#555", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(pounds))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Actual price (log scale)")
    ax.set_ylabel("Predicted price (log scale)")
    return _finish(ax, "Actual vs predicted price (test set)")


def plot_feature_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance["feature"], importance["MAE_increase_gbp"], color=NAVY)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_xlabel("Increase in MAE when the feature is shuffled")
    return _finish(ax, "What the model relies on")
