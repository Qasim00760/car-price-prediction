"""Streamlit app: UK used car price analysis and prediction.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.analysis import (
    DATA_YEAR,
    TARGET,
    clean,
    depreciation_by_manufacturer,
    depreciation_curve,
    load_raw,
    missing_report,
    plot_correlations,
    plot_depreciation,
    plot_mileage_vs_price,
    plot_price_by_category,
    plot_price_by_manufacturer,
    plot_price_distribution,
    price_by,
)
from src.model import FEATURES, train_final_model

# Test-set results from the notebook (notebooks/car_price_prediction.ipynb). They are
# shown here rather than recomputed, so the app starts quickly on Streamlit Cloud.
MODEL_RESULTS = pd.DataFrame(
    [
        ["Baseline (median price)", 6775, 47.0, -0.050],
        ["Ridge Regression", 2120, 13.2, 0.862],
        ["Random Forest", 1138, 7.2, 0.966],
        ["Gradient Boosting", 1072, 6.6, 0.970],
    ],
    columns=["model", "MAE (£)", "MAPE (%)", "R2"],
)

st.set_page_config(page_title="UK Used Car Price Prediction", layout="wide")


@st.cache_data
def get_data():
    return clean(load_raw())


@st.cache_resource
def get_model(_df):
    return train_final_model(_df)


df, cleaning_log = get_data()
with st.spinner("Training the model (first load only)..."):
    model = get_model(df)

st.title("UK Used Car Price Prediction")
st.write(
    f"Analysis of {len(df):,} used car listings from nine manufacturers in the UK ({DATA_YEAR}), "
    "and a Gradient Boosting model that estimates a fair price from a car's details."
)

tab_predict, tab_market, tab_depr, tab_model, tab_data = st.tabs(
    ["Price a car", "Market overview", "Depreciation", "Model", "Data and cleaning"]
)

# ---------------------------------------------------------------- prediction
with tab_predict:
    st.subheader("Estimate a fair price")
    makes = sorted(df["Manufacturer"].unique())
    c1, c2, c3 = st.columns(3)
    make = c1.selectbox("Manufacturer", makes, index=makes.index("Ford"))
    models_for_make = df.loc[df["Manufacturer"] == make, "model"].value_counts().index.tolist()
    car_model = c1.selectbox("Model", models_for_make)
    same = df[(df["Manufacturer"] == make) & (df["model"] == car_model)]

    year = c2.slider("Registration year", int(df["year"].min()), DATA_YEAR,
                     int(same["year"].median()))
    mileage = c2.number_input("Mileage", min_value=0, max_value=300000, step=1000,
                              value=int(round(same["mileage"].median(), -3)))
    transmission = c2.selectbox("Transmission", same["transmission"].value_counts().index.tolist())
    fuel_options = same["fuelType"].dropna().value_counts().index.tolist() or ["Petrol"]
    fuel = c3.selectbox("Fuel type", fuel_options)
    engine_options = sorted(same["engineSize"].dropna().unique().tolist()) or [1.6]
    engine = c3.selectbox("Engine size (litres)", engine_options,
                          index=engine_options.index(same["engineSize"].mode().iloc[0])
                          if same["engineSize"].notna().any() else 0)
    mpg = c3.number_input("Fuel economy (mpg)", min_value=10.0, max_value=200.0, step=0.5,
                          value=float(round(same["mpg"].median(), 1)) if same["mpg"].notna().any() else 50.0)
    tax = c3.number_input("Road tax (£ per year)", min_value=0, max_value=600, step=5,
                          value=int(same["tax"].median()))

    row = pd.DataFrame([{
        "Manufacturer": make, "model": car_model, "transmission": transmission, "fuelType": fuel,
        "age": DATA_YEAR - year, "mileage": mileage, "tax": tax, "mpg": mpg, "engineSize": engine,
    }])[FEATURES]
    price = float(model.predict(row)[0])

    similar = same[(same["year"] == year)][TARGET]
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated price", f"£{price:,.0f}")
    m2.metric("Likely range", f"£{price * 0.93:,.0f} to £{price * 1.07:,.0f}")
    m3.metric(f"Median listing, {car_model} {year}",
              f"£{similar.median():,.0f}" if len(similar) else "no listings",
              f"{len(similar)} listings" if len(similar) else None, delta_color="off")
    st.caption(
        "The range is plus or minus 7%, the model's average percentage error on unseen listings. "
        f"Prices reflect the UK market in {DATA_YEAR}."
    )

# ---------------------------------------------------------------- market
with tab_market:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listings", f"{len(df):,}")
    k2.metric("Median price", f"£{df[TARGET].median():,.0f}")
    k3.metric("Models", df["model"].nunique())
    k4.metric("Median mileage", f"{df['mileage'].median():,.0f}")
    left, right = st.columns(2)
    left.pyplot(plot_price_by_manufacturer(df))
    right.dataframe(price_by(df, "Manufacturer"), hide_index=True, width="stretch")
    st.pyplot(plot_price_by_category(df))
    left, right = st.columns(2)
    left.pyplot(plot_mileage_vs_price(df))
    right.pyplot(plot_correlations(df))

# ---------------------------------------------------------------- depreciation
with tab_depr:
    st.pyplot(plot_depreciation(df))
    left, right = st.columns([2, 3])
    left.dataframe(depreciation_curve(df), hide_index=True, width="stretch")
    right.markdown("**Value kept after 3 and 5 years, by manufacturer (median prices)**")
    right.dataframe(depreciation_by_manufacturer(df), hide_index=True, width="stretch")
    st.caption(
        "Medians compare all listings of each age, so part of the difference comes from which models "
        "are on sale at each age, not only from the same car getting older."
    )

# ---------------------------------------------------------------- model
with tab_model:
    st.dataframe(MODEL_RESULTS, hide_index=True, width="stretch")
    st.markdown(
        "- All models are trained on the log of price and scored in pounds on the same 20% test set "
        "(19,346 listings).\n"
        "- Gradient Boosting: R2 0.970, mean absolute error £1,072, mean percentage error 6.6%. "
        "79% of its test predictions are within 10% of the real price. 5-fold cross-validated R2 "
        "ranges from 0.960 to 0.971.\n"
        "- The model relies most on the car model, then age, mileage and engine size.\n"
        "- **Leakage fix:** the first version of this project added a `price_per_mileage` column, which is "
        "built from the price itself. That pushed R2 up artificially and could not be calculated for a car "
        "whose price is unknown, so it was removed."
    )

# ---------------------------------------------------------------- data
with tab_data:
    st.dataframe(cleaning_log, hide_index=True, width="stretch")
    st.markdown("**Remaining missing values** (filled inside the model: most frequent value or median)")
    st.dataframe(missing_report(df), width="stretch")
    st.markdown(
        "- Manufacturer names were fixed (`hyundi` to Hyundai, `merc` to Mercedes) and model names trimmed.\n"
        "- An engine size of 0 on a non-electric car is impossible, so those values are treated as missing.\n"
        "- Price outliers were kept. Expensive cars are real listings, and removing them would stop the model "
        "from pricing them.\n"
        "- Source: [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) on Kaggle."
    )
    st.download_button("Download the cleaned data", df.to_csv(index=False), file_name="uk_used_cars_clean.csv")
