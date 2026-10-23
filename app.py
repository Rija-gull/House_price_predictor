"""California house value predictor – Streamlit app for california_housing_kfold_10fold.ipynb."""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent

# Approximate city-centre coordinates and their usual ocean-proximity category in the dataset
CITIES = {
    "San Francisco": (37.77, -122.42, "NEAR BAY"),
    "Oakland": (37.80, -122.27, "NEAR BAY"),
    "San Jose": (37.34, -121.89, "<1H OCEAN"),
    "Los Angeles": (34.05, -118.24, "<1H OCEAN"),
    "San Diego": (32.72, -117.16, "NEAR OCEAN"),
    "Sacramento": (38.58, -121.49, "INLAND"),
    "Fresno": (36.74, -119.79, "INLAND"),
    "Bakersfield": (35.37, -119.02, "INLAND"),
}

st.set_page_config(page_title="California House Value Predictor", page_icon="🏠")


@st.cache_resource
def get_model():
    with open(HERE / "california_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_info():
    with open(HERE / "app_info.json") as f:
        return json.load(f)


model, info = get_model(), get_info()
medians = info["medians"]

st.title("🏠 California House Value Predictor")
st.write(f"Predicts the **median house value of a district** (1990 census prices) with a "
         f"**{info['algorithm']}** model selected by 10-fold cross validation.")

predict_tab, compare_tab = st.tabs(["Predict", "Model comparison"])

with predict_tab:
    st.subheader("1. Location")
    city = st.selectbox("Nearest city", list(CITIES) + ["Custom coordinates"])
    if city == "Custom coordinates":
        c1, c2 = st.columns(2)
        latitude = c1.number_input("Latitude", 32.5, 42.0, 36.0, 0.01)
        longitude = c2.number_input("Longitude", -124.4, -114.3, -119.5, 0.01)
        default_proximity = "INLAND"
    else:
        latitude, longitude, default_proximity = CITIES[city]
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5, height=250)
    ocean_proximity = st.selectbox("Ocean proximity", info["ocean_proximity"],
                                   index=info["ocean_proximity"].index(default_proximity))

    st.subheader("2. District")
    c1, c2 = st.columns(2)
    median_income = c1.slider("Median household income (× $10,000)", 0.5, 15.0,
                              round(medians["median_income"], 1), 0.1)
    housing_median_age = c2.slider("Median house age (years)", 1, 52, int(medians["housing_median_age"]))
    households = c1.number_input("Households", 1, 6000, int(medians["households"]), 10)
    population = c2.number_input("Population", 3, 36000, int(medians["population"]), 10)
    total_rooms = c1.number_input("Total rooms", 2, 40000, int(medians["total_rooms"]), 10)
    total_bedrooms = c2.number_input("Total bedrooms", 1, 6500, int(medians["total_bedrooms"]), 10)

    district = pd.DataFrame([{
        "longitude": longitude, "latitude": latitude, "housing_median_age": housing_median_age,
        "total_rooms": total_rooms, "total_bedrooms": total_bedrooms, "population": population,
        "households": households, "median_income": median_income, "ocean_proximity": ocean_proximity,
    }])
    # Same ratio features as Step 4 of the notebook
    district["rooms_per_household"] = district["total_rooms"] / district["households"]
    district["bedrooms_per_room"] = district["total_bedrooms"] / district["total_rooms"]
    district["population_per_household"] = district["population"] / district["households"]

    if total_bedrooms > total_rooms:
        st.error("Total bedrooms cannot be larger than total rooms.")
    else:
        value = float(model.predict(district[info["numeric_features"] + info["categorical_features"]])[0])
        value = min(max(value, 15_000), 500_001)
        st.subheader("3. Result")
        st.metric("Predicted median house value", f"${value:,.0f}")
        note = (f"Typical error (10-fold MAE): ±${info['mae']:,.0f}. "
                "Values in the dataset range from $15,000 to $500,001 (the census cap).")
        st.caption(note.replace("$", "\\$"))

with compare_tab:
    st.write("All five algorithms were evaluated on the same 10 folds with the same pre-processing.")
    results = pd.DataFrame(info["results"]).set_index("position")
    st.dataframe(results.style.format({"RMSE mean ($)": "${:,.0f}", "MAE mean ($)": "${:,.0f}",
                                       "R2 mean": "{:.3f}", "mean rank": "{:.1f}"}), width="stretch")
    st.write(f"Friedman test across the folds: p = {info['friedman_p']:.1e} "
             f"(the algorithms differ significantly when p < 0.05).")
    st.caption(f"{info['student']} · scikit-learn {info['scikit_learn']}")
