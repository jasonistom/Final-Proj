import streamlit as st
import pandas as pd
import joblib

st.title("Video Game Hit Predictor")


@st.cache_resource
def load_model():
    return joblib.load(r"Video Games Prediction.pki")


model = load_model()
columns = list(model.feature_names_in_)
platforms = ["2600"] + [c.replace("Platform_", "") for c in columns if c.startswith("Platform_")]
genres = ["Action"] + [c.replace("Genre_", "") for c in columns if c.startswith("Genre_")]
publishers = ["10TACLE"] + [c.replace("Publisher_", "") for c in columns if c.startswith("Publisher_")]

year = st.slider("Release year", 1980, 2026, 2010)
platform = st.selectbox("Platform", platforms)
genre = st.selectbox("Genre", genres)
publisher = st.selectbox("Publisher (choose Other if not listed)", publishers)

if st.button("Predict"):
    row = pd.DataFrame(0, index=[0], columns=columns)
    row["Year"] = (year - 2007.307) / 11.4967

    for name in [f"Platform_{platform}", f"Genre_{genre}", f"Publisher_{publisher}"]:
        if name in row.columns:
            row[name] = 1

    chance = model.predict_proba(row)[0][1]

    if chance >= 0.5:
        st.success(f"Likely a Hit ({chance:.0%} chance)")
    else:
        st.error(f"Likely Not a Hit ({chance:.0%} chance)")
