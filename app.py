import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# Load trained model
with open("model/rainfall_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset
df = pd.read_csv("data/rainfall.csv")


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide"
)


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🌧️ Rainfall Prediction System")
st.write(
    "Machine Learning based rainfall prediction using "
    "Random Forest Regression."
)

st.divider()


# ---------------------------------------------------
# Prediction inputs
# ---------------------------------------------------

st.subheader("Enter Weather Information")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input(
        "Year",
        min_value=2015,
        max_value=2035,
        value=2026
    )

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=7
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=28.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col3:
    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=100.0,
        value=8.0
    )

    previous_rainfall = st.number_input(
        "Previous Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=100.0
    )


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🌧️ Predict Rainfall", use_container_width=True):

    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "WindSpeed": [wind_speed],
        "PreviousRainfall": [previous_rainfall]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Rainfall: **{prediction:.2f} mm**"
    )

    if prediction < 50:
        st.info("☀️ Expected rainfall: Low")
    elif prediction < 150:
        st.warning("🌦️ Expected rainfall: Moderate")
    else:
        st.error("🌧️ Expected rainfall: High")


st.divider()


# ---------------------------------------------------
# Historical rainfall visualization
# ---------------------------------------------------

st.subheader("📊 Historical Rainfall Data")

yearly_rainfall = df.groupby("Year")["Rainfall"].mean()

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    yearly_rainfall.index,
    yearly_rainfall.values,
    marker="o"
)

ax.set_xlabel("Year")
ax.set_ylabel("Average Rainfall (mm)")
ax.set_title("Average Rainfall by Year")

ax.grid(True)

st.pyplot(fig)


# ---------------------------------------------------
# Dataset preview
# ---------------------------------------------------

st.subheader("📋 Dataset Preview")

st.subheader("📋 Rainfall Data by Year")

selected_year = st.number_input(
    "Enter Year",
    min_value=2015,
    max_value=2025,
    value=2020,
    step=1
)

year_data = df[df["Year"] == selected_year]

if not year_data.empty:
    st.write(f"Rainfall data for **{selected_year}**")

    st.dataframe(
        year_data,
        use_container_width=True
    )

    # Year-wise rainfall graph
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(
        year_data["Month"],
        year_data["Rainfall"],
        marker="o"
    )

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Rainfall (mm)")
    ax2.set_title(f"Monthly Rainfall - {selected_year}")
    ax2.set_xticks(range(1, 13))
    ax2.grid(True)

    st.pyplot(fig2)

else:
    st.warning("No rainfall data available for this year.")


# ---------------------------------------------------
# Project information
# ---------------------------------------------------

st.divider()

st.subheader("About the Project")

st.write("""
This project uses a Random Forest Regression machine learning
algorithm to predict rainfall based on weather conditions.

Input parameters include temperature, humidity, wind speed,
previous rainfall, year and month.

The system provides the predicted rainfall in millimeters
and displays historical rainfall information graphically.
""")