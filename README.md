# 🌧️ Rainfall Prediction System

A Machine Learning-based web application that predicts rainfall using **Random Forest Regression**. Built with Python and Streamlit, the application allows users to enter weather parameters, estimate rainfall in millimetres, and visualize historical rainfall patterns through interactive data visualizations.

## ✨ Features

- 🌦️ Rainfall prediction using Random Forest Regression
- 🌡️ Weather input parameters: temperature, humidity, wind speed, year, month, and previous rainfall
- 📊 Historical rainfall visualization using Matplotlib
- 📅 Year-wise rainfall data analysis
- 📋 Dataset preview through an interactive Streamlit interface
- 📈 Model evaluation using MAE, RMSE, and R² score

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Pickle

## ⚙️ How to Run

1. Clone this repository.
2. Install the required Python libraries.
3. Train the model using `python train.py`.
4. Launch the application using `python -m streamlit run app.py`.

## 🎯 Project Objective

To demonstrate how machine learning can be applied to rainfall estimation, weather data analysis, and interactive visualization.

## ⚠️ Dataset Note

The current training script generates synthetic, Hyderabad-like seasonal rainfall data for demonstration purposes. Predictions should not be treated as operational weather forecasts.

## 👨‍💻 Project Type

Academic Machine Learning Project
