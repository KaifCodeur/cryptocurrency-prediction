import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit UI
st.title("📈 Crypto Price Prediction App")
st.sidebar.header("Upload Cryptocurrency CSV Files")

uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    df_list = []
    coin_names = []

    for file in uploaded_files:
        df = pd.read_csv(file)
        coin_name = os.path.basename(file.name).replace("coin_", "").replace(".csv", "")
        df["Coin"] = coin_name  # Add coin column
        df_list.append(df)
        coin_names.append(coin_name)

    # Merge all data
    all_data = pd.concat(df_list, ignore_index=True)

    # Let the user select a specific coin
    selected_coin = st.sidebar.selectbox("Select a coin", coin_names)

    # Filter data for selected coin
    data = all_data[all_data["Coin"] == selected_coin].copy()

    # Display raw data
    st.subheader(f"Raw Data Preview for {selected_coin}")
    st.write(data.head())

    # "Train and Preprocess" button
    if st.button("Train and Preprocess"):
        st.subheader("🔄 Preprocessing Data...")

        # Convert Date column to datetime
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")

        # Selecting relevant columns
        data = data[["Date", "High", "Low", "Open", "Close", "Volume", "Marketcap"]]
        
        # Handle missing values
        data.fillna(method="ffill", inplace=True)

        # Feature Engineering: Creating Moving Averages
        data["MA_7"] = data["Close"].rolling(window=7).mean()
        data["MA_14"] = data["Close"].rolling(window=14).mean()
        data.dropna(inplace=True)

        st.write("✅ Data cleaned and features created!")
        st.write(data.head())

        # Splitting Data
        X = data[["High", "Low", "Open", "Volume", "Marketcap", "MA_7", "MA_14"]]
        y = data["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model Training
        st.subheader(f"🚀 Training Models for {selected_coin}...")

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }

        results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {"Model": model, "MAE": mae, "RMSE": rmse, "R²": r2, "Predictions": y_pred}

        # Select best model based on RMSE
        best_model_name = min(results, key=lambda x: results[x]["RMSE"])
        best_model = results[best_model_name]["Model"]

        st.write("✅ Model training completed!")
        st.write(f"**Best Model for {selected_coin}: {best_model_name}**")

        # Display Metrics
        st.subheader("📊 Model Performance")
        for model_name, result in results.items():
            st.write(f"### {model_name}")
            st.write(f"MAE: {result['MAE']:.4f}, RMSE: {result['RMSE']:.4f}, R²: {result['R²']:.4f}")

        # Store trained model for prediction
        st.session_state["best_model"] = best_model
        st.session_state["X"] = X

    # Future Prediction Section
    if "best_model" in st.session_state:
        st.subheader(f"🔮 Future Price Prediction for {selected_coin}")

        # Store slider state in session
        days_to_predict = st.slider("Select number of days to predict", 1, 30, 7)

        # Reset stored predictions if slider value changes
        if "previous_days_to_predict" in st.session_state and st.session_state["previous_days_to_predict"] != days_to_predict:
            st.session_state.pop("future_prices", None)
        
        st.session_state["previous_days_to_predict"] = days_to_predict

        # "Predict Future Prices" button
        if st.button("Predict Future Prices"):
            best_model = st.session_state["best_model"]
            last_X = st.session_state["X"].iloc[-1].values.reshape(1, -1)
            future_prices = []

            for _ in range(days_to_predict):
                future_price = best_model.predict(last_X)[0]
                future_prices.append(future_price)

                # Shift the input data for next prediction
                last_X = np.roll(last_X, -1)
                last_X[0, -1] = future_price

            # Store predictions in session state
            st.session_state["future_prices"] = future_prices

        # Display stored predictions if available
        if "future_prices" in st.session_state:
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days_to_predict + 1)]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(future_dates, st.session_state["future_prices"], marker='o', linestyle="dashed", label="Predicted Future Price", color="green")
            ax.set_title(f"{selected_coin} - Future Predictions")
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Close Price")
            ax.legend()
            st.pyplot(fig)

            st.write("✅ Future predictions completed!")
