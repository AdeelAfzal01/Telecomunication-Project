import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Downtime Forecasting", layout="wide")

# App title
st.title("Downtime Forecasting Application")

# Load dataset from backend
@st.cache_data
def load_data():
    csv_path = "DATA/XAA_FILTERED.csv"  # Replace with your actual file path
    df = pd.read_csv(csv_path)
    df['CALLING_DATE'] = pd.to_datetime(df['CALLING_DATE'])
    return df

df = load_data()

# Preprocess the dataset
df['downtime'] = df['TOWER_STATUS'].apply(lambda x: 1 if x == 'down' else 0)
df['CALLING_WEEK'] = df['CALLING_DATE'].dt.strftime('%Y-%U')  # Year-Week format

# Aggregate data weekly
weekly_data = df.groupby(['AREA_ID', 'CALLING_WEEK']).agg({
    'downtime': 'sum',  # Total downtimes
    'DURATION': 'mean',  # Average call duration
    'ACT_DURATION': 'mean',  # Average active duration
    'BILLAMOUNT': 'sum',  # Total bill amount
}).reset_index()

# Display a preview of the data
st.subheader("Aggregated Weekly Data")
st.dataframe(weekly_data.head())

# Select AREA_ID for forecasting
area_id = st.sidebar.selectbox("Select AREA_ID for Forecasting", weekly_data['AREA_ID'].unique())

# Filter data for the selected AREA_ID
area_data = weekly_data[weekly_data['AREA_ID'] == area_id]
area_data.set_index('CALLING_WEEK', inplace=True)

# Downtime Forecasting Section
st.header(f"Downtime Forecasting for AREA_ID {area_id}")
with st.spinner("Generating Forecast..."):
    try:
        # Fit SARIMAX model
        model = SARIMAX(area_data['downtime'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        model_fit = model.fit()

        # Forecast for the next 4 weeks
        forecast_steps = 4
        forecast = model_fit.forecast(steps=forecast_steps)

        # Prepare forecast results
        forecast_df = pd.DataFrame({
            'Week': pd.date_range(start=pd.Timestamp(area_data.index.max()), periods=forecast_steps, freq='W'),
            'Predicted Downtime': forecast
        })

        # Display forecasted downtimes
        st.subheader("Forecasted Downtimes for the Next 4 Weeks")
        st.table(forecast_df)

        # Plot forecast
        st.subheader("Downtime Forecast Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(area_data.index, area_data['downtime'], label="Historical Downtime")
        ax.plot(forecast_df['Week'], forecast_df['Predicted Downtime'], label="Forecasted Downtime", linestyle='--', color='orange')
        ax.set_xlabel("Weeks")
        ax.set_ylabel("Downtime")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Forecasting failed: {str(e)}")
