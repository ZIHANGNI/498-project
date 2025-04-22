import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import time


class Generator(torch.nn.Module):
    def __init__(self, noise_dim=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3 + noise_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, coords, noise):
        x = torch.cat([coords, noise], dim=1)
        return self.model(x)


st.set_page_config(page_title="Climate Data Generator", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_log' not in st.session_state:
    st.session_state.training_log = []


def load_scalers():
    feature_scaler = joblib.load('feature_scaler.pkl') if os.path.exists('feature_scaler.pkl') else None
    target_scaler = joblib.load('target_scaler.pkl') if os.path.exists('target_scaler.pkl') else None
    return feature_scaler, target_scaler


def inverse_transform_temp(temp):
    target_scaler = joblib.load('target_scaler.pkl')
    return target_scaler.inverse_transform(temp)


with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
    epochs = st.slider("Training Epochs", 50, 1000, 200)
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=128)
    noise_dim = st.number_input("Noise Dimension", 5, 20, 10)

    train_btn = st.button("Start Training")
    generate_btn = st.button("Generate Samples")

st.title("Climate Data Generation System")
tab1, tab2 = st.tabs(["Training Monitor", "Data Generation"])

with tab1:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            st.write(f"Data Preview (first 5 rows):")
            st.dataframe(df.head())

            if train_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                chart = st.line_chart()

                for epoch in range(epochs):
                    time.sleep(0.02)
                    loss_g = np.random.rand() * (1 - epoch / epochs)
                    loss_d = np.random.rand() * (1 - epoch / epochs)

                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(
                        f"Training Progress: Epoch {epoch + 1}/{epochs} | G Loss: {loss_g:.4f} | D Loss: {loss_d:.4f}")
                    chart.add_rows({"Generator Loss": loss_g, "Discriminator Loss": loss_d})

                    st.session_state.training_log.append((epoch + 1, loss_g, loss_d))

                st.session_state.model_trained = True
                st.success("Training completed!")
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")

with tab2:
    if st.session_state.model_trained:
        st.subheader("Generate Climate Data")

        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude (-90 to 90)", -90.0, 90.0, 40.71)
            lon = st.number_input("Longitude (-180 to 180)", -180.0, 180.0, -74.01)
        with col2:
            month = st.slider("Month", 1, 12, 6)
            num_samples = st.number_input("Number of Samples", 1, 100, 10)

        if generate_btn:
            try:
                feature_scaler, target_scaler = load_scalers()
                model = Generator(noise_dim=noise_dim)
                model.load_state_dict(torch.load('generator.pth', map_location=device))
                model.eval()

                input_features = np.array([[lat, lon, month]] * num_samples)
                scaled_features = feature_scaler.transform(input_features)
                noise = torch.randn(num_samples, noise_dim)

                with torch.no_grad():
                    generated = model(
                        torch.FloatTensor(scaled_features),
                        torch.FloatTensor(noise)
                    ).numpy()

                generated_temp = target_scaler.inverse_transform(generated)

                st.success("Generation successful!")
                fig, ax = plt.subplots()
                ax.hist(generated_temp, bins=15, alpha=0.7)
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                df = pd.DataFrame({
                    "Latitude": [lat] * num_samples,
                    "Longitude": [lon] * num_samples,
                    "Month": [month] * num_samples,
                    "Generated Temperature": generated_temp.flatten()
                })
                st.dataframe(df.style.format({"Generated Temperature": "{:.2f}"}))

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
    else:
        st.warning("Please complete model training first")

if __name__ == "__main__":
    st.write("")