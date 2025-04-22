import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")


class ClimateDataset(Dataset):
    def __init__(self, filepath):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {filepath}")

        df['dt'] = pd.to_datetime(df['dt'])
        df.set_index('dt', inplace=True)

        df['AverageTemperature'] = df['AverageTemperature'].interpolate(method='time')
        df.dropna(subset=['AverageTemperature'], inplace=True)

        def convert_coord(coord):
            value = float(coord[:-1])
            direction = coord[-1]
            return -value if direction in ['S', 'W'] else value

        df['Latitude'] = df['Latitude'].apply(convert_coord)
        df['Longitude'] = df['Longitude'].apply(convert_coord)

        df['Month'] = df.index.month

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.features = self.feature_scaler.fit_transform(df[['Latitude', 'Longitude', 'Month']])
        self.targets = self.target_scaler.fit_transform(df[['AverageTemperature']])

        joblib.dump(self.feature_scaler, 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, 'target_scaler.pkl')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.targets[idx])
        )


class Generator(nn.Module):
    def __init__(self, noise_dim=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.model = nn.Sequential(
            nn.Linear(3 + noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 匹配MinMaxScaler的[0,1]范围
        )

    def forward(self, coords, noise):
        x = torch.cat([coords, noise], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, coords, temp):
        x = torch.cat([coords, temp], dim=1)
        return self.model(x)


def train_model(filepath, epochs=500, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataset = ClimateDataset(filepath)
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    G = Generator(noise_dim=10).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    losses_G = []
    losses_D = []

    for epoch in range(epochs):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        for real_coords, real_temp in loader:
            real_coords = real_coords.to(device)
            real_temp = real_temp.to(device)
            batch_size = real_coords.size(0)

            opt_D.zero_grad()

            real_labels = torch.ones(batch_size, 1, device=device)
            real_output = D(real_coords, real_temp)
            d_loss_real = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, G.noise_dim, device=device)
            fake_temp = G(real_coords, noise)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            fake_output = D(real_coords, fake_temp.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            valid_labels = torch.ones(batch_size, 1, device=device)
            validity = D(real_coords, fake_temp)
            g_loss = criterion(validity, valid_labels)
            g_loss.backward()
            opt_G.step()

            epoch_loss_G += g_loss.item()
            epoch_loss_D += d_loss.item()

        avg_loss_G = epoch_loss_G / len(loader)
        avg_loss_D = epoch_loss_D / len(loader)
        losses_G.append(avg_loss_G)
        losses_D.append(avg_loss_D)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1:04d}/{epochs}] | D Loss: {avg_loss_D:.4f} | G Loss: {avg_loss_G:.4f}")

    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(losses_G, label='Generator Loss')
    plt.plot(losses_D, label='Discriminator Loss')
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()


if __name__ == "__main__":
    data_path = "C:/Users/nzh00/Desktop/2025 SPRING/498/project/archive/GlobalLandTemperaturesByMajorCity.csv"

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {os.path.abspath(data_path)}")
        exit(1)

    try:
        train_model(data_path, epochs=200)
        print("\nTraining completed. Saved files:")
        print("- generator.pth (Generator model)")
        print("- discriminator.pth (Discriminator model)")
        print("- feature_scaler.pkl (Feature scaler)")
        print("- target_scaler.pkl (Target scaler)")
        print("- training_loss.png (Loss curve)")
    except Exception as e:
        print(f"Training failed: {str(e)}")