import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from utils import *


class _LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(_LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTM_Model:
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        learning_rate,
        gradient_threshold,
        epoch,
    ):
        self.model = _LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )
        self.epoch = epoch
        self.gradient_threshold = gradient_threshold

    def train(self, X_train, y_train):
        print("LSTM Training")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

        self.model.train()
        for epoch in tqdm(range(self.epoch)):
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_threshold
            )

            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epoch}], Loss: {loss.item():.4f}")

            self.scheduler.step()

        print("LSTM Training Done")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).detach().numpy()
        return y_pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), f"{path}.pth")
        print("모델 상태 저장 완료")


def train_origin_model():
    df = pd.read_csv("feature_extraction.csv")

    ACC_LIST = [i for i in range(2, 38)]
    AE_LIST = [i for i in range(38, 50)]
    FORCE_LIST = [i for i in range(50, 86)]

    X_ACAE = df.iloc[:, ACC_LIST + AE_LIST].values
    X_ACF = df.iloc[:, ACC_LIST + FORCE_LIST].values
    X_AEF = df.iloc[:, AE_LIST + FORCE_LIST].values

    X = df.iloc[:, ACC_LIST + AE_LIST + FORCE_LIST].values
    y = df.iloc[:, -1].values

    # Milling Condition 에 따른 데이터 나누기
    t1_to_t4_indices = [i for i in range(0, 12 * 4)]
    t5_indices = [i for i in range(12 * 4, 12 * 5)]
    t6_indices = [i for i in range(12 * 5, 12 * 6)]
    t7_indices = [i for i in range(12 * 6, 12 * 7)]
    t8_indices = [i for i in range(12 * 7, 12 * 8)]

    X_train = X[t1_to_t4_indices]
    y_train = y[t1_to_t4_indices]
    X_test = X[t5_indices + t6_indices + t7_indices + t8_indices]
    y_test = y[t5_indices + t6_indices + t7_indices + t8_indices]

    normalizer = Normalize()
    X_train, y_train = normalizer.fit_transform(X_train, y_train)

    # LSTM Model Training
    set_seed(42)

    params = {
        "input_size": 1,
        "num_layers": 3,
        "hidden_size": 100,
        "epoch": 200,  # 100
        "gradient_threshold": 0.5,  # 0.5
        "output_size": 1,
        "learning_rate": 0.01,
    }
    lstm_model = LSTM_Model(**params)
    lstm_model.train(X_train, y_train)

    # LSTM Model Train Prediction
    y_pred_train = lstm_model.predict(X_train)
    y_train_original = normalizer.inverse_normalize_data(y_train)
    y_pred_train_original = normalizer.inverse_normalize_data(y_pred_train)

    # LSTM Model Prediction
    X_test, y_test = normalizer.normalize_data(X_test, y_test)
    y_pred_test = model.predict(X_test)
    y_test_original = normalizer.inverse_normalize_data(y_test)
    y_pred_test_original = normalizer.inverse_normalize_data(y_pred_test)
