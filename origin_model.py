import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *


class _LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(_LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

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
        device,
    ):
        self.model = _LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.epoch = epoch
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.epoch // 3, gamma=0.2
        )
        self.epoch = epoch
        self.gradient_threshold = gradient_threshold
        self.device = device

    def train(self, X_train, y_train):
        print("LSTM Training")
        X_train_tensor = (
            torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        y_train_tensor = (
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        self.model.to(self.device)

        self.model.train()
        for epoch in range(self.epoch):
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

            # self.scheduler.step()

        print("LSTM Training Done")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().detach().numpy()
        return y_pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), f"{path}.pth")
        print("모델 상태 저장 완료")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print("모델 상태 불러오기 완료")


def train_origin_model():
    df = pd.read_csv("features.csv")

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

    X_train = X_ACF[t1_to_t4_indices]
    y_train = y[t1_to_t4_indices]
    X_test = X_ACF[t5_indices + t6_indices + t7_indices + t8_indices]
    y_test = y[t5_indices + t6_indices + t7_indices + t8_indices]

    normalizer = Normalize()
    X_train, y_train = normalizer.fit_transform(X_train, y_train)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "input_size": 1,
        "num_layers": 3,
        "hidden_size": 100,
        "epoch": 300,
        "gradient_threshold": 0.5,
        "output_size": 1,
        "learning_rate": 0.01,
        "device": device,
    }
    lstm_model = LSTM_Model(**params)
    lstm_model.train(X_train, y_train)

    # LSTM Model Train Prediction
    y_pred_train = lstm_model.predict(X_train)
    y_train_original = normalizer.inverse_normalize_data(y_train)
    y_pred_train_original = normalizer.inverse_normalize_data(y_pred_train)

    # LSTM Model Prediction
    X_test, y_test = normalizer.normalize_data(X_test, y_test)
    y_pred_test = lstm_model.predict(X_test)
    y_test_original = normalizer.inverse_normalize_data(y_test)
    y_pred_test_original = normalizer.inverse_normalize_data(y_pred_test)

    mse_train, r2_train = performance(y_train_original, y_pred_train_original)
    mse_test, r2_test = performance(y_test_original, y_pred_test_original)

    print(f"Train RMSE: {np.sqrt(mse_train):.4f}, Train R2: {r2_train:.4f}")
    print(f"Test RMSE: {np.sqrt(mse_test):.4f}, Test R2: {r2_test:.4f}")

    lstm_model.save_model("origin_model_acf")
    print("모델 저장 완료")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(4):
        indices = [i for i in range(i * 12, (i + 1) * 12)]
        axs[i].scatter(
            y_test_original[indices],
            y_pred_test_original[indices],
            label="LSTM",
            color="green",
        )
        axs[i].plot([0, 350], [0, 350], "r-", label="Correct")
        axs[i].set_xlabel("Observed Wear")
        axs[i].set_ylabel("Predicted Wear")
        axs[i].legend()
        axs[i].set_title(f"T{i+5}")

    fig.suptitle("2개의 센서데이터 이용 - ACC, Force", fontsize=16)
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 그래프 출력
    plt.show()


if __name__ == "__main__":
    train_origin_model()
