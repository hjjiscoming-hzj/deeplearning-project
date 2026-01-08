import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib


class MyDataLoader(object):

    def __init__(self,X,y):
        # 划分数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 特征缩放
        self.scaler_X = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        # self.X_train_scaled = self.X_train.values
        # self.X_test_scaled = self.X_test.values
        # joblib.dump(scaler_X, 'scaler_X.joblib')

        self.scaler_y = StandardScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test_scaled = self.scaler_y.transform(self.y_test.values.reshape(-1, 1))
        # self.y_train_scaled = self.y_train.values.reshape(-1, 1)
        # self.y_test_scaled = self.y_test.values.reshape(-1, 1)
        # joblib.dump(scaler_y, 'scaler_y.joblib')
        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train_scaled, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(self.X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test_scaled, dtype=torch.float32).view(-1, 1)

        # 创建数据集
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    def inverse_transform_y(self, y_scaled):
        y_scaled_cpu = y_scaled.cpu().numpy()
        # 反向变换y值（将缩放后的y恢复为原始尺度）
        return torch.tensor(self.scaler_y.inverse_transform(y_scaled_cpu), dtype=torch.float32)