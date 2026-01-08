import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import DataLoader
from huangzhj.data_utils.data_preprocessing import *

data = DataPreprocessing_HS()
X,y = data.get_data()
# 创建数据加载器
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# joblib.dump(scaler_X, 'scaler_X.joblib')

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
# joblib.dump(scaler_y, 'scaler_y.joblib')

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)

# 创建数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for data, targets in test_loader:
    data, targets = data.to(device), targets.to(device)
    targets_scaled_cpu = targets.cpu().numpy()
    # 反向变换y值（将缩放后的y恢复为原始尺度）
    targets_inverse_tensor = torch.tensor(scaler_y.inverse_transform(targets_scaled_cpu), dtype=torch.float32)
    print(y_train)

# 配置参数
    file_path = "D:\DeepLearning\Datasets\Merak\HS_unit_dataset_1216.csv"
    output_dir = "./HS数据集相关性分析"