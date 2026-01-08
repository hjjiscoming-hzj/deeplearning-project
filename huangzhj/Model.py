import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self,y_pred,y_true):
        mse = self.mse_loss
        rmse = torch.sqrt(mse(y_pred,y_true))
        return rmse


# 定义学习率调整函数
def lr_lambda(epoch):
    if epoch < 100:
        # 线性降低学习率从 1e-3 到 1e-4
        return 1 - 0.9 * (epoch / 100)
    else:
        # 保持学习率为 1e-4
        return 0.1


class Model():

    def __init__(self,model,train_loader,test_loader,data_loader,lr,loss):
        self.data_loader = data_loader
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 将模型移到可用设备（GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-4)
        if loss == 'RMSE':
            self.criterion = RMSELoss()
        if loss == 'L1':
            self.criterion = nn.SmoothL1Loss()
        if loss == 'MSE':
            self.criterion = nn.MSELoss()



    def train(self):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # 移动数据到设备
            data, targets = data.to(self.device), targets.to(self.device)
            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播
            if 'ormer' in self.model.args.name:
                outputs = self.model(data, x_cat=None)
            else:
                outputs = self.model(data)
            # 计算损失
            loss = self.criterion(outputs, targets)
            # reg_loss = self.model.regularization_loss(regularize_activation=0, regularize_entropy=0)
            # loss = loss + reg_loss
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()
            total_loss += loss.item()
        # self.scheduler.step()
        self.average_loss = total_loss / len(self.train_loader)

    def eval(self):
        self.model.eval()
        test_loss = 0
        test_loss_inverse = 0
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                if 'ormer' in self.model.args.name:
                    outputs = self.model(data, x_cat=None)
                else:
                    outputs = self.model(data)
                test_loss += self.criterion(outputs, targets).item()
                # 计算逆缩放后的预测值和真实值以及损失
                inverse_outputs = self.data_loader.inverse_transform_y(outputs)
                inverse_targets = self.data_loader.inverse_transform_y(targets)
                test_loss_inverse += self.criterion(inverse_outputs,inverse_targets).item()
        self.test_loss = test_loss/len(self.test_loader)
        self.test_loss_inverse = test_loss_inverse/len(self.test_loader)
