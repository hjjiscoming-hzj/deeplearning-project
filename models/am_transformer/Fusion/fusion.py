import torch
from torch import nn


class StaticFeatureFusion_norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 添加特征归一化层
        self.norm_transformer = nn.LayerNorm(dim)
        self.norm_kan = nn.LayerNorm(dim)

        # 静态融合层
        self.static_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        # 静态权重因子
        self.kan_weight = 0.5
        self.transformer_weight = 0.5

    def forward(self, transformer_out, kan_out):
        # 先进行特征归一化
        norm_transformer = self.norm_transformer(transformer_out)
        norm_kan = self.norm_kan(kan_out)

        # 应用权重到归一化后的特征
        weighted_transformer = norm_transformer * self.transformer_weight
        weighted_kan = norm_kan * self.kan_weight

        # 静态融合
        combined = torch.cat([weighted_transformer, weighted_kan], dim=-1)
        static_output = self.static_fusion(combined)

        return static_output


class StaticFeatureFusion_test(nn.Module):
    def __init__(self, dim, fusion_weight, dropout = 0.2, out_dim = 1):
        super().__init__()
        # 修改比例的静态融合层
        self.static_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        # 添加权重因子用于比例调整
        self.kan_weight = fusion_weight['kan_weight']
        self.transformer_weight = fusion_weight['trans_weight']

    def forward(self, transformer_out, kan_out):
        # 应用权重
        weighted_transformer = transformer_out * self.transformer_weight
        weighted_kan = kan_out * self.kan_weight

        # 静态融合
        combined = torch.cat([weighted_transformer, weighted_kan], dim=-1)
        static_output = self.static_fusion(combined)

        return static_output

    def get_weight(self):
        return self.kan_weight, self.transformer_weight


class StaticFeatureFusion(nn.Module):
    def __init__(self, dim, fusion_weight):
        super().__init__()
        # 修改比例的静态融合层
        self.static_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        # 添加权重因子用于比例调整
        self.kan_weight = fusion_weight['kan_weight']
        self.transformer_weight = fusion_weight['trans_weight']

    def forward(self, transformer_out, kan_out):
        # 应用权重
        weighted_transformer = transformer_out * self.transformer_weight
        weighted_kan = kan_out * self.kan_weight

        # 静态融合
        combined = torch.cat([weighted_transformer, weighted_kan], dim=-1)
        static_output = self.static_fusion(combined)

        return static_output

    def get_weight(self):
        return self.kan_weight, self.transformer_weight


class DynamicFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 添加特征归一化层
        self.norm_transformer = nn.LayerNorm(dim)
        self.norm_kan = nn.LayerNorm(dim)

        # 其余部分与原始DynamicFeatureFusion相同
        self.weight_layer = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )

        self.static_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        self.current_transformer_weight = 0
        self.current_kan_weight = 0

    def forward(self, transformer_out, kan_out):
        # 先进行特征归一化
        norm_transformer = self.norm_transformer(transformer_out)
        norm_kan = self.norm_kan(kan_out)

        # 使用归一化后的特征计算权重
        concat_features = torch.cat([norm_transformer, norm_kan], dim=-1)
        weights = self.weight_layer(concat_features)

        transformer_weight = weights[..., 0].unsqueeze(-1)
        kan_weight = weights[..., 1].unsqueeze(-1)

        self.current_transformer_weight = transformer_weight.mean().item()
        self.current_kan_weight = kan_weight.mean().item()

        print(
            f"权重分布 - KAN: {self.current_kan_weight:.4f}, Transformer: {self.current_transformer_weight:.4f}")

        # 权重应用在归一化后的特征上
        weighted_transformer = norm_transformer * transformer_weight
        weighted_kan = norm_kan * kan_weight

        # 静态融合
        combined = torch.cat([weighted_transformer, weighted_kan], dim=-1)
        static_output = self.static_fusion(combined)

        return static_output


class DynamicFeatureFusion_old(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 特征归一化层
        self.norm_transformer = nn.LayerNorm(dim)
        self.norm_kan = nn.LayerNorm(dim)

        # 位置自适应权重学习器
        self.weight_learner = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),  # 为两个特征路径生成权重
            nn.Softmax(dim=-1)  # 归一化权重
        )

        # 输出层
        self.output_layer = nn.Linear(dim, dim)

    def forward(self, transformer_out, kan_out):
        # 特征归一化
        t_norm = self.norm_transformer(transformer_out)
        k_norm = self.norm_kan(kan_out)

        # 拼接特征以生成权重
        combined = torch.cat([t_norm, k_norm], dim=-1)

        # 学习自适应权重
        weights = self.weight_learner(combined)  # B x 2
        if self.training:
            # 记录批次平均权重
            avg_t_weight = weights[:, 0].mean().item()
            avg_k_weight = weights[:, 1].mean().item()
            print("transformer weight:",avg_t_weight)
            print("kan weight:",avg_k_weight)
        # 扩展权重维度以匹配特征维度
        weights = weights.unsqueeze(-1)  # B x 2 x 1

        # 堆叠特征
        stacked_features = torch.stack([t_norm, k_norm], dim=1)  # B x 2 x dim

        # 应用动态权重
        weighted_sum = (stacked_features * weights).sum(dim=1)  # B x dim

        # 最终输出
        output = self.output_layer(weighted_sum)  # B x dim

        return output


class HybridFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 原有静态融合层
        self.static_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        # 动态融合组件
        self.norm_t = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        # 动态权重生成器
        self.dynamic_weight_generator = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # 混合因子 (可学习参数)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, transformer_out, kan_out):
        # 特征归一化
        t_norm = self.norm_t(transformer_out)
        k_norm = self.norm_k(kan_out)

        # 1. 静态融合 - 使用您现有的方法
        combined = torch.cat([transformer_out, kan_out], dim=-1)
        static_output = self.static_fusion(combined)

        # 2. 动态融合
        # 生成动态权重
        norm_combined = torch.cat([t_norm, k_norm], dim=-1)
        weights = self.dynamic_weight_generator(norm_combined)

        # 应用动态权重
        dynamic_output = weights[:, 0:1] * t_norm + weights[:, 1:2] * k_norm

        # 使用sigmoid将alpha限制在[0,1]范围
        alpha = torch.sigmoid(self.alpha)

        # 混合静态和动态输出
        hybrid_output = alpha * static_output + (1 - alpha) * dynamic_output

        return hybrid_output