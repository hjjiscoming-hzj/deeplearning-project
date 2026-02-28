import argparse
import os
import torch
from torch.utils.data import DataLoader

# 导入你的自定义模块 (保持与 main.py 一致)
from models.am_transformer.am_transformer import AMTransformer
from models.ft_transformer.ft_tranformer import FTTransformer
from models.base_model import simple_MLP, simple_KAN
from models.src_kan.efficient_kan import KAN
from huangzhj.Model import Model
from huangzhj.data_utils.data_preprocessing import *
from huangzhj.data_utils.data_loader import MyDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--manual_seed', default=42, type=int)
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--loss', default='RMSE', type=str)
parser.add_argument('--model_name', default='AMFormer', type=str)

# 🌟 新增核心参数：直接接收权重文件的绝对或相对路径
parser.add_argument('--weight_path',
                    default='log/txt_log/logs_2026_2_26_HS_RMSE_fusion/AMFormer_simple_MLP_dim32_depth2_heads4_dropout0.4_dyFusion_lr0.001_6P/best_AMFormer_HS_weights.pth',
                    type=str)

opt = parser.parse_args()

if __name__ == '__main__':
    # 1. 设置全局随机种子 (必须与训练时完全一致)
    torch.manual_seed(opt.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed_all(opt.manual_seed)

    # 2. 读取数据 (必须与训练时使用的数据集完全一致)
    data = DataPreprocessing_HS()
    data_loader = MyDataLoader(*data.get_data())

    test_loader = DataLoader(data_loader.test_dataset, batch_size=opt.batch_size, shuffle=False)
    train_loader = DataLoader(data_loader.train_dataset, batch_size=opt.batch_size, shuffle=False)

    input_size = data_loader.X_train_scaled.shape[1]

    # 3. 创建模型实例 (结构必须与训练时严格一致)
    model_name = opt.model_name
    print(f"Initializing {model_name}...")

    if model_name == 'AMFormer':
        # 注意：如果你在 main.py 里修改了 depth, heads, dropout 等参数，
        # 这里的 make_default 也必须传入完全相同的参数，否则无法加载权重！
        model = AMTransformer.make_default(
            n_num_features=input_size,
            cat_cardinalities=[],
            token_dim=opt.embedding_size,
            out_dim=1)
    elif model_name == 'simple_MLP':
        model = simple_MLP(dims=[input_size, 128, 1])
    elif model_name == 'simple_KAN':
        model = simple_KAN(dims=[input_size, 2 * input_size + 1, 1])
    elif model_name == 'KAN':
        model = KAN([input_size, 2 * input_size + 1, 1])
    elif model_name == 'FTTransformer':
        model = FTTransformer.make_default(n_num_features=input_size,
                                           cat_cardinalities=[],
                                           d_out=1)
    else:
        raise ValueError(f"Unsupported model structure: {model_name}")

    # 4. 加载最佳权重
    weight_path = opt.weight_path
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"\n❌ 找不到权重文件: \n{weight_path}\n请检查路径是否拼写正确！")

    print(f"Loading weights from:\n{weight_path} ...")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    print("✅ Weights loaded successfully!")

    # 5. 封装入自定义 Model 类
    my_model = Model(model, train_loader, test_loader, data_loader, lr=1e-3, loss=opt.loss)

    if torch.cuda.is_available():
        my_model.model.cuda()

    # 6. 执行测试
    print("\n" + "=" * 40)
    print("        Begin Evaluation")
    print("=" * 40)

    my_model.eval()

    print(f"\n✅ Final Results for [ {model_name} ] on [ {data.name} ]:")
    print(f"   Test Loss ({opt.loss}): {my_model.test_loss:.6f}")
    print(f"   Inverse Test Loss: {my_model.test_loss_inverse:.6f}")
    print(f"   Test R2: {my_model.test_r2:.6f}")
    print(f"   Inverse Test R2: {my_model.test_r2_inverse:.6f}")
    print("=" * 40 + "\n")