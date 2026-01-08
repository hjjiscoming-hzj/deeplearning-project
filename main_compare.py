import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import logging
import datetime
import argparse

# 导入您的模型
from models.am_transformer.am_transformer import AMTransformer
from models.base_model import simple_MLP,simple_KAN
from models.ft_transformer.ft_tranformer import FTTransformer


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--loss', default='L1', type=str)
parser.add_argument('--output_dir',
                    default='log/analysis_log/全数据集和专用模型对比/2025_6_21_AMFormer_KAN_rc_staticfusion3-7_depth1/model_comparison_6P',
                    type=str)
parser.add_argument('--model_name', default='AMFormer', type=str)   # AMFormer / MLP / KAN / FTTransformer
args = parser.parse_args()


# 设置日志
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"comparison_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"日志记录已启动: {log_file}")
    return output_dir


# 数据加载函数
def load_data():
    # 这里需要替换为您的实际数据加载方式
    from huangzhj.data_utils.data_preprocessing import DataPreprocessing_HS_mos1, DataPreprocessing_HS_mos2

    data_mos1 = DataPreprocessing_HS_mos1()
    data_mos2 = DataPreprocessing_HS_mos2()
    target_name = data_mos1.target
    X_mos1, y_mos1 = data_mos1.get_data()
    X_mos2, y_mos2 = data_mos2.get_data()

    return X_mos1, y_mos1, X_mos2, y_mos2, target_name


# 准备数据集
def prepare_datasets(X_mos1, y_mos1, X_mos2, y_mos2):
    # 划分训练测试集
    X_mos1_train, X_mos1_test, y_mos1_train, y_mos1_test = train_test_split(
        X_mos1, y_mos1, test_size=0.2, random_state=args.seed)

    X_mos2_train, X_mos2_test, y_mos2_train, y_mos2_test = train_test_split(
        X_mos2, y_mos2, test_size=0.2, random_state=args.seed)

    # 合并训练集创建全数据集
    X_full_train = pd.concat([X_mos1_train, X_mos2_train], axis=0).reset_index(drop=True)
    y_full_train = pd.concat([y_mos1_train, y_mos2_train], axis=0).reset_index(drop=True)

    logging.info(f"数据集大小: 全数据集({len(X_full_train)}), MOS1({len(X_mos1_train)}), MOS2({len(X_mos2_train)})")

    return {
        'full': {'X_train': X_full_train, 'y_train': y_full_train},
        'mos1': {'X_train': X_mos1_train, 'y_train': y_mos1_train, 'X_test': X_mos1_test, 'y_test': y_mos1_test},
        'mos2': {'X_train': X_mos2_train, 'y_train': y_mos2_train, 'X_test': X_mos2_test, 'y_test': y_mos2_test}
    }


# 数据预处理
def preprocess_data(X_train, y_train, X_test=None, y_test=None):
    # 标准化特征
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    # 标准化目标
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    if X_test is not None and y_test is not None:
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y
    else:
        return X_train_scaled, y_train_scaled, scaler_X, scaler_y


# 创建数据加载器
def create_dataloaders(X_train_scaled, y_train_scaled, X_test_scaled=None, y_test_scaled=None):
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if X_test_scaled is not None and y_test_scaled is not None:
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test_scaled)
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        return train_loader


# 创建模型
def create_model(input_dim, output_dim, name="AMFormer"):
    if name == "AMFormer":
        # 使用AMFormer创建模型
        model = AMTransformer.make_default(
            n_num_features=input_dim,
            cat_cardinalities=[],
            token_dim=args.embedding_size,
            out_dim=output_dim
        )
    elif name == "MLP":
        model = simple_MLP(dims=[input_dim, 128, 1])
    elif name == "KAN":
        model = simple_KAN(dims=[input_dim, 2 * input_dim + 1, 1])
    elif name == 'FTTransformer':
        model = FTTransformer.make_default(n_num_features=input_dim,
                                           cat_cardinalities=[],
                                           d_out=1)
    else:
        return None

    return model


# 训练模型
def train_model(model, train_loader, test_loaders, scalers_y, test_names=None, scaler_y_full=None, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.loss == 'L1':
        criterion = torch.nn.SmoothL1Loss()
    elif args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()  # 默认使用SmoothL1Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 为每个测试集初始化最佳损失记录
    best_losses = {}
    best_epochs = {}
    for i in range(len(test_loaders)):
        name = test_names[i] if test_names and i < len(test_names) else f"test_set_{i}"
        best_losses[name] = {'scaled': float('inf'), 'unscaled': float('inf')}
        best_epochs[name] = 0

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, x_cat=None)  # AMFormer需要x_cat参数
            # outputs = model(inputs)  # AMFormer需要x_cat参数
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 在每个测试集上评估
        model.eval()
        all_test_losses = {}

        for i, (test_loader, scaler_y) in enumerate(zip(test_loaders, scalers_y)):
            name = test_names[i] if test_names and i < len(test_names) else f"test_set_{i}"

            test_loss = 0.0
            inverse_loss = 0.0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs, x_cat=None)
                    # outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    # 计算逆缩放后的损失
                    outputs_np = outputs.cpu().numpy()
                    targets_np = targets.cpu().numpy()

                    # 对模型输出使用全数据集的目标缩放器
                    if scaler_y_full is not None:
                        inverse_outputs = scaler_y_full.inverse_transform(outputs_np)
                    else:
                        inverse_outputs = scaler_y.inverse_transform(outputs_np)

                    # 对目标值使用各自数据集的缩放器
                    inverse_targets = scaler_y.inverse_transform(targets_np)

                    inverse_outputs_tensor = torch.FloatTensor(inverse_outputs).to(device)
                    inverse_targets_tensor = torch.FloatTensor(inverse_targets).to(device)

                    inverse_loss += criterion(inverse_outputs_tensor, inverse_targets_tensor).item()

            test_loss /= len(test_loader)
            inverse_loss /= len(test_loader)

            all_test_losses[name] = {'scaled': test_loss, 'unscaled': inverse_loss}

            # 更新最佳损失
            # if test_loss < best_losses[name]['scaled']:
            #     best_losses[name]['scaled'] = test_loss
            #     best_losses[name]['unscaled'] = inverse_loss
            #     best_epochs[name] = epoch + 1
            if inverse_loss < best_losses[name]['unscaled']:
                best_losses[name]['scaled'] = test_loss
                best_losses[name]['unscaled'] = inverse_loss
                best_epochs[name] = epoch + 1

        # 每个epoch都记录损失
        log_str = f"Epoch {epoch + 1}/{args.epochs}: Train Loss = {train_loss:.6f}"
        for name, losses in all_test_losses.items():
            log_str += f", {name} Loss = {losses['scaled']:.6f}, {name} Inv Loss = {losses['unscaled']:.6f}"
            if test_names != ["MOS1", "MOS2"]:
                writer.add_scalar(name+'Inv Loss', losses['unscaled'], epoch)
        logging.info(log_str)

        # 记录最佳性能
        for name, best_loss in best_losses.items():
            if all_test_losses[name]['scaled'] == best_loss['scaled']:
                logging.info(f"[新最佳-{name}] Epoch {epoch + 1}: " +
                             f"Scaled Loss = {best_loss['scaled']:.6f}, " +
                             f"Unscaled Loss = {best_loss['unscaled']:.6f}")

    # 训练完成，汇总最佳性能
    logging.info("训练完成! 各测试集上的最佳性能:")
    for name, best_loss in best_losses.items():
        logging.info(f"{name}: 最佳Epoch {best_epochs[name]}, " +
                     f"Scaled Loss = {best_loss['scaled']:.6f}, " +
                     f"Unscaled Loss = {best_loss['unscaled']:.6f}")

    return model, best_losses


def load_specialized_model_results(csv_path):
    """从CSV文件加载专用模型的损失结果"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 检查必要的列是否存在
        required_columns = ['Test Set', 'Specialized Model Scaled Loss', 'Specialized Model Unscaled Loss']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"CSV文件缺少必要的列。需要的列: {required_columns}，实际列: {df.columns.tolist()}")
            return None

        results = {}
        for _, row in df.iterrows():
            test_set = row['Test Set']
            results[test_set] = {
                'scaled': float(row['Specialized Model Scaled Loss']),
                'unscaled': float(row['Specialized Model Unscaled Loss'])
            }

        return results
    except Exception as e:
        logging.error(f"读取专用模型结果时出错: {e}")
        return None


def main():
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 设置输出目录和日志
    output_dir = setup_logging(args.output_dir)
    mos1_writer = SummaryWriter(args.output_dir+'/mos1')
    mos2_writer = SummaryWriter(args.output_dir+'/mos2')
    # 尝试从CSV文件加载专用模型的结果
    # specialized_results = load_specialized_model_results("log/analysis_log/全数据集和专用模型对比/2025_4_22_AMFormer_KAN_rc_staticfusion/model_comparison_6P/model_comparison.csv")
    # if specialized_results:
    #     logging.info("成功从CSV文件加载专用模型的结果")
    # else:
    #     logging.error("无法从CSV文件加载专用模型的结果，请检查文件路径")
    #     return

    # 加载数据
    X_mos1, y_mos1, X_mos2, y_mos2, target_name = load_data()
    logging.info("预测指标为{}".format(target_name))
    # 准备数据集
    datasets = prepare_datasets(X_mos1, y_mos1, X_mos2, y_mos2)

    # 创建和训练模型
    logging.info("开始训练模型...")
    model_name = args.model_name
    models = {}
    scalers = {}
    best_losses = {}

    # 训练MOS1模型
    logging.info(f"训练MOS1模型")
    X_train, y_train = datasets['mos1']['X_train'], datasets['mos1']['y_train']
    X_test, y_test = datasets['mos1']['X_test'], datasets['mos1']['y_test']
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = preprocess_data(X_train, y_train,
                                                                                                       X_test, y_test)
    train_loader, test_loader = create_dataloaders(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    model_mos1 = create_model(X_train.shape[1], y_train.shape[1], name=model_name)
    logging.info('Model_mos1:{0}'.format(model_mos1.get_info()))
    model_mos1, mos1_losses = train_model(model_mos1, train_loader, [test_loader], [scaler_y], ["MOS1"], writer=mos1_writer)

    models['mos1'] = model_mos1
    scalers['mos1'] = {'X': scaler_X, 'y': scaler_y}
    best_losses['mos1'] = mos1_losses['MOS1']

    # 训练MOS2模型
    logging.info(f"训练MOS2模型")
    X_train, y_train = datasets['mos2']['X_train'], datasets['mos2']['y_train']
    X_test, y_test = datasets['mos2']['X_test'], datasets['mos2']['y_test']
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = preprocess_data(X_train, y_train,
                                                                                                       X_test, y_test)
    train_loader, test_loader = create_dataloaders(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    model_mos2 = create_model(X_train.shape[1], y_train.shape[1], name=model_name)
    logging.info('Model_mos2:{0}'.format(model_mos2.get_info()))
    model_mos2, mos2_losses = train_model(model_mos2, train_loader, [test_loader], [scaler_y], ["MOS2"], writer=mos2_writer)

    models['mos2'] = model_mos2
    scalers['mos2'] = {'X': scaler_X, 'y': scaler_y}
    best_losses['mos2'] = mos2_losses['MOS2']

    # 训练全数据集模型（同时在MOS1和MOS2测试集上评估）
    # best_losses['mos1'] = specialized_results['MOS1']
    # best_losses['mos2'] = specialized_results['MOS2']
    logging.info(f"训练全数据集模型")
    X_train, y_train = datasets['full']['X_train'], datasets['full']['y_train']

    # 创建全数据集训练加载器
    X_train_scaled, y_train_scaled, scaler_X_full, scaler_y_full = preprocess_data(X_train, y_train)
    train_loader = create_dataloaders(X_train_scaled, y_train_scaled)

    # 为MOS1和MOS2测试集创建测试加载器
    # MOS1测试集
    X_test_mos1 = datasets['mos1']['X_test']
    y_test_mos1 = datasets['mos1']['y_test']
    X_test_mos1_scaled = scaler_X_full.transform(X_test_mos1)
    y_test_mos1_scaled = scalers['mos1']['y'].transform(y_test_mos1)

    test_dataset_mos1 = TensorDataset(
        torch.FloatTensor(X_test_mos1_scaled),
        torch.FloatTensor(y_test_mos1_scaled)
    )
    test_loader_mos1 = DataLoader(test_dataset_mos1, batch_size=args.batch_size, shuffle=False)

    # MOS2测试集
    X_test_mos2 = datasets['mos2']['X_test']
    y_test_mos2 = datasets['mos2']['y_test']
    X_test_mos2_scaled = scaler_X_full.transform(X_test_mos2)
    y_test_mos2_scaled = scalers['mos2']['y'].transform(y_test_mos2)

    test_dataset_mos2 = TensorDataset(
        torch.FloatTensor(X_test_mos2_scaled),
        torch.FloatTensor(y_test_mos2_scaled)
    )
    test_loader_mos2 = DataLoader(test_dataset_mos2, batch_size=args.batch_size, shuffle=False)

    # 训练全数据集模型，同时在两个测试集上评估
    model_full = create_model(X_train.shape[1], y_train.shape[1], name=model_name)
    logging.info('Model_full:{0}'.format(model_full.get_info()))
    model_full, full_losses = train_model(
        model_full,
        train_loader,
        [test_loader_mos1, test_loader_mos2],
        [scalers['mos1']['y'], scalers['mos2']['y']],
        ["MOS1", "MOS2"],
        scaler_y_full
    )

    models['full'] = model_full
    scalers['full'] = {'X': scaler_X_full}
    best_losses['full'] = full_losses

    # 比较结果
    logging.info("比较不同模型的表现...")

    comparison = pd.DataFrame([
        {
            'Test Set': 'MOS1',
            'Full Model Scaled Loss': best_losses['full']['MOS1']['scaled'],
            'Specialized Model Scaled Loss': best_losses['mos1']['scaled'],
            'Scaled Improvement (%)': (best_losses['mos1']['scaled'] - best_losses['full']['MOS1']['scaled']) /
                                      best_losses['full']['MOS1']['scaled'] * 100,
            'Full Model Unscaled Loss': best_losses['full']['MOS1']['unscaled'],
            'Specialized Model Unscaled Loss': best_losses['mos1']['unscaled'],
            'Unscaled Improvement (%)': (best_losses['mos1']['unscaled'] - best_losses['full']['MOS1']['unscaled']) /
                                        best_losses['full']['MOS1']['unscaled'] * 100
        },
        {
            'Test Set': 'MOS2',
            'Full Model Scaled Loss': best_losses['full']['MOS2']['scaled'],
            'Specialized Model Scaled Loss': best_losses['mos2']['scaled'],
            'Scaled Improvement (%)': (best_losses['mos2']['scaled'] - best_losses['full']['MOS2']['scaled']) /
                                      best_losses['full']['MOS2']['scaled'] * 100,
            'Full Model Unscaled Loss': best_losses['full']['MOS2']['unscaled'],
            'Specialized Model Unscaled Loss': best_losses['mos2']['unscaled'],
            'Unscaled Improvement (%)': (best_losses['mos2']['unscaled'] - best_losses['full']['MOS2']['unscaled']) /
                                        best_losses['full']['MOS2']['unscaled'] * 100
        }
    ])

    # 保存结果
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    logging.info(f"比较结果已保存到: {comparison_path}")
    logging.info(f"\n{comparison}")

    # 分析结论
    logging.info("\n分析结论:")
    for _, row in comparison.iterrows():
        test_name = row['Test Set']
        unscaled_improvement = row['Unscaled Improvement (%)']

        if unscaled_improvement < 0:
            logging.info(f"{test_name} 专用模型相比全数据集模型在逆放缩后有 {abs(unscaled_improvement):.2f}% 的提升")
        else:
            logging.info(f"{test_name} 专用模型相比全数据集模型在逆放缩后性能降低了 {unscaled_improvement:.2f}%")

    # 最终建议
    mos1_imp = comparison.loc[0, 'Unscaled Improvement (%)']
    mos2_imp = comparison.loc[1, 'Unscaled Improvement (%)']

    logging.info("\n最终建模建议:")
    if mos1_imp < 0 and mos2_imp < 0:
        logging.info("建议为MOS1和MOS2分别使用专用模型")
    elif mos1_imp < 0 and mos2_imp >= 0:
        logging.info("建议为MOS1使用专用模型，为MOS2使用全数据集模型")
    elif mos1_imp >= 0 and mos2_imp < 0:
        logging.info("建议为MOS1使用全数据集模型，为MOS2使用专用模型")
    else:
        logging.info("建议为MOS1和MOS2都使用全数据集模型")

    logging.info("=" * 50)
    logging.info(f"分析完成! 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n分析完成！结果已保存到 {output_dir} 目录。")


if __name__ == "__main__":
    main()