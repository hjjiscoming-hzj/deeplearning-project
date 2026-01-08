import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import datetime
from scipy.stats import pearsonr
import warnings
import pandas as pd


# 忽略字体相关的警告
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

def setup_logging(output_dir):
    """
    设置日志记录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 创建日志文件名，包含时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"correlation_analysis_{timestamp}.log")

    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"日志记录已启动，日志文件: {log_file}")

    return logger

def load_data(file_path):
    """
    加载数据集
    """
    df = pd.read_csv(file_path)
    logging.info(f"数据集加载成功，形状: {df.shape}")
    return df


def correlation_analysis(df, feature_cols, target_cols, output_dir='./output', mos_type=None):
    """
    进行相关性分析并可视化

    Parameters:
    -----------
    df : DataFrame
        数据集
    feature_cols : list
        特征列名列表
    target_cols : list
        目标列名列表
    output_dir : str
        输出目录
    mos_type : str, optional
        MOS类型 ('MOS1', 'MOS2' 或 None表示全部数据)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置字体和样式
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set(style="whitegrid")

        # 测试中文显示
        plt.figure(figsize=(1, 1))
        plt.title('测试')
        plt.close()
        use_chinese = True
    except Exception:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        use_chinese = False
        logging.info("注意: 由于字体问题，将使用英文进行可视化")

    # 确保所有列都是数值类型
    df_numeric = df.copy()
    for col in feature_cols + target_cols:
        if df_numeric[col].dtype == 'object':
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                logging.info(f"列 '{col}' 已转换为数值类型")
            except:
                logging.warning(f"警告: 列 '{col}' 无法转换为数值类型")

    # 移除常量特征
    logging.info("\n检查并移除常量特征:")
    non_constant_features = []
    for col in feature_cols:
        if df_numeric[col].nunique() > 1:  # 如果列有多个唯一值
            non_constant_features.append(col)
        else:
            logging.warning(f"警告: 列 '{col}' 在当前数据子集中是常量，将从相关性分析中排除")

    # 更新特征列列表，仅包含非常量特征
    feature_cols = non_constant_features
    logging.info(f"有效特征数量: {len(feature_cols)}")

    # 同样检查目标列
    non_constant_targets = []
    for col in target_cols:
        if df_numeric[col].nunique() > 1:
            non_constant_targets.append(col)
        else:
            logging.warning(f"警告: 目标列 '{col}' 在当前数据子集中是常量，将从相关性分析中排除")

    # 更新目标列列表
    target_cols = non_constant_targets
    logging.info(f"有效目标列数量: {len(target_cols)}")

    # 如果没有足够的非常量特征或目标列，则退出分析
    if len(feature_cols) < 2 or len(target_cols) < 1:
        logging.error(f"错误: 没有足够的非常量特征或目标列进行相关性分析")
        return None, None

    # 设置标题前缀（根据MOS类型）
    title_prefix = ""
    if mos_type:
        title_prefix = f"{mos_type} - "

    # 1. 特征与特征之间的相关性
    logging.info(f"\n=== 计算{mos_type or '全部数据'}特征之间的相关性 ===")

    try:
        # 计算特征之间的相关性
        feature_corr = df_numeric[feature_cols].corr().astype(float)

        # 找出相关性最高的特征对
        feature_corr_no_diag = feature_corr.copy()
        np.fill_diagonal(feature_corr_no_diag.values, 0)

        # 安全地查找相关性最高的对
        max_val = 0
        max_i, max_j = 0, 0
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                val = abs(feature_corr_no_diag.iloc[i, j])
                if pd.notna(val) and val > max_val:
                    max_val = val
                    max_i, max_j = i, j

        logging.info(
            f"特征之间相关性最高的一对是: {feature_cols[max_i]} 和 {feature_cols[max_j]}, 相关系数: {feature_corr.iloc[max_i, max_j]:.4f}")

        # 找出相关性最高的N对特征
        corr_pairs = []
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                if pd.notna(feature_corr.iloc[i, j]):
                    corr_pairs.append((
                        feature_cols[i],
                        feature_cols[j],
                        abs(feature_corr.iloc[i, j])
                    ))

        top_corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:10]
        logging.info("\n相关性最高的10对特征:")
        for pair in top_corr_pairs:
            logging.info(f"{pair[0]} 和 {pair[1]}: {pair[2]:.4f}")
    except Exception as e:
        logging.error(f"计算特征相关性时出错: {str(e)}")
        feature_corr = pd.DataFrame()

    # 2. 特征与目标之间的相关性
    logging.info(f"\n=== 计算{mos_type or '全部数据'}特征与目标变量之间的相关性 ===")

    try:
        # 初始化相关性矩阵
        feature_target_corr = pd.DataFrame(index=feature_cols, columns=target_cols)

        # 逐个计算特征与目标之间的相关性
        for feature in feature_cols:
            for target in target_cols:
                # 提取有效的配对数据（同时删除两列中任意一列为NaN的行）
                valid_data = df_numeric[[feature, target]].dropna()

                if len(valid_data) < 2:
                    logging.warning(f"警告: {feature} 和 {target} 的有效数据不足以计算相关性")
                    feature_target_corr.loc[feature, target] = np.nan
                else:
                    try:
                        # 使用scipy的pearsonr计算相关系数，更安全
                        x = valid_data[feature].astype(float).values
                        y = valid_data[target].astype(float).values
                        corr, _ = pearsonr(x, y)
                        feature_target_corr.loc[feature, target] = corr
                    except Exception as e:
                        logging.warning(f"计算 {feature} 和 {target} 的相关性时出错: {str(e)}")
                        feature_target_corr.loc[feature, target] = np.nan

        # 找出每个目标变量最相关的特征
        most_correlated_features = {}
        for target in target_cols:
            # 获取与当前目标相关的所有特征的相关系数
            correlations = feature_target_corr[target].dropna()

            if not correlations.empty:
                # 手动找出绝对值最大的相关系数
                max_corr = 0
                max_feature = None
                for feature, corr in correlations.items():
                    if pd.notna(corr) and abs(corr) > max_corr:
                        max_corr = abs(corr)
                        max_feature = feature
                        max_corr_value = corr

                if max_feature:
                    most_correlated_features[target] = (max_feature, max_corr_value)

        # 输出结果
        if most_correlated_features:
            logging.info("\n每个目标变量最相关的特征:")
            for target, (feature, corr) in most_correlated_features.items():
                logging.info(f"{target}: {feature} (相关系数: {corr:.4f})")

        # 计算特征的重要性（基于与所有目标的平均相关性）
        feature_importance = {}
        for feature in feature_cols:
            # 计算当前特征与所有目标的平均绝对相关性
            correlations = feature_target_corr.loc[feature, :].abs().dropna()
            if not correlations.empty:
                feature_importance[feature] = float(correlations.mean())

        # 按重要性排序特征
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            logging.info("\n特征按重要性排序（基于与所有目标的平均相关性）:")
            for i, (feature, importance) in enumerate(sorted_features):
                logging.info(f"{i + 1}. {feature}: {importance:.4f}")
    except Exception as e:
        logging.error(f"分析特征与目标变量相关性时出错: {str(e)}")
        feature_target_corr = pd.DataFrame()
        sorted_features = []

    # 3. 可视化
    logging.info(f"\n=== 生成{mos_type or '全部数据'}相关性可视化 ===")

    # 为输出文件名添加前缀
    file_prefix = ""
    if mos_type:
        file_prefix = f"{mos_type.lower()}_"

    try:
        # 可视化特征间的相关性
        if not feature_corr.empty:
            plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(feature_corr, dtype=bool))  # 只绘制下三角
            heatmap_data = feature_corr.astype(float)
            sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5, mask=mask)
            if use_chinese:
                plt.title(f'{title_prefix}特征之间的相关性')
            else:
                plt.title(f'{title_prefix}Correlation Between Features')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{file_prefix}feature_correlation.png", dpi=300)
            plt.close()

        # 可视化特征与目标的相关性
        if not feature_target_corr.empty:
            plt.figure(figsize=(16, max(6, len(feature_cols) // 3)))
            heatmap_data = feature_target_corr.fillna(0).astype(float)
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            if use_chinese:
                plt.title(f'{title_prefix}特征与目标变量的相关性')
            else:
                plt.title(f'{title_prefix}Correlation Between Features and Targets')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{file_prefix}feature_target_correlation.png", dpi=300)
            plt.close()

        # 可视化特征重要性
        if sorted_features:
            top_features = sorted_features[:min(15, len(sorted_features))]
            plt.figure(figsize=(10, 8))
            importance_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            importance_df['Importance'] = importance_df['Importance'].astype(float)
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
            if use_chinese:
                plt.title(f'{title_prefix}特征重要性（基于平均相关性）')
            else:
                plt.title(f'{title_prefix}Feature Importance (Based on Average Correlation)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{file_prefix}feature_importance.png", dpi=300)
            plt.close()

        logging.info(f"可视化结果已保存到 {output_dir} 目录")
    except Exception as e:
        logging.error(f"生成可视化图表时出错: {str(e)}")
    return feature_corr, feature_target_corr


def main():
    # 配置参数
    file_path = "D:\DeepLearning\Datasets\Merak\HS_unit_dataset_1216.csv"
    output_dir = "../../log/analysis_log/HS数据集相关性分析"

    # 设置日志记录
    logger = setup_logging(output_dir)

    # 记录分析开始
    logging.info("=" * 50)
    logging.info(f"开始相关性分析，时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"数据文件: {file_path}")
    logging.info(f"输出目录: {output_dir}")
    logging.info("=" * 50)

    # 加载数据
    df = load_data(file_path)

    # 定义特征列和目标列（排除前两列非特征列）
    # 根据您的数据集定义
    feature_cols = [
        'unit_height', 'unit_width', 'fp_row', 'fp_col',
        'main_width', 'main_output_height', 'adv1_width',
        'adv1_output_height', 'adv2_width', 'adv2_output_height',
        'output_pad_num', 'Iload', 'vnet_pad_to_gap',
        'output_pad_to_gap', 'pad_to_split_up', 'pad_to_split_down',
        'last_adv_width', 'next_main_width', 'next_main_height',
        'next_adv_width'
    ]

    target_cols = [
        'metal1_Area', 'metal1_Peak', 'metal2_Area', 'metal2_Peak',
        'metal3_Area', 'metal3_Peak', 'metal4_Area', 'metal4_Peak',
        'metal5_Area', 'metal5_Peak', 'metal6_Area', 'metal6_Peak'
    ]

    # 检查列是否在数据集中
    available_cols = df.columns.tolist()
    feature_cols = [col for col in feature_cols if col in available_cols]
    target_cols = [col for col in target_cols if col in available_cols]

    logging.info(f"\n使用的特征列 ({len(feature_cols)}): {', '.join(feature_cols)}")
    logging.info(f"使用的目标列 ({len(target_cols)}): {', '.join(target_cols)}")

    # 1. 对全部数据执行相关性分析
    logging.info("\n===== 对全部数据进行相关性分析 =====")
    correlation_analysis(df, feature_cols, target_cols, output_dir)

    # 2. 根据fp_row划分MOS类型，按具体值进行细分
    logging.info("\n===== 按具体的fp_row值进行相关性分析 =====")

    # 检查数据集中fp_row的取值
    fp_row_values = sorted(df['fp_row'].unique())
    logging.info(f"数据集中fp_row的取值: {fp_row_values}")

    # 定义MOS1和MOS2的fp_row值
    mos1_values = [9, 12, 15]
    mos2_values = [3, 4, 5]

    # 为每个具体的fp_row值创建子目录和进行分析
    for fp_value in fp_row_values:
        # 确定MOS类型
        if fp_value in mos1_values:
            mos_type = f"MOS1_{fp_value}"
            parent_dir = "mos1"
        elif fp_value in mos2_values:
            mos_type = f"MOS2_{fp_value}"
            parent_dir = "mos2"
        else:
            mos_type = f"Other_{fp_value}"
            parent_dir = "other"

        # 筛选数据
        subset_df = df[df['fp_row'] == fp_value].copy()

        if not subset_df.empty:
            logging.info(f"\n分析{mos_type} (fp_row={fp_value}) 数据, 共 {len(subset_df)} 条记录")
            # 创建特征列的深拷贝，避免修改原始列表
            subset_feature_cols = feature_cols.copy()
            subset_target_cols = target_cols.copy()

            # 创建输出目录路径
            subset_output_dir = os.path.join(output_dir, parent_dir, f"fp_row_{fp_value}")

            # 执行相关性分析
            correlation_analysis(subset_df, subset_feature_cols, subset_target_cols, subset_output_dir, mos_type)
        else:
            logging.warning(f"警告: 数据集中不存在fp_row={fp_value}的记录")

    # 3. 同时也为MOS1和MOS2的所有数据分别进行汇总分析
    logging.info("\n===== 按MOS大类进行汇总相关性分析 =====")

    # MOS1汇总分析
    mos1_df = df[df['fp_row'].isin(mos1_values)].copy()
    if not mos1_df.empty:
        logging.info(f"\n分析MOS1 (fp_row in {mos1_values}) 汇总数据, 共 {len(mos1_df)} 条记录")
        mos1_feature_cols = feature_cols.copy()
        mos1_target_cols = target_cols.copy()
        correlation_analysis(mos1_df, mos1_feature_cols, mos1_target_cols, os.path.join(output_dir, 'mos1', 'all'),
                             'MOS1_All')

    # MOS2汇总分析
    mos2_df = df[df['fp_row'].isin(mos2_values)].copy()
    if not mos2_df.empty:
        logging.info(f"\n分析MOS2 (fp_row in {mos2_values}) 汇总数据, 共 {len(mos2_df)} 条记录")
        mos2_feature_cols = feature_cols.copy()
        mos2_target_cols = target_cols.copy()
        correlation_analysis(mos2_df, mos2_feature_cols, mos2_target_cols, os.path.join(output_dir, 'mos2', 'all'),
                             'MOS2_All')

    # 记录分析结束
    logging.info("=" * 50)
    logging.info(f"相关性分析完成! 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 50)

    # 关闭日志处理器
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    print(f"\n分析完成！日志已保存到 {output_dir} 目录中。")


if __name__ == "__main__":
    main()