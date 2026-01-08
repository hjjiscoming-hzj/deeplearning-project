import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import datetime
import warnings

# 忽略警告
warnings.filterwarnings("ignore")


def setup_logging(output_dir="./output"):
    """设置日志记录"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"scale_analysis_{timestamp}.log")

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"日志记录已启动，日志文件: {log_file}")
    logging.info("=" * 50)
    logging.info(f"开始特征尺度分析，时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 50)

    return output_dir


def analyze_datasets():
    """分析各数据集中金属层指标的统计特性"""
    try:
        # 加载数据集
        logging.info("正在加载数据集...")
        full_df = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\HS_unit_dataset_1216.csv")
        mos1_df = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\HS_mos1.csv")
        mos2_df = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\HS_mos2.csv")

        logging.info(f"数据集加载成功: 全数据集({len(full_df)}行), MOS1({len(mos1_df)}行), MOS2({len(mos2_df)}行)")

        # 金属层指标列表
        metal_layers = [
            ('metal1_Area', 'metal1_Peak', '1P'),
            ('metal2_Area', 'metal2_Peak', '2P'),
            ('metal3_Area', 'metal3_Peak', '3P'),
            ('metal4_Area', 'metal4_Peak', '4P'),
            ('metal5_Area', 'metal5_Peak', '5P'),
            ('metal6_Area', 'metal6_Peak', '6P')
        ]

        # 统计每个数据集中金属层指标的基本统计量
        datasets = {
            "全数据集": full_df,
            "MOS1": mos1_df,
            "MOS2": mos2_df
        }

        logging.info("\n分析各数据集中金属层指标的统计特性:")

        # 创建统计结果DataFrame
        stats_results = []

        for layer_area, layer_peak, layer_name in metal_layers:
            logging.info(f"\n{layer_name} 指标统计分析:")

            # 分析Area指标
            logging.info(f"\n{layer_name} Area 统计量:")
            for name, df in datasets.items():
                if layer_area in df.columns:
                    stats = df[layer_area].describe()
                    stats_results.append({
                        'Dataset': name,
                        'Layer': layer_name,
                        'Metric': 'Area',
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Range': stats['max'] - stats['min']
                    })
                    logging.info(
                        f"{name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}, 范围={stats['max'] - stats['min']:.4f}")

            # 分析Peak指标
            logging.info(f"\n{layer_name} Peak 统计量:")
            for name, df in datasets.items():
                if layer_peak in df.columns:
                    stats = df[layer_peak].describe()
                    stats_results.append({
                        'Dataset': name,
                        'Layer': layer_name,
                        'Metric': 'Peak',
                        'Mean': stats['mean'],
                        'Std': stats['std'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Range': stats['max'] - stats['min']
                    })
                    logging.info(
                        f"{name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}, 范围={stats['max'] - stats['min']:.4f}")

        # 将结果转换为DataFrame
        stats_df = pd.DataFrame(stats_results)

        # 计算MOS1和MOS2相对于全数据集的统计量比例
        logging.info("\n计算MOS1和MOS2相对于全数据集的统计量比例:")

        ratios = []
        for layer in stats_df['Layer'].unique():
            for metric in ['Area', 'Peak']:
                full_stats = stats_df[
                    (stats_df['Dataset'] == '全数据集') & (stats_df['Layer'] == layer) & (stats_df['Metric'] == metric)]
                if full_stats.empty:
                    continue

                full_range = full_stats['Range'].values[0]
                full_std = full_stats['Std'].values[0]

                for dataset in ['MOS1', 'MOS2']:
                    subset_stats = stats_df[(stats_df['Dataset'] == dataset) & (stats_df['Layer'] == layer) & (
                                stats_df['Metric'] == metric)]
                    if subset_stats.empty:
                        continue

                    subset_range = subset_stats['Range'].values[0]
                    subset_std = subset_stats['Std'].values[0]

                    range_ratio = subset_range / full_range if full_range != 0 else float('nan')
                    std_ratio = subset_std / full_std if full_std != 0 else float('nan')

                    ratios.append({
                        'Dataset': dataset,
                        'Layer': layer,
                        'Metric': metric,
                        'Range_Ratio': range_ratio,
                        'Std_Ratio': std_ratio
                    })

                    logging.info(
                        f"{layer} {metric}: {dataset}/全数据集 范围比={range_ratio:.4f}, 标准差比={std_ratio:.4f}")

        ratios_df = pd.DataFrame(ratios)

        # 提示与逆放缩误差的关系
        logging.info("\n分析结果与逆放缩误差的关系:")

        # 逆放缩前后的损失数据（从您前面提供的数据）
        before_unscale = {
            'old': [0.00388, 0.02845, 0.05047, 0.02951, 0.04123, 0.06933],
            'mos1': [0.00976, 0.07878, 0.07099, 0.07957, 0.07647, 0.10089],
            'mos2': [0.01566, 0.07027, 0.06241, 0.08474, 0.10349, 0.08753]
        }

        after_unscale = {
            'old': [0.00236, 0.06724, 0.02696, 0.07086, 0.24454, 1.65343],
            'mos1': [0.00201, 0.08953, 0.03506, 0.09479, 0.37749, 2.12628],
            'mos2': [0.00308, 0.03764, 0.01146, 0.03996, 0.11312, 1.23477]
        }

        # 计算逆放缩放大倍数
        logging.info("\n计算逆放缩放大倍数:")
        amplification = {}
        for model in ['old', 'mos1', 'mos2']:
            amplification[model] = []
            for i in range(6):
                amp_factor = after_unscale[model][i] / before_unscale[model][i] if before_unscale[model][
                                                                                       i] != 0 else float('nan')
                amplification[model].append(amp_factor)

        for i, layer in enumerate(['1P', '2P', '3P', '4P', '5P', '6P']):
            logging.info(
                f"{layer} 逆放缩放大倍数: 全数据集={amplification['old'][i]:.2f}, MOS1={amplification['mos1'][i]:.2f}, MOS2={amplification['mos2'][i]:.2f}")

        # 分析结论
        logging.info("\n分析结论:")

        # 查找各层放大倍数与范围比例的关系
        for i, layer in enumerate(['1P', '2P', '3P', '4P', '5P', '6P']):
            mos1_ratio = ratios_df[
                (ratios_df['Dataset'] == 'MOS1') & (ratios_df['Layer'] == layer) & (ratios_df['Metric'] == 'Area')][
                'Range_Ratio'].values
            mos2_ratio = ratios_df[
                (ratios_df['Dataset'] == 'MOS2') & (ratios_df['Layer'] == layer) & (ratios_df['Metric'] == 'Area')][
                'Range_Ratio'].values

            mos1_amp = amplification['mos1'][i]
            mos2_amp = amplification['mos2'][i]

            if len(mos1_ratio) > 0 and len(mos2_ratio) > 0:
                mos1_ratio = mos1_ratio[0]
                mos2_ratio = mos2_ratio[0]

                if not np.isnan(mos1_ratio) and not np.isnan(mos2_ratio) and not np.isnan(mos1_amp) and not np.isnan(
                        mos2_amp):
                    logging.info(
                        f"{layer}: MOS1的范围比={mos1_ratio:.2f}, 放大倍数={mos1_amp:.2f}; MOS2的范围比={mos2_ratio:.2f}, 放大倍数={mos2_amp:.2f}")

        # 整体结论
        logging.info("\n特征尺度分析总结:")
        if np.mean([amplification['mos2'][i] for i in range(6) if not np.isnan(amplification['mos2'][i])]) < np.mean(
                [amplification['mos1'][i] for i in range(6) if not np.isnan(amplification['mos1'][i])]):
            logging.info("MOS2的逆放缩放大倍数平均低于MOS1，这与MOS2在逆放缩后表现更好相符。")
        else:
            logging.info("数据分析结果与预期不符，可能存在其他因素影响模型性能。")

        # 保存统计结果到CSV
        stats_df.to_csv('metal_layer_statistics.csv', index=False)
        ratios_df.to_csv('metal_layer_ratios.csv', index=False)
        logging.info("\n统计结果已保存到CSV文件。")

        return stats_df, ratios_df

    except Exception as e:
        logging.error(f"分析过程中出错: {str(e)}")
        return None, None


def create_visualizations(stats_df, ratios_df, output_dir):
    """创建可视化图表"""
    try:
        logging.info("\n创建数据可视化图表...")

        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("Set2")

        # 1. 绘制金属层范围比较图
        plt.figure(figsize=(14, 8))

        # 只选择Area指标
        area_stats = stats_df[stats_df['Metric'] == 'Area']

        # 按数据集分组
        for i, dataset in enumerate(['全数据集', 'MOS1', 'MOS2']):
            subset = area_stats[area_stats['Dataset'] == dataset]
            plt.bar(
                np.arange(len(subset)) + i * 0.25 - 0.25,
                subset['Range'].values,
                width=0.25,
                label=dataset
            )

        plt.xlabel('金属层')
        plt.ylabel('数值范围 (max-min)')
        plt.title('不同数据集中金属层Area指标的数值范围比较')
        plt.xticks(np.arange(6), area_stats[area_stats['Dataset'] == '全数据集']['Layer'].values)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'metal_layer_range_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. 绘制逆放缩放大倍数比较图
        amplification = {
            'old': [],
            'mos1': [],
            'mos2': []
        }

        # 逆放缩前后的损失数据
        before_unscale = {
            'old': [0.00388, 0.02845, 0.05047, 0.02951, 0.04123, 0.06933],
            'mos1': [0.00976, 0.07878, 0.07099, 0.07957, 0.07647, 0.10089],
            'mos2': [0.01566, 0.07027, 0.06241, 0.08474, 0.10349, 0.08753]
        }

        after_unscale = {
            'old': [0.00236, 0.06724, 0.02696, 0.07086, 0.24454, 1.65343],
            'mos1': [0.00201, 0.08953, 0.03506, 0.09479, 0.37749, 2.12628],
            'mos2': [0.00308, 0.03764, 0.01146, 0.03996, 0.11312, 1.23477]
        }

        # 计算并绘制逆放缩放大倍数
        plt.figure(figsize=(14, 8))

        for model, label in zip(['old', 'mos1', 'mos2'], ['全数据集', 'MOS1', 'MOS2']):
            amp_factors = []
            for i in range(6):
                amp_factor = after_unscale[model][i] / before_unscale[model][i] if before_unscale[model][i] != 0 else 0
                amp_factors.append(amp_factor)

            plt.bar(
                np.arange(6) + (['old', 'mos1', 'mos2'].index(model) * 0.25 - 0.25),
                amp_factors,
                width=0.25,
                label=label
            )

        plt.xlabel('金属层')
        plt.ylabel('逆放缩放大倍数 (放缩后/放缩前)')
        plt.title('不同数据集中金属层指标的逆放缩放大倍数比较')
        plt.xticks(np.arange(6), ['1P', '2P', '3P', '4P', '5P', '6P'])
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 限制y轴以便更好地查看前几层
        plt.ylim(0, 30)  # 根据实际情况调整

        # 添加5P和6P的实际值标签
        for i, model in enumerate(['old', 'mos1', 'mos2']):
            for j in [4, 5]:  # 5P和6P的索引
                amp_factor = after_unscale[model][j] / before_unscale[model][j] if before_unscale[model][j] != 0 else 0
                if amp_factor > 30:  # 如果超出显示范围
                    plt.text(
                        j + (i * 0.25 - 0.25),
                        28,  # 稍低于上限
                        f"{amp_factor:.1f}",
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

        plt.savefig(os.path.join(output_dir, 'unscale_amplification_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 3. 绘制范围比例与放大倍数的关系图
        plt.figure(figsize=(14, 8))

        # 只考虑Area指标
        area_ratios = ratios_df[ratios_df['Metric'] == 'Area']

        # 准备数据
        layers = ['1P', '2P', '3P', '4P', '5P', '6P']
        mos1_range_ratios = []
        mos2_range_ratios = []
        mos1_amp_factors = []
        mos2_amp_factors = []

        for layer in layers:
            # 范围比例
            mos1_ratio = area_ratios[(area_ratios['Dataset'] == 'MOS1') & (area_ratios['Layer'] == layer)][
                'Range_Ratio'].values
            mos2_ratio = area_ratios[(area_ratios['Dataset'] == 'MOS2') & (area_ratios['Layer'] == layer)][
                'Range_Ratio'].values

            mos1_range_ratios.append(mos1_ratio[0] if len(mos1_ratio) > 0 else np.nan)
            mos2_range_ratios.append(mos2_ratio[0] if len(mos2_ratio) > 0 else np.nan)

            # 放大倍数
            i = layers.index(layer)
            mos1_amp = after_unscale['mos1'][i] / before_unscale['mos1'][i] if before_unscale['mos1'][
                                                                                   i] != 0 else np.nan
            mos2_amp = after_unscale['mos2'][i] / before_unscale['mos2'][i] if before_unscale['mos2'][
                                                                                   i] != 0 else np.nan

            mos1_amp_factors.append(mos1_amp)
            mos2_amp_factors.append(mos2_amp)

        # 绘制MOS1的图
        width = 0.35
        x = np.arange(len(layers))

        ax1 = plt.subplot(111)
        ax1.bar(x - width / 2, mos1_range_ratios, width, label='MOS1 范围比例', color='darkorange')
        ax1.bar(x + width / 2, mos2_range_ratios, width, label='MOS2 范围比例', color='forestgreen')

        ax1.set_xlabel('金属层')
        ax1.set_ylabel('数值范围比例 (MOS/全数据集)')
        ax1.set_title('MOS1和MOS2相对于全数据集的范围比例与逆放缩放大倍数关系')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend(loc='upper left')

        # 创建第二个y轴显示放大倍数
        ax2 = ax1.twinx()
        ax2.plot(x, mos1_amp_factors, 'o-', color='darkred', label='MOS1 放大倍数')
        ax2.plot(x, mos2_amp_factors, 's-', color='darkgreen', label='MOS2 放大倍数')
        ax2.set_ylabel('逆放缩放大倍数')
        ax2.legend(loc='upper right')

        # 限制y轴显示范围
        ax2.set_ylim(0, 30)

        # 为超出范围的5P和6P标注实际值
        for i, layer in enumerate(layers):
            for j, (amp_factors, color) in enumerate([(mos1_amp_factors, 'darkred'), (mos2_amp_factors, 'darkgreen')]):
                if i >= 4 and amp_factors[i] > 30:  # 5P和6P
                    ax2.text(
                        i,
                        28,  # 稍低于上限
                        f"{amp_factors[i]:.1f}",
                        ha='center',
                        va='bottom',
                        color=color,
                        fontsize=9
                    )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'range_ratio_vs_amplification.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"可视化图表已保存到 {output_dir} 目录")

    except Exception as e:
        logging.error(f"创建可视化图表时出错: {str(e)}")


def main():
    # 设置日志和输出目录
    output_dir = setup_logging("../../log/analysis_log/HS数据集相关性分析/2025_3_20/feature_scale_analysis_output")

    # 分析数据集
    stats_df, ratios_df = analyze_datasets()

    # 创建可视化图表
    if stats_df is not None and ratios_df is not None:
        create_visualizations(stats_df, ratios_df, output_dir)

    # 记录分析结束
    logging.info("=" * 50)
    logging.info(f"特征尺度分析完成! 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 50)

    print(f"\n分析完成！结果已保存到 {output_dir} 目录。")


if __name__ == "__main__":
    main()