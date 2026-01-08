import pandas as pd

# 输入CSV文件路径
input_csv = "D:\DeepLearning\Datasets\Merak\HS_unit_dataset_1216.csv"  # 替换为你的CSV文件路径

# 读取CSV文件
df = pd.read_csv(input_csv)

# 根据fp_row的值将数据分为两组
mos1_df = df[df['fp_row'].isin([9, 12, 15])]
mos2_df = df[df['fp_row'].isin([3, 4, 5])]

# 将结果保存到新的CSV文件中
mos1_df.to_csv('D:\DeepLearning\Datasets\Merak\HS_mos1.csv', index=False)  # 保存为mos1.csv，不保存索引
mos2_df.to_csv('D:\DeepLearning\Datasets\Merak\HS_mos2.csv', index=False)  # 保存为mos2.csv，不保存索引

print("数据已成功分割并保存到mos1.csv和mos2.csv文件中。")