import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集
data = pd.read_csv('../data/csv/dataset.csv', delimiter='|')

# 将数据集分为训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)  # 设置验证集比例，例如 20%

# 将训练集和验证集保存为新的CSV文件
train_data.to_csv('../data/process/train.process', index=False, sep='|')
val_data.to_csv('../data/process/val.process', index=False, sep='|')
