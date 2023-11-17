import pandas as pd

# 读取原始CSV文件
file_path = 'metadata.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path, sep='|', header=None, names=['filename', 'text'])

# 移除中文text超过81个字符的行
data['text_length'] = data['text'].apply(lambda x: len(str(x).strip()))  # 计算文本长度
data_filtered = data[data['text_length'] <= 81]  # 保留长度不超过81的文本行

# 保存新的CSV文件
new_file_path = 'metadata_handle.csv'  # 新文件路径
data_filtered[['filename', 'text']].to_csv(new_file_path, sep='|', index=False, header=False)
