import os
import pandas as pd
import re

# 读取原始CSV文件
file_path = 'metadata.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path, sep='|', header=None, names=['filename', 'text'])

# 移除中文text超过81个字符的行
data['text_length'] = data['text'].apply(lambda x: len(str(x).strip()))  # 计算文本长度
data_filtered = data[(data['text_length'] < 40) & (data['text_length'] > 4)].copy()  # 保留长度不超过81的文本行，使用copy()

# 去除text中的<>标签括起来的部分
data_filtered['text'] = data_filtered['text'].apply(lambda x: re.sub(r'<[^>]*>', '', x))

# 去除text中的空格和&
data_filtered['text'] = data_filtered['text'].apply(lambda x: x.replace(' ', '').replace('&', ''))

# 检查 wavs/ 目录下是否有对应的音频文件，如果没有则移除数据
data_filtered['wav_exists'] = data_filtered['filename'].apply(lambda x: os.path.exists(f'wavs/{x}.wav')).copy()
data_filtered = data_filtered[data_filtered['wav_exists']]

# 移除名称和问题数据匹配的行
problematic_data = [
    '20200630_S_R001S05C01_7136965_7159443',
    '20200805_S_R001S08C01_29317680_29323215',
    '20200630_S_R001S05C01_22846098_22868014',
    '20200623_S_R001S07C01_19899375_19906278',
    '20200623_S_R001S06C01_11460892_11466921',
    '20200707_L_R001S01C01_16734911_16738564',
    '20200707_L_R001S01C01_16734911_16738564',  # 如果这是重复数据，可以删除其中一个
    '20200707_L_R001S08C01_19336752_19519388'
]
data_filtered = data_filtered[~data_filtered['filename'].isin(problematic_data)].copy()

# 保存新的CSV文件
new_file_path = 'metadata_handle.csv'  # 新文件路径
data_filtered[['filename', 'text']].to_csv(new_file_path, sep='|', index=False, header=False)
