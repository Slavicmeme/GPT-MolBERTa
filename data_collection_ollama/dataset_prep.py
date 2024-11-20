import numpy as np
import pandas as pd
import os
import yaml

location = os.getcwd()  # 현재 작업 디렉토리

# 설정 파일 읽기
with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

# 데이터셋 경로 설정
path_to_dataset = os.path.join(location, 'datasets')
dataset_path = path_to_dataset + '/' + str(config['dataframe']) + '.csv'

df = pd.read_csv(dataset_path)
length = len(df['smiles'])
number = np.arange(1, length + 1)
directory = os.path.join(location, 'data_gen', 'text_files_' + config['dataframe'])
descriptions = []

# 파일에서 내용 읽어오기
for i in number:
    with open(directory + '/' + 'Molecule ' + str(i) + '.txt', 'r') as f:
        content = f.read()  # 텍스트 내용을 그대로 읽어옴
        descriptions.append(content)

# 데이터프레임에 내용 추가
df.loc[:, 'Descriptions'] = descriptions

output_directory = os.path.join(location, 'data')
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

df.to_csv(os.path.join(output_directory, str(config['dataframe']) + '_dataset.csv'), sep=',', encoding='utf-8', index=False)

