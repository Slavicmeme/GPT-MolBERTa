import concurrent.futures
import pandas as pd
import os
import glob2
import numpy as np
import yaml
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

path = os.getcwd()  # 현재 작업 디렉토리를 가져옵니다.

# YAML 설정 파일 로드
with open(os.path.join(path, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

filename = os.path.join(path, 'data_gen', 'text_files_' + config['dataframe'])

p = filename + '/' + '**.txt'

def done(path):
    new_list = []
    list_done = glob2.glob(path)
    print(len(list_done))
    for i in range(len(list_done)):
        name = list_done[i].split("/")[-1]
        mol_number = int(name.replace("Molecule ", "").split(".")[0])
        new_list.append(mol_number)
    return new_list

# 데이터셋 경로 설정
path_to_dataset = os.path.join(path, 'datasets')
dataset_path = os.path.join(path_to_dataset, str(config['dataframe']) + '.csv')
length = len(pd.read_csv(dataset_path)['smiles'])

array = np.arange(1, length + 1).tolist()
done_p = done(p)
left_over = [i for i in array if i not in done_p]

# LLaMA/Ollama 모델 초기화
llm = OllamaLLM(model='llama3.1:70b')  # 'llama-model'은 실제 모델 이름으로 바꾸어야 합니다.

prompt_template = PromptTemplate(
    input_variables=["smiles"],
    template="Generate a description about the following SMILES molecule {smiles}"
)

def generate_completion(name, value):
    prompt = prompt_template.format(smiles=value)
    response = llm.invoke(prompt)
    with open(filename + "/" + str(name) + '.txt', 'w') as f:
        f.write(str(response))

def data_gen(left_over):
    df = pd.read_csv(dataset_path)  # 수정된 부분
    x_data = df['smiles']
    mapping = {}
    for idx in left_over:
        number = 'Molecule' + ' ' + str(idx)
        smiles = x_data[idx - 1]
        mapping[number] = smiles

    max_threads = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(generate_completion, name, value) for name, value in mapping.items()]
        concurrent.futures.wait(futures)

data_gen(left_over)
