import concurrent.futures
import pandas as pd
import os
import yaml
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

location = os.getcwd()  # '/home/suryabalaji/GPT_MolBERTa'

with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

filename = os.path.join(location, 'data_gen', 'text_files_' + config['dataframe'])

if not os.path.exists(filename):
    os.makedirs(filename)

# LLaMA/Ollama model initialization
llm = OllamaLLM(model='gemma2:27b') # 모델 이름을 'gemma2:27b'로 지정

prompt_template = PromptTemplate(
    input_variables=["smiles"],
    template="Generate a description about the following SMILES molecule {smiles}"
)

def generate_completion(name, value):
    prompt = prompt_template.format(smiles=value)
    response = llm.invoke(prompt)
    with open(filename + "/" + str(name) + '.txt', 'w') as f:
        f.write(str(response))

def data_gen(idx):
    df = pd.read_csv(dataset_path)
    x_data = df['smiles'][idx:]

    mapping = {}
    for index, row in pd.DataFrame(x_data).iterrows():
        smiles = row['smiles']
        name = 'Molecule' + ' ' + str(index + 1)
        mapping[name] = smiles

    max_threads = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(generate_completion, name, value) for name, value in mapping.items()]
        concurrent.futures.wait(futures)

path_to_dataset = os.path.join(location, 'datasets')
dataset_path = path_to_dataset + '/' + str(config['dataframe']) + '.csv'
length = len(pd.read_csv(dataset_path)['smiles'])

current_number = 0

while current_number < length:
    try:
        data_gen(current_number)
    except:
        for root, dirs, files in os.walk(filename, topdown=False):
            current_number = len(files)
