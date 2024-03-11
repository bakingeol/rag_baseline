#%%
import faiss
import numpy as np
import pandas as pd
import torch

from beir.datasets.data_loader import GenericDataLoader

from sentence_transformers import SentenceTransformer

data_path = '/home/baekig/practice/beir_practice/beir/examples/retrieval/training/datasets/msmarco_v2'
cot_5_path = '/home/baekig/practice/pyterrier_practice/msmarco-passage_dev/ingeol_cot_5/faiss/index.faiss'
cot_data_path = '/home/baekig/practice/pyterrier_practice/data_final/train_final_cot.csv'
model_id = 'ingeol/cot_5'

corpus, queries, _qrels = GenericDataLoader(data_path).load(split="train")
index = faiss.read_index(cot_5_path)
df_cot=pd.read_csv(cot_data_path)
#%%
def return_input_sentence(num):
    return df_cot.loc[:,'query'][num]+'. ' + df_cot.loc[:,'query'][num]+'. '+\
        df_cot.loc[:,'query'][num]+'. '+df_cot.loc[:,'cot'][num], df_cot.loc[:,'query_id'][num]

#%%

model = SentenceTransformer(model_id)
#%% 64개 한번에 batch retrieve
sentences = [return_input_sentence(i)[0] for i in range(0,64)]
qids = [return_input_sentence(i)[1] for i in range(0,64)]
#%%
# D: 관련도 점수, I: index, k: 검색을 통해 가져올 corpus 갯수
D,I = index.search(np.array(torch.tensor(model.encode(sentences))), k=1000)
# %%
# 아래의 형태로 생성 모델 input으로 사용
new_input_query = df_cot.loc[:,'query'][1] + corpus[str(I[1,0])]['text']
# %% ---------------------- qa task -------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'snorkelai/Snorkel-Mistral-PairRM-DPO'

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
model.eval()
model.to(device)
#%%
from utils import prompt

def gen(query, max_new_tokens):
        gened=model.generate(**tokenizer(query, return_tensors='pt').to(device),
                            max_new_tokens = max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id
                            )
        return tokenizer.decode(gened[0], skip_special_tokens=True)
    
output=gen(query = prompt(new_input_query), max_new_tokens=150)
# %%
print(output)