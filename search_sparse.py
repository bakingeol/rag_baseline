#%%
from pyterrier.measures import *
import pyterrier as pt
if not pt.started():
    pt.init(mem=32000)
    
from beir.datasets.data_loader import GenericDataLoader

import logging
import os

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
from utils import cleanStrDF

data_path = '/home/baekig/practice/beir_practice/beir/examples/retrieval/training/datasets/msmarco_v2'
corpus, queries, _qrels = GenericDataLoader(data_path).load(split="train")

logging.info('DB에 저장된 corpus 로드')
path = '/home/baekig/practice/pyterrier_practice/indices/msmarco-passage/data.properties'
index = pt.IndexFactory.of(path)   
_bm25 = pt.BatchRetrieve(index, wmodel='BM25')

bm25_clean = pt.apply.query(cleanStrDF) >> _bm25
#%%
import pandas as pd
queries_df = pd.DataFrame({'qid': [''], 'query': ["what is project charter in project management"]})

logging.info('top 1000개 검색결과')
results = bm25_clean.transform(queries_df)

logging.info(f'검색결과: \n{results}')

#%%
logging.info('1000개중 top10개 courpus 내용 출력')
rel_corpus_list = []
for i in results.loc[:,'docid'][:10].tolist():
    rel_corpus_list.append(corpus[str(i)])
logging.info(rel_corpus_list)
