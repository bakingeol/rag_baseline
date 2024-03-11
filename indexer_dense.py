from pyterrier.measures import *
import pyterrier as pt
if not pt.started():
    pt.init(mem=32000)
from pyterrier_sentence_transformers import (
    SentenceTransformersIndexer,
    SentenceTransformerConfig
)
import faiss

from datasets import load_dataset

import springs as sp
from pathlib import Path
from typing import Optional
import argparse

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# sparse_indexer = False
# dense_indexer = False

def main(args):
    DATASET = 'msmarco-passage/dev'

    dataset = pt.get_dataset(f'irds:{DATASET}')
    index_root = Path(args.path) / DATASET.replace('/', '_')

    NEU_MODEL_NAME = args.model_name 
    
    logging.info('*'*30,NEU_MODEL_NAME,'*'*30)
    neu_index_path = index_root / NEU_MODEL_NAME.replace('/', '_')
    
    
    class SentenceTransformerConfigWithDefaults(SentenceTransformerConfig):
        model_name_or_path: Optional[str] = None    # type: ignore
        index_path: Optional[str] = None
    
    config=SentenceTransformerConfigWithDefaults(model_name_or_path=NEU_MODEL_NAME,
                                                index_path=neu_index_path)
    
    logging.info(f'{NEU_MODEL_NAME} properties download')
    indexer = SentenceTransformersIndexer(
        model_name_or_path=NEU_MODEL_NAME,
        index_path=str(neu_index_path),
        overwrite=True,
        normalize=False,
        config=sp.to_dict(config),
        text_attr=['text']
    )
    logging.info('***** faiss 만드는 중... *****')
    indexer.index(dataset.get_corpus_iter())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ingeol/cot_5')
    parser.add_argument('--path', default='./output/faiss_indexer') # '/home/baekig/practice/pyterrier_practice'
    args=parser.parse_args()
    main(args)