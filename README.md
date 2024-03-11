# rag_baseline
'''python
cp -r /home/baekig/base_line [본인 경로]
conda create -n [env명] python=3.8.5 ipykernel 
cd [본인경로]/base_line
pip install -r req.txt
'''

### sparse database 만드는 과정입니다. Output/bm25_indexer 파일안에database가 저장됩니다.
'''python
python indexer_sparse.py
'''
### sparse database 만드는 과정입니다. Output 파일에 database가 저장됩니다.
### 서버 접속 후 사용
'''python
CUDA_VISIBLE_DEVICES=0 python indexer_dense.py 
'''
### search_sparse.py 는 bm25를 사용해 search 후 관련 corpus를 가져오는 코드입니다.
'''python
python search_sparse.py
'''
### search_dense.py 는 사전학습된 dense 모델을 사용해 검색 후 생성과정까지 작성해 놓았습니다.
'''python
python search_dense.py
'''
### dense retrieval finetuning code
Biencoder, cross encoder, dpr 방식등 학습 방법이 다양합니다. 아래 링크에서 학습코드를 확인하실 수 있습니다.
sentencetransformer: https://www.sbert.net/docs/training/overview.html
beir(biencoder + loss 함수 변형): https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3_bpr.py
Haystack(dpr): https://github.com/deepset-ai/haystack-tutorials/tree/main/tutorials
