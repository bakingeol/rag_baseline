from pyterrier.measures import *
import pyterrier as pt
if not pt.started():
    pt.init(mem=32000)


dataset = pt.get_dataset('irds:msmarco-passage/dev') 
path = './output/bm25_indexer'
indexer = pt.IterDictIndexer(path, verbose=True)
index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text'])
# ref: https://ir-datasets.com/msmarco-passage.html#msmarco-passage/dev


# path = '/home/baekig/practice/pyterrier_practice/indices/msmarco-passage/data.properties'