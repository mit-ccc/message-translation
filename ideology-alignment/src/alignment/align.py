from pathlib import Path
import sys
import logging
import json
import AlignUtils as AU
import pickle
import os
import pandas as pd
from gensim.models.word2vec import Word2Vec
import nltk
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)

# ./src/alignment/align_svd.py ./data/models/ ./data/aligners/cca/

default_args = {'k':None,
               'method':'svd'}

def get_modelfiles(model_dir, selected_ids=None):
    res = []
    if selected_ids and len(selected_ids) > 0:
        for i in selected_ids:
            res.append(Path(os.path.join(source_dir, "{}.model".format(i))))
    else:
        for model_file in source_dir.glob("*.model"):
            res.append(model_file)
    return sorted(res)

def build_aligner(a, b, counts, name1, name2, opts=default_args):
    k = opts['k']
    
    # get the shared vocab
    shared_vocab = list(sorted(list(set.intersection(set(a.wv.vocab), set(b.wv.vocab)))))
    
    # get the anchors
    v_counts = [(w, (counts[name1][w] if w in counts[name1] else 0) + (counts[name2][w] if w in counts[name2] else 0)) for w in shared_vocab]
    sorted_v = sorted(v_counts, key=lambda x: x[1], reverse=True)
    sorted_v = [x for x,y in sorted_v]
    if k is None:
        anchors = sorted_v
    elif k == 0:
        anchors = list(set(stopwords.words('english')).intersection(sorted_v))
    else:
        anchors = sorted_v[:k]
        
    # get the aligner
    if opts['method'] == 'svd':
        aligner = AU.get_svd_aligner(a, b, shared_vocab, anchors)
    if opts['method'] == 'lstsq':
        aligner = AU.get_lstsq_aligner(a, b, shared_vocab, anchors)
    if opts['method'] == 'cca':
        aligner = AU.get_cca_aligner(a, b, shared_vocab, anchors)
    return aligner

def model_iterator(files, counts, target, opts):
    for i in range(len(files)):
        model_a = Word2Vec.load(str(files[i]))
        for j in range(len(files)):
            if i == j:
                continue
            else:
                model_b = Word2Vec.load(str(models[j]))
                
            name_i = files[i].stem
            name_j = files[j].stem
            aligner = build_aligner(model_a, model_b, counts, name_i, name_j, opts)
            aligner_name = f"{name_i}-{name_j}.pkl"
            with open(target / aligner_name, 'wb') as fp:
                pickle.dump(aligner, fp)
            
if __name__ == "__main__":
    assert(len(sys.argv) == 6)
    
    source_dir = Path(sys.argv[1])
    target_dir = Path(sys.argv[2])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    counts_dir = Path(sys.argv[3])
    method = sys.argv[4]
    assert(method in ['svd', 'cca', 'lstsq'])
    k = int(sys.argv[5])
        
    with open(counts_dir) as fp:
        counts = json.load(fp)
    
    opts = {}
    opts['method'] = method
    if k == -1:
        k = None
    opts['k'] = k

    # # HACK
    # df_selected = pd.read_csv("./selected_clusters.csv")
    # selected_ids = df_selected.community_id.values
    # selected_ids = [-1] + list(selected_ids)
    # models = get_modelfiles(source_dir, selected_ids=selected_ids)
    models = get_modelfiles(source_dir)
    model_iterator(models, counts, target_dir, opts)

    