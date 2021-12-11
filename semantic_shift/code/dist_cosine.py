# python3
# coding: utf-8

import numpy as np
import argparse
import logging
from sklearn import preprocessing
import pickle

# Cosine similarity (COS) algorithm

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input0', '-i0', help='Path to 1st npz file with the embeddings', required=True)
    arg('--input1', '-i1', help='Path to 2nd npz file with the embeddings', required=True)
    arg('--target', '-t', help='Path to target words', required=True)
    arg('--output', '-o', help='Output path (csv)', required=False)
    arg('--mode', '-m', default='mean', choices=['mean', 'pca', 'sum'])
    arg('--num_layers', '-l', type=int, default=1)
    # arg('-f', action='store_true', help='Output frequencies?')

    args = parser.parse_args()
    data_path0 = args.input0
    data_path1 = args.input1

    target_words = set([w.strip() for w in open(args.target, 'r', encoding='utf-8').readlines()])

    # Note that we only save the average token embeddings for each word, so array0 and array1 will have
    # a dictionar of words {w: [v]]} where v is a vector of 12*768
    with open(args.input0, 'rb') as handle:
        array0 = pickle.load(handle)
    logger.info('Loaded an array of {0} entries from {1}'.format(len(array0), data_path0))

    with open(args.input1, 'rb') as handle:
        array1 = pickle.load(handle)
    logger.info('Loaded an array of {0} entries from {1}'.format(len(array1), data_path1))

    try:
        f_out = open(args.output, 'w', encoding='utf-8')
    except TypeError:
        f_out = None

    for word in target_words:
        # we are not aware of the frequency here
        if (word in array0) and (word in array1):
            vectors0 = array0[word]
            vectors1 = array1[word]
            vectors = []
            
            for vector in [vectors0, vectors1]:
                vector = vector[-args.num_layers*768:]
                vectors.append(vector)

            vectors = [preprocessing.normalize(v.reshape(1, -1), norm='l2') for v in vectors]
            shift = 1 - np.dot(vectors[0].reshape(-1), vectors[1].reshape(-1))

            print('\t'.join([word, str(shift)]), file=f_out)

    if f_out:
        f_out.close()
