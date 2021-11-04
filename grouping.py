import numpy as np
import sys
import warnings
import pickle
from scipy import sparse

def load_embeddings(filename):
    vectors = np.loadtxt(filename, delimiter=',')
    return vectors

def grouping(U_path, V_path, feature_path):
    vectors_U = load_embeddings(U_path)
    vectors_V = load_embeddings(U_path)

    # n*2d matrix
    concat = np.concatenate((vectors_U,vectors_V), axis=1)

    print(concat.shape)

    # n*f matrix: train_test split
    features = pickle.load(open(feature_path, 'rb'), encoding='iso-8859-1')
    print(features.shape)
    
    d2 = concat.shape[1]
    n = features.shape[0]
    f = features.shape[1]
    print(type(features))

    embed = np.zeros((f,d2))

    # Compressed matrix to normal matrix
    features1 = features.A

    for row in range(n):
        for col in range(f):
            if features1[row][col] == 1:
                embed[col] += concat[row]
    
    # f*2d matrix
    return concat, embed

def save_emb(embed, task_type):
    folder="emb/"
    suffix = ".bin"
    emb_file = folder+'Cora'+'_'+task_type+'_'+suffix

    with open(emb_file, "wb") as fout:
        np.asarray(embed, dtype=np.float64).tofile(fout)

if __name__=='__main__':

    # dataset = sys.argv[1]
    dataset = 'Cora'

    U_path = 'EB/' + dataset + '_strap_svd_d_U.csv'
    V_path = 'EB/' + dataset + '_strap_svd_d_V.csv'

    feature_path = 'data/' + dataset + '/' + 'attrs.0.7.pkl'

    concat, embedding = grouping(U_path, V_path, feature_path)
    save_emb(embedding, 'attr_emb_0.7')
    save_emb(concat, 'node_emb')
