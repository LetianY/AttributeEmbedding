import numpy as np
from sklearn.metrics import *
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity,  manhattan_distances
from scipy.spatial.distance import hamming
from scipy.spatial.distance import correlation
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import chebyshev
import pickle
import math

def get_roc_score(X, Y, F, attr_pos, attr_neg):
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    preds = []
    for (v,r) in attr_pos:
        x = np.dot(X[v,:],Y[r,:])
        preds.append(x)

    preds_neg = []
    for (v,r) in attr_neg:
        x = np.dot(X[v,:],Y[r,:])
        preds_neg.append(x)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

if __name__ == '__main__':
    n = 2708
    d2 = 256
    f = 1433
    dataset = 'Cora'

    feature_path = 'data/' + dataset + '/' + "attrs.pkl"
    features = pickle.load(open(feature_path, 'rb'), encoding='iso-8859-1')
    print(features.shape)

    path_edge_train = 'data/' + dataset + '/' + 'edgelist.txt'
    edges_train, max_id = utils.load_edges(path_edge_train)

    path_attr_emb = 'emb/Cora_attr_emb_0.7_.bin'
    path_emb = 'emb/Cora_node_emb_.bin'

    X = utils.load_emd(path_emb, n, d2, max_id)
    print(X.shape)
    
    Y = utils.load_attr_emd(path_attr_emb, d2)
    print(Y.shape)

    path_attr_pos = 'data/' + dataset + '/' + 'attr.test' + '.0.7' + '.txt'
    attr_pos,_ = utils.load_edges(path_attr_pos)

    path_attr_neg = 'data/' + dataset + '/' + 'attr.neg' + '.0.7' + '.txt'
    attr_neg,_ = utils.load_edges(path_attr_neg)

    roc_score, ap_score = get_roc_score(X, Y, features, attr_pos, attr_neg)
    print("%f %f"%(roc_score,ap_score))