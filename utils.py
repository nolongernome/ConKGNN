# coding=UTF-8
import pickle as pkl
import re
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from tqdm import tqdm


# import sparse

def clear_short_word(words):
    if len(words) > 1:
        de_t = []
        for t in range(len(words) - 1):
            for t_next in range(t + 1, len(words)):
                if words[t] in words[t_next]:
                    if words[t] not in de_t:
                        de_t.append(words[t])
                elif words[t_next] in words[t]:
                    if words[t_next] not in de_t:
                        de_t.append(words[t_next])
        for t in de_t:
            words.remove(t)
    return words


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, path):
    """
    Loads input data from gcn/data directory
    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed', 'y', 'data_id', 'triple','b_mask','tx_adj', 'tx_embed', 'ty','t_data_id', 't_triple','t_b_mask',
             'allx_adj', 'allx_embed', 'ally', 'all_data_id', 'all_triple','v_b_mask']
    objects = []

    for i in range(len(names)):
        with open(path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, id, triple, b_mask,tx_adj, tx_embed, ty, tid, t_triple,t_b_mask, allx_adj, allx_embed, ally, allid, all_triple,v_b_mask = tuple(
        objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    train_id = []
    train_triple = []
    train_b_mask=[]
    val_adj = []
    val_embed = []
    val_id = []
    val_triple = []
    val_b_mask = []
    test_adj = []
    test_embed = []
    test_id = []
    test_triple = []
    test_b_mask = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        b_mask_tmp=np.array(b_mask[i])
        train_adj.append(adj)
        train_embed.append(embed)
        train_id.append(str(id[i]))
        if len(triple) == 0:
            train_triple.append([])
        else:
            train_triple.append(triple[i])
        train_b_mask.append(b_mask_tmp)

    for i in range(len(ally)):  # val_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        b_mask_tmp = np.array(v_b_mask[i])
        val_adj.append(adj)
        val_embed.append(embed)
        val_id.append(allid[i])
        if len(all_triple) == 0:
            val_triple.append([])
        else:
            val_triple.append(all_triple[i])
        val_b_mask.append(b_mask_tmp)
    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        b_mask_tmp = np.array(t_b_mask[i])
        test_adj.append(adj)
        test_embed.append(embed)
        test_id.append(tid[i])
        if len(t_triple) == 0:
            test_triple.append([])
        else:
            test_triple.append(t_triple[i])
        test_b_mask.append(b_mask_tmp)

    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    train_y = np.array(y)
    val_y = np.array(ally)  # train_size])
    test_y = np.array(ty)
    train_id = np.array(train_id)
    val_id = np.array(val_id)
    test_id = np.array(test_id)
    train_triple = np.array(train_triple)
    val_triple = np.array(val_triple)
    test_triple = np.array(test_triple)
    train_b_mask=np.array(train_b_mask)
    val_b_mask = np.array(val_b_mask)
    test_b_mask = np.array(test_b_mask)

    return train_adj, train_embed, train_y, train_id, train_triple,train_b_mask, val_adj, val_embed, val_y, val_id, val_triple, val_b_mask,test_adj, test_embed, test_y, test_id, test_triple,test_b_mask

def load_data_5(dataset_str, path):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed', 'y', 'data_id', 'triple','tx_adj', 'tx_embed', 'ty','t_data_id', 't_triple',
             'allx_adj', 'allx_embed', 'ally', 'all_data_id', 'all_triple']
    objects = []

    for i in range(len(names)):
        with open(path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, id, triple, tx_adj, tx_embed, ty, tid, t_triple, allx_adj, allx_embed, ally, allid, all_triple = tuple(
        objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    train_id = []
    train_triple = []
    val_adj = []
    val_embed = []
    val_id = []
    val_triple = []
    test_adj = []
    test_embed = []
    test_id = []
    test_triple = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)
        train_id.append(str(id[i]))
        if len(triple) == 0:
            train_triple.append([])
        else:
            train_triple.append(triple[i])

    for i in range(len(ally)):  # val_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)
        val_id.append(allid[i])
        if len(all_triple) == 0:
            val_triple.append([])
        else:
            val_triple.append(all_triple[i])

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)
        test_id.append(tid[i])
        if len(t_triple) == 0:
            test_triple.append([])
        else:
            test_triple.append(t_triple[i])

    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    train_y = np.array(y)
    val_y = np.array(ally)  # train_size])
    test_y = np.array(ty)
    train_id = np.array(train_id)
    val_id = np.array(val_id)
    test_id = np.array(test_id)
    train_triple = np.array(train_triple)
    val_triple = np.array(val_triple)
    test_triple = np.array(test_triple)

    return train_adj, train_embed, train_y, train_id, train_triple, val_adj, val_embed, val_y, val_id, val_triple, test_adj, test_embed, test_y, test_id, test_triple

def load_data_augment(dataset_str, path):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed', 'y','b_mask','tx_adj', 'tx_embed', 'ty','t_b_mask',
             'allx_adj', 'allx_embed', 'ally', 'v_b_mask']
    objects = []

    for i in range(len(names)):
        with open(path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, b_mask,tx_adj, tx_embed, ty,t_b_mask, allx_adj, allx_embed, ally, v_b_mask = tuple(
        objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    train_b_mask=[]
    val_adj = []
    val_embed = []
    val_b_mask = []
    test_adj = []
    test_embed = []
    test_b_mask = []

    for i in range(len(y)*2):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        b_mask_tmp=np.array(b_mask[i])
        train_adj.append(adj)
        train_embed.append(embed)
        train_b_mask.append(b_mask_tmp)

    for i in range(len(allx_adj)):  # val_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        b_mask_tmp = np.array(v_b_mask[i])
        val_adj.append(adj)
        val_embed.append(embed)
        val_b_mask.append(b_mask_tmp)

    for i in range(len(tx_adj)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        b_mask_tmp = np.array(t_b_mask[i])
        test_adj.append(adj)
        test_embed.append(embed)
        test_b_mask.append(b_mask_tmp)


    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    train_y = np.array(y)
    val_y = np.array(ally)  # train_size])
    test_y = np.array(ty)
    train_b_mask=np.array(train_b_mask)
    val_b_mask = np.array(val_b_mask)
    test_b_mask = np.array(test_b_mask)

    return train_adj, train_embed, train_y,train_b_mask, val_adj, val_embed, val_y, val_b_mask,test_adj, test_embed, test_y, test_b_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo_to_tuple(sparse_coo):
    return (sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape)


def preprocess_features(features):
    """Row-normalize feature matrix and convert
    to tuple representation"""
    max_length = max([len(f) for f in features])
    # print(features[0])
    # print(max_length)

    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    return np.array(list(features))


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    a = adj.dot(d_mat_inv_sqrt)
    b = a.transpose()
    c = b.dot(d_mat_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    for a in adj:
        break
    mask = np.zeros((adj.shape[0], max_length, 1))  # mask for padding
    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adj[i].shape[0], :] = 1.
        adj[i] = adj_normalized
    return np.array(list(adj)), mask


def preprocess_adj_mask(adj,b_mask):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    for a in adj:
        break
    mask = np.zeros((adj.shape[0], max_length, 1))  # mask for padding
    b_mask_pad = np.zeros((adj.shape[0], max_length, 1))
    b_mask_ud = np.zeros((adj.shape[0], max_length, 1))
    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adj[i].shape[0], :] = 1.
        for m in range(len(b_mask[i])):
            if int(b_mask[i][m])==1:
                    b_mask_pad[i,m,:]=1
            elif m<=adj[i].shape[0]:
                b_mask_ud[i,m,:]=1

        adj[i] = adj_normalized
    return np.array(list(adj)), mask , b_mask_pad, b_mask_ud# coo_to_tuple(sparse.COO(np.array(list(adj)))), mask


def construct_feed_dict(features, support, mask, labels, b_mask,b_mask_ud,supcon_mask,supcon_logits_mask,placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['b_mask']: b_mask})
    feed_dict.update({placeholders['b_mask_ud']: b_mask_ud})
    feed_dict.update({placeholders['supcon_mask']: supcon_mask})
    feed_dict.update({placeholders['supcon_logits_mask']: supcon_logits_mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if (len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    '''
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\[", "", string)
    string = re.sub(r"\]", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"，", "", string)
    string = re.sub(r"。", "", string)
    string = re.sub(r"\（", "", string)
    string = re.sub(r"\）", "", string)
    string = re.sub(r"、", "", string)
    string = re.sub(r"……", "", string)
    string = re.sub(r"：", "", string)
    string = re.sub(r"“", "", string)
    string = re.sub(r"”", "", string)
    string = re.sub(r"……", "", string)
    '''
    return string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def construct_supcon_para(labels,bitch_size):
    labels=np.argmax(labels,1)
    labels=labels.reshape(-1,1)
    mask=np.equal(labels, np.transpose(labels)).astype(np.float)
    mask = np.tile(mask, [2, 2])
    logits_mask=np.ones((bitch_size*2,bitch_size*2))
    row, col = np.diag_indices_from(logits_mask)
    logits_mask[row, col] = 0
    # indices = np.array([[i, i] for i in range(bitch_size*2)])
    # logits_mask = np.tensor_scatter_nd_update(np.ones((bitch_size*2,bitch_size*2)), indices, [0] * bitch_size*2)
    return mask,logits_mask

def cut_off(list,keep_pro,seed):
    #np.random.seed(seed)
    rand_arr=np.random.randint(0,len(list),len(list))
    #rand_arr = np.arange(0, len(list))
    index=np.where(rand_arr>round(len(list)*keep_pro))
    raw=np.array(list)
    new=np.delete(raw,index)
    if new.shape[0]<2:
        new=raw
    return new.tolist()