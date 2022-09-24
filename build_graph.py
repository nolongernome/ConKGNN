import csv
import os
import random
from utils import *

dataset = "Drugs"
embedding="node2vec"
augmentation_seed=0
perserve_rate=0.85
print("dataset: "+dataset+" | embedding: "+embedding +" | perserve_rate: "+perserve_rate)
window_size = 3
print('using default window size = 3')
weighted_graph = False
print('using default unweighted graph')
truncate = False  # whether to truncate long document
MAX_TRUNC_LEN = 170

label_list = [x.strip() for x in
              open(os.path.abspath('..') + '/dataset/untils_dataset/output/' + dataset + '/' + dataset + '_class.txt',
                   encoding='utf-8').readlines()]
label_list = [str(i) for i in range(0, len(label_list))]

print('loading triple')
# load triple and transform it to dic
nodes = []
node_relation_dic = {}

classnames=dataset.split("_")

for c in classnames:
    with open(os.path.abspath('..') + '/dataset/embeddings/output/' + c + '/'+c+'_node.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            nodes.append(line.strip())
nodes=list(set(nodes))
print('knowledge node size: ' + str(len(nodes)) + " ...")

node_relation_dic = {}
for c in classnames:
    triple = open(os.path.abspath('..') + '/dataset/embeddings/output/' + c + '/' + c + '_triples.csv', 'r', encoding='utf-8')
    for line in csv.reader(triple):
        if "," not in line[2]:
            if line[0] not in node_relation_dic:
                node_relation_dic[line[0]] = [(line[1], line[2])]
            else:
                node_relation_dic[line[0]].append((line[1], line[2]))

for n in node_relation_dic:
    list_end = []
    new_r_e = []
    for r_e in node_relation_dic[n]:
        if r_e[1] not in list_end:
            list_end.append(r_e[1])
            new_r_e.append(r_e)
    node_relation_dic[n] = new_r_e
print(node_relation_dic[n])

print('loading raw data')

# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open(os.path.abspath('..') + '/dataset/embeddings/sgns.sogou.char', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))

for c in classnames:
    with open(os.path.abspath('..') + '/dataset/embeddings/output/'+c+'/'+embedding+'.hita.' + c + '.char', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.split()
            try:
                word_embeddings[str(data[0])] = list(map(float, data[1:]))
                word_embeddings[str(data[0])] = word_embeddings[str(data[0])][0:300]
            except:
                print("GRAPH EMBEDDING WARNING!")

# load document list
doc_name_list = []
doc_train_list = []
doc_test_list = []
doc_dev_list = []

with open(os.path.abspath('..') + '/dataset/untils_dataset/output/' + dataset + '/' + dataset + '_label.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
        elif temp[1].find('dev') != -1:
            doc_dev_list.append(line.strip())

# load raw text
doc_content_list = []

with open(os.path.abspath('..') + '/dataset/untils_dataset/output/' + dataset + '/' + dataset + '_clean_knowledge.txt', 'r',
          encoding="utf-8") as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())

# map and shuffle
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

dev_ids = []
for dev_name in doc_dev_list:
    dev_id = doc_name_list.index(dev_name)
    dev_ids.append(dev_id)
random.shuffle(dev_ids)

print("train: "+str(len(train_ids))+"| dev: "+str(len(dev_ids))+"| test: "+str(len(test_ids)))
ids = train_ids + dev_ids+test_ids

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])

# build corpus vocabulary
word_set = set()

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(set(list(word_set) + nodes))
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

train_size = len(train_ids)
val_size=len(dev_ids)
real_train_size = train_size
test_size = len(test_ids)

# build graph function
def build_joint_graph(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    doc_len_raw=[]
    subkg=[]
    x_triples = []
    b_mask=[]
    seg_lists=[]
    vocab_set = set()
    for i in tqdm(range(start, end)):
        doc_words = shuffle_doc_words_list[i].split()
        doc_node_relation_dic = {}
        w_size = 0
        tmp_w = []
        x_triple=[]
        for w in doc_words:
            if w in node_relation_dic:
                tmp_w.append(w)
        tmp_w = clear_short_word(tmp_w)
        for w in doc_words:
            if w in tmp_w:
                doc_node_relation_dic[w] = node_relation_dic[w]
                w_size = w_size + 1

        delete_node = []
        tmp_w = list(set(tmp_w))
        if len(tmp_w) > 1:
            for j in range(len(tmp_w) - 1):
                for k in range(j + 1, len(tmp_w)):
                    if tmp_w[j] in tmp_w[k]:
                        delete_node.append(tmp_w[j])
                        break
                    elif tmp_w[k] in tmp_w[j]:
                        delete_node.append(tmp_w[k])
                        break
        delete_node = list(set(delete_node))
        for d in delete_node:
            tmp_w.remove(d)
        for w in tmp_w:
            doc_node_relation_dic[w] = node_relation_dic[w]
        for w in doc_words:
            if w in tmp_w:
                w_size = w_size + 1

        diff = MAX_TRUNC_LEN - len(doc_words)
        #diff=0
        doc_words_add_triples = []
        add_relations = []
        add_triples=[]
        background_nodes=[]
        mask_background=[]
        sub=[]
        if diff > 0:
            # 得到每个词最多可取多少三元组
            if w_size > 0:
                triple_max = diff // 2 // w_size
            else:
                triple_max = 0
            for w in doc_words:
                doc_words_add_triples.append(w)
                if w in doc_node_relation_dic:
                    sub.append(w)
                    num = triple_max
                    for r_e in doc_node_relation_dic[w]:
                        if num > 0:
                            doc_words_add_triples.append(r_e[1])
                            sub.append(r_e[1])
                            add_triples.append((w, r_e[0], r_e[1]))
                            add_relations.append((w,r_e[1]))
                            num = num - 1
                        else:
                            break
        else:
            if truncate:
                doc_words = doc_words[:MAX_TRUNC_LEN]
        x_triple.append(add_triples)

        #遮蔽列表
        for a in add_relations:
            if a[1] not in doc_words:
                background_nodes.append(a[1])

        doc_word_id_map = {}
        if len(doc_words_add_triples) > len(doc_words):
            doc_vocab = list(set(doc_words_add_triples))
        else:
            doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)
        doc_len_list.append(doc_nodes)
        doc_len_raw.append(len(list(set(doc_words))))
        subkg.append(len(list(set(sub))))
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        doc_len = len(doc_words)
        vocab_set.update(doc_vocab)

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        for t in add_relations:
            window = list(t)
            windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))

        seg_list = []
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            seg_list.append(k)
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])
            mask_background.append(0 if k in background_nodes else 1)

        seg_lists.append(seg_list)
        x_adj.append(adj)
        x_feature.append(features)
        b_mask.append(mask_background)
        x_triples.append(x_triple)
    # data_id
    data_id = []

    # one-hot labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        data_id.append(temp[0])
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    data_id = np.array(data_id)
    x_triples = np.array(x_triples)

    return x_adj, x_feature, y, doc_len_list, vocab_set, data_id, x_triples,b_mask,seg_lists,doc_len_raw,subkg

def build_joint_graph_augment(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    doc_len_raw=[]
    subkg=[]
    x_triples = []
    b_mask=[]
    seg_lists=[]
    vocab_set = set()
    for i in tqdm(range(start, end)):
        doc_words = shuffle_doc_words_list[i].split()
        doc_words = cut_off(doc_words,perserve_rate,augmentation_seed)
        doc_node_relation_dic = {}
        w_size = 0
        tmp_w = []
        x_triple=[]
        for w in doc_words:
            if w in node_relation_dic:
                tmp_w.append(w)
        tmp_w = clear_short_word(tmp_w)
        for w in doc_words:
            if w in tmp_w:
                doc_node_relation_dic[w] = node_relation_dic[w]
                w_size = w_size + 1

        delete_node = []
        tmp_w = list(set(tmp_w))
        if len(tmp_w) > 1:
            for j in range(len(tmp_w) - 1):
                for k in range(j + 1, len(tmp_w)):
                    if tmp_w[j] in tmp_w[k]:
                        delete_node.append(tmp_w[j])
                        break
                    elif tmp_w[k] in tmp_w[j]:
                        delete_node.append(tmp_w[k])
                        break
        delete_node = list(set(delete_node))
        for d in delete_node:
            tmp_w.remove(d)
        for w in tmp_w:
            doc_node_relation_dic[w] = node_relation_dic[w]
        for w in doc_words:
            if w in tmp_w:
                w_size = w_size + 1

        diff = MAX_TRUNC_LEN - len(doc_words)
        doc_words_add_triples = []
        add_relations = []
        add_triples=[]
        background_nodes=[]
        mask_background=[]
        sub=[]
        if diff > 0:
            # 得到每个词最多可取多少三元组
            if w_size > 0:
                triple_max = diff // 2 // w_size
                #triple_max=10
            else:
                triple_max = 0
            for w in doc_words:
                doc_words_add_triples.append(w)
                if w in doc_node_relation_dic:
                    sub.append(w)
                    num = triple_max
                    for r_e in doc_node_relation_dic[w]:
                        if num > 0:
                            doc_words_add_triples.append(r_e[1])
                            sub.append(r_e[1])
                            add_triples.append((w, r_e[0], r_e[1]))
                            add_relations.append((w,r_e[1]))
                            num = num - 1
                        else:
                            break
        else:
            if truncate:
                doc_words = doc_words[:MAX_TRUNC_LEN]
        x_triple.append(add_triples)

        #mask list
        for a in add_relations:
            if a[1] not in doc_words:
                background_nodes.append(a[1])

        doc_word_id_map = {}
        if len(doc_words_add_triples) > len(doc_words):
            doc_vocab = list(set(doc_words_add_triples))
        else:
            doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)
        doc_len_list.append(doc_nodes)
        doc_len_raw.append(len(list(set(doc_words))))
        subkg.append(len(list(set(sub))))
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        doc_len = len(doc_words)
        vocab_set.update(doc_vocab)

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        for t in add_relations:
            window = list(t)
            windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))

        seg_list = []
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            seg_list.append(k)
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])
            mask_background.append(0 if k in background_nodes else 1)

        seg_lists.append(seg_list)
        x_adj.append(adj)
        x_feature.append(features)
        b_mask.append(mask_background)
        x_triples.append(x_triple)
    data_id = []

    # one-hot labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        data_id.append(temp[0])
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    data_id = np.array(data_id)
    x_triples = np.array(x_triples)

    return x_adj, x_feature, y, doc_len_list, vocab_set, data_id, x_triples,b_mask,seg_lists,doc_len_raw,subkg


print('building graphs for training')
x_adj, x_feature, y,doc_len_list_train, vocab_train, data_id, triple,b_mask,seg_lists_train ,doc_len_raw_train,subkg_train= build_joint_graph(start=0, end=real_train_size)
#augment
x_adj_tg, x_feature_tg, y_tg, doc_len_list_tg, vocab_set_tg, data_id_tg, x_triple_tg,b_mask_tg,seg_lists_tg,_,_= build_joint_graph_augment(start=0, end=real_train_size)
x_adj=x_adj+x_adj_tg
x_feature=x_feature+x_feature_tg
b_mask=b_mask+b_mask_tg

print('building graphs for validation')
allx_adj, allx_feature, ally, doc_len_list_dev, vocab_dev, all_data_id, all_triple,v_b_mask,seg_lists_dev,doc_len_raw_dev,subkg_dev = build_joint_graph(start=real_train_size,
                                                                                            end=real_train_size+val_size)
#augmented
x_adj_tg, x_feature_tg, y_tg, doc_len_list_tg, vocab_set_tg, data_id_tg, x_triple_tg,b_mask_tg,seg_lists_tg,_,_= build_joint_graph_augment(start=real_train_size, end=real_train_size+val_size)
allx_adj=allx_adj+x_adj_tg
allx_feature=allx_feature+x_feature_tg
v_b_mask=v_b_mask+b_mask_tg

print('building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test, t_data_id, t_triple,t_b_mask,seg_lists_test,doc_len_raw_test,subkg_test = build_joint_graph(start=real_train_size+val_size,
                                                                                         end=real_train_size+val_size + test_size)
#augmented
x_adj_tg, x_feature_tg, y_tg, doc_len_list_tg, vocab_set_tg, data_id_tg, x_triple_tg,b_mask_tg,seg_lists_tg,_,_= build_joint_graph_augment(start=real_train_size+val_size, end=real_train_size+val_size + test_size)
tx_adj=tx_adj+x_adj_tg
tx_feature=tx_feature+x_feature_tg
t_b_mask=t_b_mask+b_mask_tg

# statistics
doc_len_list = doc_len_list_train + doc_len_list_test+doc_len_list_dev
doc_len_raw=doc_len_raw_train+doc_len_raw_dev+doc_len_raw_test
subkg=subkg_train+subkg_dev+subkg_test
print('max_doc_length', max(doc_len_list), 'min_doc_length', min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
print('max_doc_raw_length', max(doc_len_raw), 'min_doc_length', min(doc_len_raw),
      'average {:.2f}'.format(np.mean(doc_len_raw)))
print('max_subkg', max(subkg), 'min_subkg', min(subkg),
      'average {:.2f}'.format(np.mean(subkg)))
print('training_vocab', len(vocab_train), 'test_vocab', len(vocab_test),
      'intersection', len((vocab_train) & vocab_test))

path = os.path.abspath('..') + '/dataset/untils_dataset/output/' + dataset+'/'+embedding+'_knowledge_'
path_att=os.path.abspath('..') + '/dataset/untils_dataset/output/' + dataset
# dump objects
with open(path_att+'/nodes_train_kg.csv','w',encoding='utf-8',newline='')as f:
    f_csv = csv.writer(f)
    for s in seg_lists_train:
        f_csv.writerow(s)
with open(path_att+'/nodes_dev_kg.csv','w',encoding='utf-8',newline='')as f:
    f_csv = csv.writer(f)
    for s in seg_lists_dev:
        f_csv.writerow(s)
with open(path_att+'/nodes_test_kg.csv','w',encoding='utf-8',newline='')as f:
    f_csv = csv.writer(f)
    for s in seg_lists_test:
        f_csv.writerow(s)

with open(path + 'ind.{}.x_adj'.format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open(path + 'ind.{}.x_embed'.format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open(path + 'ind.{}.y'.format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open(path + 'ind.{}.b_mask'.format(dataset), 'wb') as f:
    pkl.dump(b_mask, f)

with open(path + 'ind.{}.tx_adj'.format(dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open(path + 'ind.{}.tx_embed'.format(dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open(path + 'ind.{}.ty'.format(dataset), 'wb') as f:
    pkl.dump(ty, f)


with open(path + 'ind.{}.t_b_mask'.format(dataset), 'wb') as f:
    pkl.dump(t_b_mask, f)

with open(path + 'ind.{}.allx_adj'.format(dataset), 'wb') as f:
    pkl.dump(allx_adj, f)

with open(path + 'ind.{}.allx_embed'.format(dataset), 'wb') as f:
    pkl.dump(allx_feature, f)

with open(path + 'ind.{}.ally'.format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open(path + 'ind.{}.v_b_mask'.format(dataset), 'wb') as f:
    pkl.dump(v_b_mask, f)
