# ConKGNN

# Requirements

* Python 3.7
* Tensorflow/Tensorflow-gpu 2.4.1
* Scipy 1.5.1

# train and test：
python train.py

# build united graph：
##
Since the Authorization in DXY is very slow, and keeping people always waiting is embarrassing, I summarize the pre-training process as follow so you can use your own KG to play the model!

KG format.
Our framework only need to use entities( with their corresponding alias and types) and relations bewteen them in your KG. In fact, the alias are only used to linking the spans in the input sentence to the enities in KG, and so when a custom entit-linker is available for your KG, the alias are not necessary.

Step 1: Train TransR embedding with your data.
The entity, relation and transfer matrix weights are nessary to use our framework, as you see in there. I recommend to use DGL-KE to train the embedding since it is fast and scale to very large KG.

Step 2: Train the entities rank weights.
As we mention in the paper, for a entity in KG, it may has too many neighbours and we have to decide use which of them. We perform PageRank on the KG and the value for each entity(node) is used as weight as shown in there. You need to arrange it into the json foramt.

Step 3: Prepare the entity2neighbours dict for quick search.
As we often need to use the neighbours of a linked entity, we decide to build a dict beforehand to avoid unnecessary computing. We need two files, 'ent2outRel.pkl' and 'ent2inRel.pkl' respectively for out and in directions relations. The format should be ent_name -> [(rel_name,ent_name), ..., (rel_name,ent_name)].

Step 4: Prepare the entity2type dict.
As we propose the hyper-attention that makes use of entity types knowledge, we need a dict to provide our model with such information. The format should be ent_name -> type_val, the type_val could be type name or type id.

Step 5: Prepare the name2id dict.
As shown in there, name2id files are needed to provide the mapping bewteen entities and their corrponding resouces. The format is obvious as you see.

Step 6: Run Pre-training!!!
python -m torch.distributed.launch --nproc_per_node=4 run_pretraining_stream.py
Note that since the there are very large files need to be loaded into memory, the program may appear to freeze at first.

# dataset：
[Drugs(https://pan.baidu.com/s/13vk2tNhcJamBdT-TFUYuVA))

password: Drug

