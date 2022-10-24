# ConKGNN

# Requirements

* Python 3.7
* Tensorflow/Tensorflow-gpu 2.4.1
* Scipy 1.5.1

# Downloading the united graph dataset of Drugs：

Click the link below to download the graph dataset of Drugs. This dataset include preprocessed datas. Please put them into data/

[Drugs(https://pan.baidu.com/s/13vk2tNhcJamBdT-TFUYuVA))

password: Drug

# Train and Test:

python train.py

# Use your own dataset and KG to build united graph：

## Step 1: Train node2vec embedding with your KG

Our framework need to use entities and relations bewteen them in your KG.  You can use any open source code to train your nodes KG embeddings.

## Step 2: Prepare KG nodes file and triple file

Collect all KG nodes and saved them to a node.txt. Transfer the KG to a triples.csv, the format should be [head,relation,tail].

## Step 3: Prepare the dataset.

Prepare your dataset to two files, label.txt and clean_knowledge.txt. The format of label.txt is such like "id	set	 class". For Chinese dataset, every line in clean_knowledge.txt is a text and need to be word segmentation.

## Step 4: Run build_graph.py.

You can obtain the embedding united graph dataset files as we provided!


