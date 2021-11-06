# GiBERT 

This repository provides code for the paper "GiBERT: Enhancing BERT with Linguistic Information using a Lightweight Gated Injection Method" published at EMNLP Findings 2021.

![Alt text](GiBERT.png?raw=true "GiBERT model")

## Setup

### Download pretrained BERT

- Create cache folder in home directory:
```
cd ~
mkdir tf-hub-cache
cd tf-hub-cache
```
- Download pretrained BERT model and unzip:
```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### Download data

- Go to the GiBERT repository:
```
cd /path/to/GiBERT/
```
- Download datasets from dropbox:
```
wget "https://www.dropbox.com/s/6icqwmaif746seu/data.tar.gz"
```
- Uncompress data.tar.gz:
```
tar zxvf data.tar.gz &
```
- Your GiBERT directory should now have the following structure:
```
.
├── MSRP
│   └── MSRParaphraseCorpus
├── Quora
│   └── Quora_question_pair_partition
├── Semeval2017
│   └── Semeval2017
├── cache
├── embeddings
├── logs
└── models
```
- Download embeddings:
	- Download counter-fitted embeddings:
	```
	cd data/embeddings
	wget https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip
	unzip counter-fitted-vectors.txt.zip
	```
	- Download dependency-based embeddings from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ and place in `data/embeddings`


### Install requirements

- This code has been tested with Python 3.6 and Tensorflow 1.11.
- Install the required Python packages as defined in requirements.txt:
```
pip install -r requirements.txt
```

## Usage

- You can try out if everything works by training a model on a small portion of the data (you can play around with different model options by changing the opt dictionary). Please make sure you are in the top GiBERT directory when executing the following commands (`ls` should show `GiBERT.png   data  data.tar.gz  README.md  requirements.txt  src` as output):
```
python src/models/base_model_bert.py
```
- The model will be saved under data/models/model_0/ and the training log is available under data/logs/test.json
- You can also run an experiment on the complete dataset and alter different commandline flags, e.g.:
```
python src/experiments/gibert.py -epochs 2 -datasets 'MSRP' -learning_rate 5e-05 -location 5 -seed 1 -embd_type counter_fitted; python src/experiments/gibert.py -datasets 'MSRP' -learning_rate 5e-05 -location 5 -seed 3 -embd_type counter_fitted -epochs 2```
```