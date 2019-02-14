# CNN for Sentence Classification

## Introduction

This is the implementation of CNN for sentence classification on Pytorch. The accuracy for validation dataset reaches **83.3%**. Development environment is shown as follows.

|  Name   | Version |
| :-----: | :-----: |
| Python  |   3.5   |
| Pytorch |  1.0.1  |
| Gensim  |  3.4.0  |



## Specification:

- pretrain_embeddings.py: Transform pre-trained Glove to Word2vec, and create embeddings for this dataset.
- main.py: Define all parameters, and define main routine for calling all the functions.
- data_loader.py: Load train, validation and test dataset, create padded sentences, create mapping between index and words.
- model.py: Build a classification model use CNN.
- util.py: Define the training process and testing process, the prediction is printed as a text file. 
- params.pkl: The trained parameters.

**Usage:** 

```python
python main.py
```

**Output:**

The prediction for validation dataset and test dataset is stored in "/result/xx_pred.txt"



## Pre-trained Word vectors

Initializing word vectors with a publicly available Glove vectors that were trained on common crawl, it contains 400K vocabs. The vectors have 300 dimensions, words not present in the set of pre-trained words are initialized to all zeros. 

Before implementation the training, Glove embeddings are transformed to Word2vec.



## Result

The key parameters are set as follows:

|     parameters      |  value  |
| :-----------------: | :-----: |
|     window size     | 2, 3, 4 |
|     batch size      |   32    |
| embedding dimension |   300   |
|       dropout       |   0.3   |
|       filters       |   100   |

**Accuracy** for validation dataset is:  83.3%



## Cite:

- [Convolutional Neural Networks for Sentence Classification (Y.Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181) 
- [Dataset][http://www.phontron.com/class/nn4nlp2019/assignments.html]

- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). 

 

