# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:05:32 2019

@author: dell
"""
import gensim
from gensim.test.utils import datapath, get_tmpfile
import torch
import torch.utils.data
from collections import defaultdict
from gensim.scripts.glove2word2vec import glove2word2vec

def read_dataset(filename, w2i, t2i):
        with open(filename, "r",encoding="latin-1") as f:
            for line in f:
                tag, words = line.lower().strip().split(" ||| ")
                yield ([w2i[x] for x in words.split(" ")], t2i[tag])


# Existed Glove
glove_file = datapath('E:/pytorch_tutorial/lstm_tutorial/topicclass/glove_300d.txt')
# Location for changed word2vec
tmp_file = get_tmpfile("E:/pytorch_tutorial/lstm_tutorial/topicclass/word2vec_300d.txt")
glove2word2vec(glove_file, tmp_file)


# Dictionaries for word to index and vice versa
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
# Adding unk token
UNK = w2i["<unk>"]

# Read in the data and store the dicts
train = list(read_dataset("topicclass/topicclass_train.txt", w2i, t2i))
        
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt", w2i, t2i))
        
test = list(read_dataset("topicclass/topicclass_test.txt", w2i, t2i))

nwords = len(w2i)
ntags = len(t2i)
   
embed_size=300     
        
i2t = {value:key for key, value in t2i.items()}
i2w = {value:key for key, value in w2i.items()}
      
wvmodel = gensim.models.KeyedVectors.load_word2vec_format('E:/pytorch_tutorial/lstm_tutorial/topicclass/word2vec_300d.txt', binary=False, encoding='utf-8')

weight = torch.zeros(nwords, embed_size)

for i in range(len(wvmodel.index2word)):
    
    try:
        index = w2i[wvmodel.index2word[i]]
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            i2w[w2i[wvmodel.index2word[i]]]))
    except:
        continue
    
torch.save(weight, "weight.txt")
