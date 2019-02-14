import torch
import torch.nn as nn


class Classify(torch.nn.Module):
    def __init__(self, params, vocab_size, ntags):
        super(Classify, self).__init__()
        
        self.weight = torch.load("weight.txt")
        self.word_embeddings = nn.Embedding.from_pretrained(self.weight)
        
        self.word_embeddings.weight.requires_grad = False
        
        
        # conv 1d
        self.conv_1d_1 = torch.nn.Conv1d(in_channels=params.emb_dim, 
                                       out_channels=params.filters, 
                                       kernel_size=params.window_size1,
                                       stride=1, 
                                       padding=0, 
                                       dilation=1, 
                                       groups=1, 
                                       bias=True)
        self.conv_1d_2 = torch.nn.Conv1d(in_channels=params.emb_dim, 
                                       out_channels=params.filters, 
                                       kernel_size=params.window_size2,
                                       stride=1, 
                                       padding=1, 
                                       dilation=1, 
                                       groups=1, 
                                       bias=True)
        self.conv_1d_3 = torch.nn.Conv1d(in_channels=params.emb_dim, 
                                       out_channels=params.filters, 
                                       kernel_size=params.window_size3,
                                       stride=1, 
                                       padding=2, 
                                       dilation=1, 
                                       groups=1, 
                                       bias=True)


        self.dropout = nn.Dropout(params.dropout)
        self.relu = nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=params.filters, 
                                                out_features=ntags, 
                                                bias=True)
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)


    def forward(self, input_sents, input_lens):
        embeds = self.word_embeddings(input_sents)     # bs * max_seq_len * emb
        embeds = embeds.permute(0,2,1)                 # bs * emb * max_seq_len
        
                          
        h1 = self.conv_1d_1(embeds)                    
                         
        h2 = self.conv_1d_2(embeds)
       
        h3 = self.conv_1d_3(embeds)
       
        
        h = torch.cat((h1,h2,h3),dim=2)
       
        
        h = h.max(dim=2)[0]
       
        h = self.dropout(self.relu(h))                 # Relu activation and dropout
        
        h = self.projection_layer(h)                   # bs * ntags
        
        return h




