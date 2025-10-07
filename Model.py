import torch
import torch.nn as nn
import math

# Defining Embedding class

class Inputembedding(nn.Module):
    # defining constructor

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        
        self.d_model= d_model # dimension of the word vector
        self.vocab_size= vocab_size # total number of word in vocubalory
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self,x):
        return self.embedding(x) *  math.sqrt(self.d_model)

 # positional encoding   
class positionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()

        self.d_model= d_model
        self.seq_len= seq_len
        self.dropout= nn.Dropout(dropout)
        pe= torch.zeros(seq_len,d_model) # create a matrix of shape. (seq_len,d_model)
        position= torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) # position shape will be (seq_len,1) it will show position of each word

        div_term= torch.exp(torch.arange(0,d_model,2).float()* (-math.log(10000)/d_model))
        # apply the sin to the even position word
        pe[:,0::2]= torch.sin(position*div_term)
        pe[:,1::2]= torch.cos(position*div_term)
        # Batch dimension = number of sentences (or sequences) processed in parallel.
        # Positional encoding is shared across all sentences in the batch.
        pe= pe.unsqueeze(0) # so it will be dimensioin (1,seq_len,d_model)
        self.register_buffer('pe',pe) # add into register buffer 

    def forward(self,x):
        x= x + (self.pe[:,:x.shape[1],:]).requieres_grad(False)
        return self.dropout(x)


# layer Normalization.....

class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps= eps
        self.alpha= nn.parameter(torch.ones(1))
        self.bias= nn.parameter(torch.zeros(0))

    def forward(self,x):
        mean= x.mean(dim= -1,keepdim=True)
        std= x.std(dim= -1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias


# Feed Forword Neural Network class

class FNN(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear_1= nn.Linear(d_model,d_ff)
        self.dropout= nn.Dropout(dropout)
        self.linear_2= nn.Linear(d_ff,d_model)

    def forward(self,x):
        # (Batch,seq-len,d_model)----> (Batch,seq-len,d_ff)----> (Batch,seq-len,d_model):

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

        

