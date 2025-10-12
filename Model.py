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

        

# Multihead attention

       # d_model: total embedding dimension (eg.512)
       # h: number of attenttion heads (eg.8)
       # dropout: droput rate for regularization


class MultiheadAttention(nn.Module):

    # input: (Batch,seq_len, d_model)
    # output: (Batch,seq_len,d_model)
    def __init__(self,d_model:int,h:int,dropout:float)-> None:
        super().__init__()
        self.d_model= d_model # dimension of the embedding vector
        self.h= h # number of head
        assert d_model%h==0," d_model is not divisible by h"
        self.d_k= d_model //h # compute per head dimension , Each attention head will work on vector of size d_k
        # Each head doesn't learn seprate matrices instead we project the input embedding into query , key,value spaces using fully connected layer
        self.w_q= nn.Linear(d_model,d_model) #wq
        self.w_k= nn.Linear(d_model,d_model) #wk
        self.w_v= nn.Linear(d_model,d_model) #wv
        self.w_o= nn.Linear(d_model,d_model) #W_o
        self.dropout= nn.Dropout(dropout) # Used later on attention weight to prevent overfitting
    
    # This performs scaled-dot product Attention
    def attention(query,key,value,mask,dropout:nn.Dropout):
        # query: (batch,h,seq_len,d_k)
        # key: (batch,h,seq_len,d_k)
        # attention_scores= (batch,h,seq_len,seq_len)


        d_k= query.shape[-1]
        # query @ key.T computes pairwise similarity between tokens.
        # Divide by √dₖ to normalize the magnitude,
        # query: (batch,h,seq_len,d_k) @ key: (batch,h,d_k,seq_len) ( -2,-1 ).transpose we just swap last two dimension to do matrix multiplication
        attention_score= (query @ key.transpose(-2,-1))/ math.sqrt(d_k)


        if mask is not None:
            attention_score.masked_fill(mask==0,-1e9)
        
        # apply softmax to get attention weight
        # (Batch,h,seq_len,seq.len)
        attention_score= attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score= dropout(attention_score)

        return (attention_score @ value),attention_score
    
        
    def forward(self,q,k,v,mask):

        query= self.w_q(q) # (Batch,seq_len,d_model,seq,d_model)
        key= self.w_k(k) # (Batch,seq_len,d_model,seq,d_model)
        v= self.w_v(v) # (Batch,seq_len,d_model,seq,d_model)

        # split query ,key ,value matrix

        # (Batch,seq_len,d_model) ----> (Batch,seq_len,h,d_k)-------> (Batch,h,seq_len,d_k)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key= key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value= value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        x,self.attention_scores= MultiheadAttention.attention(query,key,value,mask,self.dropout)

        #(Batch ,h,seq_len,d_k)----> ( Batch,seq_len,h,d_k)-------> (Batch,seq_len,d_model)

        x= x.transpose(1,2).contigous().view(x.shape[0],-1, self.h*self.d_K)

        # (Batch,seq_len,d_model)---> (Batch,seq_len,d_mdoel)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self,dropout:float)-> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) 
        self.norm= LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))







