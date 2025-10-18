import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len)->None:
        super().__init__()
        self.ds= ds
        self.tokenizer_src= tokenizer_src
        self.tokenizer_tgt= tokenizer_tgt
        self.src_lang= src_lang
        self.tgt_lang=  tgt_lang

        self.sos_token= torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])],dtype=torch.int64)
        self.eos_token= torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])],dtype=torch.int64)
        self.pad_token= torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index: Any)-> Any:

        src_target_pair= self.ds[index]
        src_text= src_target_pair['translation'][self.src_lang]
        tgt_text= src_target_pair['translation'][self.tgt_lang]
        
        enc_input_token= self.tokenizer_src.encode(src_text).ids
        dec_input_token= self.tokenizer_src.encode(tgt_text).ids

        enc_num_padding_token= self.seq_len - len(enc_input_token)-2
        dec_num_padding_token= self.seq_len- len(dec_input_token)-1

        if enc_num_padding_token<0 or dec_num_padding_token<0:
            raise ValueError('sentence is too long')
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_token,dtype= torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_token,dtype= torch.int64),
            ]
        )
        decoder_input= torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token,dtype=torch.int64),
                torch.tensor([self.pad_token]* dec_num_padding_token,dtype=torch.int64)

            ]
        )

        label= torch.cat(
            [
                torch.tensor(dec_input_token,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* dec_num_padding_token,dtype=torch.int64)

            ]
        )

        assert encoder_input.size(0)== self.seq_len
        assert decoder_input.size(0)== self.seq_len
        assert label.size(0)== self.seq_len

        
        


  






