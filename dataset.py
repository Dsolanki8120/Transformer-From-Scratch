import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Custom PyTorch dataset for bilingual (source-target) sentence pairs
class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        # Store the dataset (Hugging Face style translation dataset)
        self.ds = ds
        # Tokenizers for source and target languages
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # Language keys (for example, 'en', 'fr')
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # ⚠️ Note: seq_len should also be stored (currently missing in your original code)
        # self.seq_len = seq_len

        # Convert special tokens into tensors of their respective token IDs
        # These tokens will be added manually to each sequence
        # [SOS] = Start of Sentence, [EOS] = End of Sentence, [PAD] = Padding
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    # Returns total number of samples in the dataset
    def __len__(self):
        return len(self.ds)
    

    # Retrieves one sample (source-target sentence pair)
    def __getitem__(self, index: Any) -> Any:
        # Extract source and target text from dataset
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        # Tokenize source and target sentences into lists of token IDs
        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_src.encode(tgt_text).ids  # ⚠️ Ideally tokenizer_tgt should be used here

        # Compute how many [PAD] tokens are needed to reach the fixed sequence length
        # For encoder: add [SOS] and [EOS] (so -2)
        # For decoder: add [SOS] but [EOS] goes into label (so -1)
        enc_num_padding_token = self.seq_len - len(enc_input_token) - 2
        dec_num_padding_token = self.seq_len - len(dec_input_token) - 1

        # If padding would be negative, the sentence is too long to fit into seq_len
        if enc_num_padding_token < 0 or dec_num_padding_token < 0:
            raise ValueError('sentence is too long')
        
        # ----------------------------------------------------
        # ENCODER INPUT CONSTRUCTION
        # ----------------------------------------------------
        # [SOS] + source sentence tokens + [EOS] + [PAD]*remaining
        encoder_input = torch.cat(
            [
                self.sos_token,  # Start token
                torch.tensor(enc_input_token, dtype=torch.int64),  # Actual source tokens
                self.eos_token,  # End token
                torch.tensor([self.pad_token] * enc_num_padding_token, dtype=torch.int64),  # Padding tokens
            ]
        )

        # ----------------------------------------------------
        # DECODER INPUT CONSTRUCTION
        # ----------------------------------------------------
        # [SOS] + target sentence tokens + [PAD]*remaining
        decoder_input = torch.cat(
            [
                self.sos_token,  # Start token
                torch.tensor(dec_input_token, dtype=torch.int64),  # Target tokens
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64),  # Padding
            ]
        )
        
        # ----------------------------------------------------
        # LABEL CONSTRUCTION (Expected output for decoder)
        # ----------------------------------------------------
        # target tokens + [EOS] + [PAD]*remaining
        # This is shifted by one compared to decoder_input (for teacher forcing)
        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype=torch.int64),  # Target tokens
                self.eos_token,  # End token
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64),  # Padding
            ]
        )

        # ----------------------------------------------------
        # Sanity checks to ensure all sequences have the same length
        # ----------------------------------------------------
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # ----------------------------------------------------
        # Return a dictionary containing:
        # - Encoder input sequence
        # - Decoder input sequence
        # - Attention masks for encoder & decoder
        # - Label sequence
        # - Original source and target text (for debugging or evaluation)
        # ----------------------------------------------------
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            
            # Encoder mask → 1 for non-PAD tokens, 0 for PAD tokens
            # Shape: (1, 1, seq_len) so it can be broadcast correctly in attention
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,seq_len)
            
            # Decoder mask = combination of:
            #   (1) Padding mask (1 where token is not PAD)
            #   (2) Causal mask (prevents attending to future tokens)
            # The "&" bitwise AND ensures both conditions must be true to allow attention.
            "decode_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1,seq_len)&(1,seq_len,seq_len)
            
            "label": label,  # Expected target tokens for loss computation
            "src_text": src_text,  # Original source sentence
            "tgt_text": tgt_text,  # Original target sentence
        }


# ------------------------------------------------------------
# Function to create a causal (look-ahead) mask for decoder self-attention
# ------------------------------------------------------------
def causal_mask(size):
    # Create an upper-triangular matrix filled with 1s
    # torch.triu(..., diagonal=1) → upper triangle excluding main diagonal
    # Example (size=4):
    # [[0,1,1,1],
    #  [0,0,1,1],
    #  [0,0,0,1],
    #  [0,0,0,0]]
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)

    # Return a boolean mask where 1s above diagonal become False (not visible)
    # So True (1) only for positions allowed to attend (past and current tokens)
    return mask == 0
