import torch
import torch.nn as nn
import math

# --------------------------- Embedding Layer -----------------------------

class Inputembedding(nn.Module):
    # Constructor for input embedding
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        
        # d_model → dimension of word vector
        # vocab_size → total number of words in vocabulary
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding lookup table: maps each word index to a dense vector
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # Multiply embedding by sqrt(d_model) for better scaling (from Transformer paper)
        return self.embedding(x) * math.sqrt(self.d_model)


# --------------------------- Positional Encoding -----------------------------

class positionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  # Dropout regularization

        # Initialize positional encoding matrix (seq_len × d_model)
        pe = torch.zeros(seq_len, d_model)

        # Positions: [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the denominator term (10000^(2i/d_model)) for sin/cos waves
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine on even indices and cosine on odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer so it's not trained but still part of model state
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings (no gradient for PE)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)  # Apply dropout


# --------------------------- Layer Normalization -----------------------------

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # Small constant for numerical stability

        # Learnable scale and bias parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Compute mean and standard deviation along feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Normalize and apply learned scale/bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# --------------------------- Feed Forward Network -----------------------------

class FNN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # First linear layer expands dimensions (d_model → d_ff)
        self.linear_1 = nn.Linear(d_model, d_ff)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Second linear layer reduces back to d_model
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Sequence: Linear → ReLU → Dropout → Linear
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# --------------------------- Multi-Head Attention -----------------------------

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()

        # Store model parameters
        self.d_model = d_model
        self.h = h  # Number of heads

        # Ensure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        # Per-head dimension
        self.d_k = d_model // h

        # Define linear transformations for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout layer for attention weights
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Scaled Dot-Product Attention

        # Compute attention scores: (QK^T / √d_k)
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided) → hide future tokens or padding
        if mask is not None:
            attention_score.masked_fill(mask == 0, -1e9)

        # Softmax normalization across sequence length
        attention_score = attention_score.softmax(dim=-1)

        # Apply dropout (if provided)
        if dropout is not None:
            attention_score = dropout(attention_score)

        # Return weighted sum (attention applied to values)
        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        # Linear projections for Q, K, V
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape for multiple heads: (batch, seq_len, d_model) → (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        # Combine heads: (batch, h, seq_len, d_k) → (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear projection
        return self.w_o(x)


# --------------------------- Residual Connection -----------------------------

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Apply: x + Dropout(Sublayer(LayerNorm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


# --------------------------- Encoder Block -----------------------------

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, feed_forward_block: FNN, dropout: float) -> None:
        super().__init__()

        # Self-Attention and Feed-Forward components
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # Two residual connections: one for attention, one for FNN
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # First sublayer: Self-Attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # Second sublayer: Feed-Forward
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


# --------------------------- Encoder -----------------------------

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Stack of N Encoder Blocks
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # Pass input through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # Final normalization


# --------------------------- Decoder Block -----------------------------

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttention, cross_attention_block: MultiheadAttention, feed_forward_block: FNN, dropout: float) -> None:
        super().__init__()

        # Sublayers: self-attention, cross-attention, and feed-forward
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # Three residual connections for the three sublayers
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        # Masked self-attention (prevents seeing future tokens)
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))

        # Cross-attention (decoder attends to encoder outputs)
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # Feed-forward network
        x = self.residual_connection[2](x, self.feed_forward_block)

        return x


# --------------------------- Decoder -----------------------------

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Stack of N Decoder Blocks
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, target_mask):
        # Sequentially apply all decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)


# --------------------------- Projection Layer -----------------------------

class projectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        # Linear projection from model dimension → vocab size
        self.proj = nn.Linear(d_model, vocab_size)
   
    def forward(self, x):
        # Convert decoder output to log-probabilities across vocabulary
        return torch.log_softmax(self.proj(x), dim=-1)


# --------------------------- Transformer Model -----------------------------

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: Inputembedding, target_embedding: Inputembedding,
                 src_pos: positionalEncoding, target_pos: positionalEncoding, projection_layer: projectionLayer) -> None:
        super().__init__()

        # Components of the Transformer
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # Apply embedding + positional encoding → pass through encoder
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target, target_mask):
        # Apply embedding + positional encoding → pass through decoder
        target = self.target_embedding(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        # Convert final decoder output to word probabilities
        return self.projection_layer(x)


# --------------------------- Transformer Builder -----------------------------

def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    # 1️ Create embedding layers for source and target
    src_embedding = Inputembedding(d_model, src_vocab_size)
    target_embedding = Inputembedding(d_model, target_vocab_size)

    # 2️ Create positional encoding layers
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    target_pos = positionalEncoding(d_model, target_seq_len, dropout)

    # 3️ Build encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FNN(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # 4️ Build decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FNN(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # 5️ Create encoder and decoder stacks
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # 6️ Create projection layer (to vocab probabilities)
    projection_layer = projectionLayer(d_model, target_vocab_size)

    # 7️ Build final Transformer model
    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_pos, target_pos, projection_layer)

    # 8️ Initialize all parameters (Xavier uniform)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 9️ Return fully built Transformer model
    return transformer
