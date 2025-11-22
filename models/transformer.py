import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import MLP

class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Heads.
    
    Args:
        input_dim: The dimension of input tokens.
        input_size: The (maximal) number of input tokens.
        num_heads: The number of heads.
        out_dim: The dimension of output tokens.
        dropout: The fraction of weights to zero via dropout.
        decoder: True for one-directional attention.
        
    """
    def __init__(
        self, input_dim, input_size, num_heads, out_dim, dropout=0, decoder=False
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "inner dim. must be multiple of num. heads"

        self.input_dim = input_dim
        self.input_size = input_size
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout
        self.decoder = decoder

        self.key=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        self.query=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        self.value=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        self.projection=nn.Parameter(
            torch.randn( self.out_dim, self.out_dim)
        )
        if decoder:
            self.register_buffer('tril', torch.tril(torch.ones(self.input_size, self.input_size)))
        else:
            self.register_buffer('tril', torch.ones(self.input_size, self.input_size))
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).
        
        Returns:
            The output of a multi-head attention layer,
            of size (batch_size, input_size, output_dim)
        """
        B,T,C = x.size()
        k = F.linear( x, self.key, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) * C**-.5    # [bs, num_heads, seq_len, head_dim]
        q = F.linear( x, self.query, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) * C**-.5  # [bs, num_heads, seq_len, head_dim]
        v = F.linear( x, self.value, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) * C**-.5  # [bs, num_heads, seq_len, head_dim]

        weight = q @ k.transpose(-2,-1) * self.head_dim**-.5             # [bs, num_heads, seq_len, seq_len]
        weight = weight.masked_fill(self.tril[:T,:T]==0, float('-inf'))  #  //
        weight = F.softmax(weight, dim=-1)                               #  //
        weight = self.dropout(weight)

        out = (weight @ v).transpose(1,2).reshape(B,T,-1) # [bs, seq_len, out_dim]
        out = F.linear( out, self.projection, bias=None) * self.projection.size(-1)**-.5

        return out


class AttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim, input_size, num_heads, dropout=0, decoder=False
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            input_size=input_size,
            num_heads=num_heads,
            out_dim=embedding_dim,
            dropout=dropout,
            decoder=decoder
        )

    def forward(self, x):
        x = self.sa(x)
        return x


class DecoderBlock(nn.Module):
    """
    One Decoder Block.
    
    Args:
        embedding_dim: The dimension of the tokens (kept constant past embedding).
        input_size: The (maximal) number of input tokens.
        num_heads: The number of attention heads.
        dropout: The fraction of weights to zero via dropout.
        ffwd_size: Size of the MLP is ffwd_size*embedding_dim.        
    """
    def __init__(
        self, embedding_dim, input_size, num_heads, ffwd_size=4, dropout=0, decoder=False
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.attn = MultiHeadAttention(
            input_dim=embedding_dim,
            input_size=input_size,
            num_heads=num_heads,
            out_dim=embedding_dim,
            dropout=dropout,
            decoder=decoder
        )
        self.ffwd = MLP(
            input_dim=embedding_dim, 
            nn_dim=ffwd_size*embedding_dim, 
            out_dim=embedding_dim, 
            num_layers=1
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x


class MLA(nn.Module):
    """
    Multi-Layer Multi-Head Attention for last token prediction

    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, num_layers
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding=nn.Parameter(
            torch.randn( self.embedding_dim, self.vocab_size)
        )
        self.position_embedding = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                ) for _ in range(self.num_layers)
            ]
        )
        self.readout = nn.Parameter(
            torch.randn(self.vocab_size, self.embedding_dim)
        )


    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, seq_len, vocab_size).
        
        Returns:
            Output of multilayer self-attention, tensor of size (batch_size, seq_len, vocab_size)
        """
        B,T,C = x.size()
        token_emb = F.linear( x, self.token_embedding, bias=None) *C**-.5   # [bs, seq_len, embedding_dim]
        pos_emb = self.position_embedding(torch.arange(T, device=x.device)) # [seq_len, embedding_dim]
        x = token_emb + pos_emb # [bs, seq_len, embedding_dim]
        x = self.blocks(x)
        logits = F.linear( x[:,-1,:], self.readout, bias=None) * self.readout.size(-1)**-.5

        return logits


class CLM(nn.Module):
    """
    Causal (decoder-only) Language Model.
    
    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        dropout: The fraction of weights to zero via dropout.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, ffwd_size, num_layers, dropout=0, share_emb=True
    ):
        super().__init__()

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffwd_size = ffwd_size
        self.num_layers = num_layers

        self.token_embedding_table = nn.Embedding(vocab_size, self.embedding_dim)
        # TODO: different kind of positional encoding?
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                    ffwd_size=self.ffwd_size,
                    decoder=True,
                    dropout=dropout
                ) for _ in range(self.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size)
        if share_emb:
            self.lm_head.weight = self.token_embedding_table.weight


    def forward(self, idx, targets=None):

        B,T = idx.size()

        token_emb = self.token_embedding_table(idx) # [bs, seq_len, embedding_dim]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # [seq_len, embedding_dim]
        x = token_emb + pos_emb  # [bs, seq_len, embedding_dim]

        x = self.blocks(x)       # [bs, seq_len, embedding_dim]
        x = self.ln_f(x)         # [bs, seq_len, embedding_dim]
        logits = self.lm_head(x) # [bs, seq_len, input_dim]

        return logits


    def generate(self, idx, num_tokens):

        for _ in range(num_tokens):
            # get the prediction
            idx_cond = idx[:,-self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # (bs, input_dim)
            probs = F.softmax(logits, dim=-1) # (bs, input_dim)
            #TODO: possibly restrict probs to the top k values, might be useful for large vocab sizes
            idx_next = torch.multinomial(probs, num_samples=1) # (bs, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (bs, seq_len+1)

        return idx
