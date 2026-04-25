import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, positions, dim):
    """
    Apply Rotary Position Embedding (RoPE) to query/key tensors
    x: [B, h, N, d] - query/key tensors
    positions: [N] - position indices
    dim: dimension size
    """
    B, h, N, d = x.shape
    
    # Generate theta values for each dimension pair
    theta = 1.0 / (10000 ** (torch.arange(0, d, 2).float().to(x.device) / d))
    
    # Generate position-dependent angles
    m = positions.unsqueeze(-1).float() * theta  # [N, d//2]
    
    # Create cos and sin components
    cos_m = torch.cos(m)  # [N, d//2]
    sin_m = torch.sin(m)  # [N, d//2]
    
    # Apply rotation to pairs of dimensions
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x[..., 0::2] * cos_m.unsqueeze(0).unsqueeze(0) - x[..., 1::2] * sin_m.unsqueeze(0).unsqueeze(0)
    x_rot[..., 1::2] = x[..., 0::2] * sin_m.unsqueeze(0).unsqueeze(0) + x[..., 1::2] * cos_m.unsqueeze(0).unsqueeze(0)
    
    return x_rot


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        d_ff = int((8/3) * d_model)
        self.l1 = nn.Linear(d_model, d_ff, bias=False)
        self.l2 = nn.Linear(d_model, d_ff, bias=False)
        self.l3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, X):
        return self.l3(self.l1(X) * F.silu(self.l2(X)))


class TransformerBlock(nn.Module):
    def __init__(self, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.dim_ = 256
        self.num_heads = num_heads
        self.head_dim = self.dim_ // num_heads
        
        assert self.dim_ % num_heads == 0, "dim must be divisible by num_heads"
        
        self.Q = nn.Linear(self.dim_, self.dim_)
        self.K = nn.Linear(self.dim_, self.dim_)
        self.V = nn.Linear(self.dim_, self.dim_)
        
        self.lin = nn.Linear(self.dim_, self.dim_)
        self.ln1 = nn.LayerNorm(self.dim_)

        self.ffn = SwiGLUFFN(self.dim_)
        self.ln2 = nn.LayerNorm(self.dim_)

    def forward(self, X):
        B, N, D = X.shape

        q:torch.Tensor = self.Q(X)
        v:torch.Tensor = self.V(X)
        k:torch.Tensor = self.K(X)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)            #[B, h, N, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to query and key
        positions = torch.arange(N, device=X.device)
        q = apply_rope(q, positions, self.head_dim)
        k = apply_rope(k, positions, self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-1, -2))/(self.head_dim**0.5)              #[B, h, N, N]
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        
        Y = self.lin(out)
        out = self.ln1(X + Y) # FIXED: Added X (original input) instead of out
        Y = self.ffn(out)

        return self.ln2(out + Y)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_dim, model_dim=256, num_heads=8, num_layers=6):
        super(TransformerEncoder, self).__init__()
        self.dim_ = model_dim
        self.num_layers_ = num_layers
        self.embedding = nn.Embedding(vocab_dim, model_dim)
        
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(num_heads) for _ in range(num_layers)])

    def forward(self, X):
        B, N = X.shape
        X = self.embedding(X)
        X = self.transformer_blocks(X) 
        
        return X
    

# --- NEW DECODER COMPONENTS ---

class DecoderBlock(nn.Module):
    def __init__(self, dim_=256, num_heads=8):
        super(DecoderBlock, self).__init__()
        self.dim_ = dim_
        self.num_heads = num_heads
        self.head_dim = self.dim_ // num_heads
        
        assert self.dim_ % num_heads == 0, "dim_ must be divisible by num_heads"

        # === 1. Masked Self-Attention Layers ===
        self.q_self = nn.Linear(self.dim_, self.dim_)
        self.k_self = nn.Linear(self.dim_, self.dim_)
        self.v_self = nn.Linear(self.dim_, self.dim_)
        self.lin_self = nn.Linear(self.dim_, self.dim_)
        self.ln1 = nn.LayerNorm(self.dim_)

        # === 2. Cross-Attention Layers ===
        self.q_cross = nn.Linear(self.dim_, self.dim_)
        self.k_cross = nn.Linear(self.dim_, self.dim_)
        self.v_cross = nn.Linear(self.dim_, self.dim_)
        self.lin_cross = nn.Linear(self.dim_, self.dim_)
        self.ln2 = nn.LayerNorm(self.dim_)

        # === 3. Feed Forward Network ===
        self.ffn = SwiGLUFFN(self.dim_)
        self.ln3 = nn.LayerNorm(self.dim_)

        
    def forward(self, Y, enc_out, mask):
        B, N_dec, D = Y.shape
        _, N_enc, _ = enc_out.shape

        # === 1. Masked Self-Attention ===
        q = self.q_self(Y).reshape(B, N_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_self(Y).reshape(B, N_dec, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_self(Y).reshape(B, N_dec, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Y's queries and keys
        positions = torch.arange(N_dec, device=Y.device)
        q = apply_rope(q, positions, self.head_dim)
        k = apply_rope(k, positions, self.head_dim)

        scores_self = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        if mask is not None:
            scores_self = scores_self.masked_fill(mask == 0, float('-inf'))
            
        attn_self = torch.softmax(scores_self, dim=-1)
        out_self = torch.matmul(attn_self, v).transpose(1, 2).contiguous().reshape(B, N_dec, D)
        
        # Residual 1
        attn_proj = self.lin_self(out_self)
        Y = self.ln1(Y + attn_proj)

        # === 2. Cross-Attention ===
        # Q comes from Decoder (Y), K and V come from Encoder (enc_out)
        q_c = self.q_cross(Y).reshape(B, N_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k_c = self.k_cross(enc_out).reshape(B, N_enc, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = self.v_cross(enc_out).reshape(B, N_enc, self.num_heads, self.head_dim).transpose(1, 2)

        scores_cross = torch.matmul(q_c, k_c.transpose(-1, -2)) / (self.head_dim**0.5)
        attn_cross = torch.softmax(scores_cross, dim=-1)
        out_cross = torch.matmul(attn_cross, v_c).transpose(1, 2).contiguous().reshape(B, N_dec, D)

        # Residual 2
        cross_proj = self.lin_cross(out_cross)
        Y = self.ln2(Y + cross_proj)

        # === 3. Feed Forward ===
        Y_ffn = self.ffn(Y)
        Y = self.ln3(Y + Y_ffn)

        return Y

class TransformerDecoder(nn.Module):
    # ... (__init__ stays the same) ...

    def forward(self, Y, enc_out):
        B, N = Y.shape
        Y = self.embedding(Y)

        mask = torch.tril(torch.ones((N, N), device=Y.device)).view(1, 1, N, N)

        for layer in self.decoder_blocks:
            Y = layer(Y, enc_out, mask)
            
        logits = self.lm_head(Y)

        return logits
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        return self.decoder(tgt, enc_out)

class FullDecoder(TransformerDecoder):
    def __init__(self, vocab_dim, model_dim=256, num_heads=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, model_dim)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(model_dim, num_heads) for _ in range(num_layers)])
        self.lm_head = nn.Linear(model_dim, vocab_dim)
