import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project inputs to multi-head space
        q = self.q_proj(query)  # (B, seq_len_q, embed_dim)
        k = self.k_proj(key)    # (B, seq_len_k, embed_dim)
        v = self.v_proj(value)  # (B, seq_len_k, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len_q, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len_k, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len_k, head_dim)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (B, num_heads, seq_len_q, head_dim)
        
        # Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (B, seq_len_q, embed_dim)
        output = self.out_proj(context)
        return output

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention with residual connection
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual connection
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.dropout(ffn_output)
        tgt = self.norm3(tgt)
        return tgt

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=512, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        # Convert image patches to embeddings using a convolutional layer
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Positional embeddings for 196 patches (assuming 224x224 images with 16x16 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))
        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input shape: (B, 3, 224, 224)
        x = self.patch_embed(x)  # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        x = x + self.pos_embed  # Add positional embeddings
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x  # Output shape: (B, 196, embed_dim)

# Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, embed, max_seq_len=128, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.embed = embed  # Shared embedding layer
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed.embedding_dim))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed.embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed.embedding_dim)

    def forward(self, x):
        # Input shape: (B, seq_len)
        x = self.embed(x)  # (B, seq_len, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]  # Add positional embeddings
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x  # Output shape: (B, seq_len, embed_dim)

# Fusion Module
class FusionModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)

    def forward(self, image_seq, text_seq):
        # Image features as queries, text features as keys/values
        fused_seq = self.attention(image_seq, text_seq, text_seq)
        return fused_seq  # Output shape: (B, 196, embed_dim)

# Decoder
class Decoder(nn.Module):
    def __init__(self, embed, max_seq_len=128, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.embed = embed  # Shared embedding layer
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed.embedding_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed.embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed.embedding_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Input shape: (B, tgt_seq_len)
        x = self.embed(tgt)  # (B, tgt_seq_len, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]  # Add positional embeddings
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        x = self.norm(x)
        # Output projection using shared embedding weights
        logits = F.linear(x, self.embed.weight)  # (B, tgt_seq_len, vocab_size)
        return logits

# Multimodal Model
class MultimodalModel(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=512, max_seq_len=128, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        # Shared token embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        self.text_encoder = TextEncoder(embed=self.embed, max_seq_len=max_seq_len, num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        self.fusion = FusionModule(embed_dim=embed_dim, num_heads=num_heads)
        self.decoder = Decoder(embed=self.embed, max_seq_len=max_seq_len, num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, image=None, text=None, tgt=None, tgt_mask=None):
        if image is not None and text is not None:
            # Multimodal mode
            image_seq = self.image_encoder(image)  # (B, 196, embed_dim)
            text_seq = self.text_encoder(text)     # (B, seq_len, embed_dim)
            fused_seq = self.fusion(image_seq, text_seq)  # (B, 196, embed_dim)
            memory = fused_seq
        elif text is not None:
            # Text-only mode
            text_seq = self.text_encoder(text)  # (B, seq_len, embed_dim)
            memory = text_seq
        else:
            raise ValueError("At least one of image or text must be provided.")
        
        # Decode to generate output sequence
        logits = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # (B, tgt_seq_len, vocab_size)
        return logits

# Example usage (optional, for testing)
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 30000
    embed_dim = 512
    max_seq_len = 128
    num_layers = 4
    num_heads = 8
    ff_dim = 2048
    
    # Initialize model
    model = MultimodalModel(vocab_size, embed_dim, max_seq_len, num_layers, num_heads, ff_dim)
    
    # Dummy inputs
    image = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    text = torch.randint(0, vocab_size, (2, 50))  # Batch of 2 text sequences
    tgt = torch.randint(0, vocab_size, (2, 60))  # Batch of 2 target sequences
    
    # Generate causal mask for decoder
    tgt_mask = torch.triu(torch.ones(60, 60), diagonal=1).bool().logical_not()[None, None, :, :]
    
    # Forward pass
    logits = model(image=image, text=text, tgt=tgt, tgt_mask=tgt_mask)
    print(logits.shape)  # Should be (2, 60, 30000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")  # Should be < 60M