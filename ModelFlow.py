import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################
# Basic Multi-Head Attention Module
#######################################
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, and V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project input embeddings and reshape for multi-head attention: (B, seq_len, embed_dim)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Multiply attention weights with value vectors
        context = torch.matmul(attn, v)  # shape: (B, num_heads, seq_len_q, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)
        return output

#######################################
# Transformer Encoder Layer for Text Encoder
#######################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention (text-only)
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

#######################################
# Text Encoder Module
#######################################
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text):
        # text: (batch, seq_len)
        x = self.embed(text)  # (B, seq_len, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x  # (B, seq_len, embed_dim)

#######################################
# Image Encoder Module
#######################################
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # A simple CNN to obtain patches from an image sized 256x256.
        # Using a kernel_size and stride of 16, we obtain 16x16 = 256 patches.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=16, stride=16),  # from 256x256 -> 16x16 feature map (64 channels)
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=1)
        )
        self.flatten = nn.Flatten(start_dim=2)  # flatten the spatial dimensions

    def forward(self, x):
        # x: (batch, 3, 256, 256)
        x = self.conv(x)  # (batch, embed_dim, 16, 16)
        x = self.flatten(x).transpose(1, 2)  # (batch, 256, embed_dim) with 256 patches per image
        return x

#######################################
# Fused Knowledge Module
#######################################
class FusedKnowledge(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # We use multi-head attention to fuse the text encoder output (query)
        # with image encoder output (keys and values)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_repr, image_repr, mask=None):
        # text_repr: (batch, text_seq_len, embed_dim) used as query
        # image_repr: (batch, image_patches, embed_dim) used as key and value
        fused = self.cross_attn(text_repr, image_repr, image_repr, mask)
        # Add residual connection and layer normalization
        fused = self.norm(text_repr + self.dropout(fused))
        return fused  # (batch, text_seq_len, embed_dim)

#######################################
# Transformer Decoder for Captioning
#######################################
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Tie the output weights with the embedding matrix
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (batch, tgt_seq_len)
        x = self.embed(tgt)  # (batch, tgt_seq_len, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

#######################################
# Transformer Decoder Layer (for decoder)
#######################################
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention on the target sequence
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)

        # Cross-attention with memory from the fusion module
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)

        # Feed-forward network
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.dropout(ffn_output)
        tgt = self.norm3(tgt)
        return tgt
#######################################
# Full Multimodal Fusion Model for Image Captioning
#######################################
class MultimodalFusionModel(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim=512,
                 num_layers_enc=2,
                 num_layers_dec=4,
                 num_heads=8,
                 ff_dim=2048,
                 max_seq_len=128):
        super().__init__()
        # Text encoder for any prompt or auxiliary text input
        self.text_encoder = TextEncoder(vocab_size, embed_dim, num_layers_enc, num_heads, ff_dim, max_seq_len)
        # Image encoder to extract patch features from 256x256 images
        self.image_encoder = ImageEncoder(embed_dim)
        # Fusion module to fuse text (query) with image (key,value)
        self.fusion = FusedKnowledge(embed_dim, num_heads)
        # Decoder to generate captions from fused memory representation
        self.decoder = Decoder(vocab_size, embed_dim, num_layers_dec, num_heads, ff_dim, max_seq_len)
        self.vocab_size = vocab_size

    def forward(self, image, tgt, text_input):
        """
        image: tensor of shape (batch, 3, 256, 256)
        tgt: caption token sequence (batch, tgt_seq_len) for teacher forcing
        text_input: text tokens for the text encoder (batch, text_seq_len)
                    This could be a fixed prompt or additional text description.
        """
        # Obtain text and image representations
        text_repr = self.text_encoder(text_input)         # (B, text_seq_len, embed_dim)
        image_repr = self.image_encoder(image)              # (B, num_patches, embed_dim)

        # Fuse text and image features; here text representation serves as queries
        fused_memory = self.fusion(text_repr, image_repr)   # (B, text_seq_len, embed_dim)

        # Create the causal mask for the decoder (to ensure auto-regressive decoding)
        batch_size, tgt_seq_len = tgt.size(0), tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device), diagonal=1).bool().logical_not()
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (causal_mask[None, None, :, :] & tgt_padding_mask).to(torch.float)

        # Decode captions using the fused memory as encoder output
        logits = self.decoder(tgt, fused_memory, tgt_mask=tgt_mask)
        return logits

    def generate(self, image, text_input, max_len=50):
        """
        Auto-regressively generate captions.
        image: (batch, 3, 256, 256)
        text_input: (batch, text_seq_len)
        """
        self.eval()
        batch_size = image.size(0)
        # Compute fused memory from text and image representations
        text_repr = self.text_encoder(text_input)
        image_repr = self.image_encoder(image)
        fused_memory = self.fusion(text_repr, image_repr)

        # Initialize generated caption with the <start> token (assumed id=1)
        tgt = torch.ones(batch_size, 1, device=image.device, dtype=torch.long) * 1
        generated_tokens = []

        for _ in range(max_len):
            logits = self.decoder(tgt, fused_memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            tgt = torch.cat([tgt, next_token], dim=1)
            # Check if all sequences predicted the <end> token (assumed id=2)
            if (next_token == 2).all():
                break

        # Concatenate and return generated tokens (without the initial <start>)
        return torch.cat(generated_tokens, dim=1)

#######################################
# Example instantiation and usage
#######################################
if __name__ == '__main__':
    # Assume vocabulary size of 10000 tokens
    vocab_size = 10000
    model = MultimodalFusionModel(vocab_size)
    # Dummy inputs: a batch of 2 images (3,256,256) and text prompt tokens (batch, text_seq_len)
    dummy_images = torch.randn(2, 3, 256, 256)
    dummy_text = torch.randint(0, vocab_size, (2, 10))  # e.g., a prompt with 10 tokens
    dummy_tgt = torch.randint(0, vocab_size, (2, 12))   # target captions for teacher forcing
    # Forward pass
    logits = model(dummy_images, dummy_tgt, dummy_text)
    print("Logits shape:", logits.shape)
    # Generation (auto-regressive decoding)
    generated = model.generate(dummy_images, dummy_text)
    print("Generated shape:", generated.shape)
