# Fused-Knowledge-MultiModal

## 1. Architecture Overview
This project aims to build a multimodal model that relies solely on attention mechanisms. The goal is to process images and text using dedicated attention-based encoders, fuse their representations using an attention-based fusion module, and generate output sequences (captions or answers) through an attention-based decoder.

## 2. Core Components

### Image Transformer Encoder
- **Purpose:** Processes images using a transformer-based approach.
- **How It Works:**
  - The image is split into patches (like in Vision Transformers) and linearly embedded.
  - Positional encodings are added to the patch embeddings.
  - Transformer encoder layers process these embeddings.
  - The final patch representations are aggregated (e.g., via mean pooling) to yield a single feature vector representing the image.

### Text Transformer Encoder
- **Purpose:** Processes textual input (captions, questions) using a transformer.
- **How It Works:**
  - Token indices are embedded into vectors with positional encodings.
  - The sequence is passed through transformer encoder layers.
  - The output is aggregated (using mean pooling or a special token) into a fixed-size text representation.

### Fusion Module via Attention
- **Purpose:** Fuses image and text features into a unified representation.
- **How It Works:**
  - A multi-head attention block treats one modality as the query and the other as key/value.
  - In our setup, image features serve as the query, while text features are the key/value.
  - The fused representation is learned by attending to relevant text features conditioned on image features.

### Transformer Decoder
- **Purpose:** Generates the final output sequence (captions or answers).
- **How It Works:**
  - Transformer decoder layers receive a target sequence (for teacher forcing during training) and the fused representation (as memory from the encoder).
  - The decoder generates token logits at each step to produce output sequences.

## 3. Training Modes

### Multimodal Mode (Image Captioning)
- **Inputs:** Both an image and its associated text (caption).
- **Flow:**
  - The image and text are encoded separately.
  - Features are fused via attention.
  - The decoder generates a caption based on the fused representation.

### Text-Only Mode (Q/A)
- **Inputs:** Only text (e.g., a question).
- **Flow:**
  - Only the text encoder processes the input.
  - The text representation is fed into the decoder to generate an answer.

This design enables the model to learn robust representations that capture both visual and textual information, helping it resolve modality conflicts (e.g., an image showing a cat while the text says "dog"). The attention mechanism ensures that the decoder generates accurate and meaningful outputs.

---
### 🚀 Future Enhancements
- Experimenting with different fusion techniques.
- Optimizing attention computation for large-scale datasets.
- Extending support for additional modalities like audio.

