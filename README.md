# txttyt
medical reporting demo
---
language: 
- en
tags:
- medical
- radiology
- image-captioning
- vision-language
- pytorch
license: mit
datasets:
- raidium/RadImageNet-VQA
---

# Medical Image Captioning Model (RadImageNet)

This model is a specialized image captioning architecture designed to generate radiology reports from medical images. It was trained on the **RadImageNet-VQA** dataset to identify imaging modalities (MRI, CT, X-Ray), anatomical structures, and pathologies.

## Model Details

### Architecture
The model utilizes a hybrid architecture combining CNN features, graph-based reasoning, and autoregressive text generation:

1.  **Encoder:** Uses a MobileNet V3 Small backbone to extract visual features from input images (resized to $224 \times 224$).
2.  **Graph Propagation Layer:** A custom Graph Neural Network (GNN) layer processes the extracted visual features (image tokens) using a sliding window approach. Crucially, this layer operates **only on image tokens**, preventing information leakage (look-ahead) from the text tokens.
3.  **Memory Bank:** The refined image features from the graph layer serve as the memory context for the text decoder.
4.  **Decoder:** A standard Transformer Decoder (5 layers, 8 heads) generates the caption autoregressively. It employs a causal mask to ensure that the prediction of the next word depends only on previously generated words and the image memory.

### Training
*   **Dataset:** [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA)
*   **Hardware:** TPU v3 (PyTorch/XLA)
*   **Optimizer:** Adam
*   **Loss Function:** Cross Entropy Loss
*   **Tokenizer:** BERT Tokenizer (Base Uncased)

#### Training Metrics (After Epoch 1)
Training was conducted for 1 epoch over 2,250 steps on the TPU. The model achieved the following metrics:
*   **Final Loss:** 0.7637
*   **Final Perplexity:** 2.1462
*   **Training Time:** ~11 minutes (TPU)

### Inference Performance
Despite only one epoch of training, the model demonstrates the ability to correlate imaging modalities and anatomical regions with the correct text generation.

**Sample 1 (Ankle/MRI):**
*   **Input:** Ankle MRI
*   **Ground Truth:** "diagnostic magnetic resonance imaging of the ankle foot ankle displays bone inflammation"
*   **Generated:** "mri scan of the ankle foot foot with findings consistent with bone inflammation"

**Sample 2 (Chest/CT):**
*   **Input:** Chest CT
*   **Ground Truth:** "ct examination of the lung respiratory showing evidence of airspace opacity"
*   **Generated:** "ct scan of the lung chest with findings consistent with airspace opacity"

**Sample 3 (Spine/MRI):**
*   **Input:** Spine MRI
*   **Ground Truth:** "mr scan of the spine spinal with findings consistent with disc pathology"
*   **Generated:** "mri examination of the spine spinal column with findings consistent with disc pathology"

## How to Use

Since this model uses a custom architecture, you must define the classes `GraphPropagationLayer` and `ImageCaptioningModel` (or import them) before loading the weights.

### Installation
```bash
pip install torch torchvision transformers pillow huggingface_hub

### Inference Script
python
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from PIL import Image

# 1. Define the Model Architecture (Must match training)
class GraphPropagationLayer(nn.Module):
def __init__(self, embed_dim, window_size=3, max_steps=3):
super().__init__()
self.window_size = window_size
self.max_steps = max_steps
self.msg_linear = nn.Linear(embed_dim, embed_dim)
self.activation = nn.Hardtanh()

def forward(self, x):
B, L, D = x.shape
pad = self.window_size // 2
for _ in range(self.max_steps):
neighbors = []
for i in range(-pad, pad + 1):
neighbors.append(torch.roll(x, shifts=i, dims=1))

stacked = torch.stack(neighbors, dim=2)
aggregated = stacked.mean(dim=2)
messages = self.msg_linear(aggregated)
x = self.activation(x + messages)
return x

class ImageCaptioningModel(nn.Module):
def __init__(self, vocab_size, embed_dim=512, num_layers=5, num_heads=8):
super().__init__()

# Encoder
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
self.encoder = create_feature_extractor(mobilenet, return_nodes={'features': 'features'})
self.vis_proj = nn.Linear(576, embed_dim)

# Graph
self.graph_layer = GraphPropagationLayer(embed_dim, window_size=3, max_steps=3)

# Decoder
self.embedding = nn.Embedding(vocab_size, embed_dim)
self.pos_encoder = nn.Parameter(torch.randn(1, 2000, embed_dim)) 

decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=1024, batch_first=True)
self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
self.fc_out = nn.Linear(embed_dim, vocab_size)

def encode_images(self, images):
visual_features = self.encoder(images)['features']
B, C, H, W = visual_features.shape
visual_tokens = visual_features.view(B, C, -1).permute(0, 2, 1)
visual_tokens = self.vis_proj(visual_tokens)
memory = self.graph_layer(visual_tokens)
return memory

@torch.no_grad()
def generate(self, images, max_len=80, tokenizer=None):
self.eval()
memory = self.encode_images(images)
batch_size = images.shape[0]

generated = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=images.device)

for _ in range(max_len):
tgt_emb = self.embedding(generated)
seq_len = tgt_emb.size(1)
tgt_emb = tgt_emb + self.pos_encoder[:, :seq_len, :]

tgt_mask = torch.triu(torch.ones((seq_len, seq_len), device=images.device) * float('-inf'), diagonal=1)
tgt_key_padding_mask = (generated == tokenizer.pad_token_id).bool().to(images.device)

output = self.decoder(
tgt=tgt_emb, memory=memory, 
tgt_mask=tgt_mask,
tgt_key_padding_mask=tgt_key_padding_mask
)

next_token = self.fc_out(output[:, -1, :]).argmax(dim=-1).unsqueeze(1)
if (next_token == tokenizer.sep_token_id).all():
break
generated = torch.cat([generated, next_token], dim=1)

return generated

# 2. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = ImageCaptioningModel(tokenizer.vocab_size).to(device)

# Download weights from Hugging Face Hub
model_path = hf_hub_download(repo_id="erfansghariyan/txttyt", filename="pytorch_model.bin")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3. Preprocess & Inference
transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("your_medical_image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

output_ids = model.generate(image_tensor, max_len=80, tokenizer=tokenizer)
caption = tokenizer.decode(output_ids.squeeze(0).cpu().numpy(), skip_special_tokens=True)

print(f"Generated Caption: {caption}")

## Limitations
*   **Training Duration:** The model was trained for only 1 epoch. While it captures high-level concepts (modality, location), it may hallucinate specific details or lack fine-grained medical accuracy compared to a fully trained model.


## Usage

This model uses a custom PyTorch architecture. Therefore, you need to instantiate the `GraphPropagationLayer` and `ImageCaptioningModel` classes to load the weights.

### Installation

First, install the required Python packages:
```bash
pip install torch torchvision transformers pillow huggingface_hub

### Inference in Python

Here is a complete script to load the model and generate a caption for a medical image.

python
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from PIL import Image

# --- 1. Define the Model Architecture ---
# These classes must match the training script exactly.

class GraphPropagationLayer(nn.Module):
def __init__(self, embed_dim, window_size=3, max_steps=3):
super().__init__()
self.window_size = window_size
self.max_steps = max_steps
self.msg_linear = nn.Linear(embed_dim, embed_dim)
self.activation = nn.Hardtanh()

def forward(self, x):
B, L, D = x.shape
pad = self.window_size // 2
for _ in range(self.max_steps):
neighbors = []
for i in range(-pad, pad + 1):
neighbors.append(torch.roll(x, shifts=i, dims=1))
stacked = torch.stack(neighbors, dim=2)
aggregated = stacked.mean(dim=2)
messages = self.msg_linear(aggregated)
x = self.activation(x + messages)
return x

class ImageCaptioningModel(nn.Module):
def __init__(self, vocab_size, embed_dim=512, num_layers=5, num_heads=8):
super().__init__()
# Encoder
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
self.encoder = create_feature_extractor(mobilenet, return_nodes={'features': 'features'})
self.vis_proj = nn.Linear(576, embed_dim)

# Graph
self.graph_layer = GraphPropagationLayer(embed_dim, window_size=3, max_steps=3)

# Decoder
self.embedding = nn.Embedding(vocab_size, embed_dim)
self.pos_encoder = nn.Parameter(torch.randn(1, 2000, embed_dim)) 
decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=1024, batch_first=True)
self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
self.fc_out = nn.Linear(embed_dim, vocab_size)

def encode_images(self, images):
visual_features = self.encoder(images)['features']
B, C, H, W = visual_features.shape
visual_tokens = visual_features.view(B, C, -1).permute(0, 2, 1)
visual_tokens = self.vis_proj(visual_tokens)
return self.graph_layer(visual_tokens)

@torch.no_grad()
def generate(self, images, max_len=80, tokenizer=None):
self.eval()
memory = self.encode_images(images)
batch_size = images.shape[0]

generated = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=images.device)

for _ in range(max_len):
tgt_emb = self.embedding(generated)
seq_len = tgt_emb.size(1)
tgt_emb = tgt_emb + self.pos_encoder[:, :seq_len, :]

# Causal Mask
tgt_mask = torch.triu(torch.ones((seq_len, seq_len), device=images.device) * float('-inf'), diagonal=1)
# Padding Mask
tgt_key_padding_mask = (generated == tokenizer.pad_token_id).bool().to(images.device)

output = self.decoder(
tgt=tgt_emb, memory=memory, 
tgt_mask=tgt_mask,
tgt_key_padding_mask=tgt_key_padding_mask
)

next_token = self.fc_out(output[:, -1, :]).argmax(dim=-1).unsqueeze(1)
if (next_token == tokenizer.sep_token_id).all():
break
generated = torch.cat([generated, next_token], dim=1)
return generated

# --- 2. Load Model & Tokenizer ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize model architecture
model = ImageCaptioningModel(tokenizer.vocab_size).to(device)

# Download weights from Hub
model_path = hf_hub_download(repo_id="erfansghariyan/txttyt", filename="pytorch_model.bin")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"Model loaded on {device}")

# --- 3. Prepare Data ---

transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load an image
image_path = "path_to_your_medical_image.jpg"  # Replace with actual path
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# --- 4. Run Inference ---

output_ids = model.generate(image_tensor, max_len=80, tokenizer=tokenizer)
caption = tokenizer.decode(output_ids.squeeze(0).cpu().numpy(), skip_special_tokens=True)

print(f"Generated Caption: {caption}")

### Hugging Face Space Demo

You can also try the model without writing code by visiting the **Hugging Face Space**:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/erfansghariyan/rad-vqa-demo)
*(Note: Update the link above once you have created and deployed your Space)*

*   **Input Size:** Images must be resized to $224 \times 224$, which may result in loss of fine detail for high-resolution radiology scans.
*   **Citation/Reporting:** This is a research-grade model and should not be used for direct diagnostic assistance without clinical validation.
