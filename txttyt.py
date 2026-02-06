import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.auto import tqdm
import math
from huggingface_hub import login, HfApi, create_repo
from PIL import Image
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer

# TPU Support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("TPU not available (torch_xla not found).")

# ==========================================
# 1. Configuration and Authentication
# ==========================================
HF_TOKEN = ""
REPO_ID = "username/txttyt"
DATASET_ID = "raidium/RadImageNet-VQA"

# Login to Hugging Face
login(token=HF_TOKEN)

# Device Setup
if TPU_AVAILABLE:
    device = xm.xla_device()
    print(f"Using TPU device: {device}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 300
LEARNING_RATE = 1e-4
EPOCHS = 1
EMBED_DIM = 512
FF_DIM = 1024
NUM_HEADS = 8
MAX_LEN = 80
GRAPH_STEPS = 3
WINDOW_SIZE = 3

# Optimization Settings
NUM_WORKERS = 16
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# ==========================================
# 2. Dataset Preparation
# ==========================================
print(f"Loading dataset {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, 'alignment', token=HF_TOKEN)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RadImageNetVQADataset(Dataset):
    def __init__(self, hf_dataset, transform, tokenizer, max_len):
        self.dataset = hf_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'].convert('RGB')
        img_tensor = self.transform(img)
        
        caption_text = "unknown"
        conv_data = item['conversations']
        if conv_data and isinstance(conv_data, list) and len(conv_data) > 0:
            last_turn = conv_data[-1]
            if isinstance(last_turn, dict) and 'value' in last_turn:
                caption_text = last_turn['value']
        
        cap_tokens = self.tokenizer(
            caption_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        caption_ids = torch.tensor(cap_tokens['input_ids'], dtype=torch.long)
        
        return {
            'image_tensor': img_tensor,
            'caption_ids': caption_ids
        }

print("Preparing datasets...")
full_train_dataset = RadImageNetVQADataset(raw_dataset['train'], transform, tokenizer, MAX_LEN)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)

# ==========================================
# 3. Optimized Model Architecture
# ==========================================

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
    def __init__(self, vocab_size, embed_dim, num_layers=5, num_heads=8):
        super().__init__()
        
        # 1. Encoder: Small MobileNet
        try:
            mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except TypeError:
            mobilenet = models.mobilenet_v3_small(pretrained=True)
            
        self.encoder = create_feature_extractor(mobilenet, return_nodes={'features': 'features'})
        self.vis_proj = nn.Linear(576, embed_dim)
        
        # 2. Graph Propagation Layer (Image Only)
        self.graph_layer = GraphPropagationLayer(embed_dim, window_size=WINDOW_SIZE, max_steps=GRAPH_STEPS)
        
        # 3. Decoder: 5 Layer Transformer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 2000, embed_dim)) 
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=FF_DIM, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def encode_images(self, images):
        """
        Pre-computes image features (Memory).
        Used in fast inference to avoid re-running CNN.
        """
        with torch.no_grad():
            visual_features = self.encoder(images)['features']
            B, C, H, W = visual_features.shape
            visual_tokens = visual_features.view(B, C, -1).permute(0, 2, 1)
            visual_tokens = self.vis_proj(visual_tokens)
            # Apply Graph Propagation here once
            memory = self.graph_layer(visual_tokens)
        return memory

    def forward(self, images, captions):
        # --- Encoder ---
        # Note: In training, we run the graph layer here every batch. 
        # This is fine as batch sizes are large.
        visual_features = self.encoder(images)['features']
        B, C, H, W = visual_features.shape
        visual_tokens = visual_features.view(B, C, -1).permute(0, 2, 1)
        visual_tokens = self.vis_proj(visual_tokens)
        
        # Graph Propagation on Image Tokens ONLY
        memory = self.graph_layer(visual_tokens)
        
        # --- Text Embedding ---
        caption_emb = self.embedding(captions)
        seq_len = caption_emb.size(1)
        caption_emb = caption_emb + self.pos_encoder[:, :seq_len, :]
        
        # --- Decoder ---
        tgt_key_padding_mask = (captions == tokenizer.pad_token_id).to(device)
        
        # CRITICAL FIX: Add Causal Mask (Square Subsequent Mask)
        # This prevents the model from "cheating" by seeing future tokens.
        # Without this, inference will likely repeat words or fail.
        tgt_mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float('-inf'), diagonal=1)

        output = self.decoder(
            tgt=caption_emb, 
            memory=memory, 
            tgt_mask=tgt_mask, # Apply causal mask
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)

    @torch.no_grad()
    def generate(self, images, max_len=80):
        """
        Fast Inference Method:
        1. Encodes image once.
        2. Loops only through the text decoder.
        """
        self.eval()
        
        # 1. Compute Image Features (Cached)
        memory = self.encode_images(images)
        batch_size = images.shape[0]
        
        # 2. Initialize with [CLS] token
        generated = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=images.device)
        
        for _ in range(max_len):
            # Prepare Input
            tgt_emb = self.embedding(generated)
            seq_len = tgt_emb.size(1)
            tgt_emb = tgt_emb + self.pos_encoder[:, :seq_len, :]
            
            # Masks
            tgt_mask = torch.triu(torch.ones((seq_len, seq_len), device=images.device) * float('-inf'), diagonal=1)
            tgt_key_padding_mask = (generated == tokenizer.pad_token_id).to(images.device)
            
            # Decoder Step
            output = self.decoder(
                tgt=tgt_emb, 
                memory=memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # Predict Next Token
            next_token_logits = self.fc_out(output[:, -1, :]) # Take last time step
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)
            
            # Check for Stop Condition
            if (next_token == tokenizer.sep_token_id).all():
                break
                
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated

# ==========================================
# 4. Training and Inference Setup
# ==========================================

model = ImageCaptioningModel(tokenizer.vocab_size, EMBED_DIM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

try:
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository {REPO_ID} ready.")
    except Exception as repo_e:
        print(f"Repo creation warning: {repo_e}")
except Exception as e:
    print(f"HF Hub setup info: {e}")

def inference(model, dataset, num_samples=3):
    """
    Updated inference function using the new fast generate() method.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    samples = []
    with torch.no_grad():
        for idx in indices:
            data = dataset[idx] 
            # Prepare image (add batch dim)
            img = data['image_tensor'].unsqueeze(0).to(device)
            
            # --- FAST GENERATION ---
            # Uses cached image features, runs CNN only ONCE.
            generated_ids = model.generate(img, max_len=MAX_LEN)
            
            # Decode
            # We squeeze(0) because batch size is 1
            gen_text = tokenizer.decode(generated_ids.squeeze(0).cpu().numpy(), skip_special_tokens=True)
            real_text = tokenizer.decode(data['caption_ids'].cpu().numpy(), skip_special_tokens=True)
            
            samples.append((gen_text, real_text))
            
    model.train()
    return samples

# ==========================================
# 5. Training Loop
# ==========================================

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_items = 0
    
    if TPU_AVAILABLE:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in pbar:
        images = batch['image_tensor']
        captions = batch['caption_ids']
        
        if not TPU_AVAILABLE:
            images = images.to(device)
            captions = captions.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (with new causal mask logic inside)
        outputs = model(images, captions[:, :-1])
        
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = captions[:, 1:].reshape(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        if TPU_AVAILABLE:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
            
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
        
        current_loss = total_loss / total_items
        ppl = math.exp(current_loss)
        
        pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'PPL': f'{ppl:.4f}'})
    
    print(f"\n--- Inference after Epoch {epoch+1} ---")
    samples = inference(model, val_dataset, num_samples=3)
    for i, (gen, real) in enumerate(samples):
        print(f"Sample {i+1}:")
        print(f"Generated: {gen}")
        print(f"Ground Truth: {real}\n")

    print(f"Attempting to push model to {REPO_ID}...")
    try:
        os.makedirs("model_checkpoint", exist_ok=True)
        torch.save(model.state_dict(), "model_checkpoint/pytorch_model.bin")
        
        config_content = f"""Embed Dim: {EMBED_DIM}
Layers: 5
Encoder: MobileNetV3_Small
Graph: Window {WINDOW_SIZE}, Steps {GRAPH_STEPS} (Image Only)
Dataset: {DATASET_ID} (Alignment)
Mode: No Mapping (Dynamic) + Causal Mask + Cached Inference
"""
        with open("model_checkpoint/config.txt", "w") as f:
            f.write(config_content)
        
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path="model_checkpoint",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Model pushed successfully.")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")

print("Training complete.")
