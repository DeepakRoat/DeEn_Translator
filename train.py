import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter
from tqdm.auto import tqdm
import os

# Import your custom architectures
from model import FullDecoder, Seq2SeqTransformer, TransformerEncoder

# --- 1. SETUP DATA ---
try:
    nlp_de = spacy.load('de_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download de_core_news_sm')
    os.system('python -m spacy download en_core_web_sm')
    nlp_de = spacy.load('de_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')

# Load dataset
print("Loading datasets...")
df_train = pd.read_json("hf://datasets/bentrevett/multi30k/train.jsonl", lines=True)
df_val = pd.read_json("hf://datasets/bentrevett/multi30k/val.jsonl", lines=True)

def tokenize(text, nlp):
    return [tok.text.lower() for tok in nlp.tokenizer(text)]

class Vocab:
    def __init__(self, tokens_generator, specials):
        self.itos = specials[:]
        self.stoi = {tok: i for i, tok in enumerate(specials)}
        
        counter = Counter()
        for tokens in tokens_generator:
            counter.update(tokens)
        
        for tok, freq in counter.items():
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        
        self.unk_idx = self.stoi.get('<unk>', 0)

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        return [self.stoi.get(tok, self.unk_idx) for tok in tokens]

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_idx)

print("Building vocabularies...")
specials = ['<unk>', '<pad>', '<bos>', '<eos>']
vocab_src = Vocab((tokenize(x, nlp_de) for x in df_train['de']), specials)
vocab_tgt = Vocab((tokenize(x, nlp_en) for x in df_train['en']), specials)

PAD_IDX = vocab_src['<pad>']
BOS_IDX = vocab_src['<bos>']
EOS_IDX = vocab_src['<eos>']

class Multi30kDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx]['de'], self.df.iloc[idx]['en']


# --- 2. TRAINING SETUP & UTILS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

encoder = TransformerEncoder(len(vocab_src), 256, 8, 3)
decoder = FullDecoder(len(vocab_tgt), 256, 8, 3)
model = Seq2SeqTransformer(encoder, decoder).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Initialize the gradient scaler for mixed precision (FP16)
scaler = torch.cuda.amp.GradScaler()

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor([BOS_IDX] + vocab_src(tokenize(src_sample, nlp_de)) + [EOS_IDX]))
        tgt_batch.append(torch.tensor([BOS_IDX] + vocab_tgt(tokenize(tgt_sample, nlp_en)) + [EOS_IDX]))
    return pad_sequence(src_batch, padding_value=PAD_IDX).T.to(device), \
           pad_sequence(tgt_batch, padding_value=PAD_IDX).T.to(device)

train_dataset = Multi30kDataset(df_train)
val_dataset = Multi30kDataset(df_val)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)


# --- 3. TRAIN AND EVAL FUNCTIONS ---
def train_epoch(model, optimizer, dataloader, epoch, scaler):
    model.train()
    losses = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]")
    
    for i, (src, tgt) in progress_bar:
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward Pass
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(src, tgt_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses += loss.item()
        
        if i % 10 == 0:
            progress_bar.set_postfix(loss=loss.item())
            
    return losses / len(dataloader)


def evaluate_epoch(model, dataloader, epoch):
    model.eval() 
    losses = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, (src, tgt) in progress_bar:
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            # Mixed Precision Evaluation
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(src, tgt_input)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                
            losses += loss.item()
            
            if i % 10 == 0:
                progress_bar.set_postfix(val_loss=loss.item())
                
    return losses / len(dataloader)


# --- 4. MAIN EXECUTION LOOP ---
os.makedirs("saved_weights", exist_ok=True)

print("Starting training pipeline...")
best_val_loss = float('inf')
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    # Train
    train_loss = train_epoch(model, optimizer, train_dataloader, epoch, scaler)
    
    # Evaluate
    val_loss = evaluate_epoch(model, val_dataloader, epoch)
    
    print(f"\n--- Epoch {epoch} Summary ---")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Early Stopping / Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = "saved_weights/transformer_best.pt"
        torch.save(model.state_dict(), save_path)
        print(f"--> Validation loss improved! Saved best weights to {save_path}\n")
    else:
        print(f"--> Validation loss did not improve.\n")

print("Training pipeline complete! You can now load 'saved_weights/transformer_best.pt' for inference.")