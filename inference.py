import torch
import pandas as pd
import spacy
from collections import Counter
from model import FullDecoder, Seq2SeqTransformer, TransformerEncoder

# --- 1. SETUP VOCABULARY (Must match training exactly) ---
try:
    nlp_de = spacy.load('de_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
except:
    import os
    os.system('python -m spacy download de_core_news_sm')
    os.system('python -m spacy download en_core_web_sm')
    nlp_de = spacy.load('de_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')

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

    def __len__(self): return len(self.itos)
    def __call__(self, tokens): return [self.stoi.get(tok, self.unk_idx) for tok in tokens]
    def __getitem__(self, token): return self.stoi.get(token, self.unk_idx)

print("Loading Vocabularies...")
df_train = pd.read_json("hf://datasets/bentrevett/multi30k/train.jsonl", lines=True)
specials = ['<unk>', '<pad>', '<bos>', '<eos>']
vocab_src = Vocab((tokenize(x, nlp_de) for x in df_train['de']), specials)
vocab_tgt = Vocab((tokenize(x, nlp_en) for x in df_train['en']), specials)

BOS_IDX, EOS_IDX = vocab_src['<bos>'], vocab_src['<eos>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. LOAD MODEL ---
print("Loading Model...")
encoder = TransformerEncoder(len(vocab_src), 256, 8, 3)
decoder = FullDecoder(len(vocab_tgt), 256, 8, 3)
model = Seq2SeqTransformer(encoder, decoder).to(device)

# Load the best weights you saved!
model.load_state_dict(torch.load("saved_weights/transformer_best.pt", weights_only=True))
model.eval() # CRITICAL for inference

# --- 3. INFERENCE LOGIC ---
def translate_sentence(sentence, model, max_len=50):
    model.eval()
    
    # 1. Tokenize and convert to tensor
    tokens = tokenize(sentence, nlp_de)
    token_ids = [BOS_IDX] + vocab_src(tokens) + [EOS_IDX]
    src_tensor = torch.tensor(token_ids).unsqueeze(0).to(device) # [1, Seq_Len]
    
    with torch.no_grad():
        # 2. Get Encoder Output
        enc_out = model.encoder(src_tensor)
        
        # 3. Autoregressive Decoding
        tgt_indices = [BOS_IDX]
        
        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            
            # Forward pass through decoder
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model.decoder(tgt_tensor, enc_out)
            
            # Get the predicted token (the one with the highest probability at the last position)
            next_token = logits.argmax(-1)[0, -1].item()
            tgt_indices.append(next_token)
            
            # Stop if the model predicts the End-Of-Sequence token
            if next_token == EOS_IDX:
                break
                
    # 4. Convert IDs back to words
    translated_words = [vocab_tgt.itos[idx] for idx in tgt_indices]
    
    # Clean up tokens
    return " ".join(translated_words[1:-1]) # Exclude <bos> and <eos>

# --- 4. TEST IT ---
if __name__ == "__main__":
    german_sentence = "Ein Hund rennt über das Gras." # "A dog runs over the grass."
    translation = translate_sentence(german_sentence, model)
    print(f"\nGerman: {german_sentence}")
    print(f"English Prediction: {translation}")