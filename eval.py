import torch
import pandas as pd
from tqdm.auto import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from inference import translate_sentence, model, vocab_tgt, tokenize, nlp_en

# Make sure you pip install nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. LOAD TEST DATA ---
print("Loading Test Dataset...")
# We use the official test split to get a valid metric
df_test = pd.read_json("hf://datasets/bentrevett/multi30k/test.jsonl", lines=True)

# --- 2. EVALUATE BLEU SCORE ---
def calculate_bleu(model, df_test):
    references = []
    candidates = []
    
    print("Translating Test Set (This may take a few minutes)...")
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        src_sentence = row['de']
        trg_sentence = row['en']
        
        # Get model prediction
        prediction = translate_sentence(src_sentence, model)
        
        # Tokenize prediction and target for BLEU math
        pred_tokens = tokenize(prediction, nlp_en)
        trg_tokens = tokenize(trg_sentence, nlp_en)
        
        # NLTK expects references as a list of lists (in case there are multiple valid translations)
        references.append([trg_tokens])
        candidates.append(pred_tokens)
        
    # Calculate Corpus BLEU
    bleu_score = corpus_bleu(references, candidates)
    return bleu_score * 100 # Multiply by 100 to get the standard 0-100 scale

if __name__ == "__main__":
    bleu = calculate_bleu(model, df_test)
    print("\n" + "="*40)
    print(f"🏆 Final Test BLEU Score: {bleu:.2f}")
    print("="*40)
    print("\n(Note: A BLEU score above 25-30 for a from-scratch model trained on 4GB VRAM is highly impressive!)")