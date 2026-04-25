Here is a complete, professional README.md file tailor-made for your repository. It highlights your custom architecture, prominently displays your massive 30.90 BLEU score, and gives clear instructions for anyone visiting your page.

You can copy this entire block and save it as README.md in your E:\AI\trans directory.
Markdown

# DeEn_Translator: Seq2Seq Transformer from Scratch 🚀

A complete Sequence-to-Sequence (Encoder-Decoder) Transformer built entirely from the ground up in native PyTorch. 

This project was built to establish a rigorous, independent technical foundation in deep learning architecture. It intentionally avoids high-level abstractions like `nn.Transformer` or Hugging Face's `transformers` library to maintain absolute control over the underlying matrix multiplications, attention mechanisms, and information flow.

## 🏆 Performance
* **Task:** Machine Translation (German to English)
* **Dataset:** Multi30k
* **Evaluation Metric:** Corpus BLEU Score
* **Result:** **30.90 BLEU** on the unseen test set.

*(A BLEU score of ~31 demonstrates highly fluent and accurate translation capabilities, which is especially impressive for a from-scratch model trained under strict 4GB VRAM hardware constraints).*

## 🧠 Architectural Highlights
While based on the original "Attention Is All You Need" paper, this implementation integrates modern advancements found in state-of-the-art LLMs (like LLaMA):

* **Custom Attention:** Hand-coded Multi-Head Self-Attention and Cross-Attention blocks with causal masking.
* **RoPE:** Rotary Position Embeddings applied directly to Queries and Keys, replacing legacy absolute sinusoidal encodings.
* **SwiGLU FFN:** Custom SwiGLU activation layers in the Feed-Forward Networks for improved non-linear representation.
* **Hardware Optimization:** Native Automatic Mixed Precision (AMP / FP16) training pipeline to maximize batch sizes on GPUs with limited VRAM (e.g., RTX 3050M).

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/DeepakRoat/DeEn_Translator.git](https://github.com/DeepakRoat/DeEn_Translator.git)
   cd DeEn_Translator

    Install dependencies:
    Make sure you have PyTorch installed with CUDA support. Then install the required NLP tools:
    Bash

    pip install pandas tqdm spacy nltk
    python -m spacy download de_core_news_sm
    python -m spacy download en_core_web_sm

🚀 Usage
1. Training the Model

To train the model from scratch on the Multi30k dataset:
Bash

python train.py

Note: The script automatically handles downloading the Hugging Face dataset, building the vocabularies, and tracking the best validation loss. Weights are saved in the saved_weights/ directory.
2. Live Inference

To test the model interactively and watch it autoregressively translate your own German sentences:
Bash

python inference.py

(You can modify the german_sentence variable at the bottom of the script to test different inputs).
3. Evaluating BLEU Score

To run a full evaluation on the test set and calculate the corpus BLEU score:
Bash

python eval.py

📂 Project Structure

    model.py - The core neural network architecture (Encoder, Decoder, Attention, RoPE, SwiGLU).

    train.py - Data loading, vocabulary generation, mixed-precision training loop, and early stopping.

    inference.py - Autoregressive decoding logic for translating raw text.

    eval.py - NLTK integration for evaluating model performance against human references.

    .gitignore - Prevents large .pt weight files from being pushed to the repository.
