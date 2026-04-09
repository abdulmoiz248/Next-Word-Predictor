# Next Word Predictor

A next-word prediction model built with PyTorch, using an LSTM neural network trained on a custom text dataset.

---

## 📖 Overview

This project builds a next-word predictor from scratch. Given a sequence of words, the model predicts the most likely next word. The pipeline covers data preprocessing, vocabulary construction, sequence generation, model training, and inference — all implemented in a single Jupyter notebook (`next_word.ipynb`).

---

## 🗂️ Project Structure

```
Next-Word-Predictor/
├── next_word.ipynb   # Main notebook: data prep, model, training, inference
└── dataset.csv       # Input dataset with a 'Text' column (and optional 'Label' column)
```

---

## 🔧 Requirements

- Python 3.7+
- PyTorch
- NLTK
- NumPy
- pandas

Install all dependencies:

```bash
pip install torch nltk numpy pandas
```

NLTK tokenizer data is downloaded automatically within the notebook:

```python
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## 📊 Dataset

The model expects a CSV file (`dataset.csv`) with at least a `Text` column containing the raw text samples. A `Label` column, if present, is dropped before training.

```
Text,Label
"The quick brown fox jumps over the lazy dog",positive
...
```

---

## 🚀 How It Works

### 1. Data Preprocessing

- All text rows are concatenated into a single document.
- The document is **tokenized** using NLTK's `word_tokenize` (lowercased).
- A **vocabulary** (`voc`) is built mapping each unique token to an integer index. An `<UNK>` token handles out-of-vocabulary words.

### 2. Sequence Construction

- The document is split into individual sentences.
- Each sentence is converted to its token indices.
- **N-gram training sequences** are generated: for a sentence of length *n*, sequences of lengths 1, 2, …, *n* are produced.
- All sequences are **zero-padded** to the maximum sequence length.
- Input (`X`) = all tokens except the last; Target (`y`) = the last token.

### 3. Model Architecture

```
Input (token indices)
       │
 Embedding Layer  (vocab_size → 100)
       │
   LSTM Layer     (100 → 150 hidden units)
       │
  Linear Layer    (150 → vocab_size)
       │
  Predicted word index
```

| Layer     | Input dim | Output dim   |
|-----------|-----------|--------------|
| Embedding | vocab_size | 100          |
| LSTM      | 100        | 150 (hidden) |
| Linear    | 150        | vocab_size   |

### 4. Training

| Hyperparameter | Value            |
|----------------|------------------|
| Epochs         | 25               |
| Learning Rate  | 0.001            |
| Optimizer      | Adam             |
| Loss Function  | CrossEntropyLoss |
| Batch Size     | 32               |

Training automatically uses **GPU** if available, falling back to CPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 5. Inference

```python
def prediction(model, vocab, text):
    # Tokenize and encode input text
    # Pad to max_length
    # Run through model
    # Return input text + predicted next word
    ...
```

Example usage:

```python
result = prediction(model, voc, "the quick brown")
print(result)  # e.g., "the quick brown fox"
```

---

## 📓 Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/abdulmoiz248/Next-Word-Predictor.git
   cd Next-Word-Predictor
   ```

2. Place your `dataset.csv` in the working directory (or update the path in the notebook).

3. Open and run the notebook:
   ```bash
   jupyter notebook next_word.ipynb
   ```
   Or open it directly in [Google Colab](https://colab.research.google.com/).

---

## 📈 Results

Loss decreases across 25 epochs as the LSTM learns word co-occurrence patterns from the dataset. The final model can predict contextually relevant next words for a given input phrase.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).