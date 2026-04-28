# Assignment 3: Natural Language Processing

## 1. Project Overview

This project explores deep learning approaches for two core Natural Language Processing (NLP) tasks:

* **Text Generation / Sequence Completion**
* **Machine Translation (English → Spanish)**

We implement and compare recurrent neural network architectures (**LSTM** and **GRU**) and evaluate the impact of different word representations (**one-hot encoding vs pre-trained embeddings**).

The goal is to understand how model architecture and embedding choice affect performance on sequence modeling tasks.

---

## 2. How to Run the Experiments

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pre-train Embeddings

```bash
python3 utils/pretrain_word2vec.py
```

---

### Run Text Generation

```bash
python3 train_text_generation.py --model lstm --embedding onehot
python3 train_text_generation.py --model gru --embedding onehot
python3 train_text_generation.py --model lstm --embedding pretrained
python3 train_text_generation.py --model gru --embedding pretrained
```

---

### Run Machine Translation

```bash
python3 train_translation.py --model lstm --embedding onehot
python3 train_translation.py --model gru --embedding onehot
python3 train_translation.py --model lstm --embedding pretrained
python3 train_translation.py --model gru --embedding pretrained
```

---

## 3. Dataset Description

### Text Generation Dataset

A small custom corpus was used containing simple English sentences related to machine learning and general topics. The dataset was intentionally kept small to focus on model behavior rather than large scale training.

### Machine Translation Dataset

A small English-Spanish sentence pair dataset was used, consisting of common phrases such as greetings, questions, and simple statements.

### Pre-trained Embeddings

Word embeddings were pre-trained using a **word2vec (skip-gram)** model with negative sampling on the Penn Treebank (PTB) dataset.

---

## 4. Model Architectures Used

### LSTM (Long Short-Term Memory)

* Captures long-range dependencies using gated memory cells
* More expressive but slightly heavier

### GRU (Gated Recurrent Unit)

* Simpler architecture with fewer parameters
* Faster training and often better for small datasets

### Seq2Seq Model (for Translation)

* Encoder-decoder architecture
* Encoder processes input sentence
* Decoder generates translated output

---

## 5. Word Embedding Methods

### One-Hot Encoding

* Sparse representation
* Each word represented as a unique index vector
* No semantic information

### Pre-trained Embeddings (Word2Vec)

* Learned using skip-gram model on PTB dataset
* Dense vector representation
* Captures semantic relationships between words

---

## 6. Experimental Results

### Text Generation

All models showed strong convergence:

* Initial loss ≈ 4.1
* Final loss ≈ 0.03–0.09
* Final perplexity ≈ 1.03–1.10

Example:

```
Input: the sun rises  
Prediction: in
```

---

### Text Generation Analysis

We evaluated text generation using both LSTM and GRU models with one-hot and pre-trained embeddings. Performance was measured using **loss** and **perplexity**.

#### Observations

* All models converged quickly from high loss to very low values.
* Final perplexity (~1.03–1.10) indicates strong learning of sequence patterns.
* GRU generally converged faster than LSTM.
* Pre-trained embeddings improved early training stability.

#### Model Comparison

**LSTM vs GRU**

* GRU converged faster and required fewer parameters
* Both models achieved similar final performance

**One-hot vs Pretrained**

* One-hot encoding started slower and required more epochs
* Pre-trained embeddings improved early-stage learning
* Final performance was similar due to small dataset size

#### Summary

| Model | Embedding  | Behavior                |
| ----- | ---------- | ----------------------- |
| LSTM  | One-hot    | Slower convergence      |
| LSTM  | Pretrained | Stable early learning   |
| GRU   | One-hot    | Faster than LSTM        |
| GRU   | Pretrained | Best overall efficiency |

#### Interpretation

* Very low perplexity suggests **overfitting due to small dataset**
* Pre-trained embeddings help more in larger datasets
* Models successfully learned local word relationships

---

### Machine Translation Results

#### BLEU Score Comparison

| Model | Embedding  | Avg BLEU   |
| ----- | ---------- | ---------- |
| LSTM  | One-hot    | 0.4222     |
| LSTM  | Pretrained | 0.5968     |
| GRU   | One-hot    | 0.4764     |
| GRU   | Pretrained | **0.6205** |

---

### Qualitative Examples

Correct translations:

```
Source: this is my book  
Prediction: este es my libro  

Source: where is the school  
Prediction: donde esta la escuela
```

Partial predictions:

```
Source: good morning  
Prediction: dias
```

Incorrect predictions (mostly one-hot):

```
Source: how are you  
Prediction: estamos aprendiendo python
```

---

## 7. Comparison of Models

### LSTM vs GRU

* GRU achieved lower loss and higher BLEU scores
* Better suited for small datasets
* Faster convergence

### One-hot vs Pretrained Embeddings

* Pre-trained embeddings significantly improved translation quality
* One-hot encoding produced more errors and incomplete outputs
* Embeddings provided better semantic understanding

---

## 8. Challenges Faced

* Handling variable-length sequences and padding
* Implementing Seq2Seq architecture from scratch
* Debugging tensor shapes and training loops
* Multiprocessing issues in word2vec pretraining
* Vocabulary mismatch between PTB embeddings and dataset

---

## 9. Limitations

* Very small dataset → limited generalization
* No attention mechanism in translation model
* Pretrained embeddings trained only on English
* Models may memorize rather than generalize
* BLEU evaluation limited due to small dataset

---

## 10. Future Improvements

* Add attention mechanism to Seq2Seq
* Use larger datasets (WikiText, Multi30K)
* Train multilingual embeddings
* Use transformer-based models (BERT, GPT)
* Improve evaluation with larger test sets

---

## Conclusion

This project demonstrates that:

* Recurrent neural networks effectively model sequential language tasks
* GRU provides a good balance of performance and efficiency
* Pre-trained embeddings significantly improve model quality
* Dataset size plays a critical role in model performance

Overall, **GRU with pre-trained embeddings achieved the best performance across tasks**.


