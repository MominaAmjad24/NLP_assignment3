import torch

def load_text_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower().split()

    vocab = sorted(set(text))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    encoded = [word_to_idx[word] for word in text]
    return encoded, word_to_idx, idx_to_word, len(vocab)

def create_sequences(encoded_text, seq_length=3):
    X, y = [], []
    for i in range(len(encoded_text) - seq_length):
        X.append(encoded_text[i:i + seq_length])
        y.append(encoded_text[i + seq_length])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
