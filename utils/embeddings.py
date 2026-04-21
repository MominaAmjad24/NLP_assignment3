import torch
import torch.nn as nn

def load_pretrained_embedding_layer(vocab_words, embedding_path, embedding_dim):
    saved = torch.load(embedding_path, map_location="cpu")
    pretrained_vocab = saved["vocab_to_idx"]
    pretrained_vectors = saved["embeddings"]

    embedding_matrix = torch.randn(len(vocab_words), embedding_dim) * 0.01

    for i, word in enumerate(vocab_words):
        if word in pretrained_vocab:
            embedding_matrix[i] = pretrained_vectors[pretrained_vocab[word]]

    layer = nn.Embedding(len(vocab_words), embedding_dim)
    layer.weight.data.copy_(embedding_matrix)
    return layer
