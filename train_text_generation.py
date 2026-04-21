import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.text_models import TextRNNModel
from utils.text_preprocessing import load_text_data, create_sequences
from utils.embeddings import load_pretrained_embedding_layer

def train(args):
    encoded, word_to_idx, idx_to_word, vocab_size = load_text_data("data/text_generation/corpus.txt")
    X, y = create_sequences(encoded, seq_length=3)

    pretrained_layer = None
    if args.embedding == "pretrained":
        if not os.path.exists("data/ptb_embeddings.pt"):
            raise FileNotFoundError("Run python3 utils/pretrain_word2vec.py first")
        vocab_words = [idx_to_word[i] for i in range(vocab_size)]
        pretrained_layer = load_pretrained_embedding_layer(
            vocab_words=vocab_words,
            embedding_path="data/ptb_embeddings.pt",
            embedding_dim=50
        )

    model = TextRNNModel(
        vocab_size=vocab_size,
        embedding_dim=50 if args.embedding == "pretrained" else 16,
        hidden_dim=32,
        model_type=args.model,
        use_one_hot=(args.embedding == "onehot"),
        pretrained_embedding_layer=pretrained_layer
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        perplexity = math.exp(loss.item())
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}")

    model.eval()
    sample_input = X[0].unsqueeze(0)
    pred = model(sample_input).argmax(dim=1).item()
    input_words = [idx_to_word[idx.item()] for idx in X[0]]
    print("Input sequence:", " ".join(input_words))
    print("Predicted next word:", idx_to_word[pred])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--embedding", choices=["onehot", "pretrained"], default="onehot")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args)
