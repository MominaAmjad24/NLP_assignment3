import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from models.seq2seq_models import Encoder, Decoder, Seq2Seq
from utils.translation_preprocessing import prepare_translation_tensors, create_batch_tensors
from utils.embeddings import load_pretrained_embedding_layer

def translate_sentence(model, src_tensor, tgt_word_to_idx, tgt_idx_to_word, device, max_len=10):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src_tensor)

        input_token = torch.tensor([tgt_word_to_idx["<sos>"]], device=device)
        generated_tokens = []

        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            pred_token = output.argmax(1).item()

            word = tgt_idx_to_word[pred_token]
            if word == "<eos>":
                break
            generated_tokens.append(word)
            input_token = torch.tensor([pred_token], device=device)

    return generated_tokens

def compute_bleu(reference, prediction):
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference], prediction, smoothing_function=smoothie)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, src_word_to_idx, src_idx_to_word, tgt_word_to_idx, tgt_idx_to_word = prepare_translation_tensors(
        "data/translation/eng_spa.txt"
    )

    src_pad_idx = src_word_to_idx["<pad>"]
    tgt_pad_idx = tgt_word_to_idx["<pad>"]

    src_tensor, tgt_tensor = create_batch_tensors(data, src_pad_idx, tgt_pad_idx)
    src_tensor, tgt_tensor = src_tensor.to(device), tgt_tensor.to(device)

    src_pretrained = None
    tgt_pretrained = None
    emb_dim = 50 if args.embedding == "pretrained" else 16

    if args.embedding == "pretrained":
        if not os.path.exists("data/ptb_embeddings.pt"):
            raise FileNotFoundError("Run python3 utils/pretrain_word2vec.py first")

        src_vocab_words = [src_idx_to_word[i] for i in range(len(src_idx_to_word))]
        tgt_vocab_words = [tgt_idx_to_word[i] for i in range(len(tgt_idx_to_word))]

        src_pretrained = load_pretrained_embedding_layer(src_vocab_words, "data/ptb_embeddings.pt", 50)
        tgt_pretrained = load_pretrained_embedding_layer(tgt_vocab_words, "data/ptb_embeddings.pt", 50)

    encoder = Encoder(
        input_dim=len(src_idx_to_word),
        emb_dim=emb_dim,
        hid_dim=32,
        model_type=args.model,
        use_one_hot=(args.embedding == "onehot"),
        embedding_layer=src_pretrained
    )

    decoder = Decoder(
        output_dim=len(tgt_idx_to_word),
        emb_dim=emb_dim,
        hid_dim=32,
        model_type=args.model,
        use_one_hot=(args.embedding == "onehot"),
        embedding_layer=tgt_pretrained
    )

    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output = model(src_tensor, tgt_tensor)
        output_dim = output.shape[-1]

        output = output[:, 1:, :].reshape(-1, output_dim)
        target = tgt_tensor[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

    total_bleu = 0.0
    print("\nSample translations:")
    for i, (src_ids, tgt_ids) in enumerate(data):
        src_example = torch.tensor([src_ids], dtype=torch.long).to(device)
        predicted_tokens = translate_sentence(model, src_example, tgt_word_to_idx, tgt_idx_to_word, device)

        reference_tokens = [tgt_idx_to_word[idx] for idx in tgt_ids[1:-1]]
        source_tokens = [src_idx_to_word[idx] for idx in src_ids[:-1]]

        bleu = compute_bleu(reference_tokens, predicted_tokens)
        total_bleu += bleu

        print(f"Source: {' '.join(source_tokens)}")
        print(f"Reference: {' '.join(reference_tokens)}")
        print(f"Prediction: {' '.join(predicted_tokens)}")
        print(f"BLEU: {bleu:.4f}\n")

    avg_bleu = total_bleu / len(data)
    print(f"Average BLEU score: {avg_bleu:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--embedding", choices=["onehot", "pretrained"], default="onehot")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args)
