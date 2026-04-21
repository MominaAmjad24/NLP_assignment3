import torch

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def tokenize(text):
    return text.lower().strip().split()

def load_translation_data(file_path):
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            src, tgt = parts
            pairs.append((tokenize(src), tokenize(tgt)))
    return pairs

def build_vocab(sentences):
    vocab = set()
    for sent in sentences:
        vocab.update(sent)

    idx_to_word = SPECIAL_TOKENS + sorted(vocab)
    word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
    return word_to_idx, idx_to_word

def numericalize(sentence, word_to_idx):
    return [word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence]

def prepare_translation_tensors(file_path):
    pairs = load_translation_data(file_path)

    src_sentences = [src for src, _ in pairs]
    tgt_sentences = [tgt for _, tgt in pairs]

    src_word_to_idx, src_idx_to_word = build_vocab(src_sentences)
    tgt_word_to_idx, tgt_idx_to_word = build_vocab(tgt_sentences)

    data = []
    for src, tgt in pairs:
        src_ids = numericalize(src, src_word_to_idx) + [src_word_to_idx["<eos>"]]
        tgt_ids = [tgt_word_to_idx["<sos>"]] + numericalize(tgt, tgt_word_to_idx) + [tgt_word_to_idx["<eos>"]]
        data.append((src_ids, tgt_ids))

    return data, src_word_to_idx, src_idx_to_word, tgt_word_to_idx, tgt_idx_to_word

def pad_sequences(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)

def create_batch_tensors(data, src_pad_idx, tgt_pad_idx):
    src_batch = [src for src, tgt in data]
    tgt_batch = [tgt for src, tgt in data]

    src_tensor = pad_sequences(src_batch, src_pad_idx)
    tgt_tensor = pad_sequences(tgt_batch, tgt_pad_idx)
    return src_tensor, tgt_tensor
