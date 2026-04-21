import torch
from torch import nn
from d2l import torch as d2l

d2l.get_dataloader_workers = lambda: 0

embed_size = 50

def get_ptb_data_iter(batch_size=512, max_window_size=5, num_noise_words=5):
    data_iter, vocab = d2l.load_data_ptb(
        batch_size=batch_size,
        max_window_size=max_window_size,
        num_noise_words=num_noise_words
    )
    return data_iter, vocab

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_size)
        self.context_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, center, contexts_and_negatives):
        v = self.center_embeddings(center)
        u = self.context_embeddings(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        )
        return out.mean(dim=1)

def train_word2vec(num_epochs=5, lr=0.01, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_iter, vocab = get_ptb_data_iter()
    net = SkipGramNegSampling(len(vocab), embed_size).to(device)
    loss = SigmoidBCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss, count = 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [x.to(device) for x in batch]

            pred = net(center, context_negative)
            l = loss(pred.reshape(label.shape).float(), label.float(), mask.float())

            optimizer.zero_grad()
            l.sum().backward()
            optimizer.step()

            total_loss += l.sum().item()
            count += l.numel()

        print(f"epoch {epoch+1}, loss {total_loss / count:.4f}")

    return net, vocab

def save_embeddings(net, vocab, save_path="data/ptb_embeddings.pt"):
    weights = net.center_embeddings.weight.data.cpu()
    torch.save({
        "vocab_to_idx": vocab.token_to_idx,
        "idx_to_token": vocab.idx_to_token,
        "embeddings": weights
    }, save_path)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    net, vocab = train_word2vec()
    save_embeddings(net, vocab)
