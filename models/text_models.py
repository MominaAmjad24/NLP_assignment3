import torch
import torch.nn as nn

class TextRNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        model_type="lstm",
        use_one_hot=False,
        pretrained_embedding_layer=None
    ):
        super().__init__()
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size

        if use_one_hot:
            self.embedding = None
            rnn_input_dim = vocab_size
        else:
            if pretrained_embedding_layer is not None:
                self.embedding = pretrained_embedding_layer
            else:
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
            rnn_input_dim = embedding_dim

        if model_type.lower() == "lstm":
            self.rnn = nn.LSTM(rnn_input_dim, hidden_dim, batch_first=True)
        elif model_type.lower() == "gru":
            self.rnn = nn.GRU(rnn_input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        if self.use_one_hot:
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        else:
            x = self.embedding(x)

        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output
