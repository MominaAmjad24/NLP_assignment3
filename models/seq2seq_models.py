import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, model_type="lstm", use_one_hot=False, embedding_layer=None):
        super().__init__()
        self.use_one_hot = use_one_hot
        self.input_dim = input_dim

        if use_one_hot:
            self.embedding = None
            rnn_input_dim = input_dim
        else:
            self.embedding = embedding_layer if embedding_layer is not None else nn.Embedding(input_dim, emb_dim)
            rnn_input_dim = emb_dim

        if model_type == "lstm":
            self.rnn = nn.LSTM(rnn_input_dim, hid_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(rnn_input_dim, hid_dim, batch_first=True)

        self.model_type = model_type

    def forward(self, src):
        if self.use_one_hot:
            embedded = torch.nn.functional.one_hot(src, num_classes=self.input_dim).float()
        else:
            embedded = self.embedding(src)

        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, model_type="lstm", use_one_hot=False, embedding_layer=None):
        super().__init__()
        self.use_one_hot = use_one_hot
        self.output_dim = output_dim

        if use_one_hot:
            self.embedding = None
            rnn_input_dim = output_dim
        else:
            self.embedding = embedding_layer if embedding_layer is not None else nn.Embedding(output_dim, emb_dim)
            rnn_input_dim = emb_dim

        if model_type == "lstm":
            self.rnn = nn.LSTM(rnn_input_dim, hid_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(rnn_input_dim, hid_dim, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.model_type = model_type

    def forward(self, input_token, hidden):
        input_token = input_token.unsqueeze(1)

        if self.use_one_hot:
            embedded = torch.nn.functional.one_hot(input_token, num_classes=self.output_dim).float()
        else:
            embedded = self.embedding(input_token)

        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        hidden = self.encoder(src)
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs
