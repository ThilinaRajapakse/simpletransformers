import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, max_position_embeddings=256, hidden_size=768):
        super(Autoencoder, self).__init__()
        self.vector_count = max_position_embeddings
        self.hidden_size = hidden_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.vector_count * hidden_size, 2048),
            nn.ReLU(True),
            nn.Linear(2048, hidden_size),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(True),
            nn.Linear(2048, self.vector_count * hidden_size),
        )

        self.init_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        decoded_x = decoded_x.view(x.size(0), self.vector_count, self.hidden_size)

        return encoded_x, decoded_x

    def encode(self, x):
        x = x.view(x.size(0), -1)
        encoded_x = self.encoder(x)
        return encoded_x

    def decode(self, x):
        decoded_x = self.decoder(x)
        decoded_x = decoded_x.view(x.size(0), self.vector_count, self.hidden_size)
        return decoded_x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
