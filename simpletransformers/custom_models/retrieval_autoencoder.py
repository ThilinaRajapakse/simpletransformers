import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, vector_count=256):
        super(Autoencoder, self).__init__()
        self.vector_count = vector_count
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.vector_count * 768, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 768),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(True),
            nn.Linear(2048, self.vector_count * 768),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        decoded_x = decoded_x.view(x.size(0), self.vector_count, 768)

        return encoded_x, decoded_x

    def encode(self, x):
        x = x.view(x.size(0), -1)
        encoded_x = self.encoder(x)
        return encoded_x

    def decode(self, x):
        decoded_x = self.decoder(x)
        decoded_x = decoded_x.view(x.size(0), self.vector_count, 768)
        return decoded_x
