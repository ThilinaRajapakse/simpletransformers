import torch
from torch.utils.data import Sampler


class ChunkSampler(Sampler):
    def __init__(self, data_source, chunk_size, batch_size):
        assert (
            batch_size % chunk_size == 0
        ), "Batch size must be divisible by chunk size"
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size

    def __iter__(self):
        # Create a list of chunk indices
        chunk_indices_list = torch.randperm(self.num_chunks).tolist()

        # Create indices by chunk
        indices = []
        for chunk_idx in chunk_indices_list:
            start_idx = chunk_idx * self.chunk_size
            chunk_indices = list(range(start_idx, start_idx + self.chunk_size))
            indices.extend(chunk_indices)

        return iter(indices)

    def __len__(self):
        return len(self.data_source)
