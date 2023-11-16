import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel


class RerankingModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = AutoModel.from_config(config)
        self.reranking_head = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs_embeds, attention_mask, token_type_ids, **kwargs):
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        outputs = outputs.last_hidden_state
        # Ignoring the batch dimension, we have:
        # outputs[0].shape = (sequence_length, hidden_size)
        # The actual sequence length comes from the attention mask
        sequence_length = attention_mask.sum(dim=1)[0]
        batch_size = attention_mask.shape[0]
        hidden_dim = outputs[0].shape[1]

        # Truncate outputs to the actual sequence length
        outputs = outputs[:, :sequence_length, :]

        # Step 1: Separate query and document vectors
        query_vectors = outputs[:, 0, :]  # Shape: (batch_size, hidden_dim)
        document_vectors = outputs[
            :, 1:, :
        ]  # Shape: (batch_size, sequence_length - 1, hidden_dim)

        # Step 2: Repeat query vectors
        query_vectors_repeated = query_vectors.repeat_interleave(
            sequence_length - 1, dim=0
        )  # Shape: (batch_size*(sequence_length - 1), hidden_dim)

        # Step 3: Reshape document vectors
        # Use .reshape() instead of .view() to handle non-contiguous memory layout
        document_vectors_reshaped = document_vectors.reshape(
            -1, hidden_dim
        )  # Shape: (batch_size*(sequence_length - 1), hidden_dim)

        # Step 4: Concatenate query and document vectors
        concatenated = torch.cat(
            (query_vectors_repeated, document_vectors_reshaped), dim=1
        )  # Shape: (batch_size*(sequence_length - 1), hidden_dim*2)

        print(concatenated.shape)

        pass
