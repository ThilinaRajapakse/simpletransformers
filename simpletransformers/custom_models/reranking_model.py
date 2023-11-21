import torch
from torch import nn
from transformers import AutoModel, BertModel


class RerankingModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.reranking_head = nn.Linear(config.hidden_size * 4, 1)
        self.init_weights()

    def forward(self, inputs_embeds, attention_mask, token_type_ids, **kwargs):
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, inputs_embeds.size()[:-1]
        )
        outputs = self.encoder(
            hidden_states=inputs_embeds,
            attention_mask=extended_attention_mask,
        )

        outputs = outputs.last_hidden_state
        sequence_length = attention_mask.sum(dim=1)[0]
        batch_size = attention_mask.shape[0]
        hidden_dim = outputs[0].shape[1]

        # Truncate outputs to the actual sequence length
        outputs = outputs[:, :sequence_length, :]

        # Separate query and document vectors
        query_vectors = outputs[:, 0, :]
        document_vectors = outputs[:, 1:, :]

        # Repeat query vectors
        query_vectors_repeated = query_vectors.repeat_interleave(
            sequence_length - 1, dim=0
        )

        # Reshape document vectors
        document_vectors_reshaped = document_vectors.reshape(-1, hidden_dim)

        # Concatenate query and document vectors
        concatenated = torch.cat(
            (query_vectors_repeated, document_vectors_reshaped), dim=1
        )

        # Skip connection: Concatenate original query and document vectors
        query_vectors_expanded = (
            query_vectors.unsqueeze(1)
            .expand(-1, sequence_length - 1, -1)
            .reshape(-1, hidden_dim)
        )
        document_vectors_flattened = document_vectors.reshape(-1, hidden_dim)
        skip_concatenated = torch.cat(
            (query_vectors_expanded, document_vectors_flattened), dim=1
        )

        # Final concatenation for the reranking head input
        reranking_input = torch.cat((concatenated, skip_concatenated), dim=1)

        # Get logits from the reranking head
        logits = self.reranking_head(reranking_input)

        # Reshape logits to get a score for each document
        logits = logits.reshape(batch_size, sequence_length - 1)

        return logits
