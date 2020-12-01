from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel


# supports both BERT & ROBERTA BASED MODELS
class GPT2ForTextRepresentation(GPT2PreTrainedModel):
    r"""
   Outputs: `List` of token vectors, 1 list of max_seq token vectors per sentence given
    """  # noqa: ignore flake8"

    def __init__(self, config, weight=None):
        super(GPT2ForTextRepresentation, self).__init__(config)
        self.gpt2 = GPT2Model(config)
        self.weight = weight
        self.init_weights()

    def resize_token_embeddings(self, new_len):
        return self.gpt2.resize_token_embeddings(new_len)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.gpt2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        return hidden_states[-1]
