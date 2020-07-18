from transformers.modeling_bert import BertModel, BertPreTrainedModel


# supports both BERT & ROBERTA BASED MODELS
class BertForTextRepresentation(BertPreTrainedModel):
    r"""
   Outputs: `List` of token vectors, 1 list of max_seq token vectors per sentence given
    """  # noqa: ignore flake8"

    def __init__(self, config, weight=None):
        super(BertForTextRepresentation, self).__init__(config)
        self.bert = BertModel(config)
        self.weight = weight
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        return hidden_states[-1]
