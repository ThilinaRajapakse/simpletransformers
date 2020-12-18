import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.mmbt.modeling_mmbt import MMBTModel


class MMBTForClassification(nn.Module):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        # For example purposes. Not runnable.
        transformer = BertModel.from_pretrained('bert-base-uncased')
        encoder = ImageEncoder(args)
        model = MMBTForClassification(config, transformer, encoder)
        outputs = model(input_modal, input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.num_labels = config.num_labels

        self.mmbt = MMBTModel(config, transformer, encoder)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.mmbt(
            input_modal=input_modal,
            input_ids=input_ids,
            modal_start_tokens=modal_start_tokens,
            modal_end_tokens=modal_end_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            modal_token_type_ids=modal_token_type_ids,
            position_ids=position_ids,
            modal_position_ids=modal_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.weight is not None:
                    weight = self.weight.to(labels.device)
                else:
                    weight = None
                loss_fct = CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
