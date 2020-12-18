import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.camembert.modeling_camembert import (
    CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    CamembertConfig,
    CamembertModel,
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForSequenceClassification


class CamembertForSequenceClassification(RobertaForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
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
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        model = CamembertForSequenceClassification.from_pretrained('camembert-base')
        input_ids = torch.tensor(tokenizer.encode("J'aime le camembert !")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    config_class = CamembertConfig
    pretrained_model_archive_map = CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "camembert"

    def __init__(self, config, weight=None, sliding_window=False):
        super(CamembertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.camembert = CamembertModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.weight = weight
        self.sliding_window = sliding_window

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None
    ):
        all_outputs = []
        if self.sliding_window:
            # input_ids is really the list of inputs for each "sequence window"
            labels = input_ids[0]["labels"]
            for inputs in input_ids:
                ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
                outputs = self.camembert(
                    ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                )
                all_outputs.append(outputs[0])

            sequence_output = torch.mean(torch.stack(all_outputs), axis=0)
        else:
            outputs = self.camembert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
            )
            sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
