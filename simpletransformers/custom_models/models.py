import torch
from torch import nn

from transformers import BertPreTrainedModel, BertModel
from transformers import XLNetPreTrainedModel, XLNetModel
from transformers import XLMPreTrainedModel, XLMModel
from transformers import DistilBertModel
from transformers.configuration_distilbert import DistilBertConfig
from transformers.modeling_utils import SequenceSummary, PreTrainedModel
from transformers import RobertaModel
from transformers.configuration_roberta import RobertaConfig

from transformers.modeling_albert import AlbertConfig, AlbertPreTrainedModel, AlbertModel


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
}

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin",
    'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-pytorch_model.bin"
}

from torch.nn import BCEWithLogitsLoss


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """
    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Roberta model adapted for multi-label sequence classification
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, pos_weight=None):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLNetForMultiLabelSequenceClassification(XLNetPreTrainedModel):
    """
    XLNet model adapted for multi-label sequence classification
    """
    def __init__(self, config, pos_weight=None):
        super(XLNetForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask,
                                               head_mask=head_mask)
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLMForMultiLabelSequenceClassification(XLMPreTrainedModel):
    """
    XLM model adapted for multi-label sequence classification
    """
    def __init__(self, config, pos_weight=None):
        super(XLMForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None, position_ids=None,
                lengths=None, cache=None, head_mask=None, inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               langs=langs,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               lengths=lengths, 
                                               cache=cache,
                                               head_mask=head_mask)

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistilBertForMultiLabelSequenceClassification(DistilBertPreTrainedModel):
    """
    DistilBert model adapted for multi-label sequence classification
    """
    def __init__(self, config, pos_weight=None):
        super(DistilBertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]                    # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                    # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
        pooled_output = self.dropout(pooled_output)         # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class AlbertForMultiLabelSequenceClassification(AlbertPreTrainedModel):
    """
    Alber model adapted for multi-label sequence classification
    """
    def __init__(self, config, pos_weight=None):
        super(AlbertForMultiLabelSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
