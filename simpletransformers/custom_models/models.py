import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    DistilBertModel,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    FlaubertModel,
    RobertaModel,
    XLMModel,
    XLMPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
)
from transformers.configuration_distilbert import DistilBertConfig
from transformers.configuration_roberta import RobertaConfig
from transformers.configuration_xlm_roberta import XLMRobertaConfig
from transformers.modeling_albert import AlbertConfig, AlbertModel, AlbertPreTrainedModel
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.modeling_electra import (
    ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaClassificationHead
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.modeling_roberta import RobertaForQuestionAnswering


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

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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


class RobertaForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Roberta model adapted for multi-label sequence classification
    """

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, pos_weight=None):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
        )
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

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
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
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

    def forward(
        self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None,
    ):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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


class FlaubertForMultiLabelSequenceClassification(FlaubertModel):
    """
    Flaubert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(FlaubertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = FlaubertModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLMRobertaForMultiLabelSequenceClassification(RobertaForMultiLabelSequenceClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super(ElectraForLanguageModelingModel, self).__init__(config, **kwargs)
        if "generator_config" in kwargs:
            generator_config = kwargs["generator_config"]
        else:
            generator_config = config
        self.generator_model = ElectraForMaskedLM(generator_config)
        if "discriminator_config" in kwargs:
            discriminator_config = kwargs["discriminator_config"]
        else:
            discriminator_config = config
        self.discriminator_model = ElectraForPreTraining(discriminator_config)
        self.vocab_size = generator_config.vocab_size
        if kwargs.get("tie_generator_and_discriminator_embeddings", True):
            self.tie_generator_and_discriminator_embeddings()

    def tie_generator_and_discriminator_embeddings(self):
        gen_embeddings = self.generator_model.electra.embeddings
        disc_embeddings = self.discriminator_model.electra.embeddings

        # tie word, position and token_type embeddings
        gen_embeddings.word_embeddings.weight = disc_embeddings.word_embeddings.weight
        gen_embeddings.position_embeddings.weight = disc_embeddings.position_embeddings.weight
        gen_embeddings.token_type_embeddings.weight = disc_embeddings.token_type_embeddings.weight

    def forward(self, inputs, masked_lm_labels, attention_mask=None, token_type_ids=None):
        d_inputs = inputs.clone()

        # run masked LM.
        g_out = self.generator_model(
            inputs, masked_lm_labels=masked_lm_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # get samples from masked LM.
        sample_probs = torch.softmax(g_out[1], dim=-1, dtype=torch.float32)
        sample_probs = sample_probs.view(-1, self.vocab_size)

        sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
        sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

        # labels have a -100 value to mask out loss from unchanged tokens.
        mask = masked_lm_labels.ne(-100)

        # replace the masked out tokens of the input with the generator predictions.
        d_inputs[mask] = sampled_tokens[mask]

        # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
        # if the prediction was correct, mark it as uncorrupted.
        correct_preds = sampled_tokens == masked_lm_labels
        d_labels = mask.long()
        d_labels[correct_preds] = 0

        # run token classification, predict whether each token was corrupted.
        d_out = self.discriminator_model(
            d_inputs, labels=d_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        g_loss = g_out[0]
        d_loss = d_out[0]
        g_scores = g_out[1]
        d_scores = d_out[1]
        return g_loss, d_loss, g_scores, d_scores, d_labels


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    r"""
    Mostly the ssame as BertForSequenceClassification. A notable difference is that this class contains a pooler while
    BertForSequenceClassification doesn't. This is because pooling happens internally in a BertModel but not in an
    ElectraModel.
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
    """  # noqa
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.weight = weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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


class ElectraForMultiLabelSequenceClassification(ElectraPreTrainedModel):
    """
    ElectraForSequenceClassification model adapted for multi-label sequence classification
    """

    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, pos_weight=None):
        super(ElectraForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    """
    Identical to BertForQuestionAnswering other than using an ElectraModel
    """

    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
