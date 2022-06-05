import torch
from torch import nn

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    QuestionAnsweringModelOutput,
    ModelOutput,
)
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
)
from transformers.models.dpr.modeling_dpr import (
    DPRPretrainedContextEncoder,
    DPREncoder,
    DPRContextEncoderOutput,
)


class PretrainRetrievalContextEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.start_idx_prediction_head = nn.Linear(config.hidden_size, 1)

        self.pooler = BertPooler(config)
        self.length_prediction_head = nn.Linear(config.hidden_size, 1)
        self.span_selector = STESpanSelect()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        indexing=False,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if indexing:
            if return_dict:
                representation_outputs = DPRContextEncoderOutput(
                    pooler_output=sequence_output[:, 0, :],
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                representation_outputs = outputs[1:]
            return representation_outputs

        logits = self.start_idx_prediction_head(sequence_output)
        start_logits = logits.squeeze(-1).contiguous()

        pooled_output = self.pooler(sequence_output)
        span_lengths = (
            self.length_prediction_head(pooled_output).squeeze(-1).contiguous()
        )

        (
            query_input_ids,
            query_attention_mask,
            average_span_length,
        ) = self.span_selector.apply(
            start_logits, span_lengths, input_ids, attention_mask
        )

        if not return_dict:
            representation_outputs = outputs[1:]
            span_prediction_outputs = (
                query_input_ids,
                query_attention_mask,
                span_lengths,
                average_span_length,
            )
            return representation_outputs, span_prediction_outputs

        representation_outputs = DPRContextEncoderOutput(
            pooler_output=sequence_output[:, 0, :],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        span_prediction_outputs = SpanPredictionLayerOutput(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            span_lengths=span_lengths,
            average_span_length=average_span_length,
        )

        return representation_outputs, span_prediction_outputs


class STESpanSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, start_logits, span_lengths, input_ids, attention_mask):
        ctx.save_for_backward(start_logits, span_lengths)
        start_positions = start_logits.argmax(dim=-1)
        span_lengths = span_lengths.clamp(min=1)
        average_span_length = torch.mean(span_lengths)
        span_lengths = span_lengths.int()

        query_inputs = {
            "input_ids": torch.zeros_like(input_ids),
            "attention_mask": torch.zeros_like(attention_mask),
        }

        # query_inputs.input_ids are the spans selected from the context_inputs.input_ids
        for i in range(len(start_positions)):
            selected_span = input_ids[
                i,
                start_positions[i] : torch.clamp(
                    start_positions[i] + span_lengths[i], max=500
                ),
            ]
            query_inputs["input_ids"][i][: len(selected_span)] = selected_span
            query_inputs["attention_mask"][i][: len(selected_span)] = 1

        return (
            query_inputs["input_ids"],
            query_inputs["attention_mask"],
            average_span_length,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class SpanPredictionLayerOutput(ModelOutput):
    query_input_ids = None
    query_attention_mask = None
    span_lengths = None
    average_span_length = None
