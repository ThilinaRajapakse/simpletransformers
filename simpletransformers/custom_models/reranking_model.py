import torch
from torch import nn
from transformers import AutoModel, BertModel
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration,
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from torch.nn import CrossEntropyLoss


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


class EET5(T5ForConditionalGeneration):
    def __init__(self, config, max_seq_length=512):
        super().__init__(config)
        self.__delattr__("encoder")

        self.external_embedding_projection = nn.Linear(1536, config.hidden_size)
        self.max_seq_length = max_seq_length

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            raise ValueError("encoder_outputs must be passed for EET5.")

        # encoder_outputs is going to be (batch_size, 1536)
        # We need to project it to (batch_size, 1, 768) and then repeat it to (batch_size, max_length, 768)
        encoder_outputs = self.external_embedding_projection(encoder_outputs)
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat_interleave(
            self.max_seq_length, dim=1
        )

        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _tie_weights(self):
        pass

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor, model_kwargs, model_input_name
    ):
        return model_kwargs
