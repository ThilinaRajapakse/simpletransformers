from abc import ABC

from torch import nn
from transformers import ElectraForPreTraining, LongformerModel
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions, ElectraGeneratorPredictions, \
    ElectraForMaskedLM


class LongformerElectraForPreTraining(ElectraForPreTraining, ABC):
    def __init__(self, config):
        super().__init__(config)

        self.electra = LongformerModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        # Initialize weights and apply final processing
        self.post_init()


class LongformerElectraForMaskedLM(ElectraForMaskedLM, ABC):
    def __init__(self, config):
        super().__init__(config)

        self.electra = LongformerModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)

        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        # Initialize weights and apply final processing
        self.post_init()
