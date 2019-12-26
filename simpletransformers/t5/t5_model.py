from transformers.modeling_t5 import *
from transformers.tokenization_t5 import T5Tokenizer
from transformers import T5Config
import torch


class T5Model:
    def __init__(self, model_name, use_cuda=True, cuda_device=-1):
        self.config = T5Config.from_pretrained(model_name)

        if False:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError("'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False.")
        else:
            self.device = "cpu"

        if not use_cuda:
            self.args['fp16'] = False

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model =  T5WithLMHeadModel.from_pretrained(model_name, config=self.config)
        print(self.model)


    def predict(self, to_predict):
        self._move_model_to_device()
        print(self.config.vocab_size)
        print(to_predict)
        input_ids = torch.tensor(self.tokenizer.encode(to_predict)).unsqueeze(0)
        outputs = self.model(input_ids=input_ids)[0]
        
        output_ids = self.clean_decodes(torch.argmax(outputs[0], dim=1).tolist(), self.config.vocab_size)

        print([self.tokenizer.convert_ids_to_tokens(output_id) for output_id in output_ids])


    def clean_decodes(self, ids, vocab_size, eos_id=1):
        """Stop at EOS or padding or OOV.
        Args:
            ids: a list of integers
            vocab_size: an integer
            eos_id: EOS id
        Returns:
            a list of integers
        """
        ret = []
        for i in ids:
            if i == eos_id:
                break
            if i >= vocab_size:
                Sbreak
            ret.append(int(i))
        return ret


    def _move_model_to_device(self):
        self.model.to(self.device)

