from pathlib import Path

import torch
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def generate_translations(lns, output_file_path, model_size, batch_size, device):
    

    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("translation_en_to_de", {}))

    with Path(output_file_path).open("w") as output_file:
        for batch in tqdm(list(chunks(lns, batch_size))):
            batch = [model.config.prefix + text for text in batch]

            dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)

            input_ids = dct["input_ids"].to(device)
            attention_mask = dct["attention_mask"].to(device)

            translations = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            dec = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translations
            ]

            for hypothesis in dec:
                output_file.write(hypothesis + "\n")


class T5Model:
    def __init__(self, model_name, use_cuda=True, cuda_device=-1):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        if use_cuda:
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

        self.model.to(self.device)
        print(self.model)

    def generate_translations(self, input_path, output_file_path="data/translated.txt", batch_size=8):
        dash_pattern = (" ##AT##-##AT## ", "-")

        # Read input lines into python
        with open(input_path, "r") as input_file:
            lns = [x.strip().replace(dash_pattern[0], dash_pattern[1]) for x in input_file.readlines()]

        # update config with summarization specific params
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None:
            self.model.config.update(task_specific_params.get("translation_en_to_de", {}))

        with Path(output_file_path).open("w") as output_file:
            for batch in tqdm(list(chunks(lns, batch_size))):
                batch = [self.model.config.prefix + text for text in batch]

                dct = self.tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)

                input_ids = dct["input_ids"].to(self.device)
                attention_mask = dct["attention_mask"].to(self.device)

                translations = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                dec = [
                    self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translations
                ]

                for hypothesis in dec:
                    output_file.write(hypothesis + "\n")

    def predict(self, to_predict):
        input_ids = self.tokenizer.encode(to_predict, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.model.generate(input_ids=input_ids)

        # output_ids = self.clean_decodes(torch.argmax(outputs[0], dim=1).tolist(), self.config.vocab_size)

        print([self.tokenizer.convert_ids_to_tokens(output_id) for output_id in outputs])

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
                break
            ret.append(int(i))
        return ret

    def _move_model_to_device(self):
        self.model.to(self.device)

