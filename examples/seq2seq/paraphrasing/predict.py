import logging

from simpletransformers.seq2seq import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name="outputs")


while True:
    original = input("Enter text to paraphrase: ")
    to_predict = [original]

    preds = model.predict(to_predict)

    print("---------------------------------------------------------")
    print(original)

    print()
    print("Predictions >>>")
    for pred in preds[0]:
        print(pred)

    print("---------------------------------------------------------")
    print()
