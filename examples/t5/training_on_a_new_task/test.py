from pprint import pprint

import pandas as pd

from simpletransformers.t5 import T5Model

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "eval_batch_size": 128,
    "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "use_multiprocessing": False,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

model = T5Model("test_outputs_large/best_model", args=model_args)

df = pd.read_csv("data/eval_df.tsv", sep="\t").astype(str)
preds = model.predict(
    ["ask_question: " + description for description in df["input_text"].tolist()]
)

questions = df["target_text"].tolist()

with open("test_outputs_large/generated_questions.txt", "w") as f:
    for i, desc in enumerate(df["input_text"].tolist()):
        pprint(desc)
        pprint(preds[i])
        print()

        f.write(str(desc) + "\n\n")

        f.write("Real question:\n")
        f.write(questions[i] + "\n\n")

        f.write("Generated questions:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )
