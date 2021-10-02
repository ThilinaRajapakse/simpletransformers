from simpletransformers.t5 import T5Model

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "eval_batch_size": 16,
    "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "use_multiprocessing": False,
    # "silent": True,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

model = T5Model("t5", "outputs/best_model", args=model_args)

query = (
    "ask_question: "
    + """ANTIQUE CAST METAL 3 GLOBE CANDLABRA JADITE LAMP.

Stunning antique lamp with three candle style globes. Cast metal base with jadite green glass insert. Has been rewired with new braided cord. In excellent condition with only one chip (as pictured) on the edge of the glass insert. E9 69 on underside of metal base. Missing finial. New low wattage globes.
"""
)

preds = model.predict([query])

print(preds)
