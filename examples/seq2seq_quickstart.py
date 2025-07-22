from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd

# Sample data: English to French
train_data = [
    ["translate English to French: Hello, how are you?", "Bonjour, comment ça va ?"],
    ["translate English to French: What is your name?", "Comment tu t'appelles ?"],
]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

# Define model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 2
model_args.train_batch_size = 2
model_args.max_seq_length = 64
model_args.eval_batch_size = 2
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.use_multiprocessing = False
model_args.no_cache = True
model_args.overwrite_output_dir = True

model = Seq2SeqModel(
    encoder_decoder_type="t5",
    encoder_decoder_name="t5-small",
    args=model_args,
    use_cuda=False  # Change to True if you have a GPU
)

# Train the model
model.train_model(train_df)

# Predict
inputs = ["translate English to French: I am learning AI."]
predictions = model.predict(inputs)
print(predictions)
