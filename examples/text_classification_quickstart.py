from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Sample training data
train_data = [
    ["I love this product!", 1],
    ["This is the worst movie ever", 0],
    ["Such a fantastic experience!", 1],
    ["I hate this so much", 0],
]
train_df = pd.DataFrame(train_data, columns=["text", "labels"])

# Sample eval data
eval_data = [
    ["I enjoyed it", 1],
    ["Terrible service", 0],
]
eval_df = pd.DataFrame(eval_data, columns=["text", "labels"])

# Model arguments
model_args = ClassificationArgs()
model_args.num_train_epochs = 1
model_args.overwrite_output_dir = True
model_args.output_dir = "outputs/"

# Create a ClassificationModel
model = ClassificationModel(
    model_type="distilbert",
    model_name="distilbert-base-uncased",
    args=model_args,
    use_cuda=False
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print("Evaluation:", result)

# Make predictions
predictions, raw_outputs = model.predict(["I love it!", "I hate it!"])
print("Predictions:", predictions)
