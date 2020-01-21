# from simpletransformers.classification import SentPairClassificationModel
# import pandas as pd
# import sys
#
# # from pyspark.sql import SparkSession
# # spark = SparkSession\
# # .builder\
# # .getOrCreate()
# #
# # print("Testing simple count")
# # # The Spark code will execute on the Databricks cluster.
# # print(spark.range(100).count())
# #
# # spark.udf('train.py')
#
#
#
# # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
# train_data = [["I'm\n\nExample sentence 1 \n\nfor multilabel\nclassification.".lower(),
#                "This is example sentence 2".lower(),
#                1]] + \
#              [['This is another example sentence. '.lower(),
#                'This is hhhhhhhh'.lower(),
#                0]]
#
# train_df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])
#
# eval_data = [['Example eval sentence for multilabel classification.'.lower(),
#               'what is this sentence'.lower(),
#               1],
#              ['Example eval senntence belonging to class 2'.lower(),
#               'this is another sentence.'.lower(),
#               0]]
#
# eval_df = pd.DataFrame(eval_data, columns=['text_a', 'text_b', 'labels'])
#
# # Create a MultiLabelClassificationModel
# model = SentPairClassificationModel('bert', 'bert-base-uncased', num_labels=2, args={
#     'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5,
#     'evaluate_during_training': True, 'evaluate_during_training_steps': 1,
#     }, use_cuda=False)
# # You can set class weights by using the optional weight argument
# print(train_df.head())
#
# # Train the model
# # model.train_model(train_df)
# model.train_model(train_df, eval_df=eval_df)
#
# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
# print(result)
# print(model_outputs)
#
# predictions, raw_outputs = model.predict([['This thing is entirely different from the other thing. ', 'yet another'],
#                                           ['This thing is entirely different from the rest thing. ', 'I am dead another']])
# print(predictions)
# print(raw_outputs)


from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn


train_data = [
    ['Example sentence belonging to class 1', 'Yep, this is 1', 1],
    ['Example sentence belonging to class 0', 'Yep, this is 0', 0],
    ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 0]
]

train_df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])

eval_data = [
    ['Example sentence belonging to class 1', 'Yep, this is 1', 1],
    ['Example sentence belonging to class 0', 'Yep, this is 0', 0],
    ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 0]
]

eval_df = pd.DataFrame(eval_data, columns=['text_a', 'text_b', 'labels'])

train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,
}

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False, cuda_device=0, args=train_args)
print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)

predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
print(predictions)
print(raw_outputs)

model = ClassificationModel('bert', './outputs', use_cuda=False)
