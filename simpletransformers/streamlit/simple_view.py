import os
import json
import logging
from dataclasses import asdict

import streamlit as st
import pandas as pd
from streamlit import table
from scipy.special import softmax
from itertools import chain
import numpy as np

from simpletransformers.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
    ClassificationArgs,
    MultiLabelClassificationArgs,
)
from simpletransformers.ner import NERModel
from simpletransformers.question_answering import QuestionAnsweringModel
from simpletransformers.t5 import T5Model
from simpletransformers.seq2seq import Seq2SeqModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

prediction_history = []
raw_history = []
probablity_history = []

model_class_map = {
    "ClassificationModel": "Classification Model",
    "MultiLabelClassificationModel": "Multi-Label Classification Model",
    "QuestionAnsweringModel": "Question Answering Model",
}


@st.cache(allow_output_mutation=True)
def load_model(
    selected_dir=None,
    model_class=None,
    model_type=None,
    model_name=None,
    num_labels=None,
    weight=None,
    args=None,
    use_cuda=True,
    cuda_device=-1,
    **kwargs,
):
    if not (model_class and model_type and model_name):
        try:
            with open(os.path.join(selected_dir, "model_args.json"), "r") as f:
                model_args = json.load(f)
            model_class = model_args["model_class"]
            model_type = model_args["model_type"]
            model_name = selected_dir
        except KeyError as e:
            raise KeyError(
                "model_class and/or model_type keys missing in {}."
                "If this model was created with Simple Transformers<0.46.0, "
                "the model must be loaded by specifying model_class, model_type, and model_name".format(
                    os.path.join(selected_dir, "model_args.json")
                )
            ) from e
    model = create_model(
        model_class, model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs
    )
    return model, model_class


def create_model(model_class, model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs):
    if model_class == "ClassificationModel":
        return ClassificationModel(model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs)
    elif model_class == "MultiLabelClassificationModel":
        return MultiLabelClassificationModel(
            model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs
        )
    elif model_class == "QuestionAnsweringModel":
        return QuestionAnsweringModel(model_type, model_name, args, use_cuda, cuda_device, **kwargs)
    else:
        raise ValueError("{} is either invalid or not yet implemented.".format(model_class))


def find_all_models(current_dir, model_list):
    for directory in os.listdir(current_dir):
        if os.path.isdir(os.path.join(current_dir, directory)):
            model_list = find_all_models(os.path.join(current_dir, directory), model_list)
    if os.path.isfile(os.path.join(current_dir, "model_args.json")):
        with open(os.path.join(current_dir, "model_args.json"), "r") as f:
            model_args = json.load(f)
        if "model_type" in model_args and "model_class" in model_args:
            model_list.append(model_args["model_class"] + ":- " + current_dir)
    return model_list


def streamlit_runner(
    selected_dir=None,
    model_class=None,
    model_type=None,
    model_name=None,
    num_labels=None,
    weight=None,
    args=None,
    use_cuda=True,
    cuda_device=-1,
    **kwargs,
):
    if not (model_class and model_type and model_name):
        model_list = find_all_models(".", [])
        selected_dir = st.sidebar.selectbox("Choose Model", model_list)
        if selected_dir:
            selected_dir = selected_dir.split(":- ")[-1]
        else:
            st.subheader("No models found in current directory.")
            st.markdown('''
            Simple Viewer looked everywhere in this directory and subdirectories but didn't find any Simple Transformers models. :(

            If you are trying to load models saved with an older Simple Transformers version, make sure the `model_args.json` file
            contains the `model_class`, `model_type`, and `model_name`.

            Or, you can write a Python script like the one below and save it to `view.py`.

            ```python
            from simpletransformers.streamlit.simple_view import streamlit_runner


            streamlit_runner(model_class="ClassificationModel", model_type="distilbert", model_name="outputs")

            ```

            You can execute this with `streamlit run view.py`.

            The `streamlit_runner()` function accepts all the same arguments as the corresponding Simple Transformers model.
            ''')
            return

    model, model_class = load_model(
        selected_dir, model_class, model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs
    )
    model.args.use_multiprocessing = False

    global raw_history

    st.title("Simple Transformers Viewer")
    st.markdown("---")
    st.header(model_class_map[model_class])

    if model_class in ["ClassificationModel", "MultiLabelClassificationModel"]:
        st.subheader("Enter text to perform a prediction: ")
        input_text = st.text_area("")

        if model_class == "ClassificationModel":
            max_seq_length = st.sidebar.number_input(
                'Max Seq Length',
                min_value=1, max_value=512, value=model.args.max_seq_length
            )
            if max_seq_length != model.args.max_seq_length:
                model.args.max_seq_length = max_seq_length

            sliding_window = st.sidebar.radio("Sliding Window", ("Enable", "Disable"), index=1)
            if sliding_window == "Enable":
                model.args.sliding_window = True
            else:
                model.args.sliding_window = False

        if input_text:
            prediction, raw_values = model.predict([input_text])
            raw_values = [list(np.squeeze(raw_values))]

            if model.args.sliding_window:
                raw_values = np.mean(raw_values, axis=1)

            st.subheader(f"Predictions")
            st.text(f"Predicted label: {prediction[0]}")
            prediction_history.append([input_text, prediction[0]])

            st.subheader(f"Model outputs")
            st.text("Raw values: ")
            try:
                raw_df = pd.DataFrame(raw_values, columns=[f"Label {label}" for label in model.args.labels_list])
            except Exception:
                raw_df = pd.DataFrame(raw_values, columns=[f"Label {label}" for label in range(len(raw_values[0]))])
            st.dataframe(raw_df)
            raw_history.append(list(chain([input_text], raw_values[0])))

            st.text("Probabilities: ")
            try:
                raw_df = pd.DataFrame(
                    softmax(raw_values, axis=1), columns=[f"Label {label}" for label in model.args.labels_list]
                )
            except Exception:
                raw_df = pd.DataFrame(
                    softmax(raw_values, axis=1), columns=[f"Label {i}" for i in range(len(raw_values[0]))]
                )
            st.dataframe(raw_df)
            probablity_history.append(list(chain([input_text], softmax(raw_values)[0])))

            st.markdown("---")
            st.header("History")

            st.subheader("Prediction History")
            history_df = pd.DataFrame(prediction_history, columns=["Text", "Predicted Label"])
            st.dataframe(history_df)

            st.subheader("Model Output History")
            st.text("Raw values: ")
            try:
                raw_history_df = pd.DataFrame(
                    raw_history, columns=["Text"] + [f"Label {label}" for label in model.args.labels_list]
                )
            except Exception:
                raw_history_df = pd.DataFrame(
                    raw_history, columns=["Text"] + [f"Label {i}" for i in range(len(raw_values[0]))]
                )
            st.dataframe(raw_history_df)

            st.text("Probabilities: ")
            try:
                probablity_history_df = pd.DataFrame(
                    probablity_history, columns=["Text"] + [f"Label {label}" for label in model.args.labels_list]
                )
            except Exception:
                probablity_history_df = pd.DataFrame(
                    probablity_history, columns=["Text"] + [f"Label {i}" for i in range(len(raw_values[0]))]
                )
            st.dataframe(probablity_history_df)
    elif model_class == "QuestionAnsweringModel":
        st.subheader("Enter context: ")
        context_text = st.text_area("", key="context")

        st.subheader("Enter question: ")
        question_text = st.text_area("", key="question")

        n_best_size = st.sidebar.slider(
            'Number of answers to generate',
            min_value=1, max_value=20
        )

        model.args.n_best_size = n_best_size

        if context_text and question_text:
            to_predict = [
                {
                    "context": context_text,
                    "qas": [
                        {
                            "id": 0,
                            "question": question_text,
                        }
                    ]
                }
            ]

            answers, probabilities = model.predict(to_predict)

            st.subheader(f"Predictions")
            answers = answers[0]["answer"]
            probabilities = probabilities[0]["probability"]

            # outputs = [list(chain(*zip(answers, probabilities)))]
            # columns = list(chain(*zip([f"Answer {i + 1}" for i in range(len(answers))], [f"Confidence {i + 1}" for i in range(len(probabilities))])))

            # output_df = pd.DataFrame(outputs, columns=columns)

            output_df = pd.DataFrame({"Answer": answers, "Confidence": probabilities})
            st.dataframe(output_df)
