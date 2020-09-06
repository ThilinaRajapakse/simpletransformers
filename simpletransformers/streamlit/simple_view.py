import os
import json
import logging

import streamlit as st
from torch.cuda import is_available

from simpletransformers.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from simpletransformers.ner import NERModel
from simpletransformers.question_answering import QuestionAnsweringModel
from simpletransformers.t5 import T5Model
from simpletransformers.seq2seq import Seq2SeqModel
from simpletransformers.streamlit.streamlit_utils import cache_on_button_press
from simpletransformers.streamlit.qa_view import qa_viewer
from simpletransformers.streamlit.classification_view import classification_viewer
from simpletransformers.streamlit.ner_view import ner_viewer


logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


model_class_map = {
    "ClassificationModel": "Classification Model",
    "MultiLabelClassificationModel": "Multi-Label Classification Model",
    "QuestionAnsweringModel": "Question Answering Model",
    "NERModel": "NER Model",
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
    elif model_class == "NERModel":
        return NERModel(model_type, model_name, args=args, use_cuda=use_cuda, cuda_device=cuda_device, **kwargs)
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


@cache_on_button_press("Load Model")
def manual_model_load(model_class, model_type, model_name):
    return model_class and model_type and model_name


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
    st.title("Simple Transformers Viewer")
    st.markdown("---")
    st.sidebar.subheader("CUDA")
    use_cuda = st.sidebar.checkbox("Use CUDA", value=is_available())
    st.sidebar.subheader("Models")
    info_text = st.empty()
    if not (model_class and model_type and model_name):
        model_list = find_all_models(".", [])
        selected_dir = st.sidebar.selectbox("Choose Model", model_list)
        if selected_dir:
            selected_dir = selected_dir.split(":- ")[-1]
        else:
            info_text.markdown(
                """
                ### No models found in current directory.

                Simple Viewer looked everywhere in this directory and subdirectories but didn't find any Simple Transformers models. :worried:

                If you are trying to load models saved with an older Simple Transformers version, make sure the `model_args.json` file
                contains the `model_class`, `model_type`, and `model_name`.

                Or, you can specify the model paths manually through the "Specify model manually" option on the sidebar.

                ---

                """  # noqa
            )

        manual_model = st.sidebar.checkbox("Specify model manually", value=False if selected_dir else True)
        if manual_model:
            st.sidebar.subheader("Model Details")
            fill_info = st.empty()
            fill_info.markdown("Please fill the Model details on the sidebar and click `Load Model`.")
            model_class = st.sidebar.selectbox("Simple Transformers task", list(model_class_map.keys()))
            model_type = st.sidebar.text_input("Model type (e.g. bert, roberta, xlnet)")
            model_name = st.sidebar.text_input("Model name (e.g. bert-base-cased, roberta-base)")

            if manual_model_load(model_class, model_type, model_name):
                selected_dir = None
                info_text.markdown("")
                fill_info.markdown("")

    model, model_class = load_model(
        selected_dir, model_class, model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs
    )
    model.args.use_multiprocessing = False

    st.header(model_class_map[model_class])

    if model_class in ["ClassificationModel", "MultiLabelClassificationModel"]:
        model = classification_viewer(model, model_class)
    elif model_class == "QuestionAnsweringModel":
        model = qa_viewer(model)
    elif model_class == "NERModel":
        model = ner_viewer(model)
