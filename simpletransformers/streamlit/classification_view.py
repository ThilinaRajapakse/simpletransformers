import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import softmax

from simpletransformers.classification import (
    ClassificationModel,
    MultiLabelClassificationModel,
)
from simpletransformers.streamlit.streamlit_utils import get, simple_transformers_model


def get_states(model, session_state=None):
    if session_state:
        setattr(session_state, "sliding_window", model.args.sliding_window)
        setattr(session_state, "stride", model.args.stride)
    else:
        session_state = get(
            max_seq_length=model.args.max_seq_length,
            sliding_window=model.args.sliding_window,
            stride=model.args.stride,
        )
    if session_state.sliding_window == "Enable":
        model.args.sliding_window = True
    else:
        model.args.sliding_window = False

    model.args.max_seq_length = session_state.max_seq_length
    model.args.stride = session_state.stride

    return session_state, model


@st.cache(
    hash_funcs={
        ClassificationModel: simple_transformers_model,
        MultiLabelClassificationModel: simple_transformers_model,
    }
)
def get_prediction(model, input_text):
    prediction, raw_values = model.predict([input_text])

    return prediction, raw_values


def classification_viewer(model, model_class):
    st.subheader("Enter text: ")
    input_text = st.text_area("")
    st.sidebar.subheader("Parameters")

    if model_class == "ClassificationModel":
        try:
            session_state, model = get_states(model)
        except AttributeError:
            session_state = get(
                max_seq_length=model.args.max_seq_length,
                sliding_window=model.args.sliding_window,
                stride=model.args.stride,
            )
            session_state, model = get_states(model, session_state)

        model.args.max_seq_length = st.sidebar.slider(
            "Max Seq Length",
            min_value=1,
            max_value=512,
            value=model.args.max_seq_length,
        )

        sliding_window = st.sidebar.radio(
            "Sliding Window",
            ("Enable", "Disable"),
            index=0 if model.args.sliding_window else 1,
        )
        if sliding_window == "Enable":
            model.args.sliding_window = True
        else:
            model.args.sliding_window = False

        if model.args.sliding_window:
            model.args.stride = st.sidebar.slider(
                "Stride (Fraction of Max Seq Length)",
                min_value=0.0,
                max_value=1.0,
                value=model.args.stride,
            )
    elif model_class == "MultiLabelClassificationModel":
        try:
            session_state, model = get_states(model)
        except AttributeError:
            session_state = get(max_seq_length=model.args.max_seq_length,)
            session_state, model = get_states(model, session_state)

        model.args.max_seq_length = st.sidebar.slider(
            "Max Seq Length",
            min_value=1,
            max_value=512,
            value=model.args.max_seq_length,
        )

    if input_text:
        prediction, raw_values = get_prediction(model, input_text)
        raw_values = [list(np.squeeze(raw_values))]

        if model.args.sliding_window and isinstance(raw_values[0][0], np.ndarray):
            raw_values = np.mean(raw_values, axis=1)

        st.subheader(f"Predictions")
        st.text(f"Predicted label: {prediction[0]}")

        st.subheader(f"Model outputs")
        st.text("Raw values: ")
        try:
            raw_df = pd.DataFrame(
                raw_values,
                columns=[f"Label {label}" for label in model.args.labels_list],
            )
        except Exception:
            raw_df = pd.DataFrame(
                raw_values,
                columns=[f"Label {label}" for label in range(len(raw_values[0]))],
            )
        st.dataframe(raw_df)

        st.text("Probabilities: ")
        try:
            prob_df = pd.DataFrame(
                softmax(raw_values, axis=1),
                columns=[f"Label {label}" for label in model.args.labels_list],
            )
        except Exception:
            prob_df = pd.DataFrame(
                softmax(raw_values, axis=1),
                columns=[f"Label {i}" for i in range(len(raw_values[0]))],
            )
        st.dataframe(prob_df)

    return model
