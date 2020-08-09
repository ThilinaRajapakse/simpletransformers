import streamlit as st
import pandas as pd

from simpletransformers.streamlit.streamlit_utils import get


ENTITY_WRAPPER = (
    """<mark style="background: {}; border-radius: 0.25rem; padding: 0.25rem; display: inline-block">{} {}</mark>"""
)
ENTITY_LABEL_WRAPPER = """<span style="background: #fff; font-size: 0.56em; font-weight: bold; padding: 0.3em 0.3em; vertical-align: middle; margin: 0 0 0.15rem 0.5rem; line-height: 1; display: inline-block">{}</span>"""


def format_word(word, entity, entity_checkboxes, entity_color_map):
    if entity_checkboxes[entity]:
        return ENTITY_WRAPPER.format(entity_color_map[entity], word, ENTITY_LABEL_WRAPPER.format(entity))
    else:
        return word


def ner_viewer(model):
    session_state = get(
        max_seq_length=model.args.max_seq_length,
    )
    model.args.max_seq_length = session_state.max_seq_length

    entity_list = model.args.labels_list

    st.sidebar.subheader("Entities")
    entity_checkboxes = {entity: st.sidebar.checkbox(entity, value=True) for entity in entity_list}
    entity_color_map = {entity: "#a6e22d" for entity in entity_list}

    st.sidebar.subheader("Parameters")
    model.args.max_seq_length = st.sidebar.slider(
        "Max Seq Length", min_value=1, max_value=512, value=model.args.max_seq_length
    )

    st.subheader("Enter text: ")
    input_text = st.text_area("")

    predictions, _ = model.predict([input_text])
    prediction = predictions[0]

    to_write = " ".join([format_word(word, entity, entity_checkboxes, entity_color_map) for pred in prediction for word, entity in pred.items()])

    st.subheader(f"Predictions")
    st.write(to_write, unsafe_allow_html=True)
