import streamlit as st
import pandas as pd

from simpletransformers.t5 import T5Model
from simpletransformers.streamlit.streamlit_utils import get, simple_transformers_model


def get_states(model, session_state=None):
    if session_state:
        setattr(session_state, "max_length", model.args.max_length)
        setattr(session_state, "decoding_algorithm", model.args.do_sample)
        setattr(session_state, "length_penalty", model.args.length_penalty)
        setattr(session_state, "num_beams", model.args.num_beams)
        setattr(session_state, "early_stopping", model.args.early_stopping)
        setattr(session_state, "top_k", model.args.top_k)
        setattr(session_state, "top_p", model.args.top_p)
    else:
        session_state = get(
            max_seq_length=model.args.max_seq_length,
            max_length=model.args.max_length,
            decoding_algorithm="Sampling" if model.args.do_sample else "Beam Search",
            length_penalty=model.args.length_penalty,
            early_stopping=model.args.early_stopping,
            num_beams=model.args.num_beams,
            top_k=model.args.top_k,
            top_p=model.args.top_p,
        )
    model.args.max_seq_length = session_state.max_seq_length
    model.args.max_length = session_state.max_length
    model.args.length_penalty = session_state.length_penalty
    model.args.early_stopping = session_state.early_stopping
    model.args.top_k = session_state.top_k
    model.args.top_p = session_state.top_p

    if session_state.decoding_algorithm == "Sampling":
        model.args.do_sample = True
        model.args.num_beams = None
    elif session_state.decoding_algorithm == "Beam Search":
        model.args.do_sample = False
        model.args.num_beams = session_state.num_beams

    return session_state, model


@st.cache(hash_funcs={T5Model: simple_transformers_model})
def get_prediction(model, input_text, prefix_text):
    if prefix_text:
        predictions = model.predict([prefix_text + ": " + input_text])
    else:
        predictions = model.predict([input_text])

    return predictions


def t5_viewer(model):
    try:
        session_state, model = get_states(model)
    except AttributeError:
        session_state = get(
            max_seq_length=model.args.max_seq_length,
            max_length=model.args.max_length,
            decoding_algorithm=model.args.do_sample,
        )
        session_state, model = get_states(model, session_state)

    st.sidebar.subheader("Parameters")
    model.args.max_seq_length = st.sidebar.slider(
        "Max Seq Length", min_value=1, max_value=512, value=model.args.max_seq_length
    )

    st.sidebar.subheader("Decoding")

    model.args.max_length = st.sidebar.slider(
        "Max Generated Text Length", min_value=1, max_value=512, value=model.args.max_length
    )

    model.args.length_penalty = st.sidebar.number_input("Length Penalty", value=model.args.length_penalty)

    model.args.early_stopping = st.sidebar.radio(
        "Early Stopping", ("True", "False"), index=0 if model.args.early_stopping else 1
    )

    decoding_algorithm = st.sidebar.radio(
        "Decoding Algorithm", ("Sampling", "Beam Search"), index=0 if model.args.do_sample else 1
    )

    if decoding_algorithm == "Sampling":
        model.args.do_sample = True
        model.args.num_beams = None
    elif decoding_algorithm == "Beam Search":
        model.args.do_sample = False
        model.args.num_beams = 1

    if model.args.do_sample:
        model.args.top_k = st.sidebar.number_input("Top-k", value=model.args.top_k if model.args.top_k else 50)

        model.args.top_p = st.sidebar.slider(
            "Top-p", min_value=0.0, max_value=1.0, value=model.args.top_p if model.args.top_p else 0.95
        )
    else:
        model.args.num_beams = st.sidebar.number_input("Number of Beams", value=model.args.num_beams)

    st.markdown("## Instructions: ")
    st.markdown("The input to a T5 model can be providied in two ways.")
    st.markdown("### Using Prefix")
    st.markdown(
        "If you provide a value for the `prefix`, Simple Viewer will automatically insert `: ` between the `prefix` text and the `input` text."
    )
    st.markdown("### Blank prefix")
    st.markdown(
        "You may also leave the `prefix` blank. In this case, you can provide a prefix and a separator at the start of the `input` text (if your model requires a prefix)."
    )

    st.subheader("Enter prefix: ")
    prefix_text = st.text_input("")

    st.subheader("Enter input text: ")
    input_text = st.text_area("")

    if input_text:
        prediction = get_prediction(model, input_text, prefix_text)[0]

        st.subheader(f"Generated output: ")
        st.write(prediction)
