import streamlit as st
import pandas as pd

from simpletransformers.streamlit.streamlit_utils import get


QA_ANSWER_WRAPPER = """{} <span style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0.25rem; background: #a6e22d">{}</span> {}"""
QA_EMPTY_ANSWER_WRAPPER = """{} <span style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0.25rem; background: #FF0000">{}</span> {}"""


def get_states(model, session_state=None):
    if session_state:
        setattr(session_state, "max_answer_length", model.args.max_answer_length)
        setattr(session_state, "max_query_length", model.args.max_query_length)
    else:
        session_state = get(
            max_seq_length=model.args.max_seq_length,
            max_answer_length=model.args.max_answer_length,
            max_query_length=model.args.max_query_length,
        )
    model.args.max_seq_length = session_state.max_seq_length
    model.args.max_answer_length = session_state.max_answer_length
    model.args.max_query_length = session_state.max_query_length

    return session_state, model


def qa_viewer(model):
    try:
        session_state, model = get_states(model)
    except AttributeError:
        session_state = get(
            max_seq_length=model.args.max_seq_length,
            max_answer_length=model.args.max_answer_length,
            max_query_length=model.args.max_query_length,
        )
        session_state, model = get_states(model, session_state)

    st.subheader(f"Predictions")
    model.args.max_seq_length = st.sidebar.slider(
        "Max Seq Length", min_value=1, max_value=512, value=model.args.max_seq_length
    )

    model.args.max_answer_length = st.sidebar.slider(
        "Max Answer Length", min_value=1, max_value=512, value=model.args.max_answer_length
    )

    model.args.max_query_length = st.sidebar.slider(
        "Max Query Length", min_value=1, max_value=512, value=model.args.max_query_length
    )

    model.args.n_best_size = st.sidebar.slider("Number of answers to generate", min_value=1, max_value=20)

    st.subheader("Enter context: ")
    context_text = st.text_area("", key="context")

    st.subheader("Enter question: ")
    question_text = st.text_area("", key="question")

    if context_text and question_text:
        to_predict = [{"context": context_text, "qas": [{"id": 0, "question": question_text}]}]

        answers, probabilities = model.predict(to_predict)

        st.subheader(f"Predictions")
        answers = answers[0]["answer"]

        context_pieces = context_text.split(answers[0])

        if answers[0] != "empty":
            st.write(QA_ANSWER_WRAPPER.format(context_pieces[0], answers[0], context_pieces[-1]), unsafe_allow_html=True)
        else:
            st.write(QA_EMPTY_ANSWER_WRAPPER.format("", answers[0], ""), unsafe_allow_html=True)

        probabilities = probabilities[0]["probability"]

        # outputs = [list(chain(*zip(answers, probabilities)))]
        # columns = list(chain(*zip([f"Answer {i + 1}" for i in range(len(answers))], [f"Confidence {i + 1}" for i in range(len(probabilities))])))

        # output_df = pd.DataFrame(outputs, columns=columns)

        st.subheader("Confidence")
        output_df = pd.DataFrame({"Answer": answers, "Confidence": probabilities})
        st.dataframe(output_df)

    return model
