import streamlit as st
import pandas as pd
from streamlit import table

from simpletransformers.classification import ClassificationModel


history = []


@st.cache(allow_output_mutation=True)
def load_model(model_type, model_name, **kwargs):
    model = ClassificationModel(model_type, model_name, cuda_device=1)
    return model


def streamlit_runner(model_type, model_name, **kwargs):
    model = load_model("roberta", "distilroberta-base", **kwargs)
    model.args.use_multiprocessing = False

    st.title("Streamlit Simple Transformers App")
    st.header("Classifcation Model")
    st.subheader("Enter text to perform a prediction: ")
    input_text = st.text_area("")
    if input_text:
        prediction, raw_values = model.predict([input_text])
        history.append([input_text, prediction[0]])

        df = pd.DataFrame(history, columns=["Text", "Predicted Label"])
        st.dataframe(df)
