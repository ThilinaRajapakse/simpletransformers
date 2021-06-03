import pytest

from simpletransformers.language_representation import RepresentationModel


@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("bert", "bert-base-uncased"),
        ("roberta", "roberta-base"),
        ("gpt2", "distilgpt2"),
    ],
)
@pytest.mark.parametrize("combine_strategy", ["mean", "concat", None])
def test_shapes(model_type, model_name, combine_strategy):
    sentence_list = ["Example sentence 1", "Example sentence 2"]
    # Create a ClassificationModel
    model = RepresentationModel(
        model_type,
        model_name,
        use_cuda=False,
        args={
            "no_save": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
        },
    )
    encoded_sentences = model.encode_sentences(
        sentence_list, combine_strategy=combine_strategy
    )
    longest_seq = (
        3  # RepresentationModel truncates sentences to the longest sentence in the list
    )
    if model_type == "bert" or model_type == "roberta":
        longest_seq += 2  # add [CLS] & [SEP] tokens added by BERT & ROBERTA Models
    # last dimention is the embedding dimension, it depends on the model
    if combine_strategy == None:
        assert encoded_sentences.shape == (len(sentence_list), longest_seq, 768)
    if combine_strategy == "concat":
        assert encoded_sentences.shape == (len(sentence_list), longest_seq * 768)
    if combine_strategy == "mean":
        assert encoded_sentences.shape == (len(sentence_list), 768)
