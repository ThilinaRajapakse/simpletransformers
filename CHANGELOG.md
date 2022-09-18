# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.63.9] - 2022-09-18

### Added

- Python 3.10 support [luketudge](https://github.com/luketudge)

#### Changed

- Removed usage of deprecated function `batch_encode_plus` [whr778](https://github.com/whr778)

## [0.63.8] - 2022-09-18

### Added

- Added `adam_betas` to model_args [anaconda121](https://github.com/anaconda121)

### Changed

- Switched to `torch.optim.AdamW` [whr778](https://github.com/whr778)

### Fixed

- Fixed issues with LayoutLM predictions. Updated docs. [deltaxrg](https://github.com/deltaxrg)
- Fixed issue with loading MarianMT models [Fortune-Adekogbe](https://github.com/Fortune-Adekogbe)


## [0.63.7] - 2022-05-29

### Added

- Added support for LayoutLMV2 and RemBERT [whr778](https://github.com/whr778)

### Fixed

- Fixed issue with ner_utils lazy_loading_start_line not being set correctly. [whr778](https://github.com/whr778)
- Lazy loading bug fixes [sainttttt](https://github.com/sainttttt)
- Fixed seq2seq and T5 preprocessing [MichelBartels](https://github.com/MichelBartels)

## [0.63.6] - 2022-03-24

### Added

- Added support for ByT5 models [@djstrong](https://github.com/djstrong)

### Fixed

- Fixed bug in NER ONNX prediction [gaganmanku96](https://github.com/gaganmanku96)
- Fixed bug in NER tokenization which prevented multiprocessing being used correctly [mxa4646](https://github.com/mxa4646)
- Fixed some broken links in docs [jordimas](https://github.com/jordimas)


## [0.63.5] - 2022-02-25

### Added

- Added support for selecting FAISS index type with `RetrievalModel`.

## [0.63.4] - 2021-11-20

### Added

- Added support for individual training of context/query encoders in `RetrievalModel`.

### Fixed

- Fixed a bug for pre-tokenized input to ONNX models. [whr778](https://github.com/whr778)
- BigBird bugfix where training dataset samples were being truncated at 510 token. [whr778](https://github.com/whr778)
- Fixed bug when FP16 is not used with `RetrievalModel`. [tiena2cva](https://github.com/tiena2cva)
- Fixed bug in auto model for `QuestionAnsweringModel`. [lekhang4497](https://github.com/lekhang4497)
- Fixed bug where wrong predictions weren't returned in `ClassificationModel` [jinschoi](https://github.com/jinschoi)

## [0.63.0] - 2021-11-10

### Added

- Added support for document retrieval with the `RetrievalModel` class

## [0.62.1] - 2021-09-24

### Fixed

- Fixed bug when using onnx with ClassificationModel [kongyurui](https://github.com/kongyurui)
- Fixed potential bug with NERModel when the input text is empty [whr778](https://github.com/whr778)
- Fixed bug in sentencepiece tokenizer for some models [whr778](https://github.com/whr778)
- Fixed issue with Seq2SeqModel showing the first training epoch as epoch 0 [dopc](https://github.com/dopc)
- Fixed bug where eval_file was not used with ConvAIModel [cahya-wirawan](https://github.com/cahya-wirawan)

### Changed

- Replaced tensorboardx imports with default torch version

## [0.62.0] - 2021-09-24

### Added

- Additional loss functions for `ClassificationModel` and `NERModel`. [Zhylkaaa](https://github.com/Zhylkaaa)

### Changed

- Deprecated custom classification models. [Zhylkaaa](https://github.com/Zhylkaaa)


## [0.61.14] - 2021-09-23

### Changed

- W&B run id is now assigned as a model attribute

## [0.70.0] - 2021-08

### TODO

#### DOCS

- use_hf_datasets
- RAG
- Retrieval


## [0.61.13] - 2021-07-24

### Added

- Pretraining and finetuning BigBird and XLMRoBERTa LMs [whr778](https://github.com/whr778)

## [0.61.10] - 2021-07-13

### Added

- Added class weights support for NER [tiena2cva](https://github.com/tiena2cva)
- Added Herbert [Zhylkaaa](https://github.com/Zhylkaaa)

### Fixed

- Bug fixes

## [0.61.9] - 2021-06-21

### Fixed

- Updated W&B repo label

## [0.61.8] - 2021-06-21

### Fixed

- Reverted changes to W&B repo label which weren't working.

## [0.61.7][0.61.7] - 2021-06-21

### Changed

- Updated W&B repo label

## [0.61.6][0.61.6] - 2021-05-28

### Fixed

- Fixed the onnx predict loop [whr778](https://github.com/whr778)

### Added

- Added NER support for BigBird, Deberta, Deberta-v2, and xlm pretrained models [whr778](https://github.com/whr778)
- Added BigBird for regular sequence classification (not multilabel) [@manueltonneau](https://github.com/manueltonneau)

## [0.61.5][0.61.5] - 2021-05-18

### Added

- Fixed possible bug when using HF Datasets with Seq2SeqModel
- Added `repo: simpletransformers` to W&B config

## [0.61.4][0.61.4] - 2021-03-27

### Added

- Bug fixed in LanguageModelingModel which could occur when loading a GPU trained model on CPU. [alvaroabascar](https://github.com/alvaroabascar)
- Bug fixed in NER ONNX for models with token type ids. [whr778](https://github.com/whr778)
- Bug fixed in NER lazy loading. [mhdhdri](https://github.com/mhdhdri)
- Bug fixed in sliding window tie breaking [calebchiam](https://github.com/calebchiam)

### Changed

- Thread count no longer fixed when using ONNX [rayline](https://github.com/rayline)

## [0.61.3] - 2021-03-19

### Changed

- Return full retrieved docs with RAG

### Added

- Added extra args for RAG:
  - `split_text_character`
  - `split_text_n`

## [0.61.0][0.61.0] - 2021-03-19

### Added

- Added support for RAG models (in `Seq2Seq`) - docs will be updated soon
- Added support for Huggingface Datasets library for memory efficient training. Currently supports:
  - Classification (all)
  - NER
  - Language Modeling
  - Seq2Seq
  - T5
  - QA (Note that HF Datasets might not always work with QAModel)

## [0.60.9][0.60.9] - 2021-02-19

# Added

- Added XLNet support for NER

# Fixed

- Fixed bug where `polynomial_decay_schedule_power` value was not being set correctly

### Changed

- Switched to using FastTokenizers where possible

## [0.60.8] - 2021-02-12

### Fixed

- Fixed bug in loading cached features with classification models

## [0.60.7] - 2021-02-11

### Changed

- Multiprocessing during tokenization is now turned on by default. If you experience any instability, this can be turned off by setting `use_multiprocessing=False`

## [0.60.6] - 2021-02-05

### Changed

- Multiprocessing during tokenization is now turned off by default. You can enable this by setting `use_multiprocessing=True`. However, the latest Pytorch versions seems to be unstable when using multiprocessing.

## [0.60.3] - 2021-02-02

### Changed

- Multiprocessing is now turned off by default for evaluation. This is to avoid potential errors when doing evaluation during training. You can enable this by setting `use_multiprocessing_for_evaluation` to `True`.

## [0.60.2][0.60.2] - 2021-02-02

### Fixed

- Fixed bug in ClassificationDataset [mapmeld](https://github.com/mapmeld)

## [0.60.1][0.60.1] - 2021-02-02

### Added

- Added new NER models:

  - ALBERT
  - MPNet
  - SqueezeBERT
  - MobileBERT
- Added new QA models:

  - CamemBERT [@adrienrenaud](https://github.com/adrienrenaud)
  - MPNet
  - SqueezeBERT

## [0.60.0][0.60.0] - 2021-02-02

### Added

- Added class weights support for Longformer classification
- Added new classification models:
  - SqueezeBert
  - DeBERTa
  - MPNet

### Changed

- Updated ClassificationModel logic to make it easier to add new models

## [0.51.16][0.51.16] - 2021-01-29

### Fixed

- Fixed bug in LayoutLM classification

## [0.51.15][0.51.15] - 2021-01-24

### Fixed

- Fixed bug in Language Generation models [mapmeld](https://github.com/mapmeld)
- Fixed bug in MBart models [nilboy](https://github.com/nilboy)

## [0.51.14][0.51.14] - 2021-01-24

### Fixed

- Fixed bug introduced in 0.51.12 when using sliding window

## [0.51.13][0.51.13] - 2021-01-11

### Fixed

- Fixed bug introduced in 0.51.12 with multiclass classification

## [0.51.12][0.51.12] - 2021-01-11

### Changed

- Added Area under the ROC curve (AUROC) and the Area under the Precision-Recall curve (AUPRC) as default metrics for binary classification [@manueltonneau](https://github.com/manueltonneau)

### Fixed

- Fixed issues with models not being set to train modd when evaluating while training [nilboy](https://github.com/nilboy)

## [0.51.11][0.51.11] - 2021-01-09

### Changed

- Removed `do_lower_case` when using `AutoTokenizer`

## [0.51.10][0.51.10] - 2020-12-29

### Fixed

- Fixed bug in `QuestionAnsweringModel` when using cached features for evaluation
- Fixed bugs in `ConvAIModel` due to compatibility issues

## [0.51.9][0.51.9] / [0.51.8][0.51.8] - 2020-12-29

## [0.51.9][0.51.9] / [0.51.8][0.51.8] - 2020-12-29

## [0.51.9][0.51.9] / [0.51.8][0.51.8] - 2020-12-29

### Added

- Added the `special_tokens_list` arg which can be used to add additional special tokens to the tokenizer [karthik19967829](https://github.com/karthik19967829)

## [0.51.7][0.51.7] - 2020-12-29

### Fixed

- Fixed bug during predicton when `sliding_window=True` and `output_hidden_states=True` [calebchiam](https://github.com/calebchiam)

## [0.51.6] - 2020-12-10

### Added

- Added BERTweet for multilabel classification [@manueltonneau](https://github.com/manueltonneau)

### Fixed

- Fixed bug where `T5Model` would save evaluation checkpoints even when `save_eval_checkpoints` is False.
- Fixed bug where `args.silent` was not used in `NERModel`. [mossadhelali](https://github.com/mossadhelali)

### Changed

- Changed the default value of `dataloader_num_workers` (for Pytorch Dataloaders) to 0. This is to avoid memory leak issues with Pytorch multiprocessing with text data.

## [0.51.5][0.51.5] - 2020-12-10

### Added

- Added support for T5/mT5 models in Simple Viewer

### Fixed

- Fixed bug where `QuestionAnsweringModel` and `Seq2SeqModel` would save (although not use) cached features even when `no_cache` is set

## [0.51.3][0.51.3] - 2020-12-10

### Fixed

- Fixed bug in `MultiLabelClassificationModel` evaluation. [mapmeld](https://github.com/mapmeld) [abhinavg97](https://github.com/abhinavg97)

## [0.51.2][0.51.2] - 2020-12-09

### Fixed

- Fixed bug in ConvAI interact_single() method

## [0.51.1][0.51.1] - 2020-12-08

### Fixed

- Fixed bug in `mbart` `predict()` function. [DM2493](https://github.com/DM2493)

### Added

- Added docs for language generation and multi-modal classifcation

## [0.51.0][0.51.0] - 2020-12-05

### Added

- Added support for MT5
- Added support for Adafactor optimizer
- Added support for various schedulers:
  - get_constant_schedule
  - get_constant_schedule_with_warmup
  - get_linear_schedule_with_warmup
  - get_cosine_schedule_with_warmup
  - get_cosine_with_hard_restarts_schedule_with_warmup
  - get_polynomial_decay_schedule_with_warmup

### Changed

- `T5Model` now has a required `model_type` parameter (`"t5"` or `"mt5"`)

### Fixed

- Fixed issue with class weights not working in `ClassificationModel` when using mult-GPU training

## [0.50.0][0.50.0] - 2020-12-01

### Changed

- Compatibility with Transformers 4.0.0.

## [0.49.4][0.49.4] - 2020-11-25

### Added

- Added `not_saved_args` to `model_args`. Any args specified in this set will not be saved when the model is saved.
- `RepresentationModel` improvements. [aesuli](https://github.com/aesuli)

## [0.49.3][0.49.3] - 2020-11-09

### Changed

- ROC and PR W&B charts are no longer generated when using sliding window to avoid an error.
- Fixed issue with ONNX in NER [gaganmanku96](https://github.com/gaganmanku96)
- Fixed issues with wandb sweeps [khituras](https://github.com/khituras) [ohstopityou](https://github.com/ohstopityou)

## [0.49.1][0.49.1] - 2020-11-22

### Fixed

- Fixed issue with Marian models using deprecated function. [@bestvater](https://github.com/bestvater)
- Added custom tokenizer option and random (no-pretraining) initialization option for `T5Model`. [sarapapi](https://github.com/sarapapi)

## [0.49.0][0.49.0] - 2020-11-09

### Added

- Added LayoutLM for Classification.
- Added MBart. [Zhylkaaa](https://github.com/Zhylkaaa)
- Added BERTweet for NER. [@manueltonneau](https://github.com/manueltonneau)
- Added Longformer for Multilabel Classification. [@manueltonneau](https://github.com/manueltonneau)

### Fixed

- Fixed issue with `Seq2SeqModel` when the `model_name` contained backslashes.
- Fixed issue with saving args when a `dataset_class` is specified in `Seq2SeqModel`.

### Changed

- The Electra implementation used with `ClassificationModel` is now fully compatible with Hugging Face.

## [0.48.15][0.48.15] - 2020-10-22

### Fixed

- Updated some tokenizer arguments to the new argument names. [macabdul9](https://github.com/macabdul9)
- Learning rate is now obtained from the `get_last_lr()` method. [sarapapi](https://github.com/sarapapi)

## [0.48.14][0.48.14] - 2020-10-12

### Fixed

- Fixed `predict()` function issue when using `sliding_window`.
- Fixed issues with simple-viewer (streamlit compatibility issues)

## [0.48.13][0.48.13] - 2020-10-12

### Fixed

- Fixed issues with using mixed precision training with `LanguageModelingModel`.

## [0.48.12][0.48.12] - 2020-10-12

### Fixed

- Fixed compatibility issues with W&B Sweeps. [jonatanklosko](https://github.com/jonatanklosko)

## [0.48.11][0.48.11] - 2020-10-11

### Changed

- The `train_model()` method now returns training details. Specifically;
  global_step: Number of global steps trained
  training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True

## [0.48.10][0.48.10] - 2020-10-11

### Added

- Added support for special tokens with `Seq2SeqModel`. [Zhylkaaa](https://github.com/Zhylkaaa)

## [0.48.9][0.48.9] - 2020-10-07

### Changed

- Moved `model.train()` inside `train()` method.

## [0.48.8][0.48.8] - 2020-10-07

### Added

- Added support for `BERTweet` with `ClassificationModel`. [@manueltonneau](https://github.com/manueltonneau)

## [0.48.7][0.48.7] - 2020-10-03

### Added

- Added support for multilabel classification with the CamemBERT model. [@adrienrenaud](https://github.com/adrienrenaud)

### Changed

- Output arrays in classification evaluate/predict now avoids `np.append()`. This should be more time and memory efficient.

## [0.48.6][0.48.6] - 2020-09-26

### Added

- Added `layoutlm` model for NER

### Fixed

- Potential fix for inconsistent `eval_loss` calculation

## [0.48.5][0.48.5] - 2020-09-17

### Added

- Added `convert_to_onnx` function to the following models:
  - ClassificationModel
  - NERModel
- Converted ONNX models can be loaded (requires specifying `onnx: True` in model_args) and used for prediction.
- Added `fp16` support for evaluation and prediction (requires Pytorch >= 1.6) for the following models:
  - ClassificationModel
  - ConvAI
  - MultiModalClassificationModel
  - NERModel
  - QuestionAnsweringModel
  - Seq2Seq
  - T5Model
- Added multigpu prediction/eval in
  - ClassificationModel
  - ConvAI
  - MultiModalClassificationModel
  - NERModel
  - QuestionAnsweringModel
  - Seq2Seq
  - T5Model

### Fixed

- Thread count can now be specified for MultiLabelClassificationModel.

## [0.48.4][0.48.4] - 2020-09-23

### Fixed

- Fixed compatibility issue with transformers 3.2. (BertPreTrainedModel was being imported from an incompatible path)

## [0.48.3] - 2020-09-08

- Version numbering issue fixed.

## [0.48.2] - 2020-09-08

### Fixed

- Fixed missing `padding_strategy` argument in `squad_convert_example_to_features()`

## [0.48.1][0.48.1] - 2020-09-08

### Fixed

- Bug when using sliding window with multiclass classification
- Bug in ConvAI where model was being accessed before being created

## [0.48.0][0.48.0] - 2020-09-06

### Added

- Added dynamic quantization support for all models.
- Added ConvAI docs to documentation website. [@pablonm3](https://github.com/pablonm3)

## [0.47.6] - 2020-09-01

### Fixed

- Fixed missing `padding_strategy` argument in `squad_convert_example_to_features()` [cahya-wirawan](https://github.com/cahya-wirawan)

## [0.47.5] - 2020-09-01

### Added

- Added dynamic quantization, `thread_count` arg, and avoids padding during inference for Classification models. [karthik19967829](https://github.com/karthik19967829)
-

### Fixed

- Bug fix which fixes reprocessing data after reading from cache in Seq2SeqDataset and SimpleSummarizationDataset [@Zhylkaaa](https://github.com/Zhylkaaa)

## [0.47.4][0.47.4] - 2020-08-29

### Fixed

- Bug fix in MultilabelClassificationModel when using sentence pairs.

## [0.47.3][0.47.3] - 2020-08-19

### Fixed

- Bug fix in ConvAI [Sxela](https://github.com/Sxela)

## [0.47.0][0.47.0] - 2020-08-09

### Added

- Added support for testing models through a Streamlit app. Use the command `simple-viewer". Currently supports:
  - Classification (including multilabel)
  - NER (design inspired by [displaCy Named Entity Visualizer](https://explosion.ai/demos/displacy-ent))
  - QA

See [docs](https://simpletransformers.ai/docs/tips-and-tricks/#simple-viewer-visualizing-model-predictions-with-streamlit) for details.

## [0.46.5][0.46.5] - 2020-08-05

### Changed

- Python version requirement changed back to 3.6 for Colab support.
- Miscellaneous bug fixes in 0.46.3 and 0.46.4

## [0.46.2][0.46.2] - 2020-08-01

### Fixed

- Fixed unreachable condition in Electra language modeling.

## [0.46.1][0.46.1] - 2020-08-01

### Fixed

- Bug in ConvAI models where cache_dir was not being created.

## [0.46.0][0.46.0] - 2020-08-01

### Changed

- Uses PyTorch native AMP instead of Apex. [@strawberrypie](https://github.com/strawberrypie)

## [0.45.5][0.45.5] - 2020-07-29

### Fixed

- Bug fixed in loading classiication models with a `labels_map` where labels are ints.

## [0.45.4][0.45.4] - 2020-07-28

### Fixed

- Bug fixed in lazy loading classification tasks where `lazy_text_column=0` caused an error.

## [0.45.2][0.45.2] - 2020-07-25

### Added

- Added `dataloader_num_workers` to `ModelArgs` for specifying the number of processes to be used with a Pytorch dataloader.

### Changed

- Bumped required `transformers` version to 3.0.2

## [0.45.0][0.45.0] - 2020-07-19

### Added

- Added Text Representation Generation (`RepresentationModel`). [@pablonm3](https://github.com/pablonm3)

## [0.44.0][0.44.0] - 2020-07-05

### Added

- Lazy loading support added for `QuestionAnsweringModel`.

## [0.43.6][0.43.6] - 2020-07-05

### Fixed

- Bug fixed in `Seq2Seq` tasks.
- Bug fixed in `NERModel` where the classification report was missing in checkpoints.
- Bug fixed in ELECTRA.
- Bug fixed in `Seq2Seq` generic encoder-decoder model.
- Bug fixed in `Seq2Seq` tasks.
- Bug fixed in regression prediction.
- Bug fixed in loading multiclass classification models when `num_labels` aren't specified.

## [0.43.0][0.43.0] - 2020-07-05

### Added

- Added support for custom parameter groups.

### Fixed

- ELECTRA pretraining no longer replaces 10% of masked inputs with random tokens. [@dev-sngwn](https://github.com/dev-sngwn)

## [0.42.0][0.42.0] - 2020-07-05

### Added

- Added better support for Chinese text in Language Modeling. [@taranais](https://github.com/taranais)
- Added `mobilebert` for classification, NER, QA, and Seq2Seq. [@flozi00](https://github.com/flozi00)

### Fixed

- Fixed incorrect logic when using `early_stopping_metric_minimize`. [@djstrong](https://github.com/djstrong)
- Fixed issue with cache directory being created even when `no_cache` is set. [@henninglebbaeus](https://github.com/henninglebbaeus)

### Changed

- Running loss is now shown next to the tqdm bar (with the tqdm bar description)
- Removed tokenizers and transformers version pins (added earlier to avoid compatibility issues)

## [0.41.2][0.41.2] - 2020-07-03

### Fixed

- Fixed bugs with args not being passed correctly to wandb in the following models:
  - `MultiModalClassificationModel`
  - `ConvAIModel`
  - `Seq2SeqModel`
  - `T5Model`
- Fixed bugs in `Seq2SeqModel` and `T5Model` when not using `use_multiprocessed_decoding`.

### Changed

- Set `use_multiprocessed_decoding=False` as default for Seq2Seq models to avoid a bug.

## [0.41.1] - 2020-07-02

### Fixed

- Fixed bug where the returned value from `MultiModalClassificationModel.evaluate_model()` was incorrect.

## [0.41.0][0.41.0] - 2020-07-02

### Added

- NER lazy loading support added [@Pradhy729](https://github.com/Pradhy729)

### Changed

- Added `lazy_loading` attibute to `ClassificationArgs` which now controls whether lazy loading is used.
- Replaced `lazy_loading_header_row` attribute in `ClassificationArgs` with `lazy_loading_start_line`.
- Unnecessary Docs spacing removed [@bryant1410](https://github.com/bryant1410)
- Set required tokenizer version to 0.7 until breaking changes are resolved.

## [0.40.2][0.40.2] - 2020-06-25

### Fixed

- Fixed bug in Multi-Modal classification when using `evaluate_during_training`.

## [0.40.1][0.40.1] - 2020-06-25

### Added

- Added `interact_single()` method to `ConvAIModel`. This accepts a message and conversation history (and an optional personality). [@Amit80007](https://github.com/Amit80007)

### Fixed

- Fixed bug in multi modal classification [@tekkon](https://github.com/tekkkon)

### Changed

- Cleaned `language_modeling_utils.py`. [@Pradhy729](https://github.com/Pradhy729)

## [0.40.0][0.40.0] - 2020-06-23

### Added

- All Simple Transformers models can now be used with W&B sweeps.
- `eval_model()` now logs to wandb in the following models (can be turned off with `wandb_log=False`)
  - `ClassificationModel`
  - `NERModel`
- Model args can now be specified through the relevant dataclass. (E.g. `ClassificationArgs`, `T5Args`, etc.)
- All model args are now saved and loaded automatically with Simple Transformers models.
- Multiprocessed decoding support added to Seq2Seq models
- Custom label support for Classification tasks (except multilabel).
- NER labels can be given as `NERArgs.labels_list` (persists through args saving)

### Changed

- Changed `NERModel.eval_model()` to return usable model_outputs
- Improved `tqdm` descriptions in progress bars
- ConvAIModel arg `no_sample` renamed to `do_sample` for consistency

## [0.34.4][0.34.4] - 2020-06-17

### Added

- Added `num_return_sequences`, `top_k`, and `top_p` args for `Seq2SeqModel`.

### Fixed

- Fixed bug potential bug when using `sliding_window`. [@BjarkePedersen](https://github.com/BjarkePedersen)

### Changed

- Cleaned `language_modeling_utils`. [@Pradhy729](https://github.com/Pradhy729)

## [0.34.3] - 2020-06-13

### Fixed

- Fixed bug in question answering when *not* using multiprocessing for feature conversion.

## [0.34.2] - 2020-06-13

### Fixed

- Fixed bug in sentence-pair task feature conversion.

## [0.34.1][0.34.1] - 2020-06-12

### Fixed

- Fixed bug in multi-modal classification due to compatibility issues with breaking changes in transformers==2.11.0.

## [0.34.0][0.34.0] - 2020-06-09

### Added

- Added distributed training support for language model training. [@cahya-wirawan](https://github.com/cahya-wirawan)
- Added multiprocessed decoding support for T5 models.

## [0.33.2][0.33.2] - 2020-06-08

### Fixed

- Fixed bug in adding prefix space. Included longformer in list of models where prefix spaces are added. [@guy-mor](https://github.com/guy-mor)

## [0.33.1][0.33.1] - 2020-06-08

### Changed

- Changed the tokenization logic of RoBERTa (and other models using GPT-2 tokenizer) so that a prefix space will be added to input sentences.

## [0.33.0][0.33.0] - 2020-06-08

### Added

- Added Longformer model support for;
  - Classification
  - NER
  - Seq2Seq
    tasks. [@flozi00](https://github.com/flozi00)

## [0.32.3][0.32.3] - 2020-06-04

### Fixed

- Fixed compatibility issues with breaking changes in transformers==2.11.0. [@fcggamou](https://github.com/fcggamou)

## [0.32.1][0.32.1] - 2020-06-01

### Fixed

- Fixed bug when using `output_hidden_states` with `ClassificationModel`. [@jpotniec](https://github.com/jpotoniec)

## [0.32.0][0.32.0] - 2020-06-01

### Added

- Added Lazy Loading support for classification tasks (except multi-label). ([Docs](https://simpletransformers.ai/docs/classification-specifics/#lazy-loading-data))

## [0.31.0][0.31.0] - 2020-05-30

### Added

- Added Longformer model support for Language Modeling.

## [0.30.0][0.30.0] - 2020-05-27

### Added

- Added XLM-RoBERTa support for question answering tasks.
- Added `save_optimizer_and_scheduler` (default 1) to `global_args` which controls whether optimizer and scheduler is saved along with the model. Disabling significantly reduces the disk space used by saved models.

### Fixed

- Bug in XLM tokenizer when preprocessing QA datasets.
- `QuestionAnsweringModel.predict(n_best_size=n)` now correctly returns `n` answers per question (along with `n` probabilities).

## BREAKING CHANGE

- `QuestionAnsweringModel.predict()` now returns two lists (a list of dicts with question ids mapped to answers and a list of dicts with question ids mapped to the answer probabilities).

## [0.29.0][0.29.0] - 2020-05-24

### Fixed

- Fixed issues with training ELECTRA language models from scratch. [@aced125](https://github.com/aced125) [@Laksh1997](https://github.com/Laksh1997)
- Fixed bug in save_discriminator() method.

### Changed

- The parallel process count is now limited to 61 by default on Windows systems. [@leungi](https://github.com/leungi)

## [0.28.10][0.28.10] - 2020-05-23

### Added

- Added more generation/decoding parameters for T5 models.

### Fixed

- Fixed bug with cached features not being used with T5 models.

## [0.28.9][0.28.9] - 2020-05-19

### Fixed

- Fixed bug where final model was not being saved automatically.

## [0.28.8][0.28.8] - 2020-05-19

### Fixed

- Fixed bug where some models were not using `multiprocessing_chunksize` argument.

## [0.28.7][0.28.7] - 2020-05-19

### Fixed

- Fixed bug in NERModel.predict() method when `split_on_space=False`. [@alexysdussier](https://github.com/alexysdussier)

## [0.28.6][0.28.6] - 2020-05-19

### Added

- Added multiprocessing support for Question Answering tasks for substantial performance boost where CPU-bound tasks (E.g. prediction especially with long contexts)
- Added `multiprocessing_chunksize` (default 500) to `global_args` for finer control over chunking. Usually, the optimal value will be (roughly) `number of examples / process count`.

## [0.28.5][0.28.5] - 2020-05-18

### Added

- Added `no_save` option to `global_args`. Setting this to `True` will prevent models from being saved to disk.
- Added minimal training script for `Seq2Seq` models in the examples directory.

## [0.28.4][0.28.4] - 2020-05-15

### Fixed

- Fixed potential bugs in loading weights when fine-tuning an ELECTRA language model. Fine-Tuning an ELECTRA language model now requires both `model_name` and `model_type` to be set to `electra`.

## [0.28.3][0.28.3] - 2020-05-15

### Changed

- Updated `Seq2SeqModel` to use `MarianTokenizer` with MarianMT models. [@flozi00](https://github.com/flozi00)

## [0.28.2][0.28.2] - 2020-05-14

### Fixed

- Bug fix for generic Seq2SeqModel

## [0.28.1][0.28.1] - 2020-05-14

### Fixed

- Bug when training language models from scratch

## [0.28.0][0.28.0] - 2020-05-11

### Added

- Sequence-to-Sequence task support added. This includes the following models:
  - BART
  - Marian
  - Generic Encoder-Decoder
- The `args` dict of a task-specific Simple Transformers model is now saved along with the model. When loading the model, these values will be read and used.
  Any new `args` passed into the model initialization will override the loaded values.

## [0.27.3][0.27.3] - 2020-05-10

### Added

- Support for `AutoModel` in NER, QA, and LanguageModeling. [@flozi00](https://github.com/flozi00)

### Fixed

- Now predict function from NER_Model returns a value model_outputs that contains:
  A Python list of lists with dicts containing each word mapped to its list with raw model output. [@flaviussn](https://github.com/flaviussn)

### Changed

- Pillow import is now optional. It only needs to be installed if MultiModal models are used.

## [0.27.2][0.27.2] - 2020-05-08

### Fixed

- Fixed T5 lm_labels not being masked properly

### Changed

- Torchvision import is now optional. It only needs to be installed if MultiModal models are used.

## [0.27.1][0.27.1] - 2020-05-05

### Fixed

- Fixed issue with custom evaluation metrics not being handled correctly in `MultiLabelClassificationModel`. [@galtay](https://github.com/galtay)

## [0.27.0][0.27.0] - 2020-05-05

### Added

- Added support for T5 Model.
- Added `do_sample` arg to language generation.
- `NERModel.predict()` now accepts a `split_on_space` optional argument. If set to `False`, `to_predict` must be a a list of lists, with the inner list being a list of strings consisting of the split sequences. The outer list is the list of sequences to predict on.

### Changed

- `eval_df` argument in `NERModel.train_model()` renamed to `eval_data` to better reflect the input format. Added Deprecation Warning.

## [0.26.1][0.26.1] - 2020-04-27

### Fixed

- Specifying `verbose=False` in `LanguageGenerationModel.generate()` method now correctly silences logger output.

## [0.26.0][0.26.0] - 2020-04-25

### Added

- Added Electra model support for sequence classification (binary, multiclass, multilabel)
- Added Electra model support for question answering
- Added Roberta model support for question answering

### Changed

- Reduced logger messages during question answering evaluation

## [0.25.0][0.25.0] - 2020-04-24

### Added

- Added support for Language Generation tasks.

## [0.24.9][0.24.9] - 2020-04-22

### Added

- Added support for custom metrics with `QuestionAnsweringModel`.

### Fixed

- Fixed issue with passing proxies to ConvAI models. [@Pradhy729](https://github.com/Pradhy729)

## [0.24.8][0.24.8] - 2020-04-13

### Fixed

- Fixed incorrect indexes when extracting hidden layer outputs and embedding outputs with `ClassificationModel.predict()` method.

## [0.24.7][0.24.7] - 2020-04-13

### Added

- Added option to get hidden layer outputs and embedding outputs with `ClassificationModel.predict()` method.
  - Setting `config: {"output_hidden_states": True}` will automatically return all embedding outputs and hidden layer outputs.

### Changed

- `global_args` now has a `config` dictionary which can be used to override default values in the confg class.
  - This can be used with ClassificationModel, MultiLabelClassificationModel, NERModel, QuestionAnsweringModel, and LanguageModelingModel

## [0.24.6][0.24.6] - 2020-04-12

### Added

- Added support for ELECTRA based NER models.

## [0.24.5][0.24.5] - 2020-04-11

### Fixed

- Fixed bug in `LanguageModelingModel` when loading from a training checkpoint.

## [0.24.4] - 2020-04-10

### Fixed

- Fixed bug in `LanguageModelingModel` initialization with a trained tokenizer.

### Added

- Added support for passing proxy information with ConvAI model.

## [0.24.3][0.24.3] - 2020-04-10

### Fixed

- Fixed potential bug in NERModel `predict()` method when using custom labels.
- Fixed typo in the NERModel description in the readme.

## [0.24.2][0.24.2] - 2020-04-09

### Fixed

- Fixed issues with `vocab_size` not being set properly in ELECTRA models.

## [0.24.1][0.24.1] - 2020-04-09

### Fixed

- Fixed bugs in minimal examples for language modeling.

### Changed

- Added `vocab_size` back to default `args` dict for clarity. (`vocab_size` is `None` by default)
- Changed error message when training a new tokenizer with incorrect parameters for clarity.

## [0.24.0][0.24.0] - 2020-04-09

### Added

- Added ELECTRA pretraining support.
- Added better support for configuring model architectures when training language models from scratch.
  - Any options which should be overriden from the default config can now be specified in the `args` dict. (`config` key)

### Changed

- Default entry for `vocab_size` removed from `args` for `LanguageModelingModel` as it differs for different model types.
  - `vocab_size` must now be specified whenever a new tokenizer is to be trained.

### Fixed

- Fixed bugs when training BERT (with word piece tokenization) language models from scratch.
- Fixed incorrect special tokens being used with BERT models when training a new tokenizer.
- Fixed potential bugs with BERT tokenizer training.

## [0.23.3][0.23.3] - 2020-04-05

### Fixed

- Fixed bug in `QuestionAnsweringModel` where the `save_model()` method wasn't being called properly.
- Fixed bug in calculating global step when resuming training.

## [0.23.2][0.23.2] - 2020-04-02

### Fixed

- Prevent padding tokens being added when using `openai-gpt` and `gpt2` models for language modeling.

## [0.23.1][0.23.1] - 2020-03-30

### Fixed

- Fixed bug in binary classification evaluation when data only contains one label.
- Fixed typo in readme.

### Changed

- Cache dir is no longer created when `no_cache` is used.

## [0.23.0][0.23.0] - 2020-03-30

### Added

- Added support for training custom tokenizers.
- Added improved support for training language models from scratch.
- Added improved support for resuming training in classification, NER, and QnA tasks.

## [0.22.1] - 2020-03-19

### Added

- Added support for XLMRoberta for multi-label tasks.

## [0.22.0][0.22.0] - 2020-03-14

### Added

- Added support for language model training (from scratch or fine-tuning).
- Added option to choose which metric should be used for early stopping.

### Changed

- Switched to using the logging module over print for everything except running loss. (QuestionAnsweringModel - [@travis-harper](https://github.com/travis-harper))
- Replaced more concatenated string paths with `os.path.join()` when creating `training_progress_scores.csv`.

## [0.21.5][0.21.5] - 2020-03-12

### Changed

- Replaced concatenated string paths with `os.path.join()` when creating `training_progress_scores.csv`. [@sarthakTUM](https://github.com/sarthakTUM)

## [0.21.4][0.21.4] - 2020-03-12

### Fixed

- Fixed issue with cached eval features being used even when using `predict()` in `ClassificationModel` and `NERModel`.

## [0.21.3][0.21.3] - 2020-03-03

### Added

- Added classification report for NER for per-tag scores. [@seo-95](https://github.com/seo-95)

## [0.21.2][0.21.2] - 2020-03-01

### Fixed

- Fixed bug with empty answers in `QuestionAnsweringModel`. @jacky18008

## [0.21.1][0.21.1] - 2020-02-29

### Fixed

- Fixed bug in ConvAIModel where `reprocess_input_data` and `use_cached_eval_features` args were ignored.

## [0.21.0][0.21.0] - 2020-02-29

### Added

- Added support for training Conversational AI models.
- Added `cuda_device` parameter to MultiLabelClassificationModel.

### Fixed

- Fixed bug in MultiModalClassificationModel when `num_labels` is not given.

## [0.20.3][0.20.3] - 2020-02-22

### Changed

- `reprocess_input_data` changed to `True` by default.
- `use_cached_eval_features` changed to `False` by default.

## [0.20.2][0.20.2] - 2020-02-22

### Fixed

- Fixed issue with early stopping not working with Question Answering.

## [0.20.1][0.20.1] - 2020-02-22

### Fixed

- Fixed issue with `predict()` function using cached features.

## [0.20.0][0.20.0] - 2020-02-21

### Added

- Added support for Multi Modal Classification tasks.

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

## [0.19.9][0.19.9] - 2020-02-18

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

## [0.19.8][0.19.8] - 2020-02-14

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

## [0.19.7][0.19.7] - 2020-02-11

### Changed

- Removed `wandb` as a dependency. Installing `wandb` in now optional.

## [0.19.6][0.19.6] - 2020-02-11

### Added

- Added support for multilabel classification with FlauBERT.@adrienrenaud

## [0.19.5][0.19.5] - 2020-02-11

### Added

- Added support for FlauBERT with classification tasks (except multi-label).@adrienrenaud

## [0.19.4][0.19.4] - 2020-02-04

### Fixed

- Fixed error that occured when `args` is not given when creating a Model.

## [0.19.3][0.19.3] - 2020-02-03

### Added

- Added `manual_seed` to `global_args` . Can be used when training needs to be reproducible.

## [0.19.2][0.19.2] - 2020-01-31

### Added

- Added early stopping support for NER and Question Answering tasks.

### Fixed

- Fixed issue with nested file paths not being created.
- `wandb_kwargs` not being used with NER and Question Answering.

## [0.19.1][0.19.1] - 2020-01-27

### Fixed

- Fixed issue with evaluation at the end of epochs not being considered for best model.

## [0.19.0][0.19.0] - 2020-01-26

### Added

- Added early stopping support for Classification tasks.
  - Set `use_early_stopping` to `True` to enable.
- The best model will now be saved to `{output_dir}/best_model/` when `evaluate_during_training` is used.
- Added `evaluate_during_training_verbose` to args dict to control whether evaluation during training outputs are printed to console.
- Added **all-contributors** to README to recognize contributors.

### Changed

- Evaluation during training no longer displays progress bars.
- Evaluation during training no longer prints results to console by default.
- Moved model/results saving logic to `_save_model` for readability and maintainability.
- Updated README.

## [0.18.12][0.18.12] - 2020-01-25

### Fixed

- Added missing extra SEP token in RoBERTa, CamemBERT, and XLMRoBERTA in sentence pair tasks.

## [0.18.11][0.18.11] - 2020-01-21

### Added

- Added `no_cache` option to `global_args` which disables caching (saving and loading) of features to/from disk.

## [0.18.10][0.18.10] - 2020-01-20

### Added

- Added Makefile with tests dependency installation, test code, formatter and types.
- Added setup.cfg file with Make configuration
- Added some tests for the functionality

### Changed

- Files linted using flake8
- Files formatted using black
- Test tested with pytest
- Unused variables deleted

## [0.18.9][0.18.9] - 2020-01-20

### Fixed

- Fixed bug with importing certain pre-trained models in `MultiLabelClassificationModel` .

## [0.18.8][0.18.8] - 2020-01-20

### Added

- Added `**kwargs` to the init methods of `ClassificationModel` , `MultiLabelClassificationModel` , `QuestionAnsweringModel` , and `NERModel` . These will be passed to the `from_pretrained()` method of the underlying model class.

## [0.18.6][0.18.6] - 2020-01-18

### Changed

- Reverted change made in 0.18.4 (Model checkpoint is no longer saved at the end of the last epoch as this is the same model saved in `ouput_dir` at the end of training).

Model checkpoint is now saved for all epochs again.

## [0.18.5][0.18.5] - 2020-01-18

### Fixed

- Fixed bug when using `sliding_window` .

## [0.18.4][0.18.4] - 2020-01-17

### Fixed

- Typo in `classification_utils.py` .

### Changed

- Model checkpoint is no longer saved at the end of the last epoch as this is the same model saved in `ouput_dir` at the end of training.

## [0.18.3][0.18.3] - 2020-01-15

### Fixed

- Potential bugfix for CamemBERT models which were giving identical outputs to all inputs.

## [0.18.2][0.18.2] - 2020-01-15

### Added

- Added option to turn off model saving at the end of every epoch with `save_model_every_epoch` .

### Fixed

- Fixed bug with missing `tensorboard_folder` key in certain situations.

### Changed

- Moved `args` items common to all classes to one place ( `config/global_args.py` ) for maintainability. Does not make any usage changes.

## [0.18.1][0.18.1] - 2020-01-15

### Fixed

- Fixed bug with missing `regression` key when using MultiLabelClassification.

## [0.18.0][0.18.0] - 2020-01-15

### Added

- Sentence pair tasks are now supported.
- Regression tasks are now supported.
- `use_cached_eval_features` to `args` . Evaluation during training will now use cached features by default. Set to `False` if features should be reprocessed.

### Changed

- Checkpoints saved at the end of an epoch now follow the `checkpoint-{global_step}-epoch-{epoch_number} format.

## [0.17.1][0.17.1] - 2020-01-14

### Fixed

- Fixed `wandb_kwargs` key missing in `args` bug.

## [0.17.0][0.17.0] - 2020-01-14

### Added

- Added new model XLM-RoBERTa. Can now be used with `ClassificationModel` and `NERModel` .

## [0.16.6][0.16.6] - 2020-01-13

### Added

- Added evaluation scores from end-of-epoch evaluation to `training_progress_scores.csv` .

### Fixed

- Typos in `README.md` .

## [0.16.5][0.16.5] - 2020-01-09

### Fixed

- Reverted missed logging commands to print statements.

## [0.16.4][0.16.4] - 2020-01-09

### Changed

- Removed logging import.

## [0.16.3][0.16.3] - 2020-01-09

### Fixed

- Reverted to using print instead of logging as logging seems to be causing issues.

## [0.16.2][0.16.2] - 2020-01-08

### Changed

- Changed print statements to logging.

## [0.16.1][0.16.1] - 2020-01-07

### Added

- Added `wandb_kwargs` to `args` which can be used to specify keyword arguments to `wandb.init()` method.

## [0.16.0][0.16.0] - 2020-01-07

### Added

- Added support for training visualization using the W&B framework.
- Added `save_eval_checkpoints` attribute to `args` which controls whether or not a model checkpoint will be saved with every evaluation.

## [0.15.7][0.15.7] - 2020-01-05

### Added

- Added `**kwargs` for different accuracy measures during multilabel training.

## [0.15.6][0.15.6] - 2020-01-05

### Added

- Added `train_loss` to `training_progress_scores.csv` (which contains the evaluation results of all checkpoints) in the output directory.

## [0.15.5][0.15.5] - 2020-01-05

### Added

- Using `evaluate_during_training` now generates `training_progress_scores.csv` (which contains the evaluation results of all checkpoints) in the output directory.

## [0.15.4][0.15.4] - 2019-12-31

### Fixed

- Fixed bug in `QuestonAnsweringModel` when using `evaluate_during_training` .

## [0.15.3][0.15.3] - 2019-12-31

### Fixed

- Fixed bug in MultiLabelClassificationModel due to `tensorboard_dir` being missing in parameter dictionary.

### Changed

- Renamed `tensorboard_folder` to `tensorboard_dir` for consistency.

## [0.19.8][0.19.8] - 2020-02-14

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

### Added

- Added `tensorboard_folder` to parameter dictionary which can be used to specify the directory in which the tensorboard files will be stored.

## [0.15.1][0.15.1] - 2019-12-27

### Added

- Added `**kwargs` to support different accuracy measures at training time.

## [0.15.0][0.15.0] - 2019-12-24

### Added

- Added `evaluate_during_training_steps` parameter that specifies when evaluation should be performed during training.

### Changed

- A model checkpoint will be created for each evaluation during training and the evaluation results will be saved along with the model.

## [0.14.0][0.14.0] - 2019-12-24

### Added

- Added option to specify a GPU to be used when multiple GPUs are available. E.g.: `cuda_device=1`
- Added `do_lower_case` argument for uncased models.

### Fixed

- Fixed possible bug with output directory not being created before evaluation is run when using `evaluate_during_training` .

## [0.13.4][0.13.4] - 2019-12-21

### Fixed

- Fixed bug with when using `eval_during_training` with QuestionAnswering model.

## [0.13.3][0.13.3] - 2019-12-21

### Fixed

- Fixed bug with loading Multilabel classification models.
- Fixed formatting in README.md.

## [0.13.2][0.13.2] - 2019-12-20

### Fixed

- Fixed formatting in README.md.

## [0.13.1][0.13.1] - 2019-12-20

### Fixed

- Bug in Multilabel Classification due to missing entries in default args dict.

## [0.13.0][0.13.0] - 2019-12-19

### Added

- Sliding window feature for Binary and Multiclass Classification tasks.

## [0.12.0][0.12.0] - 2019-12-19

### Added

- Minimal examples have been added to the `examples` directory as Python scripts.

### Changed

- Readme updated to include the addition of examples.

## [0.11.2][0.11.2] - 2019-12-18

### Fixed

- Evaluation during training fixed for multilabel classification.

## [0.11.1][0.11.1] - 2019-12-18

### Fixed

- Broken multiprocessing support for NER tasks fixed.

## [0.11.0][0.11.0] - 2019-12-15

### Added

- CamemBERT can now be used with NERModel.

### Changed

- Readme changed to include CamemBERT for NER.

## [0.10.8][0.10.8] - 2019-12-15

### Added

- DistilBERT can now be used with NERModel.

### Changed

- Readme changed to include DistilBERT for NER.

## [0.10.7][0.10.7] - 2019-12-15

### Added

- This CHANGELOG file to hopefully serve as an evolving example of a standardized open source project CHANGELOG.

[0.63.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/71880c2...HEAD

[0.63.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9323c03...71880c2

[0.63.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/a3ce529...9323c03

[0.62.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/fe70794...a3ce529

[0.62.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/67a2a47...fe70794

[0.61.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/a7e7fff...67a2a47

[0.61.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/281ff31...a7e7fff

[0.61.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b49bf28...281ff31

[0.61.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/87eeb0e...b49bf28

[0.61.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/76f1df5...87eeb0e

[0.60.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/de06bfb...76f1df5

[0.60.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/de989b5...de06bfb

[0.60.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6f189e0...de989b5

[0.60.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/5840749...6f189e0

[0.51.16]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b42898e...5840749

[0.51.15]: https://github.com/ThilinaRajapakse/simpletransformers/compare/2af55e9...b42898e

[0.51.14]: https://github.com/ThilinaRajapakse/simpletransformers/compare/278fca1...2af55e9

[0.51.13]: https://github.com/ThilinaRajapakse/simpletransformers/compare/4a5c295...278fca1

[0.51.12]: https://github.com/ThilinaRajapakse/simpletransformers/compare/36fc7a6...4a5c295

[0.51.11]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3ce3651...36fc7a6

[0.51.10]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3ce3651...3ce3651

[0.51.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3cfc400...3ce3651

[0.51.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3cfc400...3ce3651

[0.51.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/58a563e...3cfc400

[0.51.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/0609ccd...58a563e

[0.51.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/c733785...0609ccd

[0.51.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d583c6a...c733785

[0.51.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/5f4bc8d...d583c6a

[0.51.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/a0f382e...5f4bc8d

[0.50.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/2e2c50e...a0f382e

[0.49.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6025b6f...2e2c50e

[0.49.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9440c6e...6025b6f

[0.49.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d4d66d6...9440c6e

[0.49.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/77da311...d4d66d6

[0.48.15]: https://github.com/ThilinaRajapakse/simpletransformers/compare/38d0cd8...77da311

[0.48.14]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3ce9288...38d0cd8

[0.48.13]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6881e5c...3ce9288

[0.48.12]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b908bb5...6881e5c

[0.48.11]: https://github.com/ThilinaRajapakse/simpletransformers/compare/1b59118...b908bb5

[0.48.10]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6370a1b...1b59118

[0.48.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/1c231e1...6370a1b

[0.48.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/edb9fdd...1c231e1

[0.48.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/25fa010...edb9fdd

[0.48.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6f75f8e...25fa010

[0.48.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/39d25d0...6f75f8e

[0.48.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/7ef56b0...39d25d0

[0.48.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/8c1ae68...7ef56b0

[0.48.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d62ce56...0f678f2

[0.47.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/bd4c397...d62ce56

[0.47.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/78ffa94...bd4c397

[0.47.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d405b4a...78ffa94

[0.46.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/2cc77f7...d405b4a

[0.46.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/7f37cb7...2cc77f7

[0.46.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b64637c...7f37cb7

[0.46.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/121cba4...b64637c

[0.46.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/120d1e6...121cba4

[0.45.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/0ac6b69...120d1e6

[0.45.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ac0f1a0...0ac6b69

[0.45.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3e98361...ac0f1a0

[0.45.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/fad190f...3e98361

[0.44.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6a9beca...fad190f

[0.43.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/2ee0c0b...6a9beca

[0.43.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/e1eb826...2ee0c0b

[0.42.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/a8bb887...e1eb826

[0.41.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/eeb69fa...a8bb887

[0.41.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b4e1886...eeb69fa

[0.40.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f4ef3d3...b4e1886

[0.40.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/99ede24...f4ef3d3

[0.40.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/cf66100...99ede24

[0.34.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3e112de...cf66100

[0.34.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/19ecd79...3e112de

[0.34.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/4789a1d...19ecd79

[0.33.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/bb83151...4789a1d

[0.33.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f40331b...bb83151

[0.33.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/e96aacd...f40331b

[0.32.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f5cee79...e96aacd

[0.32.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d009aa1...f5cee79

[0.32.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b196267...d009aa1

[0.31.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d38e086...b196267

[0.30.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9699a0c...d38e086

[0.29.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/858d2b9...9699a0c

[0.28.10]: https://github.com/ThilinaRajapakse/simpletransformers/compare/a1a6473...858d2b9

[0.28.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/08a3b4c...a1a6473

[0.28.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/4e66cb8...08a3b4c

[0.28.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9077ebb...4e66cb8

[0.28.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/68d62b1...9077ebb

[0.28.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/91866e8...68d62b1

[0.28.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ac097e4...91866e8

[0.28.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ca87582...ac097e4

[0.28.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/1695fc4...ca87582

[0.28.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/4d9665d...1695fc4

[0.28.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/402bd8e...4d9665d

[0.27.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/bc94b34...402bd8e

[0.27.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d665494...bc94b34

[0.27.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/32d5a1a...d665494

[0.27.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ab1e600...32d5a1a

[0.26.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3d4f616...ab1e600

[0.26.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/aa8e6a6...3d4f616

[0.25.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/445d386...aa8e6a6

[0.24.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/52fea69...445d386

[0.24.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/e9b1f41...52fea69

[0.24.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/853ca94...e9b1f41

[0.24.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/777f78d...853ca94

[0.24.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/8f1daac...777f78d

[0.24.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ce4b925...8f1daac

[0.24.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/91b7ae1...ce4b925

[0.24.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ae6b6ea...91b7ae1

[0.24.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/17b1c23...ae6b6ea

[0.23.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3069694...17b1c23

[0.23.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/96bc291...3069694

[0.23.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/7529ee1...96bc291

[0.23.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b5cf82c...7529ee1

[0.22.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/51fc7a3...b5cf82c

[0.21.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/27ff44e...51fc7a3

[0.21.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/e9905a4...27ff44e

[0.21.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/7a9dd6f...e9905a4

[0.21.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d114c50...7a9dd6f

[0.21.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/721c55c...d114c50

[0.21.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f484717...721c55c

[0.20.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/daf5ccd...f484717

[0.20.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/538b8fa...daf5ccd

[0.20.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/c466ca6...538b8fa

[0.20.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/61952aa...c466ca6

[0.19.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b5ab978...61952aa

[0.19.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b5ab978...61952aa

[0.19.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d7a5abd...b5ab978

[0.19.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f814874...d7a5abd

[0.19.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9170750...f814874

[0.19.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/337670a...9170750

[0.19.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/337670a...34261a8

[0.19.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/bb17711...337670a

[0.19.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/489d4f7...bb17711

[0.19.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/bb67a2b...489d4f7

[0.19.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/6c6f2e9...bb67a2b

[0.18.12]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f8d0ad2...6c6f2e9

[0.18.11]: https://github.com/ThilinaRajapakse/simpletransformers/compare/65ef805...f8d0ad2

[0.18.10]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ce5afd7...65ef805

[0.18.9]: https://github.com/ThilinaRajapakse/simpletransformers/compare/8ade0f4...ce5afd7

[0.18.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/44afa70...8ade0f4

[0.18.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/aa7f650...44afa70

[0.18.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/ebef6c4...aa7f650

[0.18.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/0aa88e4...ebef6c4

[0.18.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/52a488e...0aa88e4

[0.18.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/1fb47f1...52a488e

[0.18.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9698fd3...1fb47f1

[0.18.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9c9345f...9698fd3

[0.17.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/9a39cab...9c9345f

[0.17.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/0e5dd18...9a39cab

[0.16.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/c6c1792...0e5dd18

[0.16.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/e9504b5...c6c1792

[0.16.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/5d1eaa9...e9504b5

[0.16.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/f5a7699...5d1eaa9

[0.16.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d589b75...f5a7699

[0.16.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/d8df83f...d589b75

[0.16.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/1684fff...d8df83f

[0.15.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/c2f620a...1684fff

[0.15.6]: https://github.com/ThilinaRajapakse/simpletransformers/compare/cd24331...c2f620a

[0.15.5]: https://github.com/ThilinaRajapakse/simpletransformers/compare/38cbea5...cd24331

[0.15.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/70e2a19...38cbea5

[0.15.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/a65dc73...70e2a19

[0.15.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/268ced8...a65dc73

[0.15.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/2c1e5e0...268ced8

[0.15.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/aa06528...2c1e5e0

[0.14.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/785ee04...aa06528

[0.13.4]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b6e0573...785ee04

[0.13.3]: https://github.com/ThilinaRajapakse/simpletransformers/compare/b3283da...b6e0573

[0.13.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/897ef9f...b3283da

[0.13.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/1ee6093...897ef9f

[0.13.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/04886b5...1ee6093

[0.12.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/04c1c06...04886b5

[0.11.2]: https://github.com/ThilinaRajapakse/simpletransformers/compare/bbc9d22...04c1c06

[0.11.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/191e2f0...bbc9d22

[0.11.0]: https://github.com/ThilinaRajapakse/simpletransformers/compare/92d08ae...191e2f0

[0.10.8]: https://github.com/ThilinaRajapakse/simpletransformers/compare/68d359f...92d08ae

[0.10.7]: https://github.com/ThilinaRajapakse/simpletransformers/compare/0.10.6...68d359f


