---
title: "History"
classes: wide
permalink: /docs/history/
excerpt: "Change log of enhancements and bug fixes made to the theme."
sidebar:
  nav: docs
last_modified_at: 2020-05-02 17:58:59
toc: false
---

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.26.1] - 2020-04-27

### Fixed

- Specifying `verbose=False` in `LanguageGenerationModel.generate()` method now correctly silences logger output.


## [0.26.0] - 2020-04-25

### Added

- Added Electra model support for sequence classification (binary, multiclass, multilabel)
- Added Electra model support for question answering
- Added Roberta model support for question answering

### Changed

- Reduced logger messages during question answering evaluation

## [0.25.0] - 2020-04-24

### Added

- Added support for Language Generation tasks.

## [0.24.9] - 2020-04-22

### Added

- Added support for custom metrics with `QuestionAnsweringModel`.

### Fixed

- Fixed issue with passing proxies to ConvAI models. [@Pradhy729](https://github.com/Pradhy729)

## [0.24.8] - 2020-04-13

### Fixed

- Fixed incorrect indexes when extracting hidden layer outputs and embedding outputs with `ClassificationModel.predict()` method.

## [0.24.7] - 2020-04-13

### Added

- Added option to get hidden layer outputs and embedding outputs with `ClassificationModel.predict()` method.
  - Setting `config: {"output_hidden_states": True}` will automatically return all embedding outputs and hidden layer outputs.

### Changed

- `global_args` now has a `config` dictionary which can be used to override default values in the confg class.
  - This can be used with ClassificationModel, MultiLabelClassificationModel, NERModel, QuestionAnsweringModel, and LanguageModelingModel

## [0.24.6] - 2020-04-12

### Added

- Added support for ELECTRA based NER models.

## [0.24.5] - 2020-04-11

### Fixed

- Fixed bug in `LanguageModelingModel` when loading from a training checkpoint.

## [0.24.4] - 2020-04-10

### Fixed

- Fixed bug in `LanguageModelingModel` initialization with a trained tokenizer.

### Added

- Added support for passing proxy information with ConvAI model.

## [0.24.3] - 2020-04-10

### Fixed

- Fixed potential bug in NERModel `predict()` method when using custom labels.
- Fixed typo in the NERModel description in the readme.

## [0.24.2] - 2020-04-09

### Fixed

- Fixed issues with `vocab_size` not being set properly in ELECTRA models.

## [0.24.1] - 2020-04-09

### Fixed

- Fixed bugs in minimal examples for language modeling.

### Changed

- Added `vocab_size` back to default `args` dict for clarity. (`vocab_size` is `None` by default)
- Changed error message when training a new tokenizer with incorrect parameters for clarity.

## [0.24.0] - 2020-04-09

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

## [0.23.3] - 2020-04-05

### Fixed

- Fixed bug in `QuestionAnsweringModel` where the `save_model()` method wasn't being called properly.
- Fixed bug in calculating global step when resuming training.

## [0.23.2] - 2020-04-02

### Fixed

- Prevent padding tokens being added when using `openai-gpt` and `gpt2` models for language modeling.

## [0.23.1] - 2020-03-30

### Fixed

- Fixed bug in binary classification evaluation when data only contains one label.
- Fixed typo in readme.

### Changed

- Cache dir is no longer created when `no_cache` is used.

## [0.23.0] - 2020-03-30

### Added

- Added support for training custom tokenizers.
- Added improved support for training language models from scratch.
- Added improved support for resuming training in classification, NER, and QnA tasks.

## [0.22.1] - 2020-03-19

### Added

- Added support for XLMRoberta for multi-label tasks.

## [0.22.0] - 2020-03-14

### Added

- Added support for language model training (from scratch or fine-tuning).
- Added option to choose which metric should be used for early stopping.

### Changed

- Switched to using the logging module over print for everything except running loss. (QuestionAnsweringModel - [@travis-harper](https://github.com/travis-harper))
- Replaced more concatenated string paths with `os.path.join()` when creating `training_progress_scores.csv`.

## [0.21.5] - 2020-03-12

### Changed

- Replaced concatenated string paths with `os.path.join()` when creating `training_progress_scores.csv`. [@sarthakTUM](https://github.com/sarthakTUM)

## [0.21.4] - 2020-03-12

### Fixed

- Fixed issue with cached eval features being used even when using `predict()` in `ClassificationModel` and `NERModel`.

## [0.21.3] - 2020-03-03

### Added

- Added classification report for NER for per-tag scores. [@seo-95](https://github.com/seo-95)

## [0.21.2] - 2020-03-01

### Fixed

- Fixed bug with empty answers in `QuestionAnsweringModel`. @jacky18008

## [0.21.1] - 2020-02-29

### Fixed

- Fixed bug in ConvAIModel where `reprocess_input_data` and `use_cached_eval_features` args were ignored.

## [0.21.0] - 2020-02-29

### Added

- Added support for training Conversational AI models.
- Added `cuda_device` parameter to MultiLabelClassificationModel.

### Fixed

- Fixed bug in MultiModalClassificationModel when `num_labels` is not given.

## [0.20.3] - 2020-02-22

### Changed

- `reprocess_input_data` changed to `True` by default.
- `use_cached_eval_features` changed to `False` by default.

## [0.20.2] - 2020-02-22

### Fixed

- Fixed issue with early stopping not working with Question Answering.

## [0.20.1] - 2020-02-22

### Fixed

- Fixed issue with `predict()` function using cached features.

## [0.20.0] - 2020-02-21

### Added

- Added support for Multi Modal Classification tasks.

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

## [0.19.9] - 2020-02-18

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

## [0.19.8] - 2020-02-14

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

## [0.19.7] - 2020-02-11

### Changed

- Removed `wandb` as a dependency. Installing `wandb` in now optional.

## [0.19.6] - 2020-02-11

### Added

- Added support for multilabel classification with FlauBERT.@adrienrenaud

## [0.19.5] - 2020-02-11

### Added

- Added support for FlauBERT with classification tasks (except multi-label).@adrienrenaud

## [0.19.4] - 2020-02-04

### Fixed

- Fixed error that occured when `args` is not given when creating a Model.

## [0.19.3] - 2020-02-03

### Added

- Added `manual_seed` to `global_args` . Can be used when training needs to be reproducible.

## [0.19.2] - 2020-01-31

### Added

- Added early stopping support for NER and Question Answering tasks.

### Fixed

- Fixed issue with nested file paths not being created.
- `wandb_kwargs` not being used with NER and Question Answering.

## [0.19.1] - 2020-01-27

### Fixed

- Fixed issue with evaluation at the end of epochs not being considered for best model.

## [0.19.0] - 2020-01-26

### Added

- Added early stopping support for Classification tasks.
    - Set `use_early_stopping` to `True` to enable.
- The best model will now be saved to `{output_dir}/best_model/` when `evaluate_during_training` is used.
- Added `evaluate_during_training_verbose` to args dict to control whether evaluation during training outputs are printed to console.
- Added **all-contributors** to README to recognize contributors.

### Changed

- Evaluation during training no longer displays progress bars.
- Evaluation during training no longer prints results to console by default.
- Moved model/results saving logic to `save_model(` for readability and maintainability.
- Updated README.

## [0.18.12] - 2020-01-25

### Fixed

- Added missing extra SEP token in RoBERTa, CamemBERT, and XLMRoBERTA in sentence pair tasks.

## [0.18.11] - 2020-01-21

### Added

- Added `no_cache` option to `global_args` which disables caching (saving and loading) of features to/from disk.

## [0.18.10] - 2020-01-20

### Added

- Added Makefile with tests dependency installation, test code, formatter and types.
- Added setup.cfg file with Make configuration
- Added some tests for the functionality

### Changed

- Files linted using flake8
- Files formatted using black
- Test tested with pytest
- Unused variables deleted

## [0.18.9] - 2020-01-20

### Fixed

- Fixed bug with importing certain pre-trained models in `MultiLabelClassificationModel` .

## [0.18.8] - 2020-01-20

### Added

- Added `**kwargs` to the init methods of `ClassificationModel` , `MultiLabelClassificationModel` , `QuestionAnsweringModel` , and `NERModel` . These will be passed to the `from_pretrained()` method of the underlying model class.

## [0.18.6] - 2020-01-18

### Changed

- Reverted change made in 0.18.4 (Model checkpoint is no longer saved at the end of the last epoch as this is the same model saved in `ouput_dir` at the end of training).

Model checkpoint is now saved for all epochs again.

## [0.18.5] - 2020-01-18

### Fixed

- Fixed bug when using `sliding_window` .

## [0.18.4] - 2020-01-17

### Fixed

- Typo in `classification_utils.py` .

### Changed

- Model checkpoint is no longer saved at the end of the last epoch as this is the same model saved in `ouput_dir` at the end of training.

## [0.18.3] - 2020-01-15

### Fixed

- Potential bugfix for CamemBERT models which were giving identical outputs to all inputs.

## [0.18.2] - 2020-01-15

### Added

- Added option to turn off model saving at the end of every epoch with `save_model_every_epoch` .

### Fixed

- Fixed bug with missing `tensorboard_folder` key in certain situations.

### Changed

- Moved `args` items common to all classes to one place ( `config/global_args.py` ) for maintainability. Does not make any usage changes.

## [0.18.1] - 2020-01-15

### Fixed

- Fixed bug with missing `regression` key when using MultiLabelClassification.

## [0.18.0] - 2020-01-15

### Added

- Sentence pair tasks are now supported.
- Regression tasks are now supported.
- `use_cached_eval_features` to `args` . Evaluation during training will now use cached features by default. Set to `False` if features should be reprocessed.

### Changed

- Checkpoints saved at the end of an epoch now follow the `checkpoint-{global_step}-epoch-{epoch_number} format.

## [0.17.1] - 2020-01-14

### Fixed

- Fixed `wandb_kwargs` key missing in `args` bug.

## [0.17.0] - 2020-01-14

### Added

- Added new model XLM-RoBERTa. Can now be used with `ClassificationModel` and `NERModel` .

## [0.16.6] - 2020-01-13

### Added

- Added evaluation scores from end-of-epoch evaluation to `training_progress_scores.csv` .

### Fixed

- Typos in `README.md` .

## [0.16.5] - 2020-01-09

### Fixed

- Reverted missed logging commands to print statements.

## [0.16.4] - 2020-01-09

### Changed

- Removed logging import.

## [0.16.3] - 2020-01-09

### Fixed

- Reverted to using print instead of logging as logging seems to be causing issues.

## [0.16.2] - 2020-01-08

### Changed

- Changed print statements to logging.

## [0.16.1] - 2020-01-07

### Added

- Added `wandb_kwargs` to `args` which can be used to specify keyword arguments to `wandb.init()` method.

## [0.16.0] - 2020-01-07

### Added

- Added support for training visualization using the W&B framework.
- Added `save_eval_checkpoints` attribute to `args` which controls whether or not a model checkpoint will be saved with every evaluation.

## [0.15.7] - 2020-01-05

### Added

- Added `**kwargs` for different accuracy measures during multilabel training.

## [0.15.6] - 2020-01-05

### Added

- Added `train_loss` to `training_progress_scores.csv` (which contains the evaluation results of all checkpoints) in the output directory.

## [0.15.5] - 2020-01-05

### Added

- Using `evaluate_during_training` now generates `training_progress_scores.csv` (which contains the evaluation results of all checkpoints) in the output directory.

## [0.15.4] - 2019-12-31

### Fixed

- Fixed bug in `QuestonAnsweringModel` when using `evaluate_during_training` .

## [0.15.3] - 2019-12-31

### Fixed

- Fixed bug in MultiLabelClassificationModel due to `tensorboard_dir` being missing in parameter dictionary.

### Changed

- Renamed `tensorboard_folder` to `tensorboard_dir` for consistency.

## [0.19.8] - 2020-02-14

### Fixed

- Fixed missing variable `wandb_available` in Multilabel Classification.

### Added

- Added `tensorboard_folder` to parameter dictionary which can be used to specify the directory in which the tensorboard files will be stored.

## [0.15.1] - 2019-12-27

### Added

- Added `**kwargs` to support different accuracy measures at training time.

## [0.15.0] - 2019-12-24

### Added

- Added `evaluate_during_training_steps` parameter that specifies when evaluation should be performed during training.

### Changed

- A model checkpoint will be created for each evaluation during training and the evaluation results will be saved along with the model.

## [0.14.0] - 2019-12-24

### Added

- Added option to specify a GPU to be used when multiple GPUs are available. E.g.: `cuda_device=1`
- Added `do_lower_case` argument for uncased models.

### Fixed

- Fixed possible bug with output directory not being created before evaluation is run when using `evaluate_during_training` .

## [0.13.4] - 2019-12-21

### Fixed

- Fixed bug with when using `eval_during_training` with QuestionAnswering model.

## [0.13.3] - 2019-12-21

### Fixed

- Fixed bug with loading Multilabel classification models.
- Fixed formatting in README.md.

## [0.13.2] - 2019-12-20

### Fixed

- Fixed formatting in README.md.

## [0.13.1] - 2019-12-20

### Fixed

- Bug in Multilabel Classification due to missing entries in default args dict.

## [0.13.0] - 2019-12-19

### Added

- Sliding window feature for Binary and Multiclass Classification tasks.

## [0.12.0] - 2019-12-19

### Added

- Minimal examples have been added to the `examples` directory as Python scripts.

### Changed

- Readme updated to include the addition of examples.

## [0.11.2] - 2019-12-18

### Fixed

- Evaluation during training fixed for multilabel classification.

## [0.11.1] - 2019-12-18

### Fixed

- Broken multiprocessing support for NER tasks fixed.

## [0.11.0] - 2019-12-15

### Added

- CamemBERT can now be used with NERModel.

### Changed

- Readme changed to include CamemBERT for NER.

## [0.10.8] - 2019-12-15

### Added

- DistilBERT can now be used with NERModel.

### Changed

- Readme changed to include DistilBERT for NER.

## [0.10.7] - 2019-12-15

### Added

- This CHANGELOG file to hopefully serve as an evolving example of a standardized open source project CHANGELOG.

[0.26.1]: https://github.com/ThilinaRajapakse/simpletransformers/compare/3d4f616...HEAD

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
