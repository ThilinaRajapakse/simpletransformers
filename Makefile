install:
	pip install -e .
	pip install -r requirements-dev.txt
	pip list

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +

clean-test:
	-rm -r .coverage*
	-rm -r data
	-rm -r runs
	-rm -r outputs
	-rm -r cache_dir
	-rm -r wandb
	-rm train.txt

formatter:
	black --line-length 119 simpletransformers tests --exclude simpletransformers/experimental\

lint: clean
	flake8 simpletransformers tests --exclude=simpletransformers/experimental
	black --check --line-length 119 . simpletransformers tests --exclude simpletransformers/experimental

types:
	pytype --keep-going simpletransformers --exclude simpletransformers/experimental

test: clean
	pytest tests --cov simpletransformers/classification simpletransformers/ner simpletransformers/question_answering simpletransformers/language_modeling simpletransformers/t5 simpletransformers/seq2seq

test-classification:
	pytest tests --cov simpletransformers/classification

test-ner:
	pytest tests --cov simpletransformers/ner

test-question_answering:
	pytest tests --cov simpletransformers/question_answering

test-language_modeling:
	pytest tests --cov simpletransformers/language_modeling

test-t5:
	pytest tests --cov simpletransformers/t5

test-seq2seq:
	pytest tests --cov simpletransformers/seq2seq

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext
