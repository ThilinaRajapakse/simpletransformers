install:
	pip install -e .
	pip install -r requirements-dev.txt
	pip list

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +

formatter:
	black simpletransformers tests --exclude simpletransformers/experimental\|simpletransformers/classification/transformer_models\|simpletransformers/custom_models
	
lint: clean
	flake8 simpletransformers tests --exclude=simpletransformers/experimental,simpletransformers/classification/transformer_models,simpletransformers/custom_models
	black --check simpletransformers tests --exclude simpletransformers/experimental\|simpletransformers/classification/transformer_models\|simpletransformers/custom_models

types:
	pytype --keep-going simpletransformers --exclude simpletransformers/experimental simpletransformers/classification/transformer_models simpletransformers/custom_models

test: clean 
	pytest tests --cov simpletransformers/classification simpletransformers/ner simpletransformers/question_answering

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext 
