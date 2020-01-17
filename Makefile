install:
	pip install -e .
	pip install -r requirements-dev.txt
	pip list

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +

formatter:
	black simpletransformers tests --exclude simpletransformers/experimental 
	
lint:
	flake8 simpletransformers tests --exclude simpletransformers/experimental
	black --check simpletransformers tests --exclude simpletransformers/experimental

types:
	pytype --keep-going simpletransformers --exclude simpletransformers/experimental

test: clean 
	pytest tests --cov simpletransformers

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext --strict
