#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = NetHackOdyssey
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Default dataset to "download_nao_small" unless overridden
NAO_DATA ?= download_nao_small

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download small nld-nao dataset for testing (~5.4GB)
.PHONY: download_nao_small
download_nao_small: requirements
	$(PYTHON_INTERPRETER) odyssey/commands/download_nao_data.py -f aa

## Download full nld-nao dataset
.PHONY: download_nao_full
download_nao_full: requirements
	$(PYTHON_INTERPRETER) odyssey/commands/download_nao_data.py

## Generate the small nld-nao dataset
.PHONY: generate_nao_dataset_small
generate_nao_dataset_small: requirements download_nao_small
	$(PYTHON_INTERPRETER) odyssey/commands/generate_nao_dataset.py -f aa

## Generate the full nld-nao dataset
.PHONY: generate_nao_dataset_full
generate_nao_dataset_full: requirements download_nao_full
	$(PYTHON_INTERPRETER) odyssey/commands/generate_nao_dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
