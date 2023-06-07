# This file is managed by devopsify > update strategy : replace

SHELL := /bin/bash
.PHONY: help version bump setup lint

VERSION	= v$(shell cat pyproject.toml | grep "^version = \"*\"" | cut -d'"' -f2)

help:		## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

version:	## Display the current version
	@echo "${VERSION}"

bump: 		## Bump the version
	poetry version patch -C libs/opt-submodular \
	&& poetry version patch -C libs/doc-summarize \
	&& poetry version patch -C libs/opt-network

setup:		## Setup a dev environment
	# Create new python environment
	pip install --upgrade pip wheel \
	&& pip install --upgrade poetry pre-commit \
	&& poetry install --no-ansi -C libs/opt-submodular \
	&& poetry install --no-ansi -C libs/doc-summarize \
	&& poetry install --no-ansi -C libs/opt-network \
	&& pre-commit install

lint:		## Lint the code
	. venv/bin/activate && black --preview --check .
	. venv/bin/activate && flake8 .
	. venv/bin/activate && pylint .
##
## Targets
##

requirements.txt: pyproject.toml	## Recipe for refreshing requirements.txt from pyproject.toml
	@echo "# THIS FILE IS AUTOMATICALLY GENERATED. DO NOT EDIT." > requirements.txt
	@echo "./" >> requirements.txt
	poetry export --without-hashes >> requirements.txt
