#!/bin/bash

# Warning: this script may modify code

# Runs all checks necessary for contributing. This script assumes:
# - The current working directory is the analysis-scripts root project directory
# - uv sync has been run, and the virtual environment it set up has been activated

set -xe

ruff check --fix
ruff format
pyproject-fmt pyproject.toml --column-width 88 --indent 4 --keep-full-version
pyproject-fmt analysis-lcm/pyproject.toml --column-width 88 --indent 4 --keep-full-version
pyproject-fmt analysis-ros/pyproject.toml --column-width 88 --indent 4 --keep-full-version

uv export --frozen --all-packages --no-hashes -o requirements-dev.txt
uv export --frozen --no-dev --all-packages --no-hashes -o requirements.txt
