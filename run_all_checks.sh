#!/bin/bash

# Warning: this script may modify code

# Runs all checks necessary for contributing. This script assumes:
# - The current working directory is the analysis-scripts root project directory
# - uv sync has been run, and the virtual environment it set up has been activated

set -xe

ruff check --fix
ruff format
