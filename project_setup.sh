#!/bin/bash

# Script to set up a machine learning project structure for demand-forecasting
# Run this from the demand-forecasting directory

# Create directories
mkdir -p data/raw
mkdir -p models
mkdir -p notebooks
mkdir -p reports
mkdir -p src/{data,features,model,utils}

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/model/__init__.py
touch src/utils/__init__.py

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "demand-forecasting"
version = "0.1.0"
authors = [
    {name = "Saikat Ghosh", email = "official.ghosh.saikat@gmail.com"}
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
EOF

# Generate .egg-info by installing in editable mode
#pip install -e .

echo "Project setup complete! Folders, __init__.py files, pyproject.toml, and .egg-info created."