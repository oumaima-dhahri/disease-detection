# show_execute.py - Notebook Execution Utility

A powerful utility script for executing Jupyter notebooks with live output, model name injection, and filtered warnings for better readability.

## Features

- 🚀 **Live Output**: Real-time display of notebook execution progress
- 🧠 **Model Injection**: Automatically inject model names into notebooks
- 🛡️ **Warning Filtering**: Suppress nuisance warnings (PyTorch normalize, PIL low contrast, etc.)
- 📊 **Smart Output**: Highlight important metrics (loss, accuracy, F1, precision, recall)
- 📝 **Logging**: Comprehensive logging with timestamps
- 💾 **Auto-save**: Automatically save executed notebooks with model name suffix

## Installation

```bash
pip install -r requirements_show_execute.txt
```

## Usage

### Basic Usage
```bash
python show_execute.py notebook.ipynb
```

### With Model Name Injection
```bash
python show_execute.py notebook.ipynb -m convnext_tiny
```

### With Custom Output Path
```bash
python show_execute.py notebook.ipynb -m resnet18 -o results/executed_notebook.ipynb
```

### With Logging
```bash
python show_execute.py notebook.ipynb -m vit_b_16 -l execution.log
```

## Command Line Arguments

- `notebook`: Path to the Jupyter notebook (.ipynb file)
- `-o, --output`: Output path for the executed notebook (optional)
- `-l, --log`: Log file path (optional)
- `-m, --model`: Model name to inject (e.g., resnet18, vit_b_16, convnext_tiny)

## Examples

### Execute a training notebook with ConvNeXt
```bash
python show_execute.py train_scripts/train_convnext.ipynb -m convnext_tiny
```

### Execute with custom output and logging
```bash
python show_execute.py test_scripts/test_model.ipynb -m efficientnet_b0 -o results/test_results.ipynb -l test_execution.log
```

## Output Features

The script provides:
- ✅ Real-time progress indicators
- 📊 Highlighted metrics (loss, accuracy, F1 scores)
- 🎯 Performance metrics tracking
- ❌ Clear error reporting
- ⏱️ Execution time tracking
- 📝 Comprehensive logging

## Filtered Warnings

The script automatically filters out common nuisance warnings:
- PyTorch tensor normalization warnings
- PIL low contrast warnings
- cuDNN enable warnings
- Division by 255 warnings

## Integration with Your Project

This script is particularly useful for:
- **Kaggle Notebooks**: Execute training scripts with live output
- **Model Comparison**: Easily switch between different models
- **Automated Training**: Batch execution of multiple model configurations
- **Debugging**: Clear visibility into training progress and errors

## Notes

- The script injects a `MODEL_NAME` variable at the beginning of the notebook
- Make sure your notebooks can handle this variable or provide a default case
- Execution timeout is set to 10 minutes per cell
- The script preserves all original notebook content while adding execution results

