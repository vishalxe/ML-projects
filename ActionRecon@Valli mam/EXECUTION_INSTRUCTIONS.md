# HAR Models Execution Instructions

## Overview

This document provides step-by-step instructions for running all four HAR models individually and aggregating their results for journal publication.

## Prerequisites

### Required Software

- Python 3.8 or higher
- TensorFlow 2.x
- Required Python packages (install via pip):

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

### Dataset

- Ensure your dataset is located at: `c:\Users\Vishal\Downloads\HAR_synthetic_full\HAR_synthetic_full.csv`
- If your dataset is in a different location, update the `csv_path` variable in each script

## Execution Steps

### Step 1: Run Individual Models

Execute each model script separately to collect comprehensive results:

#### 1.1 Run LSTM Model

```bash
python run_lstm_model.py
```

**Expected Output Files:**

- `lstm_model_final.keras` - Trained model
- `lstm_results.json` - Detailed results
- `lstm_classification_report.csv` - Per-class metrics
- `lstm_results.png` - Visualization plots

#### 1.2 Run ResNet-Transformer Model

```bash
python run_resnet_transformer_model.py
```

**Expected Output Files:**

- `resnet_transformer_model_final.keras` - Trained model
- `resnet_transformer_results.json` - Detailed results
- `resnet_transformer_classification_report.csv` - Per-class metrics
- `resnet_transformer_results.png` - Visualization plots

#### 1.3 Run TCN Model

```bash
python run_tcn_model.py
```

**Expected Output Files:**

- `tcn_model_final.keras` - Trained model
- `tcn_results.json` - Detailed results
- `tcn_classification_report.csv` - Per-class metrics
- `tcn_results.png` - Visualization plots

#### 1.4 Run Transformer Model

```bash
python run_transformer_model.py
```

**Expected Output Files:**

- `transformer_model_final.keras` - Trained model
- `transformer_results.json` - Detailed results
- `transformer_classification_report.csv` - Per-class metrics
- `transformer_results.png` - Visualization plots

### Step 2: Aggregate Results

After running all individual models, aggregate the results:

```bash
python aggregate_results.py
```

**Expected Output Files:**

- `performance_summary.csv` - Overall performance comparison
- `per_class_comparison.csv` - Per-class performance comparison
- `confusion_matrices_comparison.png` - Side-by-side confusion matrices
- `performance_radar_chart.png` - Radar chart comparison
- `performance_bar_chart.png` - Bar chart comparison
- `statistical_analysis.csv` - Statistical analysis
- `comprehensive_results_report.md` - Final comprehensive report

## Expected Execution Times

| Model              | Estimated Training Time | Memory Usage |
| ------------------ | ----------------------- | ------------ |
| LSTM               | 30-60 minutes           | High         |
| ResNet-Transformer | 45-90 minutes           | High         |
| TCN (Optimized)    | 10-20 minutes           | Low          |
| Transformer        | 30-60 minutes           | High         |

_Note: Times may vary based on hardware specifications and dataset size._

## Hardware Recommendations

### Minimum Requirements

- RAM: 8GB
- GPU: NVIDIA GTX 1060 or equivalent (optional but recommended)
- Storage: 5GB free space

### Recommended

- RAM: 16GB or higher
- GPU: NVIDIA RTX 3070 or better
- Storage: 10GB free space

## Troubleshooting

### Common Issues

#### 1. Memory Issues

If you encounter memory errors:

- Reduce batch size in the training scripts
- Use smaller sequence lengths
- Enable mixed precision training (already enabled)

#### 2. Dataset Path Issues

If the dataset is not found:

- Update the `csv_path` variable in each script
- Ensure the CSV file exists and is accessible

#### 3. CUDA/GPU Issues

If GPU is not being used:

- Install CUDA-compatible TensorFlow
- Check GPU availability with: `tf.config.list_physical_devices('GPU')`

#### 4. Missing Dependencies

If import errors occur:

```bash
pip install --upgrade tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

## Results Interpretation

### Key Metrics to Focus On

1. **Accuracy**: Overall classification accuracy
2. **Precision**: True positive rate for each class
3. **Recall**: Sensitivity for each class
4. **F1-Score**: Harmonic mean of precision and recall
5. **AUC**: Area under the ROC curve
6. **Training Time**: Computational efficiency
7. **Confusion Matrix**: Class-specific performance

### Journal Publication Ready Outputs

The scripts generate publication-ready outputs including:

- Comprehensive performance tables
- Statistical significance analysis
- Visualization plots (confusion matrices, ROC curves, training curves)
- Detailed per-class analysis
- Model comparison charts

## Batch Execution (Optional)

For convenience, you can create a batch script to run all models sequentially:

### Windows Batch Script (`run_all_models.bat`)

```batch
@echo off
echo Starting HAR Models Execution...

echo Running LSTM Model...
python run_lstm_model.py

echo Running ResNet-Transformer Model...
python run_resnet_transformer_model.py

echo Running TCN Model...
python run_tcn_model.py

echo Running Transformer Model...
python run_transformer_model.py

echo Aggregating Results...
python aggregate_results.py

echo All models completed successfully!
pause
```

### Linux/Mac Shell Script (`run_all_models.sh`)

```bash
#!/bin/bash
echo "Starting HAR Models Execution..."

echo "Running LSTM Model..."
python run_lstm_model.py

echo "Running ResNet-Transformer Model..."
python run_resnet_transformer_model.py

echo "Running TCN Model..."
python run_tcn_model.py

echo "Running Transformer Model..."
python run_transformer_model.py

echo "Aggregating Results..."
python aggregate_results.py

echo "All models completed successfully!"
```

## Data Collection for Journal Publication

After running all scripts, you will have:

1. **Individual Model Results**: Detailed performance metrics for each model
2. **Comparative Analysis**: Side-by-side comparison of all models
3. **Statistical Analysis**: Significance testing and confidence intervals
4. **Visualizations**: Publication-ready plots and charts
5. **Comprehensive Report**: Complete analysis suitable for journal submission

## Next Steps

1. Review the generated results and visualizations
2. Update the `Journal_Results_Section.md` template with actual results
3. Perform additional statistical tests if needed
4. Prepare figures and tables for journal submission
5. Write the discussion and conclusion sections based on results

## Support

If you encounter any issues:

1. Check the error messages and logs
2. Verify all dependencies are installed
3. Ensure sufficient system resources
4. Check dataset path and format
5. Review the troubleshooting section above

---

**Note**: This execution pipeline is designed to generate comprehensive, publication-ready results for your HAR research project. Each script is optimized for both performance and thoroughness of analysis.
