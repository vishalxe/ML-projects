# Results Section: Comparative Analysis of Deep Learning Models for Human Activity Recognition

## Abstract of Results

This study presents a comprehensive evaluation of four deep learning architectures for Human Activity Recognition (HAR) using multi-modal sensor data. We compared Long Short-Term Memory (LSTM) networks, ResNet-Transformer hybrid models, Temporal Convolutional Networks (TCN), and Transformer encoders on a standardized HAR dataset. The evaluation encompasses classification performance, computational efficiency, and real-time applicability metrics.

## Dataset and Experimental Setup

### Dataset Characteristics

- **Dataset**: HAR_synthetic_full.csv
- **Features**: 7 sensor modalities (3-axis accelerometer, 3-axis gyroscope, heart rate)
- **Classes**: 8 human activities
- **Total Samples**: [FILL IN AFTER RUNNING MODELS]
- **Train/Test Split**: 80%/20% stratified split
- **Preprocessing**: StandardScaler normalization, sequence windowing

### Experimental Configuration

- **Hardware**: [SPECIFY YOUR HARDWARE]
- **Software**: TensorFlow 2.x, Python 3.x
- **Training**: Mixed precision (float16), early stopping, learning rate scheduling
- **Evaluation**: 5-fold cross-validation for robust metrics

## Model Architectures and Configurations

### 1. LSTM with Attention Mechanism

- **Architecture**: Bidirectional LSTM(128) → LSTM(64) → Attention → Dense layers
- **Sequence Length**: 50 timesteps
- **Parameters**: [FILL IN FROM RESULTS]
- **Training Time**: [FILL IN FROM RESULTS]

### 2. ResNet-Transformer Hybrid

- **Architecture**: 1D ResNet backbone + Transformer encoder
- **Sequence Length**: 128 timesteps
- **Parameters**: [FILL IN FROM RESULTS]
- **Training Time**: [FILL IN FROM RESULTS]

### 3. Temporal Convolutional Network (TCN)

- **Architecture**: Dilated convolutions with residual connections
- **Sequence Length**: 50 timesteps (optimized)
- **Parameters**: [FILL IN FROM RESULTS]
- **Training Time**: [FILL IN FROM RESULTS]

### 4. Transformer Encoder

- **Architecture**: Multi-head self-attention with positional encoding
- **Sequence Length**: 128 timesteps
- **Parameters**: [FILL IN FROM RESULTS]
- **Training Time**: [FILL IN FROM RESULTS]

## Classification Performance Results

### Overall Performance Metrics

| Model              | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | AUC (Macro) |
| ------------------ | ------------ | ----------------- | -------------- | ---------------- | ----------- |
| LSTM               | [FILL IN]    | [FILL IN]         | [FILL IN]      | [FILL IN]        | [FILL IN]   |
| ResNet-Transformer | [FILL IN]    | [FILL IN]         | [FILL IN]      | [FILL IN]        | [FILL IN]   |
| TCN                | [FILL IN]    | [FILL IN]         | [FILL IN]      | [FILL IN]        | [FILL IN]   |
| Transformer        | [FILL IN]    | [FILL IN]         | [FILL IN]      | [FILL IN]        | [FILL IN]   |

_Note: All metrics calculated using macro-averaging across all activity classes._

### Per-Class Performance Analysis

#### LSTM Model Results

| Activity Class | Precision | Recall    | F1-Score  | Support   |
| -------------- | --------- | --------- | --------- | --------- |
| Walking        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Running        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Sitting        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Standing       | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Lying          | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Up    | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Down  | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Jumping        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

#### ResNet-Transformer Model Results

| Activity Class | Precision | Recall    | F1-Score  | Support   |
| -------------- | --------- | --------- | --------- | --------- |
| Walking        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Running        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Sitting        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Standing       | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Lying          | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Up    | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Down  | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Jumping        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

#### TCN Model Results

| Activity Class | Precision | Recall    | F1-Score  | Support   |
| -------------- | --------- | --------- | --------- | --------- |
| Walking        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Running        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Sitting        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Standing       | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Lying          | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Up    | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Down  | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Jumping        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

#### Transformer Model Results

| Activity Class | Precision | Recall    | F1-Score  | Support   |
| -------------- | --------- | --------- | --------- | --------- |
| Walking        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Running        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Sitting        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Standing       | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Lying          | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Up    | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Climbing_Down  | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| Jumping        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

## Confusion Matrices

_[Confusion matrices will be generated as PNG files and can be referenced in the paper]_

### LSTM Confusion Matrix

_Reference: lstm_results.png - Figure X_

### ResNet-Transformer Confusion Matrix

_Reference: resnet_transformer_results.png - Figure X_

### TCN Confusion Matrix

_Reference: tcn_results.png - Figure X_

### Transformer Confusion Matrix

_Reference: transformer_results.png - Figure X_

## ROC Curves and AUC Analysis

### Multi-Class ROC Curves

_[ROC curves will be generated as PNG files and can be referenced in the paper]_

### AUC Scores by Class

| Model              | Walking   | Running   | Sitting   | Standing  | Lying     | Climbing_Up | Climbing_Down | Jumping   | Macro-Avg |
| ------------------ | --------- | --------- | --------- | --------- | --------- | ----------- | ------------- | --------- | --------- |
| LSTM               | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]   | [FILL IN]     | [FILL IN] | [FILL IN] |
| ResNet-Transformer | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]   | [FILL IN]     | [FILL IN] | [FILL IN] |
| TCN                | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]   | [FILL IN]     | [FILL IN] | [FILL IN] |
| Transformer        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]   | [FILL IN]     | [FILL IN] | [FILL IN] |

## Computational Performance Analysis

### Training Efficiency Metrics

| Model              | Training Time (min) | Inference Time (ms) | Memory Usage (GB) | Parameters (M) |
| ------------------ | ------------------- | ------------------- | ----------------- | -------------- |
| LSTM               | [FILL IN]           | [FILL IN]           | [FILL IN]         | [FILL IN]      |
| ResNet-Transformer | [FILL IN]           | [FILL IN]           | [FILL IN]         | [FILL IN]      |
| TCN                | [FILL IN]           | [FILL IN]           | [FILL IN]         | [FILL IN]      |
| Transformer        | [FILL IN]           | [FILL IN]           | [FILL IN]         | [FILL IN]      |

### Training and Validation Curves

_[Loss and accuracy curves will be generated as PNG files and can be referenced in the paper]_

## Feature Importance Analysis

### Sensor Modality Importance

| Model              | Acc_X     | Acc_Y     | Acc_Z     | Gyro_X    | Gyro_Y    | Gyro_Z    | Heart_Rate |
| ------------------ | --------- | --------- | --------- | --------- | --------- | --------- | ---------- |
| LSTM               | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]  |
| ResNet-Transformer | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]  |
| TCN                | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]  |
| Transformer        | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN]  |

_Note: Feature importance calculated using permutation importance and attention weights where applicable._

## Error Analysis

### Misclassification Patterns

| Model              | Most Confused Classes | Error Rate (%) | Common Misclassifications |
| ------------------ | --------------------- | -------------- | ------------------------- |
| LSTM               | [FILL IN]             | [FILL IN]      | [FILL IN]                 |
| ResNet-Transformer | [FILL IN]             | [FILL IN]      | [FILL IN]                 |
| TCN                | [FILL IN]             | [FILL IN]      | [FILL IN]                 |
| Transformer        | [FILL IN]             | [FILL IN]      | [FILL IN]                 |

### Sample Predictions

_[Examples of correct and incorrect predictions will be shown with confidence scores]_

## Statistical Significance Testing

### Pairwise Model Comparison (McNemar's Test)

| Model Pair                        | Chi-square | p-value   | Significant Difference |
| --------------------------------- | ---------- | --------- | ---------------------- |
| LSTM vs ResNet-Transformer        | [FILL IN]  | [FILL IN] | [FILL IN]              |
| LSTM vs TCN                       | [FILL IN]  | [FILL IN] | [FILL IN]              |
| LSTM vs Transformer               | [FILL IN]  | [FILL IN] | [FILL IN]              |
| ResNet-Transformer vs TCN         | [FILL IN]  | [FILL IN] | [FILL IN]              |
| ResNet-Transformer vs Transformer | [FILL IN]  | [FILL IN] | [FILL IN]              |
| TCN vs Transformer                | [FILL IN]  | [FILL IN] | [FILL IN]              |

### Cross-Validation Results

| Model              | Mean Accuracy | Std Dev   | 95% CI    |
| ------------------ | ------------- | --------- | --------- |
| LSTM               | [FILL IN]     | [FILL IN] | [FILL IN] |
| ResNet-Transformer | [FILL IN]     | [FILL IN] | [FILL IN] |
| TCN                | [FILL IN]     | [FILL IN] | [FILL IN] |
| Transformer        | [FILL IN]     | [FILL IN] | [FILL IN] |

## Discussion of Results

### Performance Analysis

The results demonstrate that [ANALYSIS TO BE COMPLETED AFTER DATA COLLECTION]:

1. **Accuracy Performance**: [TO BE ANALYZED]
2. **Computational Efficiency**: [TO BE ANALYZED]
3. **Class-specific Performance**: [TO BE ANALYZED]
4. **Real-time Applicability**: [TO BE ANALYZED]

### Key Findings

1. [TO BE DETERMINED BASED ON RESULTS]
2. [TO BE DETERMINED BASED ON RESULTS]
3. [TO BE DETERMINED BASED ON RESULTS]

### Limitations and Future Work

1. [TO BE IDENTIFIED BASED ON RESULTS]
2. [TO BE IDENTIFIED BASED ON RESULTS]
3. [TO BE IDENTIFIED BASED ON RESULTS]

## Conclusion

This comprehensive evaluation of four deep learning architectures for HAR provides valuable insights into the trade-offs between accuracy, computational efficiency, and real-time applicability. The results indicate that [CONCLUSION TO BE WRITTEN AFTER DATA COLLECTION AND ANALYSIS].

---

## Instructions for Filling This Template

1. **Run All Models**: Execute the individual model scripts to collect results
2. **Aggregate Results**: Run the aggregation script to generate comparison tables
3. **Fill in Metrics**: Replace all [FILL IN] placeholders with actual results from the generated files
4. **Add Figures**: Reference the generated PNG files as figures in your paper
5. **Complete Analysis**: Write the discussion and conclusion sections based on the collected data
6. **Review and Edit**: Ensure all metrics are accurate and properly formatted

## Generated Files to Reference

After running all scripts, you will have the following files to reference:

- `performance_summary.csv` - Overall performance metrics
- `per_class_comparison.csv` - Detailed per-class analysis
- `comprehensive_results_report.md` - Complete analysis report
- Individual model result files (JSON, CSV, PNG)
- Visualization files (confusion matrices, ROC curves, training curves)

_Note: This template provides the structure for a comprehensive results section. All [FILL IN] placeholders should be replaced with actual experimental results after running the models._
