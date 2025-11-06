# Results Section: Comparative Analysis of Deep Learning Models for Human Activity Recognition

## Abstract of Results

This study presents a comprehensive evaluation of four deep learning architectures for Human Activity Recognition (HAR) using multi-modal sensor data. We compared Long Short-Term Memory (LSTM) networks, ResNet-Transformer hybrid models, Temporal Convolutional Networks (TCN), and Transformer encoders on a standardized HAR dataset. The evaluation encompasses classification performance, computational efficiency, and real-time applicability metrics.

## Dataset and Experimental Setup

### Dataset Characteristics

- **Dataset**: HAR_synthetic_full.csv
- **Features**: 7 sensor modalities (3-axis accelerometer, 3-axis gyroscope, heart rate)
- **Classes**: 8 human activities
- **Total Samples**: [To be filled after running models]
- **Train/Test Split**: 80%/20% stratified split
- **Preprocessing**: StandardScaler normalization, sequence windowing

### Experimental Configuration

- **Hardware**: [To be specified]
- **Software**: TensorFlow 2.x, Python 3.x
- **Training**: Mixed precision (float16), early stopping, learning rate scheduling
- **Evaluation**: 5-fold cross-validation for robust metrics

## Model Architectures and Configurations

### 1. LSTM with Attention Mechanism

- **Architecture**: Bidirectional LSTM(128) → LSTM(64) → Attention → Dense layers
- **Sequence Length**: 50 timesteps
- **Parameters**: ~[To be measured]
- **Training Time**: [To be measured]

### 2. ResNet-Transformer Hybrid

- **Architecture**: 1D ResNet backbone + Transformer encoder
- **Sequence Length**: 128 timesteps
- **Parameters**: ~[To be measured]
- **Training Time**: [To be measured]

### 3. Temporal Convolutional Network (TCN)

- **Architecture**: Dilated convolutions with residual connections
- **Sequence Length**: 50 timesteps (optimized)
- **Parameters**: ~[To be measured]
- **Training Time**: [To be measured]

### 4. Transformer Encoder

- **Architecture**: Multi-head self-attention with positional encoding
- **Sequence Length**: 128 timesteps
- **Parameters**: ~[To be measured]
- **Training Time**: [To be measured]

## Classification Performance Results

### Overall Performance Metrics

| Model              | Accuracy (%)   | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | AUC (Macro)    |
| ------------------ | -------------- | ----------------- | -------------- | ---------------- | -------------- |
| LSTM               | [To be filled] | [To be filled]    | [To be filled] | [To be filled]   | [To be filled] |
| ResNet-Transformer | [To be filled] | [To be filled]    | [To be filled] | [To be filled]   | [To be filled] |
| TCN                | [To be filled] | [To be filled]    | [To be filled] | [To be filled]   | [To be filled] |
| Transformer        | [To be filled] | [To be filled]    | [To be filled] | [To be filled]   | [To be filled] |

_Note: All metrics calculated using macro-averaging across all activity classes._

### Per-Class Performance Analysis

#### LSTM Model Results

| Activity Class | Precision      | Recall         | F1-Score       | Support        |
| -------------- | -------------- | -------------- | -------------- | -------------- |
| Walking        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Running        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Sitting        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Standing       | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Lying          | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Up    | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Down  | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Jumping        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

#### ResNet-Transformer Model Results

| Activity Class | Precision      | Recall         | F1-Score       | Support        |
| -------------- | -------------- | -------------- | -------------- | -------------- |
| Walking        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Running        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Sitting        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Standing       | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Lying          | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Up    | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Down  | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Jumping        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

#### TCN Model Results

| Activity Class | Precision      | Recall         | F1-Score       | Support        |
| -------------- | -------------- | -------------- | -------------- | -------------- |
| Walking        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Running        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Sitting        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Standing       | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Lying          | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Up    | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Down  | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Jumping        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

#### Transformer Model Results

| Activity Class | Precision      | Recall         | F1-Score       | Support        |
| -------------- | -------------- | -------------- | -------------- | -------------- |
| Walking        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Running        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Sitting        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Standing       | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Lying          | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Up    | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Climbing_Down  | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Jumping        | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

## Confusion Matrices

### LSTM Confusion Matrix

```
[To be generated - will show actual vs predicted classifications]
```

### ResNet-Transformer Confusion Matrix

```
[To be generated - will show actual vs predicted classifications]
```

### TCN Confusion Matrix

```
[To be generated - will show actual vs predicted classifications]
```

### Transformer Confusion Matrix

```
[To be generated - will show actual vs predicted classifications]
```

## ROC Curves and AUC Analysis

### Multi-Class ROC Curves

_[ROC curves will be generated for each model showing performance across all activity classes]_

### AUC Scores by Class

| Model              | Walking        | Running        | Sitting        | Standing       | Lying          | Climbing_Up    | Climbing_Down  | Jumping        | Macro-Avg      |
| ------------------ | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| LSTM               | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| ResNet-Transformer | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| TCN                | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Transformer        | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

## Computational Performance Analysis

### Training Efficiency Metrics

| Model              | Training Time (min) | Inference Time (ms) | Memory Usage (GB) | Parameters (M) |
| ------------------ | ------------------- | ------------------- | ----------------- | -------------- |
| LSTM               | [To be filled]      | [To be filled]      | [To be filled]    | [To be filled] |
| ResNet-Transformer | [To be filled]      | [To be filled]      | [To be filled]    | [To be filled] |
| TCN                | [To be filled]      | [To be filled]      | [To be filled]    | [To be filled] |
| Transformer        | [To be filled]      | [To be filled]      | [To be filled]    | [To be filled] |

### Training and Validation Curves

_[Loss and accuracy curves will be generated for each model showing convergence behavior]_

## Feature Importance Analysis

### Sensor Modality Importance

| Model              | Acc_X          | Acc_Y          | Acc_Z          | Gyro_X         | Gyro_Y         | Gyro_Z         | Heart_Rate     |
| ------------------ | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| LSTM               | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| ResNet-Transformer | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| TCN                | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |
| Transformer        | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] | [To be filled] |

_Note: Feature importance calculated using permutation importance and attention weights where applicable._

## Error Analysis

### Misclassification Patterns

| Model              | Most Confused Classes | Error Rate (%) | Common Misclassifications |
| ------------------ | --------------------- | -------------- | ------------------------- |
| LSTM               | [To be filled]        | [To be filled] | [To be filled]            |
| ResNet-Transformer | [To be filled]        | [To be filled] | [To be filled]            |
| TCN                | [To be filled]        | [To be filled] | [To be filled]            |
| Transformer        | [To be filled]        | [To be filled] | [To be filled]            |

### Sample Predictions

_[Examples of correct and incorrect predictions will be shown with confidence scores]_

## Statistical Significance Testing

### Pairwise Model Comparison (McNemar's Test)

| Model Pair                        | Chi-square     | p-value        | Significant Difference |
| --------------------------------- | -------------- | -------------- | ---------------------- |
| LSTM vs ResNet-Transformer        | [To be filled] | [To be filled] | [To be filled]         |
| LSTM vs TCN                       | [To be filled] | [To be filled] | [To be filled]         |
| LSTM vs Transformer               | [To be filled] | [To be filled] | [To be filled]         |
| ResNet-Transformer vs TCN         | [To be filled] | [To be filled] | [To be filled]         |
| ResNet-Transformer vs Transformer | [To be filled] | [To be filled] | [To be filled]         |
| TCN vs Transformer                | [To be filled] | [To be filled] | [To be filled]         |

### Cross-Validation Results

| Model              | Mean Accuracy  | Std Dev        | 95% CI         |
| ------------------ | -------------- | -------------- | -------------- |
| LSTM               | [To be filled] | [To be filled] | [To be filled] |
| ResNet-Transformer | [To be filled] | [To be filled] | [To be filled] |
| TCN                | [To be filled] | [To be filled] | [To be filled] |
| Transformer        | [To be filled] | [To be filled] | [To be filled] |

## Discussion of Results

### Performance Analysis

The results demonstrate that [analysis to be completed after data collection]:

1. **Accuracy Performance**: [To be analyzed]
2. **Computational Efficiency**: [To be analyzed]
3. **Class-specific Performance**: [To be analyzed]
4. **Real-time Applicability**: [To be analyzed]

### Key Findings

1. [To be determined based on results]
2. [To be determined based on results]
3. [To be determined based on results]

### Limitations and Future Work

1. [To be identified based on results]
2. [To be identified based on results]
3. [To be identified based on results]

## Conclusion

This comprehensive evaluation of four deep learning architectures for HAR provides valuable insights into the trade-offs between accuracy, computational efficiency, and real-time applicability. The results indicate that [conclusion to be written after data collection and analysis].

---

_Note: This template provides the structure for a comprehensive results section. All [To be filled] placeholders should be replaced with actual experimental results after running the models._
