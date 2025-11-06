# HAR Models - Comprehensive Results Report
==================================================

## Performance Summary

             Model  Accuracy (%)  Precision (%)  Recall (%)  F1-Score (%)  AUC (%)
              Lstm         91.38          95.14       91.89         91.34    99.99
Resnet-Transformer         85.92          81.25       87.50         83.33    99.88
               Tcn         99.97          99.98       99.98         99.98   100.00

**Best Performing Model**: Tcn with 99.97% accuracy

## Statistical Analysis

            Mean   Std    Min     Max  Range
accuracy   92.42  5.78  85.92   99.97  14.05
precision  92.12  7.94  81.25   99.98  18.73
recall     93.12  5.17  87.50   99.98  12.48
f1_score   91.55  6.80  83.33   99.98  16.64
auc        99.96  0.05  99.88  100.00   0.12

## Key Findings

1. **Model Performance Ranking**:
   1. Tcn: 99.97% accuracy
   2. Lstm: 91.38% accuracy
   3. Resnet-Transformer: 85.92% accuracy

2. **Performance Consistency**:
   - Accuracy (%): Standard deviation = 7.08%
   - Precision (%): Standard deviation = 9.72%
   - Recall (%): Standard deviation = 6.33%
   - F1-Score (%): Standard deviation = 8.33%
   - AUC (%): Standard deviation = 0.07%
