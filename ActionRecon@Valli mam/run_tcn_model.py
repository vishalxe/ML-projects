#!/usr/bin/env python3
"""
TCN Model for Human Activity Recognition - Standalone Execution Script
This script runs the optimized TCN model and generates comprehensive results for journal publication.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TCNModelRunner:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.history = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess sensor data"""
        print("Loading sensor data...")
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=['label'])
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique labels: {df['label'].unique()}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def create_sequences_optimized(self, df, sequence_length=50, overlap=0.3):
        """Create optimized sequences from sensor data for TCN input"""
        print(f"Creating OPTIMIZED sequences with length {sequence_length} and overlap {overlap}...")
        
        feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate_bpm']
        sequences = []
        labels = []
        
        # Pre-allocate arrays for better memory efficiency
        max_sequences = len(df) // sequence_length * 2  # Rough estimate
        sequences = np.zeros((max_sequences, sequence_length, len(feature_cols)), dtype=np.float32)
        labels = []
        seq_count = 0
        
        # Group by label to create sequences for each activity
        for label in df['label'].unique():
            label_data = df[df['label'] == label].copy()
            label_data = label_data.sort_values('timestamp_ms')
            features = label_data[feature_cols].values
            
            # Create overlapping sequences with larger step size for speed
            step_size = max(1, int(sequence_length * (1 - overlap)))
            
            for i in range(0, len(features) - sequence_length + 1, step_size):
                if seq_count >= max_sequences:
                    break
                sequence = features[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    sequences[seq_count] = sequence
                    labels.append(label)
                    seq_count += 1
        
        # Trim arrays to actual size
        sequences = sequences[:seq_count]
        labels = np.array(labels)
        
        print(f"Created {len(sequences)} sequences (OPTIMIZED)")
        print(f"Sequence shape: {sequences.shape}")
        
        return sequences, labels
    
    def preprocess_data(self, sequences, labels):
        """Preprocess sequences and labels"""
        print("Preprocessing data...")
        
        # Normalize features
        scaler = StandardScaler()
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_normalized = scaler.fit_transform(sequences_reshaped)
        sequences = sequences_normalized.reshape(original_shape)
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        labels_onehot = to_categorical(labels_encoded)
        
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Classes: {label_encoder.classes_}")
        
        return sequences, labels_onehot, label_encoder, scaler
    
    def create_optimized_temporal_block(self, filters, kernel_size, dilation_rate, dropout_rate=0.1):
        """Optimized Temporal Block with efficient convolutions"""
        def temporal_block(inputs):
            # First convolution block
            x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                      padding='causal', activation='relu')(inputs)
            x = LayerNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Second convolution block
            x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
                      padding='causal', activation='relu')(x)
            x = LayerNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Residual connection
            if inputs.shape[-1] != filters:
                residual = Conv1D(filters, 1, padding='same')(inputs)
            else:
                residual = inputs
            
            # Add residual and apply activation
            output = Add()([x, residual])
            return Activation('relu')(output)
        return temporal_block
    
    def create_optimized_tcn_model(self, input_shape, num_classes, nb_filters=32, 
                                 kernel_size=3, nb_stacks=2, dilations=[1, 2, 4, 8], 
                                 dropout_rate=0.1):
        """Create an optimized Temporal Convolutional Network model"""
        
        inputs = Input(shape=input_shape)
        
        # Initial convolution to increase channels
        x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(inputs)
        
        # Stack of optimized temporal blocks
        for stack in range(nb_stacks):
            for dilation in dilations:
                temporal_block = self.create_optimized_temporal_block(
                    filters=nb_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout_rate=dropout_rate
                )
                x = temporal_block(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Simplified classification head
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the optimized TCN model"""
        print("Building optimized TCN model...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        self.model = self.create_optimized_tcn_model(input_shape, num_classes)
        
        # Optimized loss function (no label smoothing for speed)
        def fast_categorical_crossentropy(y_true, y_pred):
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Optimized optimizer with higher learning rate for faster convergence
        optimizer = AdamW(learning_rate=0.003, weight_decay=1e-5)
        
        # Simplified metrics (only accuracy for speed)
        metrics = ['accuracy']
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=fast_categorical_crossentropy,
            metrics=metrics
        )
        
        total_params = self.model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        # Optimized callbacks for faster training
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6, verbose=1),
            ModelCheckpoint('best_tcn_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        print("Starting optimized training...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,  # Reduced epochs for faster training
            batch_size=64,  # Larger batch size for speed
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return training_time, total_params
    
    def evaluate_model(self, X_test, y_test, label_encoder):
        """Comprehensive model evaluation"""
        print("Evaluating model...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Basic metrics
        accuracy = np.mean(y_pred == y_true)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # ROC AUC (multi-class)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        return self.results
    
    def generate_plots(self, label_encoder):
        """Generate visualization plots"""
        print("Generating plots...")
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.title('TCN (Optimized) - Training Accuracy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('TCN (Optimized) - Training Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(2, 3, 3)
        cm = np.array(self.results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.title('TCN (Optimized) - Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # ROC Curves
        plt.subplot(2, 3, 4)
        y_test_onehot = to_categorical(self.results['true_labels'])
        y_pred_proba = np.array(self.results['prediction_probabilities'])
        
        for i, class_name in enumerate(label_encoder.classes_):
            fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_test_onehot[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('TCN (Optimized) - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Model Summary
        plt.subplot(2, 3, 5)
        plt.text(0.1, 0.8, 'Model: TCN (Optimized)', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'Accuracy: {self.results["accuracy"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.6, f'Precision: {self.results["precision"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.5, f'Recall: {self.results["recall"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.4, f'F1-Score: {self.results["f1_score"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.3, f'AUC: {self.results["auc"]*100:.2f}%', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('TCN (Optimized) - Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('tcn_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Save model
        self.model.save('tcn_model_final.keras')
        
        # Save results as JSON
        with open('tcn_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save detailed classification report
        class_report_df = pd.DataFrame(self.results['classification_report']).T
        class_report_df.to_csv('tcn_classification_report.csv')
        
        print("Results saved to:")
        print("- tcn_model_final.keras")
        print("- tcn_results.json")
        print("- tcn_classification_report.csv")
        print("- tcn_results.png")

def main():
    """Main execution function"""
    print("=== TCN (Optimized) Model for Human Activity Recognition ===")
    
    # Initialize model runner
    csv_path = r"c:\Users\Vishal\Downloads\HAR_synthetic_full\HAR_synthetic_full.csv"
    runner = TCNModelRunner(csv_path)
    
    # Load and preprocess data
    df = runner.load_and_preprocess_data()
    sequences, labels = runner.create_sequences_optimized(df, sequence_length=50, overlap=0.3)
    X, y, label_encoder, scaler = runner.preprocess_data(sequences, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train model
    training_time, total_params = runner.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = runner.evaluate_model(X_test, y_test, label_encoder)
    
    # Print results
    print(f"\n=== TCN (OPTIMIZED) RESULTS ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall: {results['recall']*100:.2f}%")
    print(f"F1-Score: {results['f1_score']*100:.2f}%")
    print(f"AUC: {results['auc']*100:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Parameters: {total_params:,}")
    
    # Generate plots
    runner.generate_plots(label_encoder)
    
    # Save results
    runner.save_results()
    
    print("\n=== TCN (Optimized) Model Execution Complete ===")

if __name__ == "__main__":
    main()
