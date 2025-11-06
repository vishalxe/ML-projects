#!/usr/bin/env python3
"""
LSTM Model for Human Activity Recognition - Standalone Execution Script
This script runs the LSTM model and generates comprehensive results for journal publication.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import train_test_split, cross_val_score
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

class LSTMModelRunner:
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
    
    def create_sequences(self, df, sequence_length=50, overlap=0.5):
        """Create sequences from sensor data for LSTM input"""
        print(f"Creating sequences with length {sequence_length} and overlap {overlap}...")
        
        feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate_bpm']
        sequences = []
        labels = []
        
        for label in df['label'].unique():
            label_data = df[df['label'] == label].copy()
            label_data = label_data.sort_values('timestamp_ms')
            features = label_data[feature_cols].values
            
            step_size = int(sequence_length * (1 - overlap))
            
            for i in range(0, len(features) - sequence_length + 1, step_size):
                sequence = features[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    sequences.append(sequence)
                    labels.append(label)
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels)
        
        print(f"Created {len(sequences)} sequences")
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
    
    def create_attention_lstm(self, input_shape, num_classes):
        """Create LSTM model with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = LSTM(64, return_sequences=True, dropout=0.3)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Attention mechanism
        attention = Dense(64, activation='tanh')(lstm_out)
        attention = Dense(1)(attention)
        attention_weights = Softmax(axis=1)(attention)
        
        # Apply attention
        attended = Multiply()([lstm_out, attention_weights])
        attended = GlobalAveragePooling1D()(attended)
        
        # Classification head
        x = Dense(64, activation='relu')(attended)
        x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the LSTM model"""
        print("Building LSTM model...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        self.model = self.create_attention_lstm(input_shape, num_classes)
        
        # Custom loss function with label smoothing
        def smooth_categorical_crossentropy(y_true, y_pred, alpha=0.1):
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true_smooth = y_true * (1.0 - alpha) + alpha / num_classes
            return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
        
        # Optimizer
        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
        
        # Metrics
        top_3_accuracy = TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        top_5_accuracy = TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=smooth_categorical_crossentropy,
            metrics=['accuracy', top_3_accuracy, top_5_accuracy]
        )
        
        total_params = self.model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_lstm_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # Train the model
        print("Starting training...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
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
        
        # Training curves
        plt.figure(figsize=(15, 10))
        
        # Accuracy
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.title('LSTM - Training Accuracy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('LSTM - Training Loss', fontweight='bold')
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
        plt.title('LSTM - Confusion Matrix', fontweight='bold')
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
        plt.title('LSTM - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Model Summary
        plt.subplot(2, 3, 5)
        plt.text(0.1, 0.8, 'Model: LSTM with Attention', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'Accuracy: {self.results["accuracy"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.6, f'Precision: {self.results["precision"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.5, f'Recall: {self.results["recall"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.4, f'F1-Score: {self.results["f1_score"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.3, f'AUC: {self.results["auc"]*100:.2f}%', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('LSTM - Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('lstm_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Save model
        self.model.save('lstm_model_final.keras')
        
        # Save results as JSON
        with open('lstm_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save detailed classification report
        class_report_df = pd.DataFrame(self.results['classification_report']).T
        class_report_df.to_csv('lstm_classification_report.csv')
        
        print("Results saved to:")
        print("- lstm_model_final.keras")
        print("- lstm_results.json")
        print("- lstm_classification_report.csv")
        print("- lstm_results.png")

def main():
    """Main execution function"""
    print("=== LSTM Model for Human Activity Recognition ===")
    
    # Initialize model runner
    csv_path = r"c:\Users\Vishal\Downloads\HAR_synthetic_full\HAR_synthetic_full.csv"
    runner = LSTMModelRunner(csv_path)
    
    # Load and preprocess data
    df = runner.load_and_preprocess_data()
    sequences, labels = runner.create_sequences(df, sequence_length=50, overlap=0.5)
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
    print(f"\n=== LSTM RESULTS ===")
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
    
    print("\n=== LSTM Model Execution Complete ===")

if __name__ == "__main__":
    main()
