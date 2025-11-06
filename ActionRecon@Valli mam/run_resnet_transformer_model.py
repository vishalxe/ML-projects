#!/usr/bin/env python3
"""
ResNet-Transformer Model for Human Activity Recognition - Standalone Execution Script
This script runs the ResNet-Transformer hybrid model and generates comprehensive results for journal publication.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
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

class ResNetTransformerModelRunner:
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
    
    def create_sequences(self, df, sequence_length=128, overlap=0.5):
        """Create sequences from sensor data"""
        print(f"Creating sequences with length {sequence_length} and overlap {overlap}...")
        
        feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate_bpm']
        sequences, labels = [], []
        step_size = int(sequence_length * (1 - overlap))

        for label in df['label'].unique():
            label_data = df[df['label'] == label].copy()
            label_data = label_data.sort_values('timestamp_ms')
            features = label_data[feature_cols].values
            
            for i in range(0, len(features) - sequence_length + 1, step_size):
                seq = features[i:i + sequence_length]
                if len(seq) == sequence_length:
                    sequences.append(seq)
                    labels.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels)
        print(f"Created {len(sequences)} sequences with shape {sequences.shape}")
        return sequences, labels
    
    def preprocess_data(self, sequences, labels):
        """Preprocess sequences and labels"""
        print("Preprocessing data...")
        scaler = StandardScaler()
        original_shape = sequences.shape
        flat = sequences.reshape(-1, sequences.shape[-1])
        norm = scaler.fit_transform(flat)
        sequences = norm.reshape(original_shape)

        le = LabelEncoder()
        labels_idx = le.fit_transform(labels)
        labels_oh = to_categorical(labels_idx)

        print(f"Number of classes: {len(le.classes_)}")
        print(f"Classes: {le.classes_}")
        return sequences, labels_oh, le, scaler
    
    def create_residual_block_1d(self, filters, kernel_size=3, stride=1, use_projection=False, dropout_rate=0.1):
        """Create 1D Residual Block"""
        def residual_block(inputs):
            x = Conv1D(filters, kernel_size, padding='same', strides=stride)(inputs)
            x = LayerNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters, kernel_size, padding='same', strides=1)(x)
            x = LayerNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            if use_projection:
                shortcut = Conv1D(filters, 1, strides=stride, padding='same')(inputs)
            else:
                shortcut = inputs
            
            x = Add()([x, shortcut])
            return Activation('relu')(x)
        return residual_block
    
    def create_positional_encoding(self, max_len, d_model):
        """Create positional encoding"""
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding[np.newaxis, :, :]
    
    def create_transformer_encoder_block(self, d_model=64, num_heads=8, dff=128, dropout_rate=0.1):
        """Create Transformer Encoder Block"""
        def transformer_block(inputs):
            # Multi-head attention
            attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)(inputs, inputs)
            attn = Dropout(dropout_rate)(attn)
            out1 = LayerNormalization(epsilon=1e-6)(inputs + attn)
            
            # Feed-forward network
            ffn = Sequential([Dense(dff, activation='relu'), Dense(d_model)])(out1)
            ffn = Dropout(dropout_rate)(ffn)
            out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
            
            return out2
        return transformer_block
    
    def create_resnet_transformer(self, input_shape, num_classes, resnet_filters=[32, 64], 
                                 res_blocks_per_stage=2, d_model=64, num_heads=8, 
                                 num_layers=2, dff=128, dropout_rate=0.1):
        """Create ResNet-Transformer hybrid model"""
        inputs = Input(shape=input_shape)

        # Stem
        x = Conv1D(resnet_filters[0], 7, strides=1, padding='same')(inputs)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)

        # ResNet stages
        for stage_idx, filters in enumerate(resnet_filters):
            for block_idx in range(res_blocks_per_stage):
                use_proj = (block_idx == 0 and stage_idx > 0)
                stride = 2 if use_proj else 1
                residual_block = self.create_residual_block_1d(filters, kernel_size=3, stride=stride, 
                                                             use_projection=use_proj, dropout_rate=dropout_rate)
                x = residual_block(x)

        # Project to transformer dim
        x = Conv1D(d_model, 1, padding='same')(x)

        # Positional encoding
        pos_encoding = self.create_positional_encoding(input_shape[0], d_model)
        pos_encoding_tensor = tf.constant(pos_encoding, dtype=tf.float32)
        # Add positional encoding via a Keras layer so KerasTensors aren't passed into raw TF functions
        x = Lambda(lambda t: t + pos_encoding_tensor[:, :tf.shape(t)[1], :])(x)

        # Transformer encoder stack
        for _ in range(num_layers):
            transformer_block = self.create_transformer_encoder_block(d_model=d_model, num_heads=num_heads, 
                                                                     dff=dff, dropout_rate=dropout_rate)
            x = transformer_block(x)

        # Head
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

        model = Model(inputs, outputs)
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the ResNet-Transformer model"""
        print("Building ResNet-Transformer model...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        self.model = self.create_resnet_transformer(input_shape, num_classes)
        
        # Custom loss function with label smoothing
        def smooth_categorical_crossentropy(y_true, y_pred, alpha=0.1):
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true_smooth = y_true * (1.0 - alpha) + alpha / num_classes
            return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
        
        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
        
        # Metrics
        top_3_accuracy = TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        top_5_accuracy = TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        
        self.model.compile(optimizer=optimizer,
                          loss=smooth_categorical_crossentropy,
                          metrics=['accuracy', top_3_accuracy, top_5_accuracy])
        
        total_params = self.model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_resnet_transformer_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
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
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.title('ResNet-Transformer - Training Accuracy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('ResNet-Transformer - Training Loss', fontweight='bold')
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
        plt.title('ResNet-Transformer - Confusion Matrix', fontweight='bold')
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
        plt.title('ResNet-Transformer - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Model Summary
        plt.subplot(2, 3, 5)
        plt.text(0.1, 0.8, 'Model: ResNet-Transformer', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.7, f'Accuracy: {self.results["accuracy"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.6, f'Precision: {self.results["precision"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.5, f'Recall: {self.results["recall"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.4, f'F1-Score: {self.results["f1_score"]*100:.2f}%', fontsize=10)
        plt.text(0.1, 0.3, f'AUC: {self.results["auc"]*100:.2f}%', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('ResNet-Transformer - Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('resnet_transformer_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Save model
        self.model.save('resnet_transformer_model_final.keras')
        
        # Save results as JSON
        with open('resnet_transformer_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save detailed classification report
        class_report_df = pd.DataFrame(self.results['classification_report']).T
        class_report_df.to_csv('resnet_transformer_classification_report.csv')
        
        print("Results saved to:")
        print("- resnet_transformer_model_final.keras")
        print("- resnet_transformer_results.json")
        print("- resnet_transformer_classification_report.csv")
        print("- resnet_transformer_results.png")

def main():
    """Main execution function"""
    print("=== ResNet-Transformer Model for Human Activity Recognition ===")
    
    # Initialize model runner
    csv_path = r"c:\Users\Vishal\Downloads\HAR_synthetic_full\HAR_synthetic_full.csv"
    runner = ResNetTransformerModelRunner(csv_path)
    
    # Load and preprocess data
    df = runner.load_and_preprocess_data()
    sequences, labels = runner.create_sequences(df, sequence_length=128, overlap=0.5)
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
    print(f"\n=== RESNET-TRANSFORMER RESULTS ===")
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
    
    print("\n=== ResNet-Transformer Model Execution Complete ===")

if __name__ == "__main__":
    main()
