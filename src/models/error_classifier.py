import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict, Callable, Optional
import pandas as pd
from tqdm import tqdm

class ErrorClassifier:
    def __init__(self, sequence_length: int = 100, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.model = None
        self.feature_columns = ['speed', 'rpm', 'temperature', 'fuel_level']
        self.label_mapping = {
            'normal': 0,
            'sensor_dropout': 1,
            'noise_spike': 2,
            'signal_drift': 3
        }
        
        # Disable mixed precision as it might cause numerical instability
        tf.keras.mixed_precision.set_global_policy('float32')
    
    def build_model(self):
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            self.model = models.Sequential([
                layers.Input(shape=(self.sequence_length, len(self.feature_columns))),
                # Add BatchNormalization for better training stability
                layers.BatchNormalization(),
                # Reduce LSTM complexity
                layers.LSTM(32, return_sequences=True),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.LSTM(16),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(len(self.label_mapping), activation='softmax')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.001,
                    clipnorm=1.0  # Add gradient clipping
                ),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return self.model
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(data_path)
        
        # Extract features and labels
        X = df[self.feature_columns].values
        y = df['error_type'].map(self.label_mapping).values
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])
            
        return np.array(X_seq), np.array(y_seq)
    
    def load_dataset(self, train_files: list, valid_files: list, 
                    progress_callback: Optional[Callable[[float], None]] = None
                    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        
        def load_and_process_files(files, desc):
            all_X = []
            all_y = []
            total = len(files)
            
            # Store statistics for normalization
            mean_values = None
            std_values = None
            
            # First pass: compute statistics
            print(f"\nComputing statistics for {desc}...")
            for file in tqdm(files, desc="Computing statistics"):
                df = pd.read_csv(file)
                X = df[self.feature_columns].values
                if mean_values is None:
                    mean_values = np.mean(X, axis=0)
                    std_values = np.std(X, axis=0)
                else:
                    mean_values = (mean_values + np.mean(X, axis=0)) / 2
                    std_values = (std_values + np.std(X, axis=0)) / 2
            
            # Second pass: normalize and create sequences
            print(f"\nProcessing {desc}...")
            for i, file in enumerate(tqdm(files, desc=desc)):
                try:
                    df = pd.read_csv(file)
                    X = df[self.feature_columns].values
                    y = df['error_type'].map(self.label_mapping).values
                    
                    # Normalize data
                    X = (X - mean_values) / (std_values + 1e-7)
                    
                    # Create sequences
                    for j in range(0, len(X) - self.sequence_length + 1, self.sequence_length):
                        X_seq = X[j:j + self.sequence_length]
                        y_seq = y[j + self.sequence_length - 1]
                        all_X.append(X_seq)
                        all_y.append(y_seq)
                    
                    if progress_callback:
                        progress_callback((i + 1) / total)
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue
            
            if not all_X:
                raise ValueError("No valid data found in files")
                
            return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int32)

        # Load training and validation data
        X_train, y_train = load_and_process_files(train_files, "Loading training files")
        X_valid, y_valid = load_and_process_files(valid_files, "Loading validation files")
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

        # Batch and prefetch
        train_dataset = train_dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        valid_dataset = valid_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Enable GPU acceleration for datasets
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        train_dataset = train_dataset.with_options(options)
        valid_dataset = valid_dataset.with_options(options)
        
        return train_dataset, valid_dataset
