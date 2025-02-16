import sys
import os
from datetime import datetime
from typing import Optional
import traceback  # Add traceback import

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import tensorflow as tf
from models.error_classifier import ErrorClassifier
import glob
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from utils.gpu_check import setup_gpu

def load_dataset_metadata():
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'training')
    metadata = {}
    
    for split in ['train', 'valid', 'test']:
        meta_path = os.path.join(base_dir, split, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata[split] = json.load(f)
        
    return metadata

def load_existing_model(car_type: str, classifier: ErrorClassifier) -> Optional[tf.keras.Model]:
    """Try to load an existing model for the given car type"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(model_dir, f'{car_type}_model.weights.h5')  # Changed extension
    
    if os.path.exists(model_path):
        try:
            st.info(f"Found existing model at {model_path}")
            # Rebuild model with same architecture first
            model = classifier.build_model()
            # Then load weights
            model.load_weights(model_path)
            return model
        except Exception as e:
            st.warning(f"Could not load existing model: {e}")
            st.code(traceback.format_exc())
    return None

def train_model(car_type: str, metadata: dict):
    # Check for GPU availability
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    st.write(f"GPU Available: {gpu_available}")
    if gpu_available:
        st.write("Using GPU for training")
    else:
        st.write("Using CPU for training (training might be slower)")

    # Configure memory growth for GPU
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            st.write(f"Configured memory growth for GPU: {gpu}")
        except RuntimeError as e:
            st.write(f"Error configuring GPU: {e}")

    st.write("Initializing model...")
    classifier = ErrorClassifier(batch_size=32)
    
    # Try to load existing model
    model = load_existing_model(car_type, classifier)
    if model is None:
        model = classifier.build_model()
        st.write("Created new model")
    else:
        st.write("Loaded existing model")
        
        # Add option to continue training
        if not st.checkbox("Continue training existing model", value=True):
            model = classifier.build_model()
            st.write("Created new model instead")
    
    # Add epochs selection
    epochs = st.slider("Number of epochs", min_value=1, max_value=50, value=10)
    
    # Get files for selected car type
    train_files = [f['path'] for f in metadata['train']['files'] 
                  if f['car_type'] == car_type]
    valid_files = [f['path'] for f in metadata['valid']['files'] 
                  if f['car_type'] == car_type]
    
    st.write(f"Found {len(train_files)} training files and {len(valid_files)} validation files")
    
    # Data loading progress
    data_progress = st.progress(0)
    data_status = st.empty()
    
    def update_progress(progress):
        data_progress.progress(progress)
        data_status.text(f"Loading data: {int(progress * 100)}%")
    
    st.write("Loading and preprocessing data...")
    try:
        train_dataset, valid_dataset = classifier.load_dataset(
            train_files, valid_files, 
            progress_callback=update_progress
        )
        
        # Unbatch datasets first
        train_data = [(x.numpy(), y.numpy()) for x, y in train_dataset.unbatch()]
        valid_data = [(x.numpy(), y.numpy()) for x, y in valid_dataset.unbatch()]
        
        train_size = len(train_data)
        valid_size = len(valid_data)
        
        st.write(f"Training samples: {train_size}")
        st.write(f"Validation samples: {valid_size}")
        
        # Recreate datasets from unbatched data
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor([x for x, _ in train_data]),
                tf.convert_to_tensor([y for _, y in train_data])
            )
        ).shuffle(1024).batch(classifier.batch_size)
        
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor([x for x, _ in valid_data]),
                tf.convert_to_tensor([y for _, y in valid_data])
            )
        ).batch(classifier.batch_size)
        
        # Training progress tracking - Move these before the callback class
        train_progress = st.progress(0)
        epoch_status = st.empty()
        metrics_status = st.empty()
        
        st.write("Starting training...")
        
        # Callback class that now has access to the progress variables
        class TrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.train_progress = train_progress
                self.epoch_status = epoch_status
                self.metrics_status = metrics_status
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_status.text(f'Training Epoch {epoch+1}/{epochs}')
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                self.train_progress.progress(progress)
                if logs:
                    self.metrics_status.text(
                        f"Loss: {logs.get('loss', 0):.4f} | "
                        f"Accuracy: {logs.get('accuracy', 0):.4f} | "
                        f"Val Loss: {logs.get('val_loss', 0):.4f} | "
                        f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
                    )
        
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            callbacks=[TrainingCallback()],
            verbose=1
        )
        
        return model, history, classifier  # Add classifier to return values
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def evaluate_model(model, metadata: dict, car_type: str, classifier: ErrorClassifier):
    # Get test files
    test_files = [f['path'] for f in metadata['test']['files'] 
                 if f['car_type'] == car_type]
    
    st.subheader("Model Evaluation")
    st.write(f"Evaluating model on {len(test_files)} test files...")
    
    # Data loading progress
    data_progress = st.progress(0)
    data_status = st.empty()
    
    def update_progress(progress):
        data_progress.progress(progress)
        data_status.text(f"Loading test data: {int(progress * 100)}%")
    
    try:
        test_dataset = classifier.load_dataset(
            test_files, test_files,  # Pass same files twice since load_dataset expects both train and valid
            progress_callback=update_progress
        )[0]  # Only take first dataset since we don't need validation for testing
        
        # Unbatch dataset
        test_data = [(x.numpy(), y.numpy()) for x, y in test_dataset.unbatch()]
        test_size = len(test_data)
        
        # Recreate dataset from unbatched data
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor([x for x, _ in test_data]),
                tf.convert_to_tensor([y for _, y in test_data])
            )
        ).batch(classifier.batch_size)
        
        # No need for steps parameter
        results = model.evaluate(test_dataset, verbose=0)
        
        # Display metrics - Handle different metric names
        metrics = dict(zip(model.metrics_names, results))
        col1, col2 = st.columns(2)
        
        # Handle different possible metric names
        accuracy = metrics.get('accuracy', metrics.get('acc', 0.0))
        loss = metrics.get('loss', 0.0)
        
        col1.metric("Test Accuracy", f"{accuracy:.4f}")
        col2.metric("Test Loss", f"{loss:.4f}")
        
        # Generate predictions
        st.write("Generating predictions...")
        predictions = model.predict(test_dataset)
        
        # Calculate confusion matrix
        y_true = []
        for _, labels in test_dataset.unbatch():
            y_true.append(labels.numpy())
        y_pred = np.argmax(predictions, axis=1)
        
        # Convert numeric labels back to string labels
        label_names = list(classifier.label_mapping.keys())
        conf_matrix = pd.DataFrame(
            confusion_matrix(y_true, y_pred),
            columns=label_names,
            index=label_names
        )
        
        # Display confusion matrix
        st.write("Confusion Matrix:")
        fig = px.imshow(conf_matrix,
                       labels=dict(x="Predicted", y="Actual"),
                       x=label_names,
                       y=label_names,
                       color_continuous_scale="RdBu")
        st.plotly_chart(fig)
        
        # Display classification report
        st.write("Classification Report:")
        report = classification_report(y_true, y_pred, target_names=label_names)
        st.code(report)
        
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        st.code(traceback.format_exc())

def check_gpu():
    st.write("\nChecking GPU setup...")
    st.write(f"TensorFlow version: {tf.__version__}")
    
    try:
        # List physical devices
        physical_devices = tf.config.list_physical_devices()
        st.write("\nAll physical devices:", physical_devices)
        
        # List GPU devices
        gpu_devices = tf.config.list_physical_devices('GPU')
        st.write("\nGPU devices:", gpu_devices)
        
        if gpu_devices:
            # Get GPU device details
            for device in gpu_devices:
                st.write(f"\nGPU device: {device}")
                try:
                    device_details = tf.config.experimental.get_device_details(device)
                    st.write("Device details:", device_details)
                except:
                    st.write("Could not get device details")
            
            # Test GPU availability
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                st.write("\nGPU test successful!")
                st.write("Matrix multiplication result:", c.numpy())
        else:
            st.error("No GPU devices found!")
            st.write("\nPossible issues:")
            st.write("1. CUDA not installed or not in PATH")
            st.write("2. Incompatible CUDA version")
            st.write("3. GPU drivers not installed or outdated")
            st.write("4. TensorFlow CPU version installed instead of GPU version")
    except Exception as e:
        st.error(f"Error checking GPU: {str(e)}")

def main():
    # Setup GPU first
    setup_gpu()
    
    st.title("Car Error Detection Dashboard")
    
    # Add GPU check button
    if st.sidebar.button("Check GPU Status"):
        check_gpu()
    
    # Load dataset metadata
    metadata = load_dataset_metadata()
    if not metadata:
        st.error("No dataset found. Please generate the dataset first.")
        return
    
    # Sidebar for car selection
    st.sidebar.header("Settings")
    car_types = list(set(f['car_type'] for f in metadata['train']['files']))
    selected_car = st.sidebar.selectbox("Select Car Type", car_types)
    
    # Display dataset statistics
    st.header("Dataset Statistics")
    stats = {
        'train': len([f for f in metadata['train']['files'] if f['car_type'] == selected_car]),
        'valid': len([f for f in metadata['valid']['files'] if f['car_type'] == selected_car]),
        'test': len([f for f in metadata['test']['files'] if f['car_type'] == selected_car])
    }
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", stats['train'])
    col2.metric("Validation Samples", stats['valid'])
    col3.metric("Test Samples", stats['test'])
    
    # Display model info
    st.sidebar.header("Model Settings")
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(model_dir, f'{selected_car}_model.keras')
    
    if os.path.exists(model_path):
        st.sidebar.success(f"Model exists for {selected_car}")
        model_stats = os.stat(model_path)
        st.sidebar.text(f"Last modified: {datetime.fromtimestamp(model_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.warning(f"No existing model for {selected_car}")
    
    # Train model button
    if st.button("Train Model"):
        st.header("Model Training")
        with st.spinner('Training model...'):
            model, history, classifier = train_model(selected_car, metadata)  # Get classifier from train_model
            
            if model is not None and history is not None:
                # Plot training history
                hist_df = pd.DataFrame(history.history)
                fig = px.line(hist_df, title='Model Training History')
                st.plotly_chart(fig)
                
                # Save model weights with correct extension
                model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f'{selected_car}_model.weights.h5')  # Changed extension
                model.save_weights(model_path)
                st.success(f"Model weights saved to {model_path}")
                
                # Evaluate model
                evaluate_model(model, metadata, selected_car, classifier)

if __name__ == "__main__":
    main()
