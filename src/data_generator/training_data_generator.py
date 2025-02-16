import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from .synthetic_data import SyntheticDataGenerator
import shutil
import multiprocessing
from multiprocessing import Pool

def generate_worker_init():
    """Initialize worker process"""
    np.random.seed()

def generate_sample_worker(args):
    """Standalone worker function for multiprocessing"""
    try:
        car_type, i, error_type, signal_params = args
        generator = SyntheticDataGenerator()
        data = generator.generate_mdf4_data(duration_seconds=60)
        
        # Add car-specific characteristics
        if car_type == 'sports':
            data['rpm'] = data['rpm'] * 1.2
        elif car_type == 'truck':
            data['speed'] = data['speed'] * 0.8
        
        # Generate labels
        labels = {
            'car_type': car_type,
            'error_type': error_type,
            'error_locations': {}  # Simplified for worker
        }
        
        return data, labels, car_type, i
    except Exception as e:
        print(f"Worker error: {e}")
        return None

class TrainingDataGenerator:
    def __init__(self, output_dir: str, max_workers: int = None):
        self.output_dir = output_dir
        self.max_workers = max_workers or min(32, os.cpu_count() or 1)
        self.synthetic_generator = SyntheticDataGenerator()
        self.car_types = ['sedan', 'suv', 'truck', 'sports']
        self.error_types = ['normal', 'sensor_dropout', 'noise_spike', 'signal_drift']
    
    def clean_output_directory(self):
        """Remove all existing data from output directory"""
        print("\nCleaning existing data...")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_dataset(self, samples_per_car: int = 100):
        # Clean existing data first
        self.clean_output_directory()
        
        # Create output directories for each car type
        for car_type in self.car_types:
            car_dir = os.path.join(self.output_dir, car_type)
            os.makedirs(car_dir, exist_ok=True)
        
        dataset_info = {
            'generated_date': datetime.now().isoformat(),
            'samples_per_car': samples_per_car,
            'car_types': self.car_types,
            'error_types': self.error_types,
            'files': []
        }

        # Prepare work items
        work_items = []
        signal_params = self.synthetic_generator.signal_types
        for car_type in self.car_types:
            for i in range(samples_per_car):
                error_type = np.random.choice(self.error_types)
                work_items.append((car_type, i, error_type, signal_params))

        # Process in parallel with progress bar
        print(f"\nGenerating {samples_per_car} samples for each car type using {self.max_workers} workers")
        
        if __name__ == '__main__':
            multiprocessing.freeze_support()
            
        # Use Pool instead of ProcessPoolExecutor
        with Pool(processes=self.max_workers) as pool:
            results = []
            with tqdm(total=len(work_items), desc="Generating samples") as pbar:
                for result in pool.imap_unordered(generate_sample_worker, work_items):
                    if result is not None:
                        data, labels, car_type, i = result
                        
                        # Save as CSV
                        csv_filename = f"{car_type}_sample_{i:06d}.csv"
                        csv_path = os.path.join(self.output_dir, car_type, csv_filename)
                        self._save_as_csv(data, labels, csv_path)
                        
                        # Update metadata
                        dataset_info['files'].append({
                            'filename': csv_filename,
                            'car_type': car_type,
                            'error_type': labels['error_type'],
                            'path': csv_path
                        })
                    pbar.update(1)

        # Save dataset metadata
        metadata_path = os.path.join(self.output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info
    
    def _generate_sample(self, car_type: str, error_type: str) -> Tuple[Dict, Dict]:
        data = self.synthetic_generator.generate_mdf4_data(duration_seconds=60)
        
        # Add car-specific characteristics
        if car_type == 'sports':
            data['rpm'] = data['rpm'] * 1.2  # Higher RPM range
        elif car_type == 'truck':
            data['speed'] = data['speed'] * 0.8  # Lower speed range
        
        # Generate labels
        labels = {
            'car_type': car_type,
            'error_type': error_type,
            'error_locations': self._detect_error_regions(data)
        }
        
        return data, labels
    
    def _detect_error_regions(self, data: Dict) -> Dict:
        error_regions = {}
        for signal_name in data:
            if signal_name == 'time':
                continue
            
            signal = data[signal_name]
            # Detect regions with NaN values (dropouts)
            nan_regions = np.where(np.isnan(signal))[0]
            # Detect spike regions
            spikes = np.where(np.abs(np.diff(signal)) > np.std(signal) * 3)[0]
            
            error_regions[signal_name] = {
                'dropouts': nan_regions.tolist(),
                'spikes': spikes.tolist()
            }
        
        return error_regions
    
    def _save_as_csv(self, data: Dict, labels: Dict, filepath: str):
        df = pd.DataFrame(data)
        # Add label columns
        df['car_type'] = labels['car_type']
        df['error_type'] = labels['error_type']
        df.to_csv(filepath, index=False)

