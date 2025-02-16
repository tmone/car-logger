import os
import shutil
import random
from typing import List, Dict
import json
from tqdm import tqdm

class DatasetSplitter:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.splits = {
            'train': 0.8,
            'valid': 0.1,
            'test': 0.1
        }
        
    def clean_split_directories(self):
        """Remove existing split directories"""
        print("\nCleaning existing split directories...")
        for split in self.splits.keys():
            split_dir = os.path.join(self.base_dir, split)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
    
    def split_dataset(self):
        # Clean existing splits first
        self.clean_split_directories()
        
        print("\nPreparing dataset split directories...")
        # Create split directories
        for split in self.splits.keys():
            for car_type in ['sedan', 'suv', 'truck', 'sports']:
                split_dir = os.path.join(self.base_dir, split, car_type)
                os.makedirs(split_dir, exist_ok=True)
        
        # Load metadata
        print("Loading dataset metadata...")
        metadata_path = os.path.join(self.base_dir, 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Group files by car type
        files_by_car = {}
        for file_info in metadata['files']:
            car_type = file_info['car_type']
            if (car_type not in files_by_car):
                files_by_car[car_type] = []
            files_by_car[car_type].append(file_info)
        
        # Split files for each car type
        new_metadata = {split: {'files': []} for split in self.splits.keys()}
        total_files = sum(len(files) for files in files_by_car.values())
        
        print("\nMoving files to split directories...")
        with tqdm(total=total_files, desc="Splitting dataset") as pbar:
            for car_type, files in files_by_car.items():
                random.shuffle(files)
                total = len(files)
                
                # Calculate split indices
                train_idx = int(total * self.splits['train'])
                valid_idx = train_idx + int(total * self.splits['valid'])
                
                # Split the files
                splits = {
                    'train': files[:train_idx],
                    'valid': files[train_idx:valid_idx],
                    'test': files[valid_idx:]
                }
                
                # Move files to split directories with progress update
                for split_name, split_files in splits.items():
                    for file_info in split_files:
                        src_path = file_info['path']
                        dst_path = os.path.join(self.base_dir, split_name, car_type, os.path.basename(src_path))
                        shutil.move(src_path, dst_path)
                        file_info['path'] = dst_path
                        new_metadata[split_name]['files'].append(file_info)
                        pbar.update(1)
        
        print("\nSaving split metadata...")
        for split_name, split_data in new_metadata.items():
            split_metadata_path = os.path.join(self.base_dir, split_name, 'metadata.json')
            with open(split_metadata_path, 'w') as f:
                json.dump(split_data, f, indent=2)

