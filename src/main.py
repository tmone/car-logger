# main.py

from data_loader.mdf4_reader import MDF4Reader
from analysis.analyzer import DataAnalyzer
from analysis.plotter import DataPlotter
from data_loader.file_loader import CarDataLoader
from data_generator.synthetic_data import SyntheticDataGenerator
from data_generator.training_data_generator import TrainingDataGenerator
from data_generator.dataset_splitter import DatasetSplitter
import os

def ensure_minimum_dataset_size(data_by_car: dict, min_samples_per_car: int = 100):
    generator = SyntheticDataGenerator()
    
    for car_type in data_by_car:
        mdf4_count = len(data_by_car[car_type]['mdf4'])
        if mdf4_count < min_samples_per_car:
            synthetic_samples_needed = min_samples_per_car - mdf4_count
            print(f"Generating {synthetic_samples_needed} synthetic samples for {car_type}")
            
            for _ in range(synthetic_samples_needed):
                synthetic_data = generator.generate_mdf4_data()
                data_by_car[car_type]['mdf4'].append(synthetic_data)
    
    return data_by_car

def generate_large_dataset(output_dir: str, samples_per_car: int = 1000):
    generator = TrainingDataGenerator(output_dir)
    print(f"Generating {samples_per_car} samples per car type...")
    dataset_info = generator.generate_dataset(samples_per_car)
    print("Dataset generation complete")
    return dataset_info

def split_dataset(data_dir: str):
    splitter = DatasetSplitter(data_dir)
    print("Splitting dataset into train/valid/test sets...")
    splitter.split_dataset()
    print("Dataset splitting complete")

def main():
    # Initialize the data loader
    loader = CarDataLoader()
    
    # Load all data from the data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_by_car = loader.load_directory(data_dir)
    
    # Ensure minimum dataset size for each car type
    data_by_car = ensure_minimum_dataset_size(data_by_car)
    
    # Print summary of loaded and generated data
    for car_type, data in data_by_car.items():
        print(f"\nCar Type: {car_type}")
        print(f"MDF4 files: {len(data['mdf4'])}")
        print(f"ODC files: {len(data['odc'])}")

    # Create data directories
    base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    training_data_dir = os.path.join(base_data_dir, 'training')
    os.makedirs(training_data_dir, exist_ok=True)
    
    # Generate large dataset
    dataset_info = generate_large_dataset(training_data_dir)
    
    # Split the dataset
    split_dataset(training_data_dir)
    
    # Print summary
    print("\nDataset creation complete:")
    print(f"Total samples: {len(dataset_info['files'])}")
    print("Split sizes:")
    print("- Train: 80%")
    print("- Valid: 10%")
    print("- Test: 10%")
    print("\nData location:")
    print(f"- Train: {os.path.join(training_data_dir, 'train')}")
    print(f"- Valid: {os.path.join(training_data_dir, 'valid')}")
    print(f"- Test: {os.path.join(training_data_dir, 'test')}")

if __name__ == "__main__":
    main()