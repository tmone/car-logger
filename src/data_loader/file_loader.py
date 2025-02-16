import os
from typing import Dict, List
import pandas as pd
import numpy as np

class CarDataLoader:
    def __init__(self):
        self.data_cache = {}
        self.mdf4_reader = None
        try:
            import asammdf
            self.mdf4_reader = asammdf.MDF
        except ImportError:
            print("asammdf not found. Please install it using: pip install asammdf")
            print("Alternatively, run: pip install -r requirements.txt")
    
    def load_mdf4_file(self, file_path: str) -> Dict:
        if self.mdf4_reader is None:
            raise ImportError("asammdf is required to read MDF4 files. Please install it first.")
        
        try:
            mdf = self.mdf4_reader(file_path)
            return {ch.name: ch.samples for ch in mdf.channels()}
        except Exception as e:
            print(f"Error reading MDF4 file {file_path}: {str(e)}")
            return {}
    
    def load_odc_file(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)  # Assuming ODC is CSV-based
    
    def load_directory(self, dir_path: str) -> Dict[str, Dict]:
        data_by_car = {}
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mf4') or file.endswith('.mdf4'):
                    car_type = os.path.basename(os.path.dirname(root))
                    if car_type not in data_by_car:
                        data_by_car[car_type] = {'mdf4': [], 'odc': []}
                    data_by_car[car_type]['mdf4'].append(self.load_mdf4_file(os.path.join(root, file)))
                
                elif file.endswith('.odc'):
                    car_type = os.path.basename(os.path.dirname(root))
                    if car_type not in data_by_car:
                        data_by_car[car_type] = {'mdf4': [], 'odc': []}
                    data_by_car[car_type]['odc'].append(self.load_odc_file(os.path.join(root, file)))
        
        return data_by_car
