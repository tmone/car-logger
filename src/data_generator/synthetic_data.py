import numpy as np
import pandas as pd
from typing import Dict, List

class SyntheticDataGenerator:
    def __init__(self):
        self.signal_types = {
            'speed': {'min': 0, 'max': 200, 'unit': 'km/h'},
            'rpm': {'min': 0, 'max': 8000, 'unit': 'rpm'},
            'temperature': {'min': 60, 'max': 120, 'unit': 'C'},
            'fuel_level': {'min': 0, 'max': 100, 'unit': '%'}
        }
        
        self.error_patterns = {
            'sensor_dropout': self.generate_sensor_dropout,
            'noise_spike': self.generate_noise_spike,
            'signal_drift': self.generate_signal_drift
        }
    
    def generate_sensor_dropout(self, data: np.ndarray, dropout_ratio=0.1) -> np.ndarray:
        mask = np.random.random(len(data)) > dropout_ratio
        return np.where(mask, data, np.nan)
    
    def generate_noise_spike(self, data: np.ndarray, spike_probability=0.05) -> np.ndarray:
        spikes = np.random.random(len(data)) < spike_probability
        spike_values = np.random.uniform(low=-2, high=2, size=len(data))
        return data + (spikes * spike_values)
    
    def generate_signal_drift(self, data: np.ndarray, drift_factor=0.1) -> np.ndarray:
        drift = np.cumsum(np.random.normal(0, drift_factor, len(data)))
        return data + drift
    
    def generate_mdf4_data(self, duration_seconds: int = 3600, sample_rate: int = 100) -> Dict:
        n_samples = duration_seconds * sample_rate
        time = np.linspace(0, duration_seconds, n_samples)
        
        data = {}
        for signal_name, params in self.signal_types.items():
            base_signal = np.random.uniform(params['min'], params['max'], n_samples)
            
            # Add random error pattern
            error_type = np.random.choice(list(self.error_patterns.keys()))
            error_func = self.error_patterns[error_type]
            data[signal_name] = error_func(base_signal)
            
        data['time'] = time
        return data
