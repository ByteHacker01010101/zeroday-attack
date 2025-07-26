import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_nsl_kdd(self, train_path='data/KDDTrain+.txt', test_path='data/KDDTest+.txt'):
        """Load NSL-KDD dataset"""
        # Column names for NSL-KDD dataset
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
        ]
        
        try:
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_data = pd.read_csv(train_path, names=columns)
                test_data = pd.read_csv(test_path, names=columns)
                data = pd.concat([train_data, test_data], ignore_index=True)
            else:
                print("NSL-KDD files not found. Generating synthetic data...")
                data = self.generate_synthetic_data()
                
        except Exception as e:
            print(f"Error loading NSL-KDD: {e}. Generating synthetic data...")
            data = self.generate_synthetic_data()
            
        return self.preprocess_data(data)
    
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic network traffic data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'duration': np.random.exponential(2, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples),
            'src_bytes': np.random.exponential(1000, n_samples),
            'dst_bytes': np.random.exponential(1000, n_samples),
            'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'wrong_fragment': np.random.poisson(0.1, n_samples),
            'urgent': np.random.poisson(0.05, n_samples),
            'hot': np.random.poisson(0.2, n_samples),
            'num_failed_logins': np.random.poisson(0.1, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(5, n_samples),
            'serror_rate': np.random.beta(1, 10, n_samples),
            'srv_serror_rate': np.random.beta(1, 10, n_samples),
            'rerror_rate': np.random.beta(1, 10, n_samples),
            'srv_rerror_rate': np.random.beta(1, 10, n_samples),
            'same_srv_rate': np.random.beta(5, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 5, n_samples),
        }
        
        # Add more features
        for i in range(20):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Generate attack types
        attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        attack_probs = [0.7, 0.15, 0.08, 0.05, 0.02]
        data['attack_type'] = np.random.choice(attack_types, n_samples, p=attack_probs)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, data):
        """Preprocess the dataset"""
        # Handle missing values
        data = data.fillna(0)
        
        # Encode categorical variables
        categorical_columns = ['protocol_type', 'service', 'flag']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))
        
        # Separate features and labels
        X = data.drop('attack_type', axis=1)
        y = data['attack_type']
        
        # Convert attack types to binary (normal vs attack)
        y_binary = (y != 'normal').astype(int)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y, y_binary
    
    def get_normal_data(self, X, y):
        """Extract only normal traffic data for anomaly detection"""
        normal_mask = (y == 'normal')
        return X[normal_mask]
