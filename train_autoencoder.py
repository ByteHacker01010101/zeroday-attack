import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import joblib
import os
from utils.data_loader import DataLoader

def create_autoencoder(input_dim, encoding_dim=32):
    """Create autoencoder model"""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder

def train_anomaly_detector():
    """Train autoencoder for zero-day attack prediction"""
    print("Loading data for anomaly detection...")
    loader = DataLoader()
    X, y, y_binary = loader.load_nsl_kdd()
    
    # Get only normal data for training
    normal_data = loader.get_normal_data(X, y)
    
    print(f"Training autoencoder on {len(normal_data)} normal samples...")
    
    # Create and train autoencoder
    autoencoder = create_autoencoder(X.shape[1])
    
    # Train on normal data only
    history = autoencoder.fit(
        normal_data, normal_data,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Calculate reconstruction errors on test data
    reconstructed = autoencoder.predict(X)
    mse = np.mean(np.power(X - reconstructed, 2), axis=1)
    
    # Determine threshold (95th percentile of normal data errors)
    normal_errors = mse[y == 'normal']
    threshold = np.percentile(normal_errors, 95)
    
    print(f"Anomaly threshold: {threshold:.4f}")
    
    # Save model and threshold
    os.makedirs('models', exist_ok=True)
    autoencoder.save('models/autoencoder.h5')
    joblib.dump(threshold, 'models/anomaly_threshold.pkl')
    
    print("Autoencoder saved successfully!")
    return autoencoder, threshold

if __name__ == "__main__":
    train_anomaly_detector()
