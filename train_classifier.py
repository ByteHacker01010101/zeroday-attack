import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from utils.data_loader import DataLoader

def train_attack_classifier():
    """Train Random Forest classifier for known attack detection"""
    print("Loading and preprocessing data...")
    loader = DataLoader()
    X, y, y_binary = loader.load_nsl_kdd()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print("Training Random Forest classifier...")
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/attack_classifier.pkl')
    joblib.dump(loader.scaler, 'models/scaler.pkl')
    joblib.dump(loader.feature_names, 'models/feature_names.pkl')
    
    print("Model saved successfully!")
    return rf_model, accuracy

if __name__ == "__main__":
    train_attack_classifier()
