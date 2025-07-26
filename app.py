import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from utils.data_loader import DataLoader
from utils.explainer import ModelExplainer, AnomalyVisualizer

# Page configuration
st.set_page_config(
    page_title="Zero-Day Attack Predictor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models"""
    models = {}
    try:
        if os.path.exists('models/attack_classifier.pkl'):
            models['classifier'] = joblib.load('models/attack_classifier.pkl')
        if os.path.exists('models/autoencoder.h5'):
            models['autoencoder'] = tf.keras.models.load_model('models/autoencoder.h5')
        if os.path.exists('models/anomaly_threshold.pkl'):
            models['threshold'] = joblib.load('models/anomaly_threshold.pkl')
        if os.path.exists('models/scaler.pkl'):
            models['scaler'] = joblib.load('models/scaler.pkl')
        if os.path.exists('models/feature_names.pkl'):
            models['feature_names'] = joblib.load('models/feature_names.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    loader = DataLoader()
    X, y, y_binary = loader.load_nsl_kdd()
    return X, y, y_binary

def predict_attacks(X, models):
    """Make predictions using loaded models"""
    predictions = {}
    
    # Known attack prediction
    if 'classifier' in models:
        pred_binary = models['classifier'].predict(X)
        pred_proba = models['classifier'].predict_proba(X)
        predictions['attack_pred'] = pred_binary
        predictions['attack_proba'] = pred_proba[:, 1]  # Probability of attack
    
    # Anomaly detection
    if 'autoencoder' in models and 'threshold' in models:
        reconstructed = models['autoencoder'].predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        anomaly_pred = (mse > models['threshold']).astype(int)
        predictions['anomaly_pred'] = anomaly_pred
        predictions['anomaly_score'] = mse
    
    return predictions

def get_risk_assessment(attack_pred, attack_proba, anomaly_pred, anomaly_score, threshold):
    """Assess overall risk level"""
    if attack_pred == 1:
        return "Known Attack", "high", attack_proba
    elif anomaly_pred == 1:
        if anomaly_score > threshold * 2:
            return "High-Risk Unknown", "high", anomaly_score / threshold
        else:
            return "Medium-Risk Unknown", "medium", anomaly_score / threshold
    else:
        return "Normal Traffic", "low", max(attack_proba, anomaly_score / threshold)

def main():
    st.markdown('<h1 class="main-header">🔐 Zero-Day Attack Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "📊 Data Analysis", 
        "🔍 Prediction",
        "📈 Model Performance",
        "🧬 Explainability"
    ])
    
    # Load models and data
    models = load_models()
    X, y, y_binary = load_sample_data()
    
    if page == "🏠 Home":
        show_home_page(models)
    elif page == "📊 Data Analysis":
        show_data_analysis(X, y, y_binary)
    elif page == "🔍 Prediction":
        show_prediction_page(models, X, y)
    elif page == "📈 Model Performance":
        show_performance_page(models, X, y, y_binary)
    elif page == "🧬 Explainability":
        show_explainability_page(models, X, y)

def show_home_page(models):
    """Display home page"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Known Attack Detection</h3>
            <p>Random Forest classifier trained on labeled attack data</p>
            <p><strong>Status:</strong> {'✅ Ready' if 'classifier' in models else '❌ Not Loaded'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Zero-Day Prediction</h3>
            <p>Autoencoder-based anomaly detection for unknown threats</p>
            <p><strong>Status:</strong> {'✅ Ready' if 'autoencoder' in models else '❌ Not Loaded'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧬 Feature Optimization</h3>
            <p>Genetic algorithm for feature selection and tuning</p>
            <p><strong>Status:</strong> {'✅ Ready' if os.path.exists('models/genetic_optimization_results.pkl') else '❌ Not Available'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Architecture
    st.subheader("🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔄 Data Pipeline
        1. **Data Ingestion**: NSL-KDD dataset loading
        2. **Preprocessing**: Feature encoding and normalization
        3. **Feature Selection**: Genetic algorithm optimization
        4. **Model Training**: Parallel training of classifiers
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 ML Models
        1. **Random Forest**: Known attack classification
        2. **Autoencoder**: Anomaly detection for zero-days
        3. **SHAP/LIME**: Model explainability
        4. **Genetic Algorithm**: Hyperparameter optimization
        """)
    
    # Key Features
    st.subheader("🎯 Key Capabilities")
    
    features = [
        ("Real-time Threat Assessment", "Upload network logs for instant analysis"),
        ("Multi-layer Detection", "Combines supervised and unsupervised learning"),
        ("Explainable AI", "SHAP and LIME explanations for predictions"),
        ("Risk Scoring", "Quantitative risk assessment with thresholds"),
        ("Interactive Dashboard", "User-friendly interface with visualizations"),
        ("Model Optimization", "Genetic algorithm for feature selection")
    ]
    
    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)
        with col1:
            title, desc = features[i]
            st.markdown(f"**{title}**: {desc}")
        if i + 1 < len(features):
            with col2:
                title, desc = features[i + 1]
                st.markdown(f"**{title}**: {desc}")

def show_data_analysis(X, y, y_binary):
    """Display data analysis page"""
    st.header("📊 Data Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Normal Traffic", sum(y == 'normal'))
    with col4:
        st.metric("Attack Traffic", sum(y != 'normal'))
    
    # Attack type distribution
    st.subheader("Attack Type Distribution")
    attack_counts = y.value_counts()
    fig = px.pie(values=attack_counts.values, names=attack_counts.index, 
                 title="Distribution of Traffic Types")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature statistics
    st.subheader("Feature Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Features by Variance**")
        feature_var = X.var().sort_values(ascending=False).head(10)
        fig = px.bar(x=feature_var.values, y=feature_var.index, orientation='h',
                     title="Feature Variance")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Feature Correlation Heatmap**")
        # Select top features for correlation
        top_features = X.var().sort_values(ascending=False).head(10).index
        corr_matrix = X[top_features].corr()
        fig = px.imshow(corr_matrix, title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality metrics
    st.subheader("Data Quality Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_pct = (X.isnull().sum() / len(X) * 100).max()
        st.metric("Max Missing %", f"{missing_pct:.2f}%")
    
    with col2:
        duplicates = X.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
    
    with col3:
        zero_var_features = (X.var() == 0).sum()
        st.metric("Zero Variance Features", zero_var_features)

def show_prediction_page(models, X, y):
    """Display prediction page"""
    st.header("🔍 Network Traffic Prediction")
    
    if not models:
        st.error("Models not loaded. Please train the models first.")
        return
    
    # File upload
    st.subheader("Upload Network Traffic Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            upload_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(upload_data)} samples")
            
            # Preprocess if needed (assuming same format as training data)
            if 'scaler' in models:
                upload_data_scaled = models['scaler'].transform(upload_data)
                upload_data_scaled = pd.DataFrame(upload_data_scaled, columns=models['feature_names'])
            else:
                upload_data_scaled = upload_data
            
            # Make predictions
            predictions = predict_attacks(upload_data_scaled, models)
            
            # Display results
            show_prediction_results(upload_data_scaled, predictions, models)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        # Use sample data for demonstration
        st.subheader("Demo with Sample Data")
        if st.button("Analyze Sample Traffic"):
            # Use a subset of sample data
            sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
            sample_data = X.iloc[sample_indices]
            
            # Make predictions
            predictions = predict_attacks(sample_data, models)
            
            # Display results
            show_prediction_results(sample_data, predictions, models)

def show_prediction_results(data, predictions, models):
    """Display prediction results"""
    st.subheader("🎯 Prediction Results")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    if 'attack_pred' in predictions:
        attacks_detected = sum(predictions['attack_pred'])
        with col1:
            st.metric("Known Attacks", attacks_detected)
    
    if 'anomaly_pred' in predictions:
        anomalies_detected = sum(predictions['anomaly_pred'])
        with col2:
            st.metric("Anomalies Detected", anomalies_detected)
    
    if 'attack_proba' in predictions:
        avg_attack_prob = np.mean(predictions['attack_proba'])
        with col3:
            st.metric("Avg Attack Probability", f"{avg_attack_prob:.3f}")
    
    if 'anomaly_score' in predictions:
        avg_anomaly_score = np.mean(predictions['anomaly_score'])
        with col4:
            st.metric("Avg Anomaly Score", f"{avg_anomaly_score:.3f}")
    
    # Risk assessment for each sample
    st.subheader("📋 Individual Risk Assessment")
    
    results_data = []
    for i in range(len(data)):
        attack_pred = predictions.get('attack_pred', [0])[i] if 'attack_pred' in predictions else 0
        attack_proba = predictions.get('attack_proba', [0])[i] if 'attack_proba' in predictions else 0
        anomaly_pred = predictions.get('anomaly_pred', [0])[i] if 'anomaly_pred' in predictions else 0
        anomaly_score = predictions.get('anomaly_score', [0])[i] if 'anomaly_score' in predictions else 0
        
        threshold = models.get('threshold', 1.0)
        risk_level, risk_category, risk_score = get_risk_assessment(
            attack_pred, attack_proba, anomaly_pred, anomaly_score, threshold
        )
        
        results_data.append({
            'Sample': i + 1,
            'Risk Level': risk_level,
            'Risk Score': f"{risk_score:.3f}",
            'Attack Probability': f"{attack_proba:.3f}",
            'Anomaly Score': f"{anomaly_score:.3f}"
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Color code based on risk level
    def highlight_risk(row):
        if 'High-Risk' in row['Risk Level'] or 'Known Attack' in row['Risk Level']:
            return ['background-color: #ffebee'] * len(row)
        elif 'Medium-Risk' in row['Risk Level']:
            return ['background-color: #fff3e0'] * len(row)
        else:
            return ['background-color: #e8f5e8'] * len(row)
    
    st.dataframe(results_df.style.apply(highlight_risk, axis=1), use_container_width=True)
    
    # Alert system
    high_risk_count = sum(1 for r in results_data if 'High-Risk' in r['Risk Level'] or 'Known Attack' in r['Risk Level'])
    medium_risk_count = sum(1 for r in results_data if 'Medium-Risk' in r['Risk Level'])
    
    if high_risk_count > 0:
        st.markdown(f"""
        <div class="alert-high">
            <h4>🚨 HIGH RISK ALERT</h4>
            <p>{high_risk_count} samples detected as high-risk threats requiring immediate attention!</p>
        </div>
        """, unsafe_allow_html=True)
    elif medium_risk_count > 0:
        st.markdown(f"""
        <div class="alert-medium">
            <h4>⚠️ MEDIUM RISK WARNING</h4>
            <p>{medium_risk_count} samples detected as medium-risk anomalies requiring investigation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-low">
            <h4>✅ ALL CLEAR</h4>
            <p>No high-risk threats detected in the analyzed traffic.</p>
        </div>
        """, unsafe_allow_html=True)

def show_performance_page(models, X, y, y_binary):
    """Display model performance page"""
    st.header("📈 Model Performance Analysis")
    
    if not models:
        st.error("Models not loaded. Please train the models first.")
        return
    
    # Test the models on sample data
    sample_size = min(1000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_test = X.iloc[indices]
    y_test = y.iloc[indices]
    y_binary_test = y_binary.iloc[indices]
    
    predictions = predict_attacks(X_test, models)
    
    # Attack Classifier Performance
    if 'attack_pred' in predictions:
        st.subheader("🛡️ Attack Classifier Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Classification report
            if len(predictions['attack_pred']) > 0:
                report = classification_report(y_binary_test, predictions['attack_pred'], output_dict=True)
                
                metrics_df = pd.DataFrame({
                    'Precision': [report['0']['precision'], report['1']['precision']],
                    'Recall': [report['0']['recall'], report['1']['recall']],
                    'F1-Score': [report['0']['f1-score'], report['1']['f1-score']]
                }, index=['Normal', 'Attack'])
                
                st.write("**Classification Metrics**")
                st.dataframe(metrics_df)
                
                # Overall metrics
                st.metric("Overall Accuracy", f"{report['accuracy']:.3f}")
        
        with col2:
            # Confusion Matrix
            cm = confusion_matrix(y_binary_test, predictions['attack_pred'])
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                           title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly Detector Performance
    if 'anomaly_pred' in predictions:
        st.subheader("🔍 Anomaly Detector Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC-like analysis for anomaly detection
            normal_scores = predictions['anomaly_score'][y_test == 'normal']
            attack_scores = predictions['anomaly_score'][y_test != 'normal']
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=normal_scores, name='Normal', opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=attack_scores, name='Attack', opacity=0.7, nbinsx=30))
            fig.update_layout(title="Anomaly Score Distribution", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Threshold analysis
            threshold = models.get('threshold', 1.0)
            
            # Calculate metrics at current threshold
            true_attacks = (y_test != 'normal').astype(int)
            tp = sum((predictions['anomaly_pred'] == 1) & (true_attacks == 1))
            fp = sum((predictions['anomaly_pred'] == 1) & (true_attacks == 0))
            tn = sum((predictions['anomaly_pred'] == 0) & (true_attacks == 0))
            fn = sum((predictions['anomaly_pred'] == 0) & (true_attacks == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            st.write("**Anomaly Detection Metrics**")
            metrics_data = {
                'Metric': ['Precision', 'Recall', 'Specificity', 'Threshold'],
                'Value': [f"{precision:.3f}", f"{recall:.3f}", f"{specificity:.3f}", f"{threshold:.3f}"]
            }
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
    
    # Feature Importance
    if 'classifier' in models:
        st.subheader("🎯 Feature Importance Analysis")
        
        # Get feature importance from Random Forest
        importance = models['classifier'].feature_importances_
        feature_names = models.get('feature_names', [f'Feature_{i}' for i in range(len(importance))])
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Top 15 Most Important Features")
        st.plotly_chart(fig, use_container_width=True)
    
    # Genetic Optimization Results
    if os.path.exists('models/genetic_optimization_results.pkl'):
        st.subheader("🧬 Genetic Optimization Results")
        
        try:
            ga_results = joblib.load('models/genetic_optimization_results.pkl')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Optimization Summary**")
                st.write(f"Selected Features: {len(ga_results['best_features'])}")
                st.write(f"Best Parameters: {ga_results['best_params']}")
            
            with col2:
                # Plot optimization history if available
                if 'optimization_history' in ga_results:
                    history = ga_results['optimization_history']
                    generations = list(range(len(history)))
                    max_fitness = [gen['max'] for gen in history]
                    avg_fitness = [gen['avg'] for gen in history]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=generations, y=max_fitness, name='Max Fitness'))
                    fig.add_trace(go.Scatter(x=generations, y=avg_fitness, name='Avg Fitness'))
                    fig.update_layout(title="Genetic Algorithm Convergence")
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading genetic optimization results: {e}")

def show_explainability_page(models, X, y):
    """Display explainability page"""
    st.header("🧬 Model Explainability")
    
    if 'classifier' not in models:
        st.error("Classifier model not loaded. Please train the model first.")
        return
    
    # Sample data for explanation
    sample_size = min(500, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[indices]
    y_sample = y.iloc[indices]
    
    # Create explainer
    feature_names = models.get('feature_names', X.columns.tolist())
    explainer = ModelExplainer(models['classifier'], X_sample, feature_names)
    
    # SHAP Analysis
    st.subheader("🎯 SHAP Feature Importance")
    
    try:
        # Get feature importance
        importance_df = explainer.get_feature_importance(X_sample)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top features bar chart
            top_features = importance_df.head(15)
            fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                        title="Top 15 Features by SHAP Importance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance table
            st.write("**Feature Importance Rankings**")
            st.dataframe(importance_df.head(20), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {e}")
    
    # Individual Prediction Explanation
    st.subheader("🔍 Individual Prediction Explanation")
    
    sample_idx = st.selectbox("Select a sample to explain:", range(min(50, len(X_sample))))
    
    if st.button("Explain Prediction"):
        try:
            # Get single prediction explanation
            instance = X_sample.iloc[sample_idx:sample_idx+1]
            prediction = models['classifier'].predict(instance)[0]
            probability = models['classifier'].predict_proba(instance)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Prediction Results**")
                st.write(f"Predicted Class: {'Attack' if prediction == 1 else 'Normal'}")
                st.write(f"Attack Probability: {probability[1]:.3f}")
                st.write(f"Normal Probability: {probability[0]:.3f}")
            
            with col2:
                # Show top feature values for this instance
                feature_values = instance.iloc[0]
                top_feature_indices = explainer.get_feature_importance(X_sample).head(10).index
                
                st.write("**Top Feature Values**")
                for feature in top_feature_indices:
                    if feature in feature_values.index:
                        st.write(f"{feature}: {feature_values[feature]:.3f}")
        
        except Exception as e:
            st.error(f"Error explaining individual prediction: {e}")
    
    # Model Comparison
    st.subheader("⚖️ Model Comparison")
    
    if 'anomaly_score' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Attack Classifier**")
            st.write("- Type: Supervised Learning")
            st.write("- Algorithm: Random Forest")
            st.write("- Purpose: Known attack detection")
            st.write("- Output: Binary classification + probability")
        
        with col2:
            st.write("**Anomaly Detector**")
            st.write("- Type: Unsupervised Learning")
            st.write("- Algorithm: Autoencoder")
            st.write("- Purpose: Zero-day attack prediction")
            st.write("- Output: Anomaly score + binary flag")

if __name__ == "__main__":
    main()
