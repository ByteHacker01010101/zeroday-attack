import shap
import lime
import lime.tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class ModelExplainer:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        
    def setup_shap_explainer(self):
        """Setup SHAP explainer"""
        try:
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fallback to KernelExplainer
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
    
    def get_shap_values(self, X_test, max_samples=1000):
        """Get SHAP values for test data"""
        if self.explainer is None:
            self.setup_shap_explainer()
        
        # Limit samples for performance
        if len(X_test) > max_samples:
            indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test.iloc[indices] if isinstance(X_test, pd.DataFrame) else X_test[indices]
        else:
            X_sample = X_test
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # For binary classification, take values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        return shap_values, X_sample
    
    def plot_shap_summary(self, X_test):
        """Create SHAP summary plot"""
        shap_values, X_sample = self.get_shap_values(X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        return plt.gcf()
    
    def get_feature_importance(self, X_test):
        """Get feature importance from SHAP values"""
        shap_values, X_sample = self.get_shap_values(X_test)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_single_prediction(self, X_instance):
        """Explain a single prediction using LIME"""
        # Setup LIME explainer
        lime_explainer = lime.tabular.TabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            mode='classification'
        )
        
        # Get explanation
        explanation = lime_explainer.explain_instance(
            X_instance.values.flatten(),
            self.model.predict_proba,
            num_features=10
        )
        
        return explanation
    
    def create_feature_interaction_plot(self, X_test):
        """Create feature interaction visualization"""
        shap_values, X_sample = self.get_shap_values(X_test)
        
        # Get top features
        importance = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(importance)[-10:]
        
        fig = go.Figure()
        
        for i, feature_idx in enumerate(top_features):
            fig.add_trace(go.Scatter(
                x=X_sample.iloc[:, feature_idx],
                y=shap_values[:, feature_idx],
                mode='markers',
                name=self.feature_names[feature_idx],
                opacity=0.6
            ))
        
        fig.update_layout(
            title="Feature Values vs SHAP Values",
            xaxis_title="Feature Value",
            yaxis_title="SHAP Value",
            height=600
        )
        
        return fig

class AnomalyVisualizer:
    def __init__(self, X, y, predictions, scores):
        self.X = X
        self.y = y
        self.predictions = predictions
        self.scores = scores
    
    def create_tsne_plot(self):
        """Create t-SNE visualization of anomalies"""
        # Use subset for performance
        n_samples = min(2000, len(self.X))
        indices = np.random.choice(len(self.X), n_samples, replace=False)
        
        X_subset = self.X.iloc[indices] if isinstance(self.X, pd.DataFrame) else self.X[indices]
        y_subset = self.y.iloc[indices] if isinstance(self.y, pd.Series) else self.y[indices]
        pred_subset = self.predictions[indices]
        scores_subset = self.scores[indices]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_subset)
        
        # Create plot
        fig = px.scatter(
            x=X_tsne[:, 0], y=X_tsne[:, 1],
            color=pred_subset,
            size=scores_subset,
            title="t-SNE Visualization of Anomalies",
            labels={'color': 'Anomaly', 'size': 'Anomaly Score'}
        )
        
        return fig
    
    def create_score_distribution(self):
        """Create anomaly score distribution plot"""
        fig = go.Figure()
        
        # Normal samples
        normal_scores = self.scores[self.y == 'normal'] if 'normal' in self.y.values else self.scores[self.predictions == 0]
        fig.add_trace(go.Histogram(
            x=normal_scores,
            name='Normal',
            opacity=0.7,
            nbinsx=50
        ))
        
        # Anomalous samples
        anomaly_scores = self.scores[self.y != 'normal'] if 'normal' in self.y.values else self.scores[self.predictions == 1]
        fig.add_trace(go.Histogram(
            x=anomaly_scores,
            name='Anomaly',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.update_layout(
            title="Distribution of Anomaly Scores",
            xaxis_title="Anomaly Score",
            yaxis_title="Count",
            barmode='overlay'
        )
        
        return fig
