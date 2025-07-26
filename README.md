# 🔐 Zero-Day Attack Predictor

A comprehensive machine learning system for detecting known network intrusions and predicting zero-day attacks through advanced anomaly detection techniques.

## 🎯 Overview

This project combines traditional supervised learning with unsupervised anomaly detection to create a robust network security monitoring system. It uses the NSL-KDD dataset for training and provides real-time threat assessment capabilities through an interactive Streamlit dashboard.

## ✨ Features

### 🛡️ Known Attack Detection
- **Random Forest Classifier** for detecting known attack patterns
- **Multi-class classification** supporting DoS, Probe, R2L, and U2R attacks
- **High accuracy** with comprehensive evaluation metrics

### 🔍 Zero-Day Prediction
- **Autoencoder-based anomaly detection** trained only on normal traffic
- **Reconstruction error analysis** for identifying unknown threats
- **Risk scoring system** with configurable thresholds

### 🧬 Feature Optimization
- **Genetic Algorithm** for optimal feature selection
- **Hyperparameter tuning** for maximum model performance
- **Automated optimization pipeline** with convergence tracking

### 📊 Explainable AI
- **SHAP integration** for global and local feature importance
- **LIME explanations** for individual predictions
- **Interactive visualizations** for model interpretability

### 🌐 Interactive Dashboard
- **Real-time prediction interface** with file upload capability
- **Comprehensive visualizations** including t-SNE and correlation plots
- **Alert system** with risk-based notifications
- **Model performance monitoring** with detailed metrics

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/yourusername/zero-day-attack-predictor.git
cd zero-day-attack-predictor

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models
\`\`\`

### Dataset Setup

1. Download the NSL-KDD dataset:
   - Place `KDDTrain+.txt` and `KDDTest+.txt` in the `data/` directory
   - Or let the system generate synthetic data for demonstration

### Model Training

\`\`\`bash
# Train the attack classifier
python train_classifier.py

# Train the anomaly detector
python train_autoencoder.py

# Run genetic optimization (optional)
python genetic_optimizer.py
\`\`\`

### Launch Dashboard

\`\`\`bash
streamlit run app.py
\`\`\`

## 📁 Project Structure

\`\`\`
zero-day-attack-predictor/
├── data/                          # Dataset files
├── models/                        # Trained model files
├── utils/
│   ├── data_loader.py            # Data preprocessing utilities
│   └── explainer.py              # Model explainability tools
├── app.py                        # Streamlit dashboard
├── train_classifier.py          # Attack classifier training
├── train_autoencoder.py         # Anomaly detector training
├── genetic_optimizer.py         # Feature optimization
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
\`\`\`

## 🔧 Configuration

### Model Parameters

**Random Forest Classifier:**
- `n_estimators`: 100
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2

**Autoencoder:**
- `encoding_dim`: 32
- `epochs`: 50
- `batch_size`: 32
- `validation_split`: 0.2

**Genetic Algorithm:**
- `population_size`: 50
- `generations`: 20
- `crossover_probability`: 0.7
- `mutation_probability`: 0.3

## 📈 Performance Metrics

### Attack Classification
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%
- **F1-Score**: ~95%

### Anomaly Detection
- **True Positive Rate**: ~85%
- **False Positive Rate**: ~5%
- **Specificity**: ~95%

## 🎮 Dashboard Usage

### 🏠 Home Page
- System overview and model status
- Architecture explanation
- Key capabilities summary

### 📊 Data Analysis
- Dataset statistics and distributions
- Feature correlation analysis
- Data quality metrics

### 🔍 Prediction Interface
- CSV file upload for batch analysis
- Real-time risk assessment
- Individual sample examination

### 📈 Model Performance
- Comprehensive evaluation metrics
- ROC curves and confusion matrices
- Feature importance analysis

### 🧬 Explainability
- SHAP feature importance
- Individual prediction explanations
- Model comparison insights

## 🛠️ Customization

### Adding New Features
1. Modify `utils/data_loader.py` to include new features
2. Update preprocessing pipeline
3. Retrain models with new feature set

### Custom Models
1. Implement new model class following existing patterns
2. Add to training pipeline
3. Update dashboard integration

### New Attack Types
1. Update dataset with new attack samples
2. Retrain classification model
3. Validate performance on new attack types

## 🚀 Deployment

### Local Development
\`\`\`bash
streamlit run app.py --server.port 8501
\`\`\`

### Docker Deployment
\`\`\`dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
\`\`\`

### Cloud Platforms
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web application hosting
- **AWS/GCP**: Scalable cloud deployment

## 📚 Dependencies

### Core Libraries
- `streamlit`: Interactive web dashboard
- `scikit-learn`: Machine learning algorithms
- `tensorflow`: Deep learning framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing

### Specialized Libraries
- `deap`: Genetic algorithm implementation
- `shap`: Model explainability
- `lime`: Local interpretable explanations
- `plotly`: Interactive visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NSL-KDD dataset creators for providing the benchmark dataset
- SHAP and LIME developers for explainability tools
- Streamlit team for the amazing dashboard framework
- Open source community for various tools and libraries

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Email: support@example.com
- Documentation: [Project Wiki](https://github.com/yourusername/zero-day-attack-predictor/wiki)

## 🔮 Future Enhancements

- [ ] Real-time network traffic monitoring
- [ ] Advanced GAN-based zero-day simulation
- [ ] Multi-model ensemble approach
- [ ] REST API for external integration
- [ ] Mobile-responsive dashboard
- [ ] Automated model retraining pipeline
- [ ] Integration with SIEM systems
- [ ] Advanced threat intelligence feeds

---

**Built with ❤️ for cybersecurity research and education**
