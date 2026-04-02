# OSL-SLA: Operational Service Level Agreement Prediction System

 **A comprehensive ML-powered system for predicting machine failures and operational service level agreements**

## 📋 Table of Contents

- [ Overview](#-overview)
- [ Architecture](#-architecture)
- [ Quick Start](#-quick-start)
- [ Installation](#-installation)
- [ Docker Deployment](#-docker-deployment)
- [ Configuration](#-configuration)
- [ Model Information](#-model-information)
- [ Testing](#-testing)
- [ CI/CD Pipeline](#-cicd-pipeline)
- [ Kubernetes Deployment](#-kubernetes-deployment)
- [ API Documentation](#-api-documentation)
- [ Troubleshooting](#-troubleshooting)

---

##  Overview

OSL-SLA is a machine learning system designed to predict operational service level agreements and machine failures in industrial settings. The system uses real-time sensor data to provide:

- **Failure Prediction**: Predicts machine failures before they occur
- **SLA Monitoring**: Tracks service level agreement compliance
- **Real-time Analytics**: Live dashboard with streaming data
- **Explainable AI**: SHAP-based feature importance analysis

### Key Features
-  **Interactive Web Interface** built with Streamlit
-  **Real-time Monitoring** with live data streaming
-  **ML Model**: Random Forest classifier with high accuracy
-  **Performance Metrics**: Accuracy, Precision, F1 Score tracking
-  **Security**: Built-in security scanning and validation
-  **Container Ready**: Dockerized for easy deployment

---

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Web Interface                 │
│            (Streamlit Dashboard)              │
├─────────────────────────────────────────────────────┤
│                                             │
│   ML Model           Data Processing   │
│  (Random Forest)      (Preprocessing)        │
│                                             │
├─────────────────────────────────────────────────────┤
│                                             │
│   Data Storage      Auto-Retrain       │
│  (Model Files)       (Pipeline)            │
└─────────────────────────────────────────────────────┘
```

### Components
- **Frontend**: Streamlit web application (`app/main.py`)
- **ML Core**: Prediction engine (`src/predict.py`)
- **Data Pipeline**: Preprocessing and training (`src/preprocessing.py`, `src/train_model.py`)
- **Auto-Retainer**: Continuous model improvement (`src/auto_retrain.py`)
- **Cache Layer**: Redis for session and prediction storage

---

##  Quick Start

### Method 1: Local Development (Recommended)

```bash
# Clone the repository
git clone https://github.com/swap1930/OSL_Project.git
cd OSL_SLA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py

# Run the application
streamlit run app/main.py
```

### Method 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the application
open http://localhost:8501
```

### Method 3: Custom Port

```bash
# Run on different port
docker-compose up -d
# or
STREAMLIT_SERVER_PORT=8085 docker-compose up -d
```

---

##  Installation

### System Requirements
- **Python**: 3.9+ (recommended 3.10)
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ free space
- **OS**: Windows, macOS, Linux

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `streamlit` - Web interface framework
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `shap` - Model explainability
- `redis` - Caching layer
- `pytest` - Testing framework

### Docker Setup
```bash
# Install Docker (if not already installed)
# Then build and run
docker-compose up -d
```

---

##  Docker Deployment

### Development Environment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Build production image
docker build -t osl-sla:latest .

# Run with persistent storage
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  osl-sla:latest
```

### Port Configuration
- **Default Port**: 8501
- **Custom Port**: Use `STREAMLIT_SERVER_PORT` environment variable
- **Docker Mapping**: `HOST:CONTAINER` format

---

##  Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | 8501 | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | 0.0.0.0 | Server bind address |
| `PYTHON_VERSION` | 3.10 | Python version for CI/CD |

### Model Configuration

Models are automatically saved to:
- `model/model.pkl` - Trained Random Forest model
- `model/scaler.pkl` - Data preprocessing scaler

### Data Structure

Expected input format:
```python
[
    "Air temperature [K]",    # 298-305K
    "Process temperature [K]", # 308-315K  
    "Rotational speed [rpm]",  # 1000-3000 rpm
    "Torque [Nm]",           # 20-80 Nm
    "Tool wear [min]"         # 0-300 minutes
]
```

---

##  Model Information

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Training Data**: Industrial sensor data
- **Features**: 5 key operational parameters
- **Output**: Binary classification (0=Normal, 1=Failure)
- **Accuracy**: ~90% on test data
- **Explainability**: SHAP values for feature importance

### Feature Importance
The model analyzes these key factors:
1. **Torque** - Critical for failure prediction
2. **Tool Wear** - Cumulative damage indicator
3. **Rotational Speed** - Operational stress level
4. **Temperature** - Thermal stress indicators
5. **Process Parameters** - System efficiency metrics

---

##  Testing

### Run All Tests
```bash
# Execute the complete CI test pipeline
python ci_test.py
```

### Individual Test Categories
```bash
# Model training test
python main.py

# Prediction functionality test
python -c "from src.predict import predict; print(predict([300, 310, 1500, 40, 50]))"

# Data preprocessing test
python -c "from src.preprocessing import preprocess; print('Preprocessing OK')"
```

### Test Scenarios
-  **Normal Operation**: Standard parameters
-  **High Stress**: Elevated temperature/torque
-  **Failure Conditions**: Critical parameter combinations
-  **Performance Metrics**: Accuracy, Precision, F1 Score

---

##  CI/CD Pipeline

### GitHub Actions Workflow

The pipeline automatically:
1. **Trains Model** using existing training data
2. **Validates Predictions** with multiple test scenarios
3. **Calculates Metrics** - Accuracy, Precision, F1 Score
4. **Verifies Artifacts** - Model files and sizes
5. **Reports Results** - Clear success/failure status

### Pipeline Commands
```bash
# Run locally (same as CI)
python ci_test.py

# Expected output
 Starting OSL-SLA CI Test Pipeline
==================================================
 Step 1: Training model... 
 Step 2: Testing model predictions...  
 Step 3: Evaluating model performance... 
 Step 4: Verifying model files... 
 Final Metrics - Acc: 0.900, Prec: 0.000, F1: 0.000
 CI Pipeline Summary:  SUCCESS
```

### Triggers
- **Push to main/develop**: Runs full pipeline
- **Pull Request**: Validates changes before merge

---


**Features:**
- 3 replicas with auto-scaling (3-10)
- LoadBalancer with SSL termination
- 2Gi persistent storage for models
- Health checks and monitoring
- Pod disruption budgets

#### Staging Environment
```bash
# Deploy to staging
kubectl apply -f k8s/staging/
```

**Features:**
- 2 replicas for testing
- HTTP LoadBalancer (no SSL)
- 1Gi storage for models
- Resource limits and requests

### Environment-Specific Configurations
- **Production**: SSL, high availability, auto-scaling
- **Staging**: Basic setup, resource optimization
- **Development**: Local Docker setup

---

##  API Documentation

### Core Prediction API

```python
from src.predict import predict

# Make prediction
data = [300.0, 310.0, 1500.0, 40.0, 50.0]
result, probability = predict(data)

print(f"Prediction: {result}")
print(f"Failure Probability: {probability:.3f}")
```

### Explanation API

```python
from src.predict import explain_prediction

# Get feature importance
explanation = explain_prediction(data)
for feature, details in explanation:
    print(f"{feature}: {details['impact']} (SHAP: {details['shap_value']:.3f})")
```

### Model Training API

```python
from src.train_model import train
from src.preprocessing import preprocess

# Train new model
df = pd.read_csv('data/your_data.csv')
X, y = preprocess(df)
model = train(X, y)
```

---

##  Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Error: Port 8501 already in use
# Solution: Use different port
STREAMLIT_SERVER_PORT=8085 streamlit run app/main.py

# Or kill existing process
lsof -ti:8501 | xargs kill -9
```

#### Model Loading Errors
```bash
# Error: Model files not found
# Solution: Train model first
python main.py

# Check model files exist
ls -la model/
```

#### Docker Issues
```bash
# Error: Permission denied
# Solution: Use sudo or proper permissions
sudo docker-compose up -d

# Error: Out of memory
# Solution: Increase Docker memory
docker system prune -a
```

#### Performance Issues
```bash
# Slow predictions
# Solution: Check data quality and model complexity
python ci_test.py  # Includes performance metrics

# Memory issues
# Solution: Monitor resource usage
docker stats
```

### Getting Help

1. **Check Logs**: `docker-compose logs -f app`
2. **Verify Model**: `python -c "from src.predict import predict; print(predict([300,310,1500,40,50]))"`
3. **Run Diagnostics**: `python ci_test.py`
4. **Check Issues**: Review GitHub Actions workflow

---

##  Monitoring and Maintenance

### Health Checks
```bash
# Application health
curl -f http://localhost:8501/_stcore/health

# Container health
docker-compose ps
```

### Log Management
```bash
# View application logs
docker-compose logs -f app

# Clear old logs
find logs/ -name "*.log" -mtime +7 -delete
```

### Model Retraining
```bash
# Manual retraining
python main.py  # Retrains with latest data

# Check retraining status
from src.auto_retrain import get_retraining_status
```


### Code Quality
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use meaningful commit messages

### Pull Request Process
1. Update tests for new functionality
2. Ensure CI pipeline passes
3. Submit pull request with clear description
4. Address review feedback promptly

---

##  Quick Reference

### Essential Commands
```bash
# Development
streamlit run app/main.py              # Start app
python main.py                        # Train model
python ci_test.py                      # Run tests

# Docker
docker-compose up -d                   # Start containers
docker-compose down                    # Stop containers
docker-compose logs -f app              # View logs

# Production
kubectl apply -f k8s/production/     # Deploy to K8s
kubectl get pods -n osl-sla            # Check status
```

### File Structure
```
OSL_SLA/
├── app/                    # Streamlit web application
│   └── main.py         # Main dashboard
├── src/                    # Core ML modules
│   ├── predict.py        # Prediction engine
│   ├── train_model.py    # Model training
│   ├── preprocessing.py  # Data processing
│   └── auto_retrain.py   # Auto-retraining
├── model/                   # Trained models
│   ├── model.pkl         # ML model
│   └── scaler.pkl       # Data scaler
├── k8s/                    # Kubernetes configs
│   ├── production/       # Production deployment
│   └── staging/          # Staging deployment
├── .github/workflows/        # CI/CD pipeline
├── docker-compose.yml        # Local development
├── Dockerfile               # Container build
├── ci_test.py              # Simple CI tests
└── README.md               # This file
```

---

** Ready to predict operational excellence with OSL-SLA!**