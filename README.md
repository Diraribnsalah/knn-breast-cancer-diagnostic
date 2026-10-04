#  End-to-End Breast Cancer Diagnostic System (K-NN)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://knn-breast-cancer-diagnostic.streamlit.app/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready, modular Machine Learning system for early breast cancer diagnosis using an optimized **K-Nearest Neighbors (K-NN)** algorithm. The project transitions seamlessly from exploratory data analysis to a structured, production-oriented pipeline integrated with an interactive **Streamlit** web application.

> **Note:** This project is intended for educational and research purposes. It is not a medical device and should not be used as a substitute for professional medical diagnosis.

##  Live Web Application

 **[Open the Streamlit Application](https://knn-breast-cancer-diagnostic.streamlit.app/)**

The deployed application provides an interactive interface for real-time probabilistic scoring and sample patient selection.

##  Dashboard Preview

![Streamlit Web UI Dashboard](dashboard_preview.png)

##  Key Metrics & Performance Highlights

| Metric | Result |
|---|---:|
| **Test Accuracy** | **98.25%** |
| **Train Accuracy** | **97.80%** |
| **Cross-Validation** | **5-Fold Cross-Validation** |
| **Hyperparameter Optimization** | **GridSearchCV (`n_jobs=-1`)** |
| **Final Model** | **Optimized K-Nearest Neighbors (K-NN)** |

The training and validation setup was designed to check for overfitting while preserving a clean separation between training and evaluation data.

##  Machine Learning Workflow

```text
Raw Dataset
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Data Preprocessing
    │
    ├── Missing-value / data checks
    ├── Feature preparation
    └── Feature scaling with StandardScaler
    │
    ▼
K-NN Model Training
    │
    ▼
Hyperparameter Tuning with GridSearchCV
    │
    ├── Number of neighbors: k = 1 ... 25
    ├── Distance metric: Manhattan / Euclidean
    └── Weighting: uniform / distance
    │
    ▼
Cross-Validation & Evaluation
    │
    ▼
Model Serialization with Joblib
    │
    ▼
Streamlit Inference Dashboard
```

##  Engineering & Pipeline Highlights

### 1. Data Leakage Prevention
Feature scaling with **StandardScaler** is strictly fitted on the training folds inside the cross-validation loop to avoid data contamination and ensure reliable validation results.

### 2. Decoupled Architecture
A clean separation is maintained between the core model inference layer (`app.py`) and the presentation layer (`streamlit_app.py`), following practical MLOps-style design patterns.

### 3. Hyperparameter Tuning
The model evaluates combinations of:

- **Number of neighbors:** `1 ≤ k ≤ 25`
- **Distance metric:** `p = 1` (Manhattan) vs. `p = 2` (Euclidean)
- **Neighbor weighting:** `uniform` vs. `distance`

The best configuration is selected automatically through **GridSearchCV**.

##  Repository Structure

```text
knn-project/
│
├── data/
│   ├── raw/                         # Raw dataset (breast_cancer.csv)
│   └── processed/                   # Scaled & split data artifacts
│
├── models/
│   ├── scaler.pkl                   # Fitted StandardScaler object
│   └── knn_model.pkl                # Final K-NN model artifact
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # Data preparation notebook
│   ├── 02_model_training_tuning.ipynb # Model training & tuning
│   └── 03_model_evaluation_inference.ipynb # Evaluation & inference
│
├── app.py                           # Decoupled inference engine
├── streamlit_app.py                 # Streamlit Web Dashboard UI
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

##  Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Diraribnsalah/knn-breast-cancer-diagnostic.git
cd knn-breast-cancer-diagnostic
```

### 2. Set Up Virtual Environment & Dependencies

#### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Web Dashboard

```bash
streamlit run streamlit_app.py
```

After starting Streamlit, open the local URL displayed in the terminal.

##  Model Configuration

The K-NN model is optimized using a systematic grid search across the main hyperparameters:

```python
param_grid = {
    "n_neighbors": range(1, 26),
    "p": [1, 2],
    "weights": ["uniform", "distance"]
}
```

The tuning process uses cross-validation to select the configuration that provides the strongest validation performance.

##  Model Serialization

The trained artifacts are stored with **Joblib** so that the Streamlit application can load the fitted preprocessing and model objects directly during inference.

- `models/scaler.pkl` → fitted feature scaler
- `models/knn_model.pkl` → optimized K-NN estimator

##  Notebooks

The notebooks document the project from experimentation to deployment-oriented inference:

1. **`_data_preprocessing.ipynb`** — data loading, inspection, preprocessing, and preparation.
2. **`_model_training_tuning.ipynb`** — K-NN training and systematic hyperparameter optimization.
3. **`_model_evaluation_inference.ipynb`** — final evaluation, inference workflow, and deployment preparation.

##  Tech Stack

- **Core Language:** Python 3.10
- **Machine Learning:** Scikit-learn
- **Data Engineering:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Web Framework:** Streamlit

##  Deployment

The application is deployed using **Streamlit Community Cloud** and exposes the trained K-NN pipeline through an interactive web interface for real-time probabilistic scoring and sample patient selection.

**Live app:** https://knn-breast-cancer-diagnostic.streamlit.app/

##  Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset for binary classification experiments.

- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Task: classify breast cancer cases into diagnostic classes using numerical features.

##  Project Goals

This project demonstrates an end-to-end Machine Learning workflow rather than only model training. The main goals are to:

- Build a reproducible preprocessing pipeline.
- Prevent data leakage during model validation.
- Systematically optimize K-NN hyperparameters.
- Serialize trained artifacts for reuse.
- Separate inference logic from the user interface.
- Deliver the final model through an interactive Streamlit application.

##  License

This project is licensed under the **MIT License**.

See the [MIT License](https://opensource.org/licenses/MIT) for details.
