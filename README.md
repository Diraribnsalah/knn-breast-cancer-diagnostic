#  End-to-End Breast Cancer Diagnostic System (k-NN)

An enterprise-ready, modular Machine Learning project for early breast cancer diagnosis using an optimized **k-Nearest Neighbors (k-NN)** algorithm. The system transitions from exploratory data analysis to a structured production-grade pipeline integrated with an interactive **Streamlit** web application.

---

##  Key Metrics & Highlights

- **Test Accuracy:** `98.25%`
- **Train Accuracy:** `97.80%` (Validated against overfitting via 5-Fold Cross-Validation)
- **Optimization Strategy:** Hyperparameter tuning via `GridSearchCV` (`n_jobs=-1`)
- **Architecture:** Modular ML Workflow (`data/`, `models/`, `notebooks/`, `src/`)
- **Deployment:** Interactive Web Interface with real-time confidence scores and sample patient selection.

---

##  Repository Structure

```text
knn_project/
├── data/
│   ├── raw/                      # Raw dataset (breast_cancer.csv)
│   └── processed/                # Scaled & split artifacts (scaled_data.pkl)
├── models/                       # Persisted production artifacts
│   ├── scaler.pkl                # Fitted StandardScaler object
│   └── knn_model.pkl             # Best estimator model artifact
├── notebooks/                    # Sequential modular development notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training_tuning.ipynb
│   └── 03_model_evaluation_inference.ipynb
├── app.py                        # Core inference engine module
├── streamlit_app.py              # Streamlit Web Application Dashboard
├── requirements.txt              # Project dependencies & environment specs
└── README.md                     # Project documentation
