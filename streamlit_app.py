import os
import sys
import streamlit as st
import pandas as pd

#  Add project path to system environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import previously developed inference engine
from app import predict_cancer

#  Basic page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnostics",
    page_icon="",
    layout="wide"
)

st.title(" Breast Cancer Diagnostic Assistant")
st.markdown("An interactive web application for early breast cancer diagnosis using an optimized **k-NN** model with **98.25%** test accuracy.")

#  Load raw dataset for sample patient testing
@st.cache_data
def load_sample_data():
    csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'breast_cancer.csv')
    return pd.read_csv(csv_path)

df = load_sample_data()
X_features = df.drop(columns=['target'])

#  Sidebar options for data input
st.sidebar.header(" Input Configurations")
mode = st.sidebar.radio(
    "Select patient input method:",
    ["Load Sample Patient from Dataset", "Manual Feature Input"]
)

if mode == "Load Sample Patient from Dataset":
    patient_idx = st.sidebar.number_input("Select Patient Index (0 to 568):", min_value=0, max_value=len(df)-1, value=0)
    input_data = X_features.iloc[[patient_idx]]
    actual_label = "Benign" if df['target'].iloc[patient_idx] == 1 else "Malignant"
    st.info(f" Ground Truth Diagnosis recorded in dataset: **{actual_label}**")

else:
    st.subheader(" Enter Patient Physiological Values:")
    input_dict = {}
    cols = st.columns(3)
    for idx, col_name in enumerate(X_features.columns):
        with cols[idx % 3]:
            min_val = float(X_features[col_name].min())
            max_val = float(X_features[col_name].max())
            mean_val = float(X_features[col_name].mean())
            input_dict[col_name] = st.number_input(
                label=col_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
    input_data = pd.DataFrame([input_dict])

# Display input feature values
st.write("###  Medical Readout Summary:")
st.dataframe(input_data)

#  Prediction execution and output rendering
if st.button(" Run Smart Diagnosis", type="primary"):
    res = predict_cancer(input_data)
    
    st.markdown("---")
    st.subheader(" Diagnosis Results:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if res['Diagnosis'] == 'Malignant':
            st.error(f" Diagnosis: **{res['Diagnosis']}**")
        else:
            st.success(f" Diagnosis: **{res['Diagnosis']}**")
            
    with col2:
        st.metric("Benign Probability", f"{res['Confidence_Benign_%']}%")
        
    with col3:
        st.metric("Malignant Probability", f"{res['Confidence_Malignant_%']}%")
