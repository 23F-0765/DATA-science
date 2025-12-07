import streamlit as st
import pandas as pd
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="DS Assignment 4 Dashboard",
    layout="wide"
)

# Base directory for data
BASE_DIR = "DS-Assignment 4"

def load_data(rq_folder, filename):
    """Helper to load CSV data safely."""
    file_path = os.path.join(BASE_DIR, rq_folder, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning(f"File not found: {file_path}")
        return None

def load_image(rq_folder, filename):
    """Helper to load image safely."""
    file_path = os.path.join(BASE_DIR, rq_folder, filename)
    if os.path.exists(file_path):
        return Image.open(file_path)
    else:
        st.warning(f"Image not found: {file_path}")
        return None

def display_rq_results(rq_name, rq_folder, rq_prefix):
    """Displays results for a specific Research Question."""
    st.header(f"{rq_name} Results")

    # 1. Comparison Table
    st.subheader("Model Comparison")
    comparison_df = load_data(rq_folder, f"{rq_prefix}_comparison_table.csv")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
    
    # 2. Best Model Metrics
    st.subheader("Best Model Metrics (Train vs Test)")
    metrics_df = load_data(rq_folder, f"{rq_prefix}_best_train_test_metrics.csv")
    if metrics_df is not None:
        st.table(metrics_df)

    # 3. Prediction Plot
    st.subheader("Prediction vs Actual (Best Model)")
    image = load_image(rq_folder, f"{rq_prefix}_pred_vs_actual_best.png")
    if image is not None:
        st.image(image, caption=f"{rq_name} - Best Model Predictions", use_container_width=True)
    
    # 4. Bootstrap CI (Optional, included if available)
    st.subheader("Bootstrap MAE Confidence Intervals")
    bootstrap_df = load_data(rq_folder, f"{rq_prefix}_bootstrap_mae_ci.csv")
    if bootstrap_df is not None:
        st.dataframe(bootstrap_df, use_container_width=True)

# Main App Layout
st.title("Data Science Assignment 4 Analysis")
st.markdown("This dashboard visualizes the results of the analysis performed on the dataset.")

# Sidebar Navigation
tabs = ["Project Overview", "RQ1 Results", "RQ2 Results", "RQ3 Results"]
selected_tab = st.sidebar.radio("Navigate", tabs)

if selected_tab == "Project Overview":
    st.header("Project Overview")
    st.write("This project analyzes student performance data.")
    
    st.subheader("Key Visualizations")
    
    # Display project root images
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("Data cleaning.png"):
            st.image("Data cleaning.png", caption="Data Cleaning Process", use_container_width=True)
        else:
            st.warning("Data cleaning.png not found in project root.")
            
    with col2:
        if os.path.exists("Prediction.png"):
            st.image("Prediction.png", caption="Prediction Overview", use_container_width=True)
        else:
            st.warning("Prediction.png not found in project root.")
            
    # Try to read README if it has content (we saw it was empty/header only, but good practice)
    if os.path.exists("README.md"):
        with open("README.md", "r") as f:
            readme_content = f.read()
            if len(readme_content) > 20: # arbitrary length to filter empty readmes
                st.markdown("---")
                st.markdown(readme_content)

elif selected_tab == "RQ1 Results":
    display_rq_results("RQ1", "rq1_outputs", "rq1")

elif selected_tab == "RQ2 Results":
    display_rq_results("RQ2", "rq2_outputs", "rq2")

elif selected_tab == "RQ3 Results":
    display_rq_results("RQ3", "rq3_outputs", "rq3")

