import streamlit as st
import pandas as pd
import joblib
import json
import os

# Set page config
st.set_page_config(
    page_title="Indian Water Quality Intelligence System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
    }
    .prediction-box {
        background-color: #e1f5fe;
        border-left: 5px solid #0288d1;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #01579b;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #7f8c8d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "water_quality_preprocessor.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "water_quality_gradient_boosting_model.joblib")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "water_quality_model_info.json")

@st.cache_resource
def load_models():
    """Load the preprocessor, model, and info safely with caching."""
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        return preprocessor, model, model_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    st.markdown('<div class="main-header">💧 Indian Water Quality Intelligence System</div>', unsafe_allow_html=True)
    st.write("""
    Welcome to the Indian Water Quality Intelligence System. This application leverages a classical Machine Learning 
    approach (Gradient Boosting Regression) to predict **Dissolved Oxygen (DO)** in water bodies.
    """)

    # Load components
    preprocessor, model, model_info = load_models()
    
    if preprocessor is None or model is None:
        st.stop()

    # Define Input Categories (based on preprocessor)
    countries = ['Canada', 'China', 'England', 'Ireland', 'USA']
    waterbody_types = ['Bay', 'Canal', 'Coastal', 'Drainage', 'Effluent', 'Estuarine', 'Lake', 'Marine', 'River', 'Sea Water', 'Sewage', 'Transitional']
    seasons = ['Autumn', 'Spring', 'Summer', 'Winter']

    st.sidebar.header("Input Parameters")
    st.sidebar.write("Please provide the water quality readings below:")

    # Input Form
    with st.sidebar.form("prediction_form"):
        # Categorical Inputs
        country = st.selectbox("Country", options=countries)
        waterbody = st.selectbox("Waterbody Type", options=waterbody_types)
        season = st.selectbox("Season", options=seasons)
        
        st.markdown("---")
        
        # Date Inputs
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2100, value=2020, step=1)
        with col2:
            month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
        
        st.markdown("---")
        
        # Numeric Inputs
        ammonia = st.number_input("Ammonia (mg/l)", min_value=0.0, value=0.5, step=0.1)
        bod = st.number_input("Biochemical Oxygen Demand (mg/l)", min_value=0.0, value=2.0, step=0.1)
        orthophosphate = st.number_input("Orthophosphate (mg/l)", min_value=0.0, value=0.05, step=0.01)
        ph = st.number_input("pH (ph units)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0, value=20.0, step=0.5)
        nitrogen = st.number_input("Nitrogen (mg/l)", min_value=0.0, value=1.0, step=0.1)
        nitrate = st.number_input("Nitrate (mg/l)", min_value=0.0, value=2.0, step=0.1)

        submit_button = st.form_submit_button(label="Predict Dissolved Oxygen")

    # Main area content
    if submit_button:
        # Calculate derived feature
        temp_extreme = 1 if (temp < 0 or temp > 50) else 0

        # Create DataFrame matching original feature names
        input_data = pd.DataFrame([{
            'Country': country,
            'Waterbody Type': waterbody,
            'Ammonia (mg/l)': ammonia,
            'Biochemical Oxygen Demand (mg/l)': bod,
            'Orthophosphate (mg/l)': orthophosphate,
            'pH (ph units)': ph,
            'Temperature (cel)': temp,
            'Nitrogen (mg/l)': nitrogen,
            'Nitrate (mg/l)': nitrate,
            'Year': year,
            'Month': month,
            'Season': season,
            'Temperature_Extreme': temp_extreme
        }])

        st.subheader("Input Summary")
        st.dataframe(input_data, use_container_width=True)

        with st.spinner("Processing data and generating prediction..."):
            try:
                # 1. Apply preprocessor
                processed_data = preprocessor.transform(input_data)
                
                # 2. Predict
                prediction = model.predict(processed_data)[0]
                
                # 3. Display Result
                st.markdown(f"""
                <div class="prediction-box">
                    <div>Predicted Dissolved Oxygen (DO):</div>
                    <div class="prediction-value">{prediction:.2f} mg/L</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="disclaimer">
                <b>Disclaimer:</b> This prediction is based solely on the provided inputs and historical data modeling. 
                It does not serve as a definitive statement on the safety or potability of the water.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    # Model Information Section
    st.markdown("---")
    st.subheader("Model Performance & Information")
    st.write("Transparent overview of the trained model underlying this application.")
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{model_info.get('test_mae', 1.2924):.4f}")
    with col_metric2:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{model_info.get('test_rmse', 1.9092):.4f}")
    with col_metric3:
        st.metric(label="R-Squared (R²)", value=f"{model_info.get('test_r2', 0.2464):.4f}")
    
    st.info(f"""
    **Model Type:** {model_info.get('model', 'Gradient Boosting Regressor')}  
    **Target Variable:** {model_info.get('target', 'Dissolved Oxygen (mg/l)')}  
    **Training Samples:** {model_info.get('training_samples', 'N/A')} | **Test Samples:** {model_info.get('test_samples', 'N/A')}
    
    *Note: An R² of {model_info.get('test_r2', 0.2464):.4f} indicates that the model captures roughly {model_info.get('test_r2', 0.2464)*100:.1f}% of the variance in Dissolved Oxygen. This highlights the complex, multi-faceted nature of water quality prediction in real-world scenarios.*
    """)

if __name__ == "__main__":
    main()
