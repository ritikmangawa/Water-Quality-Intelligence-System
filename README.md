# 💧 Indian Water Quality Intelligence System

Welcome to the **Indian Water Quality Intelligence System**. This project leverages classical Machine Learning approaches to predict the **Dissolved Oxygen (DO)** in water bodies across various regions. 

This repository contains the complete end-to-end pipeline, from exploratory data analysis and feature engineering to model training, and finally a production-ready Streamlit application.

## 🌟 Features
- **Accurate Predictions**: Uses a trained **Gradient Boosting Regressor** model to predict Dissolved Oxygen based on 13 environmental and chemical parameters.
- **Interactive Dashboard**: A sleek, user-friendly Streamlit web app to input water quality readings and get real-time predictions.
- **Robust Preprocessing**: Includes a robust pipeline that standardizes numerical features, encodes categorical variables, and handles missing/unknown categories gracefully.
- **No Over-promising**: Emphasizes transparency by showcasing the exact model performance metrics (MAE, RMSE, R²) within the application.

## 📁 Project Structure

```text
Water-Quality-Intelligence/
│
├── notebooks/                 # Jupyter notebooks for EDA, data cleaning, engineering, and training
│
├── models/                    # Serialized machine learning models and pipelines
│   ├── water_quality_preprocessor.joblib
│   ├── water_quality_gradient_boosting_model.joblib
│   └── water_quality_model_info.json
│
├── app.py                     # The Streamlit application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
*(Note: The `data/` and `processed_data/` directories are intentionally ignored via `.gitignore` to prevent uploading large dataset files to GitHub.)*

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
Ensure you have Python installed (preferably 3.8+). It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Streamlit dashboard by running:
```bash
streamlit run app.py
```
This will open the web application in your default browser automatically.

## 📊 Model Information
- **Target Variable**: Dissolved Oxygen (mg/l)
- **Model Used**: Gradient Boosting Regressor
- **Input Features**: Country, Waterbody Type, Ammonia, BOD, Orthophosphate, pH, Temperature, Nitrogen, Nitrate, Year, Month, Season, and derived Temperature_Extreme.

## ⚠️ Disclaimer
Predictions made by this tool are based solely on historical data modeling. This application is an educational and intelligence tool, and its results should not serve as a definitive statement on the safety or potability of the water.
