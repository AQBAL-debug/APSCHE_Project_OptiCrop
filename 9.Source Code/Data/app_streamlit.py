import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Auto-train model if it doesn't exist
@st.cache_resource
def ensure_model_exists():
    """Ensure model and data exist, train if necessary"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    data_path = os.path.join(data_dir, 'crop_recommendation.csv')
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if model exists
    model_path = os.path.join(model_dir, 'crop_recommendation_model.pkl')
    if not os.path.exists(model_path):
        st.info("Training model for first use... This may take a minute.")
        from data_preprocessing import DataPreprocessor
        from model_training import ModelTrainer
        
        # Download dataset if not exists
        if not os.path.exists(data_path):
            try:
                import kagglehub
                st.info("Downloading dataset from Kaggle...")
                path = kagglehub.dataset_download('atharvaingle/crop-recommendation-dataset')
                import shutil
                shutil.copy(f'{path}/Crop_recommendation.csv', data_path)
            except:
                st.error("Could not download dataset. Please ensure kagglehub is installed.")
                return False
        
        # Train model
        preprocessor = DataPreprocessor(data_path)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocessor.preprocess_pipeline()
        preprocessor.save_preprocessors(model_dir)
        
        trainer = ModelTrainer(model_dir)
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        trainer.training_pipeline(X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, feature_names)
        
        st.success("Model trained successfully!")
    
    return True

# Ensure model exists before loading
if ensure_model_exists():
    from crop_recommendation import CropRecommendationSystem

# Page configuration
st.set_page_config(
    page_title="OptiCrop - Smart Agricultural Production Optimization",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #556B2F;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the recommendation system
@st.cache_resource
def load_system():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    return CropRecommendationSystem(model_dir)

try:
    system = load_system()
except Exception as e:
    st.error(f"Error loading the model. Please ensure the model is trained first. Error: {e}")
    st.stop()

# Main title
st.markdown('<h1 class="main-header">🌾 OptiCrop</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Smart Agricultural Production Optimization Engine</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Scenario", [
    "🌱 Smart Crop Recommendation",
    "🔍 Crop Suitability Assessment",
    "📊 Agricultural Research Dashboard"
])

# Common input function
def get_environmental_inputs():
    """Get environmental inputs from user"""
    st.subheader("Environmental Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 90, help="Nitrogen content ratio in soil")
        P = st.slider("Phosphorus (P)", 5, 145, 42, help="Phosphorus content ratio in soil")
        K = st.slider("Potassium (K)", 5, 205, 43, help="Potassium content ratio in soil")
        temperature = st.slider("Temperature (°C)", 0.0, 45.0, 20.88, 0.01, help="Soil temperature")
    
    with col2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.00, 0.01, help="Relative humidity")
        ph = st.slider("Soil pH", 3.5, 9.5, 6.50, 0.01, help="Soil pH level")
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 202.94, 0.01, help="Annual rainfall")
    
    return N, P, K, temperature, humidity, ph, rainfall

# Scenario 1: Smart Crop Recommendation
if page == "🌱 Smart Crop Recommendation":
    st.markdown('<h2 class="sub-header">Smart Crop Recommendation for Farmers</h2>', unsafe_allow_html=True)
    
    st.write("""
    Enter your soil and environmental details below to get personalized crop recommendations 
    for maximum yield and better farming efficiency.
    """)
    
    N, P, K, temperature, humidity, ph, rainfall = get_environmental_inputs()
    
    if st.button("🔮 Get Crop Recommendation", type="primary"):
        with st.spinner("Analyzing your soil conditions..."):
            result = system.predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        
        # Display main recommendation
        st.markdown(f'<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Recommended Crop: **{result['recommended_crop'].upper()}**")
        st.markdown(f"**Confidence Level:** {result['confidence']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display top 3 recommendations
        st.subheader("Top 3 Crop Recommendations")
        
        cols = st.columns(3)
        for i, (col, rec) in enumerate(zip(cols, result['top_3_recommendations']), 1):
            with col:
                st.metric(
                    label=f"Option {i}",
                    value=rec['crop'],
                    delta=f"{rec['probability']:.1%}"
                )
        
        # Environmental analysis
        st.subheader("Environmental Factor Analysis")
        analysis = system.analyze_environmental_factors(N, P, K, temperature, humidity, ph, rainfall)
        
        analysis_df = pd.DataFrame([
            {
                'Factor': factor.capitalize(),
                'Value': data['value'],
                'Status': data['status']
            }
            for factor, data in analysis.items()
        ])
        
        st.dataframe(analysis_df, use_container_width=True, hide_index=True)

# Scenario 2: Crop Suitability Assessment
elif page == "🔍 Crop Suitability Assessment":
    st.markdown('<h2 class="sub-header">Crop Suitability and Environmental Assessment</h2>', unsafe_allow_html=True)
    
    st.write("""
    Check whether your current soil and climate conditions are suitable for a specific crop.
    Get insights about crop compatibility, environmental suitability, and productivity potential.
    """)
    
    N, P, K, temperature, humidity, ph, rainfall = get_environmental_inputs()
    
    # Get all available crops
    all_crops = system.get_all_crops()
    target_crop = st.selectbox("Select Crop to Assess", sorted(all_crops))
    
    if st.button("🔍 Assess Suitability", type="primary"):
        with st.spinner("Assessing crop suitability..."):
            suitability = system.assess_crop_suitability(
                N, P, K, temperature, humidity, ph, rainfall, target_crop
            )
        
        if 'error' in suitability:
            st.error(suitability['error'])
        else:
            # Display suitability result
            score = suitability['suitability_score']
            level = suitability['suitability_level']
            
            # Color coding based on suitability
            if score >= 0.7:
                color = "#d4edda"
                border = "#c3e6cb"
                text_color = "#155724"
            elif score >= 0.5:
                color = "#fff3cd"
                border = "#ffeaa7"
                text_color = "#856404"
            else:
                color = "#f8d7da"
                border = "#f5c6cb"
                text_color = "#721c24"
            
            st.markdown(f"""
            <div style="background-color: {color}; border: 1px solid {border}; 
                        color: {text_color}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin-top: 0;">Suitability Assessment for {target_crop}</h3>
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <strong>Suitability Level:</strong> {level}
                </p>
                <p style="font-size: 1.5rem; margin-bottom: 1rem;">
                    <strong>Score:</strong> {score:.1%}
                </p>
                <p>{suitability['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for visual representation
            st.progress(score)
            st.caption(f"Suitability Score: {score:.1%}")
            
            # Environmental analysis
            st.subheader("Environmental Factor Analysis")
            analysis = system.analyze_environmental_factors(N, P, K, temperature, humidity, ph, rainfall)
            
            analysis_df = pd.DataFrame([
                {
                    'Factor': factor.capitalize(),
                    'Value': data['value'],
                    'Status': data['status']
                }
                for factor, data in analysis.items()
            ])
            
            st.dataframe(analysis_df, use_container_width=True, hide_index=True)

# Scenario 3: Agricultural Research Dashboard
elif page == "📊 Agricultural Research Dashboard":
    st.markdown('<h2 class="sub-header">Agricultural Research and Policy Planning</h2>', unsafe_allow_html=True)
    
    st.write("""
    Analyze crop-environment relationships and identify patterns in agricultural production.
    This dashboard helps researchers and policymakers make data-driven decisions for 
    sustainable farming strategies and resource optimization.
    """)
    
    # Load the dataset for analysis
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'crop_recommendation.csv')
    df = pd.read_csv(data_path)
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Crop Distribution", "Environmental Analysis", "Crop-Environment Relationships"])
    
    with tab1:
        st.subheader("Crop Distribution in Dataset")
        
        crop_counts = df['label'].value_counts().reset_index()
        crop_counts.columns = ['Crop', 'Count']
        
        fig = px.bar(crop_counts, x='Crop', y='Count', 
                     title='Distribution of Crop Samples',
                     color='Count',
                     color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Unique Crops", df['label'].nunique())
        with col3:
            st.metric("Environmental Features", len(df.columns) - 1)
    
    with tab2:
        st.subheader("Environmental Factor Analysis")
        
        # Feature distributions
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        selected_feature = st.selectbox("Select Environmental Factor", features)
        
        fig = px.histogram(df, x=selected_feature, 
                          title=f'Distribution of {selected_feature}',
                          nbins=50,
                          color='label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Matrix of Environmental Factors")
        corr_matrix = df[features].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Crop-Environment Relationships")
        
        # Select two features to compare
        col1, col2 = st.columns(2)
        with col1:
            feature_x = st.selectbox("Select X-axis Feature", features, index=3)
        with col2:
            feature_y = st.selectbox("Select Y-axis Feature", features, index=6)
        
        # Scatter plot
        fig = px.scatter(df, x=feature_x, y=feature_y, color='label',
                        title=f'{feature_x} vs {feature_y} by Crop',
                        hover_data=['label'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots for each crop
        st.subheader("Environmental Ranges by Crop")
        selected_crop = st.selectbox("Select Crop", sorted(df['label'].unique()))
        
        crop_data = df[df['label'] == selected_crop]
        
        cols = st.columns(4)
        for i, (col, feature) in enumerate(zip(cols, features[:4])):
            with col:
                fig = px.box(crop_data, y=feature, 
                           title=f'{feature} for {selected_crop}')
                st.plotly_chart(fig, use_container_width=True)
        
        cols = st.columns(3)
        for i, (col, feature) in enumerate(zip(cols, features[4:])):
            with col:
                fig = px.box(crop_data, y=feature, 
                           title=f'{feature} for {selected_crop}')
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🌾 OptiCrop - Smart Agricultural Production Optimization Engine</p>
    <p>Empowering farmers with data-driven insights for sustainable agriculture</p>
</div>
""", unsafe_allow_html=True)
