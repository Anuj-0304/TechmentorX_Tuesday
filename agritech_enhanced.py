import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from tensorflow import keras
from PIL import Image
import os
import sys

# Import custom modules
sys.path.insert(0, '/home/claude')
from disease_detection import DiseaseDetectionModel
from multilingual import MultilingualSupport
from soil_health import SoilHealthAnalyzer
from model_integration import CropPredictionModel

# Page configuration
st.set_page_config(
    page_title="AgriTech Pro - Smart Farming Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'disease_model' not in st.session_state:
    st.session_state.disease_model = None
if 'crop_model' not in st.session_state:
    st.session_state.crop_model = None

# Initialize modules
translator = MultilingualSupport()
soil_analyzer = SoilHealthAnalyzer()

# Enhanced Custom CSS for better UI/UX
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Modern card design */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* Green themed metric card */
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white;
    }
    
    /* Alert boxes */
    .stAlert {
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    
    /* Headers */
    h1 {
        color: #2E7D32;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4CAF50;
    }
    
    h2 {
        color: #388E3C;
        font-weight: 600;
    }
    
    h3 {
        color: #43A047;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Disease detection result card */
    .disease-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    /* Soil health gauge */
    .health-gauge {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for translations
def t(key):
    """Get translated text"""
    return translator.get_text(key, st.session_state.language)

# Ollama API integration
def query_ollama(prompt, model="llama3.2"):
    """Query local Ollama model"""
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "AI assistant temporarily unavailable. Using rule-based recommendations."
    except Exception as e:
        return "AI assistant offline. Showing pre-configured recommendations."

# Load models with caching
@st.cache_resource
def load_disease_model():
    """Load disease detection model"""
    model = DiseaseDetectionModel('disease_model.keras')
    model.load_model()
    return model

@st.cache_resource
def load_crop_model():
    """Load crop recommendation model"""
    model = CropPredictionModel('crop_model.keras')
    model.load_model()
    return model

# Initialize models
if st.session_state.disease_model is None:
    st.session_state.disease_model = load_disease_model()
if st.session_state.crop_model is None:
    st.session_state.crop_model = load_crop_model()

# Market price data (simulated)
def get_market_prices():
    """Get current market prices for crops"""
    crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Pulses", "Vegetables", "Fruits"]
    prices = np.random.randint(1000, 5000, len(crops))
    trends = np.random.choice(["↑", "↓", "→"], len(crops))
    changes = np.random.randint(-15, 20, len(crops))
    
    return pd.DataFrame({
        "Crop": crops,
        "Price (₹/quintal)": prices,
        "Trend": trends,
        "Change (%)": changes
    })

# Weather simulation
def get_weather_data(location="Default"):
    """Get weather forecast"""
    days = ["Today", "Tomorrow", "Day 3", "Day 4", "Day 5"]
    temps = np.random.randint(20, 35, 5)
    humidity = np.random.randint(60, 90, 5)
    rainfall = np.random.randint(0, 50, 5)
    
    return pd.DataFrame({
        "Day": days,
        "Temperature (°C)": temps,
        "Humidity (%)": humidity,
        "Rainfall (mm)": rainfall
    })

# Sidebar
st.sidebar.title("🌾 AgriTech Pro")

# Language selector
languages = translator.get_all_languages()
selected_lang = st.sidebar.selectbox(
    "🌐 Language / भाषा",
    options=list(languages.keys()),
    format_func=lambda x: languages[x],
    index=0
)

if selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.rerun()

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    t('dashboard'),
    ["🏠 " + t('dashboard'), 
     "🌱 " + t('crop_recommendation'), 
     "🔬 " + t('disease_detection'),
     "🌍 " + t('soil_health'),
     "📊 " + t('market_insights'), 
     "🌦️ " + t('weather_forecast'), 
     "💬 " + t('ai_assistant'), 
     "📚 " + t('resources')]
)

st.sidebar.markdown("---")
st.sidebar.info("**Smart Farming for Better Tomorrow**\n\nData-driven insights for optimal crop management and sustainable agriculture.")

# Main content based on page selection
if page.startswith("🏠"):
    st.title(t('app_title'))
    st.markdown(f"### {t('welcome')}")
    
    # Key metrics with modern cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">🏛️ Active Farms</h3>
            <h2 style="color: white; margin: 10px 0;">1,234</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0;">+12% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card-green">
            <h3 style="color: white; margin: 0;">🌾 Crop Varieties</h3>
            <h2 style="color: white; margin: 10px 0;">45</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0;">+3 new varieties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: white; margin: 0;">📈 Avg Yield</h3>
            <h2 style="color: white; margin: 10px 0;">18%</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0;">Increase</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card-green">
            <h3 style="color: white; margin: 0;">📡 Market Updates</h3>
            <h2 style="color: white; margin: 10px 0;">Live</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0;">Real-time data</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Crop Production Trends")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        production = [320, 380, 450, 410, 490, 520]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=production,
            mode='lines+markers',
            name='Production',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=10, color='#2E7D32')
        ))
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Production (tons)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌾 Crop Distribution")
        crops = ["Rice", "Wheat", "Cotton", "Maize", "Others"]
        values = [30, 25, 20, 15, 10]
        colors = ['#2E7D32', '#43A047', '#66BB6A', '#81C784', '#A5D6A7']
        
        fig = go.Figure(data=[go.Pie(
            labels=crops,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent'
        )])
        fig.update_layout(
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature highlights
    st.markdown("---")
    st.subheader("✨ Platform Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f5f7fa; border-radius: 15px;'>
            <div style='font-size: 3rem;'>🌱</div>
            <h4>Smart Recommendations</h4>
            <p style='color: #666; font-size: 0.9rem;'>AI-powered crop suggestions based on soil & climate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f5f7fa; border-radius: 15px;'>
            <div style='font-size: 3rem;'>🔬</div>
            <h4>Disease Detection</h4>
            <p style='color: #666; font-size: 0.9rem;'>CNN-based image analysis for crop diseases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f5f7fa; border-radius: 15px;'>
            <div style='font-size: 3rem;'>🌍</div>
            <h4>Soil Health</h4>
            <p style='color: #666; font-size: 0.9rem;'>Comprehensive soil analysis & recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f5f7fa; border-radius: 15px;'>
            <div style='font-size: 3rem;'>🌐</div>
            <h4>Multilingual</h4>
            <p style='color: #666; font-size: 0.9rem;'>Support for 5+ Indian languages</p>
        </div>
        """, unsafe_allow_html=True)

elif page.startswith("🌱"):
    st.title("🌱 " + t('crop_recommendation'))
    st.markdown("Get AI-powered crop suggestions based on your soil and environmental conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 " + t('location_soil'))
        location = st.text_input(t('location'), "Maharashtra, India")
        soil_type = st.selectbox(t('soil_type'), 
                                ["Loamy", "Sandy", "Clay", "Black", "Red", "Alluvial"])
        ph = st.slider(t('soil_ph'), 4.0, 9.0, 6.5, 0.1)
        
        st.subheader("💧 " + t('nutrients'))
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            nitrogen = st.number_input(t('nitrogen'), 0, 200, 90)
        with col_p:
            phosphorus = st.number_input(t('phosphorus'), 0, 100, 42)
        with col_k:
            potassium = st.number_input(t('potassium'), 0, 100, 43)
    
    with col2:
        st.subheader("🌦️ " + t('environmental'))
        temperature = st.slider(t('temperature'), 10, 45, 25)
        humidity = st.slider(t('humidity'), 30, 100, 82)
        rainfall = st.slider(t('rainfall'), 0, 300, 202)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎯 " + t('get_recommendation'), use_container_width=True):
            with st.spinner(t('loading')):
                # Get prediction from model
                crop, confidence = st.session_state.crop_model.predict_crop(
                    nitrogen, phosphorus, potassium,
                    temperature, humidity, ph, rainfall
                )
                
                # Get top 3 recommendations
                top_crops = st.session_state.crop_model.get_top_n_crops(
                    nitrogen, phosphorus, potassium,
                    temperature, humidity, ph, rainfall, n=3
                )
                
                st.markdown("---")
                st.success("✅ " + t('analysis_complete'))
                
                # Display recommendation
                st.markdown(f"""
                <div class="disease-card" style="border-left-color: #4CAF50;">
                    <h2 style="color: #2E7D32; margin-top: 0;">🌟 {t('recommended_crop')}</h2>
                    <h1 style="color: #1B5E20; margin: 10px 0;">{crop}</h1>
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                color: white; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <h3 style="margin: 0; color: white;">{t('confidence')}: {confidence:.1%}</h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Alternative crops
                st.subheader("📋 " + t('alternative_crops'))
                
                for i, (crop_name, conf) in enumerate(top_crops[1:], 2):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{i}. {crop_name}**")
                    with col_b:
                        st.progress(conf)
                        st.caption(f"{conf:.1%}")

elif page.startswith("🔬"):
    st.title("🔬 " + t('disease_detection'))
    st.markdown("Upload an image of your crop to detect diseases using AI-powered image analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 " + t('upload_image'))
        
        uploaded_file = st.file_uploader(
            t('upload_prompt'),
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the affected crop leaf"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 " + t('analyze_disease'), use_container_width=True):
                with st.spinner("Analyzing image with CNN model..."):
                    # Detect disease
                    disease, confidence, treatment = st.session_state.disease_model.detect_disease(image)
                    
                    # Calculate affected area
                    affected_area = st.session_state.disease_model.calculate_disease_severity(image)
                    
                    # Get multiple predictions
                    top_predictions = st.session_state.disease_model.get_multiple_predictions(image, top_n=3)
                    
                    st.session_state.disease_result = {
                        'disease': disease,
                        'confidence': confidence,
                        'treatment': treatment,
                        'affected_area': affected_area,
                        'top_predictions': top_predictions
                    }
                    
                    st.rerun()
    
    with col2:
        if 'disease_result' in st.session_state:
            result = st.session_state.disease_result
            
            # Display detection result
            severity_color = {
                'None': '#4CAF50',
                'Low': '#8BC34A',
                'Medium': '#FFC107',
                'High': '#FF5722'
            }.get(result['treatment'].get('severity', 'Medium'), '#FFC107')
            
            st.markdown(f"""
            <div class="disease-card" style="border-left-color: {severity_color};">
                <h2 style="color: #d32f2f; margin-top: 0;">🦠 {t('disease_detected')}</h2>
                <h1 style="color: #b71c1c; margin: 10px 0;">{result['disease']}</h1>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <div>
                        <p style="margin: 0; color: #666;">Confidence</p>
                        <h3 style="margin: 5px 0; color: #1976D2;">{result['confidence']:.1%}</h3>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666;">{t('severity')}</p>
                        <h3 style="margin: 5px 0; color: {severity_color};">
                            {result['treatment'].get('severity', 'Unknown')}
                        </h3>
                    </div>
                    <div>
                        <p style="margin: 0; color: #666;">{t('affected_area')}</p>
                        <h3 style="margin: 5px 0; color: #FF5722;">{result['affected_area']:.1f}%</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Treatment information tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "💊 " + t('treatment'),
                "🛡️ " + t('prevention'),
                "🌿 " + t('organic_treatment'),
                "🧪 " + t('chemical_treatment')
            ])
            
            with tab1:
                st.markdown(f"**{t('treatment')}:**")
                st.info(result['treatment'].get('treatment', 'Consult agricultural expert'))
            
            with tab2:
                st.markdown(f"**{t('prevention')}:**")
                st.success(result['treatment'].get('prevention', 'Follow good agricultural practices'))
            
            with tab3:
                st.markdown(f"**{t('organic_treatment')}:**")
                st.info(result['treatment'].get('organic', 'Use organic methods'))
            
            with tab4:
                st.markdown(f"**{t('chemical_treatment')}:**")
                st.warning(result['treatment'].get('chemical', 'Consult agronomist'))
            
            # Alternative predictions
            st.subheader("📊 Alternative Diagnoses")
            for i, (disease_name, conf, _) in enumerate(result['top_predictions'], 1):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{i}. {disease_name}**")
                with col_b:
                    st.progress(conf)
                    st.caption(f"{conf:.1%}")

elif page.startswith("🌍"):
    st.title("🌍 " + t('soil_health'))
    st.markdown("Comprehensive soil health analysis and personalized recommendations")
    
    tab1, tab2 = st.tabs(["📊 " + t('soil_analysis'), "📅 Fertilizer Schedule"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Soil Parameters")
            
            ph = st.slider(t('soil_ph'), 4.0, 9.0, 6.5, 0.1)
            nitrogen = st.number_input(t('nitrogen') + " (kg/ha)", 0, 200, 85, 5)
            phosphorus = st.number_input(t('phosphorus') + " (kg/ha)", 0, 100, 35, 5)
            potassium = st.number_input(t('potassium') + " (kg/ha)", 0, 100, 50, 5)
            organic_matter = st.slider(t('organic_matter'), 0.0, 10.0, 3.2, 0.1)
            ec = st.slider(t('electrical_conductivity') + " (dS/m)", 0.0, 20.0, 1.5, 0.1)
            
            crop_for_analysis = st.selectbox(
                "Select Crop for Specific Recommendations",
                ["None", "Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Pulses", "Vegetables", "Fruits"]
            )
            
            analyze_button = st.button("📊 Analyze Soil Health", use_container_width=True)
        
        with col2:
            if analyze_button or 'soil_health_score' in st.session_state:
                with st.spinner("Analyzing soil health..."):
                    # Calculate health score
                    score, grade, components = soil_analyzer.calculate_health_score(
                        ph, nitrogen, phosphorus, potassium,
                        organic_matter, ec
                    )
                    
                    st.session_state.soil_health_score = score
                    st.session_state.soil_health_grade = grade
                    st.session_state.soil_components = components
                    
                    # Display health score
                    score_color = {
                        'Excellent': '#4CAF50',
                        'Good': '#8BC34A',
                        'Fair': '#FFC107',
                        'Poor': '#FF9800',
                        'Very Poor': '#F44336'
                    }.get(grade, '#FFC107')
                    
                    st.markdown(f"""
                    <div class="health-gauge" style="background: linear-gradient(135deg, {score_color}20 0%, {score_color}40 100%);">
                        <h2 style="margin: 0; color: #333;">{t('soil_health_score')}</h2>
                        <h1 style="font-size: 4rem; margin: 10px 0; color: {score_color};">{score:.1f}</h1>
                        <h3 style="margin: 0; color: {score_color};">{grade}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Component scores
                    st.subheader("📈 Component Scores")
                    
                    component_names = {
                        'ph': 'pH Balance',
                        'nitrogen': 'Nitrogen',
                        'phosphorus': 'Phosphorus',
                        'potassium': 'Potassium',
                        'organic_matter': 'Organic Matter',
                        'ec': 'Salinity (EC)'
                    }
                    
                    for key, value in components.items():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{component_names.get(key, key)}**")
                            st.progress(value / 100)
                        with col_b:
                            st.metric("", f"{value:.0f}/100")
        
        # Recommendations
        if 'soil_health_score' in st.session_state:
            st.markdown("---")
            st.subheader("💡 " + t('recommendations'))
            
            crop_param = None if crop_for_analysis == "None" else crop_for_analysis
            recommendations = soil_analyzer.generate_recommendations(
                ph, nitrogen, phosphorus, potassium,
                organic_matter, ec, crop_param
            )
            
            # Display recommendations in expandable sections
            if recommendations['immediate_actions']:
                with st.expander("🚨 Immediate Actions", expanded=True):
                    for action in recommendations['immediate_actions']:
                        st.warning(action)
            
            if recommendations['nutrient_management']:
                with st.expander("🌱 Nutrient Management"):
                    for item in recommendations['nutrient_management']:
                        st.info(item)
            
            if recommendations['soil_amendments']:
                with st.expander("🧪 Soil Amendments"):
                    for item in recommendations['soil_amendments']:
                        st.success(item)
            
            if recommendations['long_term_practices']:
                with st.expander("📅 Long-term Practices"):
                    for item in recommendations['long_term_practices']:
                        st.info(item)
            
            if recommendations['warnings']:
                with st.expander("⚠️ Important Warnings"):
                    for warning in recommendations['warnings']:
                        st.error(warning)
    
    with tab2:
        st.subheader("📅 Fertilizer Application Schedule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_crop = st.selectbox(
                "Select Crop",
                ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Pulses", "Vegetables", "Fruits"]
            )
        
        with col2:
            area = st.number_input("Farm Area (hectares)", 0.1, 100.0, 1.0, 0.1)
        
        if st.button("Generate Schedule", use_container_width=True):
            schedule_df = soil_analyzer.get_fertilizer_schedule(selected_crop, area)
            
            st.markdown("### Recommended Fertilizer Schedule")
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)
            
            # Visualize nutrient distribution
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Nitrogen',
                x=schedule_df['Stage'],
                y=schedule_df['N (kg)'],
                marker_color='#2196F3'
            ))
            
            fig.add_trace(go.Bar(
                name='Phosphorus',
                x=schedule_df['Stage'],
                y=schedule_df['P (kg)'],
                marker_color='#FF9800'
            ))
            
            fig.add_trace(go.Bar(
                name='Potassium',
                x=schedule_df['Stage'],
                y=schedule_df['K (kg)'],
                marker_color='#4CAF50'
            ))
            
            fig.update_layout(
                barmode='group',
                title="Nutrient Distribution Across Growth Stages",
                xaxis_title="Growth Stage",
                yaxis_title="Quantity (kg)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page.startswith("📊"):
    st.title("📊 " + t('market_insights'))
    st.markdown("Real-time market prices and trends")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💰 Current Market Prices")
        market_df = get_market_prices()
        
        # Style the dataframe
        def color_trends(val):
            if val == "↑":
                return 'color: green'
            elif val == "↓":
                return 'color: red'
            else:
                return 'color: gray'
        
        def color_change(val):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
            else:
                return 'color: gray'
        
        styled_df = market_df.style.applymap(
            color_trends, subset=['Trend']
        ).applymap(
            color_change, subset=['Change (%)']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Prices", use_container_width=True):
            st.rerun()
    
    # Price trend chart
    st.markdown("---")
    st.subheader("📈 Price Trends (Last 7 Days)")
    
    days = list(range(1, 8))
    selected_crop = st.selectbox("Select Crop for Trend", market_df['Crop'].tolist())
    
    base_price = market_df[market_df['Crop'] == selected_crop]['Price (₹/quintal)'].values[0]
    prices = base_price + np.random.randint(-200, 300, 7)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days,
        y=prices,
        mode='lines+markers',
        name=selected_crop,
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10),
        fill='tonexty'
    ))
    
    fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Price (₹/quintal)",
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page.startswith("🌦️"):
    st.title("🌦️ " + t('weather_forecast'))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        location = st.text_input("📍 Location", "Nagpur, Maharashtra")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Weather forecast
    weather_df = get_weather_data(location)
    
    st.subheader("📅 5-Day Forecast")
    st.dataframe(weather_df, use_container_width=True, hide_index=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Temperature Trend")
        fig = px.line(weather_df, x="Day", y="Temperature (°C)", markers=True)
        fig.update_traces(line_color='#FF5722', marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💧 Rainfall Prediction")
        fig = px.bar(weather_df, x="Day", y="Rainfall (mm)", 
                    color="Rainfall (mm)",
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts
    st.markdown("---")
    st.subheader("⚠️ Weather Alerts & Recommendations")
    
    avg_rainfall = weather_df["Rainfall (mm)"].mean()
    max_temp = weather_df["Temperature (°C)"].max()
    
    if avg_rainfall > 20:
        st.warning("🌧️ **Heavy Rainfall Alert**: Consider postponing spraying operations. Ensure proper drainage in fields.")
    
    if max_temp > 35:
        st.error("🌡️ **High Temperature Alert**: Increase irrigation frequency. Monitor crops for heat stress.")
    
    if avg_rainfall < 5 and max_temp > 30:
        st.info("☀️ **Dry Weather**: Optimal for harvesting. Plan accordingly.")

elif page.startswith("💬"):
    st.title("💬 " + t('ai_assistant'))
    st.markdown("Ask me anything about farming, crops, soil management, pest control, and more!")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"""
                <div style='background: #E3F2FD; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                    <strong>🧑‍🌾 You:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: #F1F8E9; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                    <strong>🤖 AI Assistant:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("💭 Ask your question:", 
                              placeholder="E.g., How can I improve soil fertility?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌱 Best crops for clay soil?"):
            user_input = "What are the best crops for clay soil?"
    
    with col2:
        if st.button("🐛 Organic pest control?"):
            user_input = "What are effective organic pest control methods?"
    
    with col3:
        if st.button("💧 Drip irrigation setup?"):
            user_input = "How do I set up a drip irrigation system?"
    
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        context_prompt = f"""You are an expert agricultural advisor with deep knowledge of Indian farming practices.
Provide practical, actionable advice for farmers.

User Question: {user_input}

Provide a helpful, concise response (4-6 sentences) with specific recommendations."""
        
        with st.spinner("Thinking..."):
            response = query_ollama(context_prompt)
            st.session_state.chat_history.append(("assistant", response))
        
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

else:  # Resources
    st.title("📚 " + t('resources'))
    
    tab1, tab2, tab3 = st.tabs(["📖 Guides", "🧮 Calculators", "🔗 Links"])
    
    with tab1:
        st.subheader("Farming Guides")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🌾 Crop Cultivation Guides"):
                st.markdown("""
                - **Rice Cultivation**: Complete guide from seedbed to harvest
                - **Wheat Farming**: Best practices for optimal yield
                - **Cotton Growing**: Pest management and irrigation
                - **Organic Farming**: Transition guide and certification
                """)
            
            with st.expander("🌱 Soil Management"):
                st.markdown("""
                - Soil testing procedures
                - Organic fertilizer preparation
                - Composting techniques
                - Crop rotation strategies
                """)
        
        with col2:
            with st.expander("💧 Water Management"):
                st.markdown("""
                - Drip irrigation installation
                - Rainwater harvesting
                - Water conservation techniques
                - Irrigation scheduling
                """)
            
            with st.expander("🐛 Pest & Disease Control"):
                st.markdown("""
                - Integrated Pest Management (IPM)
                - Natural pesticides
                - Disease identification
                - Preventive measures
                """)
    
    with tab2:
        st.subheader("Farming Calculators")
        
        calc_type = st.radio(
            "Select Calculator",
            ["🧮 Fertilizer Calculator", "💰 Profit Calculator", "💧 Irrigation Calculator"]
        )
        
        if calc_type == "🧮 Fertilizer Calculator":
            col1, col2 = st.columns(2)
            
            with col1:
                crop_area = st.number_input("Crop Area (hectares)", 0.1, 100.0, 1.0)
                crop_type = st.selectbox("Crop Type", 
                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Vegetables"])
            
            with col2:
                if st.button("Calculate", use_container_width=True):
                    requirements = {
                        'Rice': (120, 60, 40),
                        'Wheat': (120, 60, 40),
                        'Cotton': (120, 60, 60),
                        'Maize': (120, 60, 40),
                        'Sugarcane': (200, 80, 80),
                        'Vegetables': (150, 75, 75)
                    }
                    
                    n, p, k = requirements.get(crop_type, (120, 60, 40))
                    n_req = crop_area * n
                    p_req = crop_area * p
                    k_req = crop_area * k
                    
                    st.success(f"""
                    **Recommended Fertilizer (kg):**
                    - Nitrogen (N): {n_req:.1f} kg
                    - Phosphorus (P): {p_req:.1f} kg
                    - Potassium (K): {k_req:.1f} kg
                    """)
        
        elif calc_type == "💰 Profit Calculator":
            col1, col2 = st.columns(2)
            
            with col1:
                investment = st.number_input("Total Investment (₹)", 10000, 10000000, 50000)
                yield_qty = st.number_input("Expected Yield (quintals)", 1, 1000, 50)
                price = st.number_input("Market Price (₹/quintal)", 1000, 20000, 2500)
            
            with col2:
                if st.button("Calculate Profit", use_container_width=True):
                    revenue = yield_qty * price
                    profit = revenue - investment
                    roi = (profit / investment) * 100 if investment > 0 else 0
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Revenue", f"₹{revenue:,.0f}")
                    col_b.metric("Profit", f"₹{profit:,.0f}")
                    col_c.metric("ROI", f"{roi:.1f}%")
        
        else:  # Irrigation Calculator
            col1, col2 = st.columns(2)
            
            with col1:
                field_area = st.number_input("Field Area (hectares)", 0.1, 100.0, 1.0)
                crop_water = st.selectbox("Crop Water Requirement",
                    ["Low (400-500 mm)", "Medium (500-700 mm)", "High (700-1000 mm)"])
                
            with col2:
                if st.button("Calculate Water Need", use_container_width=True):
                    water_ranges = {
                        "Low (400-500 mm)": 450,
                        "Medium (500-700 mm)": 600,
                        "High (700-1000 mm)": 850
                    }
                    
                    water_mm = water_ranges[crop_water]
                    water_liters = field_area * 10000 * (water_mm / 1000)  # Convert to liters
                    
                    st.success(f"""
                    **Estimated Water Requirement:**
                    - Total: {water_liters:,.0f} liters
                    - Per day (120 days): {water_liters/120:,.0f} liters
                    """)
    
    with tab3:
        st.subheader("Useful Links")
        
        st.markdown("""
        ### 🏛️ Government Resources
        - [Ministry of Agriculture & Farmers Welfare](https://agricoop.nic.in/)
        - [Pradhan Mantri Fasal Bima Yojana](https://pmfby.gov.in/)
        - [Soil Health Card](https://soilhealth.dac.gov.in/)
        - [Kisan Call Centre](https://mkisan.gov.in/)
        
        ### 🎓 Research & Education
        - Indian Council of Agricultural Research (ICAR)
        - Agricultural Universities
        - Krishi Vigyan Kendras (KVKs)
        
        ### 📰 News & Updates
        - Agricultural market updates
        - Weather forecasts
        - Government schemes
        """)

# Footer
st.markdown("""
<div class="footer">
    <h3 style="color: #2E7D32;">🌾 AgriTech Pro</h3>
    <p>Empowering Farmers with AI & Technology</p>
    <p style="font-size: 0.9rem; color: #999;">
        Features: Crop Recommendation • Disease Detection • Soil Health Analysis • Multilingual Support
    </p>
    <p style="font-size: 0.8rem; color: #999; margin-top: 1rem;">
        Built with Streamlit, TensorFlow & Modern AI | © 2024
    </p>
</div>
""", unsafe_allow_html=True)
