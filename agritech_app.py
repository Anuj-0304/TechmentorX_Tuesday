import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from tensorflow import keras
import os

# Page configuration
st.set_page_config(
    page_title="AgriTech Pro - Smart Farming Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    h1 {
        color: #2E7D32;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "Error: Unable to connect to Ollama. Please ensure Ollama is running."
    except Exception as e:
        return f"Error connecting to AI model: {str(e)}\nPlease start Ollama with 'ollama serve' command."

# Load Keras model function
@st.cache_resource
def load_keras_model(model_path):
    """Load a Keras model from file"""
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Simulated crop prediction (replace with your actual model)
def predict_crop(soil_type, ph, rainfall, temperature, humidity, nitrogen, phosphorus, potassium):
    """Predict suitable crop based on soil and environmental conditions"""
    # This is a placeholder - replace with your actual Keras model prediction
    crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Pulses", "Vegetables"]
    
    # Simple rule-based prediction (replace with model.predict())
    if ph < 6.5:
        if rainfall > 150:
            return "Rice", 0.89
        else:
            return "Wheat", 0.82
    elif ph > 7.5:
        if temperature > 25:
            return "Cotton", 0.85
        else:
            return "Sugarcane", 0.78
    else:
        if humidity > 70:
            return "Maize", 0.91
        else:
            return "Pulses", 0.76

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

# Sidebar navigation
st.sidebar.title("🌾 AgriTech Pro")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🌱 Crop Recommendation", "📊 Market Insights", 
     "🌦️ Weather Forecast", "💬 AI Assistant", "📚 Resources"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Smart Farming for Better Tomorrow**\n\nData-driven insights for optimal crop management and sustainable agriculture.")

# Main content based on page selection
if page == "🏠 Dashboard":
    st.title("🌾 AgriTech Pro - Smart Farming Dashboard")
    st.markdown("### Welcome to your comprehensive agricultural management platform")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Farms", "1,234", "+12%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Crop Varieties", "45", "+3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Yield Increase", "18%", "+2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Market Updates", "Live", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Crop Production Trends")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        production = np.random.randint(100, 500, 6)
        fig = px.line(x=months, y=production, markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Production (tons)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌾 Crop Distribution")
        crops = ["Rice", "Wheat", "Cotton", "Maize", "Others"]
        values = [30, 25, 20, 15, 10]
        fig = px.pie(names=crops, values=values, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("🌱 Get Crop Recommendation", use_container_width=True)
    with col2:
        st.button("📊 View Market Prices", use_container_width=True)
    with col3:
        st.button("💬 Ask AI Assistant", use_container_width=True)

elif page == "🌱 Crop Recommendation":
    st.title("🌱 Smart Crop Recommendation System")
    st.markdown("Get AI-powered crop suggestions based on your soil and environmental conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location & Soil Data")
        location = st.text_input("Location", "Maharashtra, India")
        soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay", "Black", "Red", "Alluvial"])
        ph = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
        
        st.subheader("💧 Nutrients (kg/ha)")
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            nitrogen = st.number_input("Nitrogen (N)", 0, 200, 50)
        with col_p:
            phosphorus = st.number_input("Phosphorus (P)", 0, 100, 30)
        with col_k:
            potassium = st.number_input("Potassium (K)", 0, 100, 40)
    
    with col2:
        st.subheader("🌦️ Environmental Conditions")
        temperature = st.slider("Temperature (°C)", 10, 45, 25)
        humidity = st.slider("Humidity (%)", 30, 100, 70)
        rainfall = st.slider("Rainfall (mm)", 0, 300, 100)
        
        st.subheader("📅 Season")
        season = st.selectbox("Growing Season", ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"])
    
    if st.button("🔍 Get Recommendation", use_container_width=True):
        with st.spinner("Analyzing conditions..."):
            crop, confidence = predict_crop(soil_type, ph, rainfall, temperature, humidity, 
                                          nitrogen, phosphorus, potassium)
            
            st.success("✅ Analysis Complete!")
            
            st.markdown("---")
            st.subheader("🎯 Recommended Crop")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"## **{crop}**")
                st.markdown(f"**Confidence Score:** {confidence:.1%}")
                st.progress(confidence)
            
            with col2:
                st.metric("Expected Yield", "4.5 tons/ha", "↑ 12%")
                st.metric("Market Price", "₹2,850/quintal", "↑ 5%")
            
            # AI Insights
            st.markdown("---")
            st.subheader("🤖 AI-Generated Insights")
            
            prompt = f"""You are an agricultural expert. Based on these conditions:
- Location: {location}
- Soil: {soil_type}, pH {ph}
- Nutrients: N:{nitrogen}, P:{phosphorus}, K:{potassium}
- Weather: Temp {temperature}°C, Humidity {humidity}%, Rainfall {rainfall}mm
- Recommended crop: {crop}

Provide brief, practical farming advice (3-4 sentences) covering:
1. Why this crop is suitable
2. Key cultivation tips
3. Expected challenges"""
            
            with st.spinner("Generating AI insights..."):
                advice = query_ollama(prompt)
                st.info(advice)
            
            # Additional recommendations
            st.markdown("---")
            st.subheader("📋 Cultivation Guidelines")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🌱 Seed Rate**")
                st.write("20-25 kg/ha")
            with col2:
                st.markdown("**📅 Growing Period**")
                st.write("90-120 days")
            with col3:
                st.markdown("**💰 Investment**")
                st.write("₹25,000-30,000/ha")

elif page == "📊 Market Insights":
    st.title("📊 Agricultural Market Insights")
    st.markdown("Real-time market prices and trends")
    
    # Market prices table
    st.subheader("💰 Current Market Prices")
    market_data = get_market_prices()
    
    # Color code based on trend
    def color_trend(val):
        if val == "↑":
            return 'background-color: #d4edda'
        elif val == "↓":
            return 'background-color: #f8d7da'
        return 'background-color: #fff3cd'
    
    styled_df = market_data.style.applymap(color_trend, subset=['Trend'])
    st.dataframe(styled_df, use_container_width=True, height=350)
    
    st.markdown("---")
    
    # Market analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Price Trends (Last 30 Days)")
        days = pd.date_range(end=datetime.now(), periods=30)
        rice_prices = np.random.randint(2800, 3200, 30)
        wheat_prices = np.random.randint(2000, 2400, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=rice_prices, name="Rice", mode='lines+markers'))
        fig.add_trace(go.Scatter(x=days, y=wheat_prices, name="Wheat", mode='lines+markers'))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹/quintal)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Demand Forecast")
        crops = ["Rice", "Wheat", "Cotton", "Maize", "Pulses"]
        demand = [85, 72, 68, 78, 65]
        
        fig = px.bar(x=crops, y=demand, color=demand, 
                     color_continuous_scale='Greens')
        fig.update_layout(xaxis_title="Crop", yaxis_title="Demand Index",
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Market Analysis
    st.markdown("---")
    st.subheader("🤖 AI Market Analysis")
    
    if st.button("Generate Market Report"):
        prompt = """As an agricultural market analyst, provide a brief market outlook (4-5 sentences) covering:
1. Current market trends in Indian agriculture
2. Price predictions for major crops
3. Recommendations for farmers
Focus on practical, actionable insights."""
        
        with st.spinner("Analyzing market data..."):
            report = query_ollama(prompt)
            st.info(report)

elif page == "🌦️ Weather Forecast":
    st.title("🌦️ Weather Forecast & Alerts")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        location = st.text_input("Enter Location", "Nagpur, Maharashtra")
    
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💧 Rainfall Prediction")
        fig = px.bar(weather_df, x="Day", y="Rainfall (mm)", color="Rainfall (mm)",
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
    
    # AI Weather Insights
    st.markdown("---")
    if st.button("Get AI Weather Insights"):
        prompt = f"""Based on this 5-day weather forecast:
{weather_df.to_string()}

Provide brief farming recommendations (3-4 sentences) for the upcoming week."""
        
        with st.spinner("Generating insights..."):
            insights = query_ollama(prompt)
            st.success(insights)

elif page == "💬 AI Assistant":
    st.title("💬 AgriTech AI Assistant")
    st.markdown("Ask me anything about farming, crops, soil management, pest control, and more!")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"**🧑‍🌾 You:** {message}")
            else:
                st.markdown(f"**🤖 AI Assistant:** {message}")
            st.markdown("---")
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Ask your question:", key="user_input", 
                                   placeholder="E.g., How can I improve soil fertility?")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("Send", use_container_width=True)
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌱 Best crops for clay soil?"):
            user_input = "What are the best crops for clay soil?"
            send_button = True
    
    with col2:
        if st.button("🐛 Organic pest control methods?"):
            user_input = "What are effective organic pest control methods?"
            send_button = True
    
    with col3:
        if st.button("💧 Drip irrigation setup?"):
            user_input = "How do I set up a drip irrigation system?"
            send_button = True
    
    # Process input
    if send_button and user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        # Create agricultural context
        context_prompt = f"""You are an expert agricultural advisor with deep knowledge of Indian farming practices.
Provide practical, actionable advice for farmers.

User Question: {user_input}

Provide a helpful, concise response (4-6 sentences) with specific recommendations."""
        
        with st.spinner("Thinking..."):
            response = query_ollama(context_prompt)
            st.session_state.chat_history.append(("assistant", response))
        
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

elif page == "📚 Resources":
    st.title("📚 Agricultural Resources & Learning")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Guides", "🎥 Videos", "📱 Tools", "🔗 Links"])
    
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
        st.subheader("Educational Videos")
        st.info("📹 Video tutorials coming soon! Connect with agricultural extension services for video resources.")
    
    with tab3:
        st.subheader("Farming Tools & Calculators")
        
        st.markdown("#### 🧮 Fertilizer Calculator")
        crop_area = st.number_input("Crop Area (hectares)", 1.0, 100.0, 1.0)
        crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Cotton", "Maize"])
        
        if st.button("Calculate Fertilizer Need"):
            n_req = crop_area * 120
            p_req = crop_area * 60
            k_req = crop_area * 40
            
            st.success(f"""
            **Recommended Fertilizer (kg):**
            - Nitrogen (N): {n_req} kg
            - Phosphorus (P): {p_req} kg
            - Potassium (K): {k_req} kg
            """)
        
        st.markdown("---")
        st.markdown("#### 💰 Profit Calculator")
        investment = st.number_input("Total Investment (₹)", 10000, 1000000, 50000)
        yield_qty = st.number_input("Expected Yield (quintals)", 10, 500, 50)
        price = st.number_input("Market Price (₹/quintal)", 1000, 10000, 2500)
        
        if st.button("Calculate Profit"):
            revenue = yield_qty * price
            profit = revenue - investment
            roi = (profit / investment) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Revenue", f"₹{revenue:,.0f}")
            col2.metric("Profit", f"₹{profit:,.0f}")
            col3.metric("ROI", f"{roi:.1f}%")
    
    with tab4:
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
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 AgriTech Pro - Empowering Farmers with Technology</p>
    <p style='font-size: 0.8rem;'>Hackathon Project 2024 | Built with Streamlit & AI</p>
</div>
""", unsafe_allow_html=True)
