import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.mock_data_generator import MockDataGenerator
from models.enhanced_predictor import EnhancedHazardPredictor
from visualization.maps import create_region_specific_map

def render_drought_page():
    """
    Render the drought prediction page.
    """
    st.header("🌵 Drought Prediction & Monitoring")
    
    # Initialize session state components
    if 'mock_generator' not in st.session_state:
        st.session_state.mock_generator = MockDataGenerator()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EnhancedHazardPredictor()
    
    # Sidebar controls
    st.sidebar.subheader("Analysis Parameters")
    
    # Region selection
    selected_region = st.sidebar.selectbox(
        "Select Region",
        ["California Central Valley", "Sahel Region", "Australian Outback", 
         "Amazon Basin", "European Plains", "Custom Location"]
    )
    
    if selected_region == "Custom Location":
        custom_lat = st.sidebar.number_input("Latitude", value=40.0, min_value=-90.0, max_value=90.0)
        custom_lon = st.sidebar.number_input("Longitude", value=-120.0, min_value=-180.0, max_value=180.0)
        region_coords = (custom_lat, custom_lon)
        region_type = st.sidebar.selectbox("Region Type", 
                                         ["agricultural", "forest", "semi-arid", "arid", "temperate"])
    else:
        # Predefined coordinates
        region_coords_map = {
            "California Central Valley": (36.7783, -119.4179, "agricultural"),
            "Sahel Region": (15.0000, 0.0000, "semi-arid"),
            "Australian Outback": (-25.2744, 133.7751, "arid"),
            "Amazon Basin": (-3.4653, -62.2159, "forest"),
            "European Plains": (52.5200, 13.4050, "temperate")
        }
        region_coords = region_coords_map[selected_region][:2]
        region_type = region_coords_map[selected_region][2]
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Analysis Period",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last year"]
    )
    
    days_map = {
        "Last 30 days": 30,
        "Last 90 days": 90,
        "Last 6 months": 180,
        "Last year": 365
    }
    analysis_days = days_map[time_period]
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.8)
    prediction_horizon = st.sidebar.selectbox("Prediction Horizon", ["7 days", "14 days", "30 days", "90 days"])

    # Initialize prediction variables to default values
    risk_level = "Unknown"
    risk_probability = 0.0
    confidence = 0.0
    factors = {}
    current_ndvi = 0.5
    historical_ndvi = 0.5
    ndvi_percentile = 50.0
    current_moisture = 0.5
    historical_moisture = 0.5
    moisture_percentile = 50.0
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Drought Analysis - {selected_region}")
        
        # Generate drought-specific data
        with st.spinner("Analyzing satellite data and generating predictions..."):
            # Generate mock NDVI and soil moisture data
            ndvi_data = st.session_state.mock_generator.generate_ndvi_data(
                days=analysis_days, 
                lat=region_coords[0], 
                region_type=region_type
            )
            
            moisture_data = st.session_state.mock_generator.generate_soil_moisture_data(
                days=analysis_days, 
                lat=region_coords[0], 
                region_type=region_type
            )
            
            weather_data = st.session_state.mock_generator.generate_weather_data(
                days=analysis_days, 
                lat=region_coords[0], 
                region_type=region_type
            )
        
        # Create time series plots
        fig_ndvi = px.line(
            ndvi_data, 
            x='date', 
            y='ndvi',
            title='Normalized Difference Vegetation Index (NDVI) Trend',
            labels={'ndvi': 'NDVI Value', 'date': 'Date'}
        )
        fig_ndvi.add_hline(y=0.3, line_dash="dash", line_color="red", 
                          annotation_text="Drought Threshold")
        fig_ndvi.update_layout(height=400)
        st.plotly_chart(fig_ndvi, width='stretch')
        
        # Soil moisture plot
        fig_moisture = px.line(
            moisture_data, 
            x='date', 
            y='soil_moisture',
            title='Soil Moisture Content',
            labels={'soil_moisture': 'Soil Moisture (%)', 'date': 'Date'}
        )
        fig_moisture.add_hline(y=0.2, line_dash="dash", line_color="orange", 
                              annotation_text="Drought Risk Threshold")
        fig_moisture.update_layout(height=400)
        st.plotly_chart(fig_moisture, width='stretch')
        
        # Combined analysis
        st.subheader("🔍 Multi-factor Drought Analysis")
        
        # Create correlation plot
        combined_data = pd.merge(ndvi_data, moisture_data, on='date')
        combined_data = pd.merge(combined_data, weather_data[['date', 'precipitation', 'temperature']], on='date')
        
        fig_correlation = px.scatter(
            combined_data,
            x='soil_moisture',
            y='ndvi',
            color='precipitation',
            size='temperature',
            title='NDVI vs Soil Moisture (Color: Precipitation, Size: Temperature)',
            labels={'soil_moisture': 'Soil Moisture', 'ndvi': 'NDVI'}
        )
        st.plotly_chart(fig_correlation, width='stretch')
    
    with col2:
        st.subheader("🎯 Current Drought Assessment")
        
        # Generate drought prediction
        latest_data = {
            'ndvi': ndvi_data['ndvi'].tail(7).values,
            'soil_moisture': moisture_data['soil_moisture'].tail(7).values,
            'temperature': weather_data['temperature'].tail(7).values,
            'precipitation': weather_data['precipitation'].tail(7).values
        }
        
        st.subheader("Latest Data Snapshot for Drought Prediction")
        st.json({
            "Avg NDVI (last 30 days)": np.mean(latest_data['ndvi']),
            "Avg Soil Moisture (last 30 days)": np.mean(latest_data['soil_moisture']),
            "Avg Temperature (last 30 days)": np.mean(latest_data['temperature']),
            "Total Precipitation (last 30 days)": np.sum(latest_data['precipitation'])
        })

        if st.button("Predict Drought Risk", type="primary"):
            if st.session_state.predictor.drought_model is None or not st.session_state.predictor.is_trained:
                st.warning("Drought model is not trained. Please go to 'Model Training' page to train the models.")
            else:
                with st.spinner("Predicting drought risk..."):
                    # Call the correct prediction method with pre-processed data
                    drought_prediction = st.session_state.predictor.predict_drought_risk({
                        'ndvi': latest_data['ndvi'],
                        'soil_moisture': latest_data['soil_moisture'],
                        'temperature': latest_data['temperature'],
                        'precipitation': latest_data['precipitation']
                    })

                    if "error" in drought_prediction:
                        st.error(f"Prediction Error: {drought_prediction['error']}")
                        # Re-initialize variables to default values if there's an error
                        risk_level = "Unknown"
                        risk_probability = 0.0
                        confidence = 0.0
                        factors = {}
                        current_ndvi = np.mean(latest_data['ndvi']) if len(latest_data['ndvi']) > 0 else 0.5
                        historical_ndvi = current_ndvi
                        ndvi_percentile = 50.0
                        current_moisture = np.mean(latest_data['soil_moisture']) if len(latest_data['soil_moisture']) > 0 else 0.5
                        historical_moisture = current_moisture
                        moisture_percentile = 50.0
                    else:
                        # Display current risk
                        risk_level = drought_prediction['risk_level']
                        risk_probability = drought_prediction['risk_probability']
                        confidence = drought_prediction['confidence']
                        factors = drought_prediction['factors']

                        # Calculate historical statistics (only if prediction was successful and data is available)
                        current_ndvi = np.mean(ndvi_data['ndvi'].tail(7)) if not ndvi_data['ndvi'].empty else 0.5
                        historical_ndvi = np.mean(ndvi_data['ndvi']) if not ndvi_data['ndvi'].empty else 0.5
                        ndvi_percentile = (ndvi_data['ndvi'] < current_ndvi).mean() * 100 if not ndvi_data['ndvi'].empty else 50.0
                        
                        current_moisture = np.mean(moisture_data['soil_moisture'].tail(7)) if not moisture_data['soil_moisture'].empty else 0.5
                        historical_moisture = np.mean(moisture_data['soil_moisture']) if not moisture_data['soil_moisture'].empty else 0.5
                        moisture_percentile = (moisture_data['soil_moisture'] < current_moisture).mean() * 100 if not moisture_data['soil_moisture'].empty else 50.0
                    
                    # Display current risk (always attempt to display even if default values)
                    risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Unknown': 'gray'}
                    st.markdown(f"**Current Risk Level:** <span style='color: {risk_colors.get(risk_level, 'gray')}'>{risk_level}</span>", 
                               unsafe_allow_html=True)
                    
                    st.metric("Risk Probability", f"{risk_probability:.1%}")
                    st.metric("Model Confidence", f"{confidence:.1%}")
                    
                    # Risk factors (only display if factors are available and not empty)
                    if factors and any(factors.values()):
                        st.subheader("📈 Key Risk Factors")
                        factor_df = pd.DataFrame([
                            {'Factor': 'NDVI Trend', 'Value': f"{factors.get('ndvi_trend', 0.0):.3f}", 'Impact': 'Negative' if factors.get('ndvi_trend', 0.0) < 0 else 'Positive'},
                            {'Factor': 'Soil Moisture', 'Value': f"{factors.get('soil_moisture', 0.0):.2f}", 'Impact': 'Low' if factors.get('soil_moisture', 0.0) < 0.3 else 'Normal'},
                            {'Factor': 'Temperature Anomaly', 'Value': f"{factors.get('temperature_anomaly', 0.0):.1f}°C", 'Impact': 'High' if factors.get('temperature_anomaly', 0.0) > 2 else 'Normal'}
                        ])
                        st.dataframe(factor_df, width='stretch')
                    else:
                        st.info("No risk factors to display (prediction might be unavailable or failed).")
                    
                    # Historical comparison (only display if historical data is meaningful)
                    st.subheader("📊 Historical Context")
                    if not ndvi_data['ndvi'].empty and not moisture_data['soil_moisture'].empty:
                        st.metric(
                            "NDVI Percentile", 
                            f"{ndvi_percentile:.0f}%",
                            f"{current_ndvi - historical_ndvi:.3f}"
                        )
                        
                        st.metric(
                            "Moisture Percentile", 
                            f"{moisture_percentile:.0f}%",
                            f"{current_moisture - historical_moisture:.3f}"
                        )
                    else:
                        st.info("No historical context to display (insufficient data).")
    
    # Full width sections
    st.markdown("---")
    
    # Drought indicators and forecasting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Drought Indicators")
        
        # Calculate various drought indices
        # Standardized Precipitation Index (simplified)
        precip_values = weather_data['precipitation'].values
        precip_mean = float(np.mean(precip_values))
        precip_std = float(np.std(precip_values))
        recent_precip = float(np.mean(precip_values[-30:]))  # Last 30 days
        spi = (recent_precip - precip_mean) / precip_std if precip_std > 0 else 0
        
        # Vegetation Condition Index
        ndvi_min = np.percentile(ndvi_data['ndvi'], 5)
        ndvi_max = np.percentile(ndvi_data['ndvi'], 95)
        current_ndvi = np.mean(ndvi_data['ndvi'].tail(7))
        vci = (current_ndvi - ndvi_min) / (ndvi_max - ndvi_min) * 100 if ndvi_max > ndvi_min else 50
        
        # Soil Water Deficit
        max_moisture = np.percentile(moisture_data['soil_moisture'], 95) if not moisture_data['soil_moisture'].empty else 0.0
        current_moisture_avg = np.mean(moisture_data['soil_moisture'].tail(7)) if not moisture_data['soil_moisture'].empty else 0.0
        water_deficit = float(max(0, max_moisture - current_moisture_avg))
        
        indicators_df = pd.DataFrame([
            {'Indicator': 'Standardized Precipitation Index (SPI)', 'Value': f"{spi:.2f}", 'Status': 'Drought' if spi < -1 else 'Normal'},
            {'Indicator': 'Vegetation Condition Index (VCI)', 'Value': f"{vci:.1f}%", 'Status': 'Poor' if vci < 40 else 'Good'},
            {'Indicator': 'Soil Water Deficit', 'Value': f"{water_deficit:.3f}", 'Status': 'High' if water_deficit > 0.3 else 'Low'},
            {'Indicator': 'Temperature Anomaly', 'Value': f"{factors.get('temperature_anomaly', 0.0):.1f}°C", 'Status': 'High' if factors.get('temperature_anomaly', 0.0) > 2 else 'Normal'}
        ])
        
        st.dataframe(indicators_df, width='stretch')
    
    with col2:
        st.subheader("🔮 Forecast & Trends")
        
        # Generate forecast data (mock)
        forecast_days = int(prediction_horizon.split()[0])
        forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Simple trend extrapolation for demonstration
        ndvi_trend = np.polyfit(range(len(ndvi_data)), ndvi_data['ndvi'], 1)[0]
        moisture_trend = np.polyfit(range(len(moisture_data)), moisture_data['soil_moisture'], 1)[0]
        
        forecast_ndvi = current_ndvi + ndvi_trend * np.arange(1, forecast_days + 1)
        forecast_moisture = current_moisture_avg + moisture_trend * np.arange(1, forecast_days + 1)
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_ndvi': np.clip(forecast_ndvi, 0, 1),
            'predicted_moisture': np.clip(forecast_moisture, 0, 1)
        })
        
        # Plot forecast
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=ndvi_data['date'].tail(30), 
            y=ndvi_data['ndvi'].tail(30),
            mode='lines',
            name='Historical NDVI',
            line=dict(color='green')
        ))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['predicted_ndvi'],
            mode='lines',
            name='Predicted NDVI',
            line=dict(color='green', dash='dash')
        ))
        
        fig_forecast.update_layout(
            title=f'NDVI Forecast - {prediction_horizon}',
            xaxis_title='Date',
            yaxis_title='NDVI',
            height=350
        )
        
        st.plotly_chart(fig_forecast, width='stretch')
        
        # Forecast summary
        avg_forecast_ndvi = np.mean(forecast_df['predicted_ndvi'])
        avg_forecast_moisture = np.mean(forecast_df['predicted_moisture'])
        
        if avg_forecast_ndvi < 0.3 or avg_forecast_moisture < 0.2:
            forecast_risk = "High"
            forecast_color = "red"
        elif avg_forecast_ndvi < 0.5 or avg_forecast_moisture < 0.3:
            forecast_risk = "Medium"
            forecast_color = "orange"
        else:
            forecast_risk = "Low"
            forecast_color = "green"
        
        st.markdown(f"**{prediction_horizon} Forecast:** <span style='color: {forecast_color}'>{forecast_risk} Risk</span>", 
                   unsafe_allow_html=True)
    
    # Regional map
    st.markdown("---")
    st.subheader("🗺️ Regional Drought Monitoring")
    
    # Create region-specific map
    region_data = {
        'monitoring_stations': [
            {'name': 'Station A', 'lat': region_coords[0] + 0.1, 'lon': region_coords[1] + 0.1, 'type': 'Weather'},
            {'name': 'Station B', 'lat': region_coords[0] - 0.1, 'lon': region_coords[1] - 0.1, 'type': 'Soil'},
            {'name': 'Station C', 'lat': region_coords[0] + 0.05, 'lon': region_coords[1] - 0.05, 'type': 'Satellite'}
        ],
        'risk_grid': [
            {'lat': region_coords[0] + i*0.05, 'lon': region_coords[1] + j*0.05, 
             'risk_score': np.random.uniform(0.1, 0.8)}
            for i in range(-2, 3) for j in range(-2, 3)
        ]
    }
    
    region_map = create_region_specific_map(
        region_data, 
        region_coords[0], 
        region_coords[1], 
        zoom_start=8
    )
    
    import streamlit.components.v1 as components
    components.html(region_map._repr_html_(), height=400)
    
    # Action recommendations
    st.markdown("---")
    st.subheader("💡 Recommended Actions")
    
    # Ensure risk_level is defined before using it for recommendations
    if 'risk_level' not in locals():
        risk_level = "Unknown" # Default if not set by prediction

    if risk_level == "High":
        recommendations = [
            "🚨 Implement immediate water conservation measures",
            "📊 Increase monitoring frequency to daily",
            "🌾 Issue drought advisory to agricultural sector",
            "💧 Assess water reservoir levels",
            "📢 Prepare public communication about water restrictions"
        ]
    elif risk_level == "Medium":
        recommendations = [
            "⚠️ Monitor situation closely",
            "💧 Review water usage patterns",
            "🌱 Advise farmers on drought-resistant practices",
            "📈 Increase data collection frequency",
            "📋 Review drought contingency plans"
        ]
    else:
        recommendations = [
            "✅ Continue routine monitoring",
            "📊 Maintain current observation schedule",
            "🌿 Monitor vegetation health indicators",
            "💧 Ensure water infrastructure maintenance",
            "📚 Update historical databases"
        ]
    
    for rec in recommendations:
        st.write(rec)
