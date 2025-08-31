import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.mock_data_generator import MockDataGenerator
from models.enhanced_predictor import EnhancedHazardPredictor # Changed import from MultiHazardPredictor
from visualization.maps import create_region_specific_map

def render_flood_page():
    """
    Render the flood prediction page.
    """
    st.header("🌊 Flood Prediction & Monitoring")
    
    # Initialize session state components
    if 'mock_generator' not in st.session_state:
        st.session_state.mock_generator = MockDataGenerator()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EnhancedHazardPredictor() # Changed to EnhancedHazardPredictor
    
    # Sidebar controls
    st.sidebar.subheader("Analysis Parameters")
    
    # Region selection
    selected_region = st.sidebar.selectbox(
        "Select Region",
        ["Ganges Delta", "Amazon Basin", "European Plains", 
         "California Central Valley", "East African Rift", "Custom Location"]
    )
    
    if selected_region == "Custom Location":
        custom_lat = st.sidebar.number_input("Latitude", value=23.6345, min_value=-90.0, max_value=90.0)
        custom_lon = st.sidebar.number_input("Longitude", value=90.2934, min_value=-180.0, max_value=180.0)
        region_coords = (custom_lat, custom_lon)
        region_type = st.sidebar.selectbox("Region Type", 
                                         ["delta", "forest", "temperate", "agricultural", "semi-arid"])
    else:
        # Predefined coordinates
        region_coords_map = {
            "Ganges Delta": (23.6345, 90.2934, "delta"),
            "Amazon Basin": (-3.4653, -62.2159, "forest"),
            "European Plains": (52.5200, 13.4050, "temperate"),
            "California Central Valley": (36.7783, -119.4179, "agricultural"),
            "East African Rift": (-1.2921, 36.8219, "semi-arid")
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
    prediction_horizon = st.sidebar.selectbox("Prediction Horizon", ["3 days", "7 days", "14 days", "30 days"])

    # Initialize prediction variables to default values
    risk_level = "Unknown"
    risk_probability = 0.0
    confidence = 0.0
    factors = {}
    current_river_level = 0.5
    historical_river_level = 0.5
    river_percentile = 50.0
    recent_precip = 0.0
    avg_weekly_precip = 0.0
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Flood Analysis - {selected_region}")
        
        # Generate flood-specific data
        with st.spinner("Analyzing hydrological data and generating predictions..."):
            # Generate weather data with focus on precipitation
            weather_data = st.session_state.mock_generator.generate_weather_data(
                days=analysis_days, 
                lat=region_coords[0], 
                region_type=region_type
            )
            
            # Generate soil moisture data
            moisture_data = st.session_state.mock_generator.generate_soil_moisture_data(
                days=analysis_days, 
                lat=region_coords[0], 
                region_type=region_type
            )
            
            # Calculate river level simulation (mock)
            precipitation = weather_data['precipitation'].values
            # Simulate river levels based on precipitation with lag effect
            river_levels = []
            base_level = 0.5  # Base river level (normalized)
            current_level = base_level
            
            for i, precip in enumerate(precipitation):
                # River level responds to precipitation with some delay and persistence
                inflow = precip * 0.01  # Convert mm to level increase
                outflow = max(0, (current_level - base_level) * 0.1)  # Drainage
                current_level = max(0, min(1, current_level + inflow - outflow))
                river_levels.append(current_level)
            
            weather_data['river_level'] = river_levels
        
        # Precipitation analysis
        fig_precip = px.bar(
            weather_data.tail(30), 
            x='date', 
            y='precipitation',
            title='Daily Precipitation (Last 30 Days)',
            labels={'precipitation': 'Precipitation (mm)', 'date': 'Date'}
        )
        fig_precip.add_hline(y=25, line_dash="dash", line_color="orange", 
                            annotation_text="Heavy Rain Threshold")
        fig_precip.add_hline(y=50, line_dash="dash", line_color="red", 
                            annotation_text="Extreme Rain Threshold")
        fig_precip.update_layout(height=400)
        st.plotly_chart(fig_precip, width='stretch')
        
        # River level plot
        fig_river = px.line(
            weather_data, 
            x='date', 
            y='river_level',
            title='Simulated River Water Level',
            labels={'river_level': 'Water Level (Normalized)', 'date': 'Date'}
        )
        fig_river.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                           annotation_text="Flood Warning Level")
        fig_river.add_hline(y=0.85, line_dash="dash", line_color="red", 
                           annotation_text="Flood Danger Level")
        fig_river.update_layout(height=400)
        st.plotly_chart(fig_river, width='stretch')
        
        # Combined hydrological analysis
        st.subheader("🔍 Multi-factor Flood Analysis")
        
        # Calculate cumulative precipitation
        weather_data['cumulative_precip_7d'] = weather_data['precipitation'].rolling(window=7, min_periods=1).sum()
        weather_data['cumulative_precip_30d'] = weather_data['precipitation'].rolling(window=30, min_periods=1).sum()
        
        # Create correlation plot
        combined_data = pd.merge(weather_data, moisture_data, on='date')
        
        fig_correlation = px.scatter(
            combined_data,
            x='cumulative_precip_7d',
            y='river_level',
            color='soil_moisture',
            size='precipitation',
            title='River Level vs 7-Day Cumulative Precipitation (Color: Soil Moisture, Size: Daily Precipitation)',
            labels={'cumulative_precip_7d': '7-Day Cumulative Precipitation (mm)', 'river_level': 'River Level'}
        )
        st.plotly_chart(fig_correlation, width='stretch')
    
    with col2:
        st.subheader("🎯 Current Flood Assessment")
        
        # Generate flood prediction
        # Also generate mock precipitation forecast here
        forecast_days_int = int(prediction_horizon.split()[0])
        mock_precipitation_forecast = np.random.exponential(5, forecast_days_int).tolist() # Mock forecast

        latest_data = {
            'precipitation': weather_data['precipitation'].tail(analysis_days).values.tolist(),
            'soil_moisture': moisture_data['soil_moisture'].tail(analysis_days).values.tolist(),
            'temperature': weather_data['temperature'].tail(analysis_days).values.tolist(),
            'precipitation_forecast': mock_precipitation_forecast
        }
        
        if st.button("Predict Flood Risk", type="primary"):
            if st.session_state.predictor.flood_model is None or not st.session_state.predictor.is_trained:
                st.warning("Flood model is not trained. Please go to 'Model Training' page to train the models.")
            else:
                with st.spinner("Predicting flood risk..."):
                    # Call the correct prediction method with pre-processed data
                    flood_prediction = st.session_state.predictor.predict_flood_risk({
                        'precipitation': latest_data['precipitation'],
                        'soil_moisture': latest_data['soil_moisture'],
                        'temperature': latest_data['temperature'],
                        'precipitation_forecast': latest_data['precipitation_forecast']
                    })

                    if "error" in flood_prediction:
                        st.error(f"Prediction Error: {flood_prediction['error']}")
                        # Re-initialize variables to default values if there's an error
                        risk_level = "Unknown"
                        risk_probability = 0.0
                        confidence = 0.0
                        factors = {}
                        current_river_level = 0.5
                        historical_river_level = 0.5
                        river_percentile = 50.0
                        recent_precip = 0.0
                        avg_weekly_precip = 0.0
                    else:
                        # Display current risk
                        risk_level = flood_prediction['risk_level']
                        risk_probability = flood_prediction['risk_probability']
                        confidence = flood_prediction['confidence']
                        factors = flood_prediction['factors']
                        
                        # Calculate historical statistics (only if prediction was successful and data is available)
                        current_river_level = np.mean(weather_data['river_level'].tail(7)) if not weather_data.empty and not weather_data['river_level'].empty else 0.5
                        historical_river_level = np.mean(weather_data['river_level']) if not weather_data.empty and not weather_data['river_level'].empty else 0.5
                        river_percentile = (weather_data['river_level'] < current_river_level).mean() * 100 if not weather_data.empty and not weather_data['river_level'].empty else 50.0
                        
                        recent_precip = np.sum(weather_data['precipitation'].tail(7)) if not weather_data.empty and not weather_data['precipitation'].empty else 0.0
                        avg_weekly_precip = np.mean([np.sum(weather_data['precipitation'][i:i+7]) 
                                                   for i in range(0, len(weather_data)-7, 7)]) if not weather_data.empty and len(weather_data) >= 7 else 0.0
                    
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
                            {'Factor': 'Recent Rainfall Intensity', 'Value': f"{factors.get('recent_rainfall_intensity', 0.0):.2f}", 'Impact': 'High' if factors.get('recent_rainfall_intensity', 0.0) > 0.5 else 'Normal'},
                            {'Factor': 'Cumulative Rainfall', 'Value': f"{factors.get('cumulative_rainfall', 0.0):.2f}", 'Impact': 'High' if factors.get('cumulative_rainfall', 0.0) > 0.5 else 'Normal'},
                            {'Factor': 'Soil Saturation', 'Value': f"{factors.get('soil_saturation', 0.0):.2f}", 'Impact': 'High' if factors.get('soil_saturation', 0.0) > 0.8 else 'Normal'},
                            {'Factor': 'Forecast Risk', 'Value': f"{factors.get('forecast_risk', 0.0):.2f}", 'Impact': 'High' if factors.get('forecast_risk', 0.0) > 0.5 else 'Normal'}
                        ])
                        st.dataframe(factor_df, width='stretch')
                    else:
                        st.info("No risk factors to display (prediction might be unavailable or failed).")
                    
                    # Historical comparison (only display if historical data is meaningful)
                    st.subheader("📊 Historical Context")
                    if not weather_data.empty and not weather_data['river_level'].empty and not weather_data['precipitation'].empty:
                        st.metric(
                            "River Level Percentile", 
                            f"{river_percentile:.0f}%",
                            f"{current_river_level - historical_river_level:.3f}"
                        )
                        
                        st.metric(
                            "7-Day Precipitation", 
                            f"{recent_precip:.1f} mm",
                            f"{recent_precip - avg_weekly_precip:.1f} mm"
                        )
                    else:
                        st.info("No historical context to display (insufficient data).")
    
    # Full width sections
    st.markdown("---")
    
    # Flood indicators and forecasting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌊 Flood Indicators")
        
        # Calculate various flood indices
        if not weather_data.empty and not moisture_data.empty:
            current_river = np.mean(weather_data['river_level'].tail(7)) if not weather_data['river_level'].empty else 0.0
            recent_precip_7d = np.sum(weather_data['precipitation'].tail(7)) if not weather_data['precipitation'].empty else 0.0
            recent_precip_3d = np.sum(weather_data['precipitation'].tail(3)) if not weather_data['precipitation'].empty else 0.0
            current_soil_moisture = np.mean(moisture_data['soil_moisture'].tail(7)) if not moisture_data['soil_moisture'].empty else 0.0
            
            # Flood Potential Index
            fpi = (current_river * 0.4) + (min(recent_precip_7d/100, 1) * 0.3) + (current_soil_moisture * 0.3)
            
            # Stream Flow Index
            historical_river_max = np.percentile(weather_data['river_level'], 95) if not weather_data['river_level'].empty else 0.0
            sfi = current_river / historical_river_max if historical_river_max > 0 else 0.0
            
            # Antecedent Precipitation Index
            api = recent_precip_7d + (recent_precip_3d * 0.5)
            
            indicators_df = pd.DataFrame([
                {'Indicator': 'Flood Potential Index (FPI)', 'Value': f"{fpi:.2f}", 'Status': 'High' if fpi > 0.7 else 'Moderate' if fpi > 0.5 else 'Low'},
                {'Indicator': 'Stream Flow Index (SFI)', 'Value': f"{sfi:.2f}", 'Status': 'Critical' if sfi > 0.9 else 'High' if sfi > 0.7 else 'Normal'},
                {'Indicator': 'Antecedent Precipitation (API)', 'Value': f"{api:.1f} mm", 'Status': 'High' if api > 75 else 'Moderate' if api > 35 else 'Low'},
                {'Indicator': 'Soil Saturation Level', 'Value': f"{current_soil_moisture:.2f}", 'Status': 'Saturated' if current_soil_moisture > 0.8 else 'Normal'}
            ])
            
            st.dataframe(indicators_df, width='stretch')
        else:
            st.info("No sufficient data to calculate flood indicators.")
    
    with col2:
        st.subheader("🔮 Forecast & Trends")
        
        # Generate forecast data (mock)
        if not weather_data.empty and not moisture_data.empty:
            forecast_days = int(prediction_horizon.split()[0])
            forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days + 1)]
            
            # Simple trend extrapolation for river levels
            recent_trend = np.polyfit(range(7), weather_data['river_level'].tail(7), 1)[0] if len(weather_data) >= 7 and not weather_data['river_level'].empty else 0.0
            
            # Simulate future precipitation (mock weather forecast)
            forecast_precip = np.random.exponential(5, forecast_days)  # Mock precipitation forecast
            
            # Simulate future river levels based on forecast precipitation
            forecast_river_levels = []
            current_level = weather_data['river_level'].iloc[-1] if not weather_data['river_level'].empty else 0.5
            
            for precip in forecast_precip:
                inflow = precip * 0.01
                outflow = max(0, (current_level - 0.5) * 0.1)
                current_level = max(0, min(1, current_level + inflow - outflow))
                forecast_river_levels.append(current_level)
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_precipitation': forecast_precip,
                'predicted_river_level': forecast_river_levels
            })
            
            # Plot forecast
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=weather_data['date'].tail(14), 
                y=weather_data['river_level'].tail(14),
                mode='lines',
                name='Historical River Level',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'], 
                y=forecast_df['predicted_river_level'],
                mode='lines',
                name='Predicted River Level',
                line=dict(color='blue', dash='dash')
            ))
            
            # Add flood thresholds
            fig_forecast.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                                  annotation_text="Warning Level")
            fig_forecast.add_hline(y=0.85, line_dash="dash", line_color="red", 
                                  annotation_text="Danger Level")
            
            fig_forecast.update_layout(
                title=f'River Level Forecast - {prediction_horizon}',
                xaxis_title='Date',
                yaxis_title='River Level',
                height=350
            )
            
            st.plotly_chart(fig_forecast, width='stretch')
            
            # Forecast summary
            max_forecast_level = np.max(forecast_df['predicted_river_level'])
            total_forecast_precip = np.sum(forecast_df['predicted_precipitation'])
            
            if max_forecast_level > 0.85 or total_forecast_precip > 100:
                forecast_risk = "High"
                forecast_color = "red"
            elif max_forecast_level > 0.7 or total_forecast_precip > 50:
                forecast_risk = "Medium"
                forecast_color = "orange"
            else:
                forecast_risk = "Low"
                forecast_color = "green"
            
            st.markdown(f"**{prediction_horizon} Forecast:** <span style='color: {forecast_color}'>{forecast_risk} Risk</span>", 
                       unsafe_allow_html=True)
            
            st.metric("Forecast Max River Level", f"{max_forecast_level:.2f}")
            st.metric("Total Forecast Precipitation", f"{total_forecast_precip:.1f} mm")
        else:
            st.info("No sufficient data to generate flood forecast.")
    
    # Regional map
    st.markdown("---")
    st.subheader("🗺️ Regional Flood Monitoring")
    
    # Create region-specific map
    region_data = {
        'monitoring_stations': [
            {'name': 'River Gauge A', 'lat': region_coords[0] + 0.1, 'lon': region_coords[1] + 0.1, 'type': 'Hydrological'},
            {'name': 'Weather Station B', 'lat': region_coords[0] - 0.1, 'lon': region_coords[1] - 0.1, 'type': 'Meteorological'},
            {'name': 'Flood Sensor C', 'lat': region_coords[0] + 0.05, 'lon': region_coords[1] - 0.05, 'type': 'Flood Detection'}
        ],
        'risk_grid': [
            {'lat': region_coords[0] + i*0.05, 'lon': region_coords[1] + j*0.05, 
             'risk_score': min(1.0, max(0.1, np.random.uniform(0.2, 0.9) * (1.2 if region_type == 'delta' else 0.8)))}
            for i in range(-2, 3) for j in range(-2, 3)
        ],
        'historical_events': [
            {'lat': region_coords[0] + 0.03, 'lon': region_coords[1] + 0.02, 'type': 'flood', 
             'date': '2020-07-15', 'magnitude': 'Major'},
            {'lat': region_coords[0] - 0.02, 'lon': region_coords[1] + 0.04, 'type': 'flood', 
             'date': '2018-08-22', 'magnitude': 'Moderate'}
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
    
    if risk_level == "High":
        recommendations = [
            "🚨 Issue flood warnings to affected areas",
            "🚤 Prepare emergency evacuation plans",
            "🛡️ Deploy flood barriers and sandbags",
            "📢 Activate emergency response teams",
            "🏥 Alert hospitals and emergency services",
            "🚧 Close flood-prone roads and bridges"
        ]
    elif risk_level == "Medium":
        recommendations = [
            "⚠️ Issue flood watch advisory",
            "📊 Increase monitoring of river levels",
            "🎒 Advise residents to prepare emergency kits",
            "🚗 Monitor transportation routes",
            "📱 Test emergency communication systems",
            "🏗️ Inspect flood defenses and drainage systems"
        ]
    else:
        recommendations = [
            "✅ Continue routine monitoring",
            "📊 Maintain regular observation schedule",
            "🌧️ Monitor weather forecasts closely",
            "🏗️ Perform maintenance on flood infrastructure",
            "📚 Update flood risk assessments",
            "👥 Conduct community preparedness activities"
        ]
    
    for rec in recommendations:
        st.write(rec)
