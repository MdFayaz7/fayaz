import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from models.enhanced_predictor import EnhancedHazardPredictor
from data.data_processor import SatelliteDataProcessor
from data.mock_data_generator import MockDataGenerator
from visualization.maps import create_hazard_map
from utils.enhanced_alerts import EnhancedAlertSystem
from streamlit_js_eval import streamlit_js_eval, get_geolocation
from data.live_data_fetcher import LiveDataFetcher
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure page
st.set_page_config(
    page_title="Multi-Hazard Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = SatelliteDataProcessor()
if 'predictor' not in st.session_state:
    st.session_state.predictor = EnhancedHazardPredictor()
if 'mock_generator' not in st.session_state:
    st.session_state.mock_generator = MockDataGenerator()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = EnhancedAlertSystem()
if 'live_data_fetcher' not in st.session_state:
    st.session_state.live_data_fetcher = LiveDataFetcher()

def main():
    # Header
    st.title("🌍 Multi-Hazard Prediction System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard Overview", "Data Processing", "Drought Prediction", 
         "Flood Prediction", "Earthquake Prediction", "Alert Management", 
         "Real-time Local Predictions", "Model Training"]
    )
    
    # Main content based on selected page
    if page == "Dashboard Overview":
        dashboard_overview()
    elif page == "Data Processing":
        data_processing_page()
    elif page == "Drought Prediction":
        drought_prediction_page()
    elif page == "Flood Prediction":
        flood_prediction_page()
    elif page == "Earthquake Prediction":
        earthquake_prediction_page()
    elif page == "Alert Management":
        alert_management_page()
    elif page == "Real-time Local Predictions":
        realtime_local_predictions_page()
    elif page == "Model Training":
        model_training_page()

def dashboard_overview():
    st.header("🎯 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Monitoring Regions",
            value="127",
            delta="5"
        )
    
    with col2:
        st.metric(
            label="Current Alerts",
            value="8",
            delta="-2"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value="94.2%",
            delta="1.3%"
        )
    
    with col4:
        st.metric(
            label="Data Sources",
            value="12",
            delta="1"
        )
    
    # Real-time map
    st.subheader("🗺️ Global Hazard Overview")
    
    # Generate mock global data for demonstration
    regions_data = st.session_state.mock_generator.generate_global_hazard_data()
    
    # Create map
    hazard_map = create_hazard_map(regions_data)
    import streamlit.components.v1 as components
    components.html(hazard_map._repr_html_(), height=500)
    
    # Recent predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recent Predictions")
        predictions_df = st.session_state.mock_generator.generate_recent_predictions()
        st.dataframe(predictions_df, width='stretch')
    
    with col2:
        st.subheader("⚠️ Active Alerts")
        alerts_df = st.session_state.alert_system.get_active_alerts_enhanced()
        st.dataframe(alerts_df, width='stretch')

def data_processing_page():
    st.header("📡 Satellite Data Processing")
    
    # Data source selection
    st.subheader("Data Sources Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Optical Satellite Data**")
        sentinel2_enabled = st.checkbox("Sentinel-2", value=True)
        landsat_enabled = st.checkbox("Landsat 8/9", value=True)
        modis_enabled = st.checkbox("MODIS", value=True)
    
    with col2:
        st.markdown("**SAR & Geophysical Data**")
        sentinel1_enabled = st.checkbox("Sentinel-1 SAR", value=True)
        seismic_enabled = st.checkbox("USGS Seismic", value=True)
        weather_enabled = st.checkbox("Weather Stations", value=True)
    
    # Processing parameters
    st.subheader("Processing Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spatial_resolution = st.selectbox(
            "Spatial Resolution (m)",
            [10, 30, 100, 250, 500]
        )
    
    with col2:
        temporal_window = st.selectbox(
            "Temporal Window (days)",
            [7, 14, 30, 60, 90]
        )
    
    with col3:
        region_size = st.selectbox(
            "Region Size (km²)",
            [100, 500, 1000, 5000, 10000]
        )
    
    # Mock data processing demonstration
    if st.button("Process Sample Data", type="primary"):
        with st.spinner("Processing satellite data..."):
            # Simulate data processing
            processed_data = st.session_state.data_processor.process_mock_data(
                spatial_resolution, temporal_window, region_size
            )
            
            st.success("Data processing completed!")
            
            # Display processing results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("NDVI Time Series")
                fig = px.line(
                    processed_data['ndvi_timeseries'],
                    x='date', y='ndvi',
                    title="Normalized Difference Vegetation Index"
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("Soil Moisture")
                fig = px.line(
                    processed_data['soil_moisture'],
                    x='date', y='moisture',
                    title="Soil Moisture Content (%)"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Feature extraction results
            st.subheader("Extracted Features")
            features_df = processed_data['features']
            st.dataframe(features_df, width='stretch')

def drought_prediction_page():
    from pages.drought_prediction import render_drought_page
    render_drought_page()

def flood_prediction_page():
    from pages.flood_prediction import render_flood_page
    render_flood_page()

def earthquake_prediction_page():
    from pages.earthquake_prediction import render_earthquake_page
    render_earthquake_page()

def alert_management_page():
    st.header("⚠️ Alert Management System")
    
    # Alert configuration
    st.subheader("Alert Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Drought Alerts**")
        drought_low = st.slider("Low Risk", 0.0, 1.0, 0.3, key="drought_low")
        drought_high = st.slider("High Risk", 0.0, 1.0, 0.7, key="drought_high")
    
    with col2:
        st.markdown("**Flood Alerts**")
        flood_low = st.slider("Low Risk", 0.0, 1.0, 0.4, key="flood_low")
        flood_high = st.slider("High Risk", 0.0, 1.0, 0.8, key="flood_high")
    
    with col3:
        st.markdown("**Earthquake Alerts**")
        earthquake_low = st.slider("Low Risk", 0.0, 1.0, 0.2, key="earthquake_low")
        earthquake_high = st.slider("High Risk", 0.0, 1.0, 0.6, key="earthquake_high")
    
    # Alert history
    st.subheader("Alert History")
    
    # Generate alert history data
    alert_history = st.session_state.alert_system.get_alert_history_dataframe()
    
    # Alert timeline
    if not alert_history.empty:
        fig = px.timeline(
            alert_history,
            x_start="Start Time",
            x_end="End Time",
            y="Region",
            color="Hazard Type",
            title="Alert Timeline"
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No alert history available to display timeline.")
    
    # Alert statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alert Distribution by Hazard Type")
        if not alert_history.empty:
            alert_counts = alert_history['Hazard Type'].value_counts()
            fig = px.pie(
                values=alert_counts.values,
                names=alert_counts.index,
                title=""
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No alerts to display distribution by hazard type.")
    
    with col2:
        st.subheader("Alert Distribution by Severity")
        if not alert_history.empty:
            severity_counts = alert_history['Severity'].value_counts()
            fig = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                title=""
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No alerts to display distribution by severity.")

def realtime_local_predictions_page():
    st.header("📍 Real-time Local Predictions")
    st.write("Get real-time hazard predictions based on your current location.")

    # Placeholder for location data
    location_data = st.empty()

    # Button to trigger location request
    if st.button("Get My Current Location"):
        location_data.info("Attempting to get your location...")
        
        # Get location using get_geolocation from streamlit_js_eval
        try:
            location = get_geolocation()
            
            if location:
                latitude = location['latitude']
                longitude = location['longitude']
                st.session_state.current_location = {'latitude': latitude, 'longitude': longitude}
                location_data.success("Location obtained.")
            else:
                location_data.warning("Could not retrieve location. Please ensure location services are enabled and granted.")
        except Exception as e:
            location_data.error(f"Error getting location: {e}")



    
    if 'current_location' in st.session_state:
        st.write(f"Latitude: {st.session_state.current_location['latitude']}, Longitude: {st.session_state.current_location['longitude']}")
        st.subheader("Hazard Prediction for Your Location")
        
        # Fetch real-time data for the current location
        with st.spinner("Fetching real-time hazard data..."):
            lat = st.session_state.current_location['latitude']
            lon = st.session_state.current_location['longitude']
            
            live_data = st.session_state.live_data_fetcher.get_comprehensive_data(lat, lon)
            st.session_state.live_hazard_data = live_data # Store for later use in prediction
            st.success("Real-time hazard data fetched.")

            # Display some fetched data (for verification)
            st.subheader("Fetched Data Snapshot:")
            st.json({
                "Weather (last day temp)": live_data['weather']['temperature'][-1],
                "NDVI (last value)": live_data['ndvi']['ndvi'][-1] if live_data['ndvi']['ndvi'] else "N/A",
                "Seismic events (count)": len(live_data['seismic']),
                "Precipitation forecast (next 7 days total)": sum(live_data['precipitation_forecast']['precipitation'])
            })
        
        # Placeholder for prediction - to be implemented in the next step
        st.info("Data fetched. Now preparing for hazard prediction using the models.")

        # Make predictions for each hazard type
        st.subheader("Hazard Predictions:")

        predictor = st.session_state.predictor
        lat = st.session_state.current_location['latitude']
        lon = st.session_state.current_location['longitude']
        
        # Drought Prediction
        drought_prediction = predictor.predict_drought_risk_live(lat, lon)
        st.write("**Drought Risk:**")
        st.write(f"Level: {drought_prediction['risk_level']} (Probability: {drought_prediction['risk_probability']:.2f}, Confidence: {drought_prediction['confidence']:.2f})")
        st.write(f"Recommendations: {', '.join(drought_prediction['recommendation'])}")
        
        # Flood Prediction
        flood_prediction = predictor.predict_flood_risk_live(lat, lon)
        st.write("**Flood Risk:**")
        st.write(f"Level: {flood_prediction['risk_level']} (Probability: {flood_prediction['risk_probability']:.2f}, Confidence: {flood_prediction['confidence']:.2f})")
        st.write(f"Recommendations: {', '.join(flood_prediction['recommendation'])}")

        # Earthquake Prediction
        earthquake_prediction = predictor.predict_earthquake_risk_live(lat, lon)
        st.write("**Earthquake Risk:**")
        st.write(f"Magnitude: {earthquake_prediction['magnitude_prediction']:.2f} (Level: {earthquake_prediction['risk_level']}, Probability: {earthquake_prediction['probability']:.2f}, Confidence: {earthquake_prediction['confidence']:.2f})")
        st.write(f"Recommendations: {', '.join(earthquake_prediction['recommendation'])}")

def model_training_page():
    st.header("⚙️ Model Training and Management")
    st.write("Here you can train your hazard prediction models using mock data.")

    # Ensure a fresh scaler is used for training page to avoid feature mismatch errors
    st.session_state.predictor.feature_scaler = StandardScaler()

    # Training parameters
    st.subheader("Training Data Generation Parameters")
    num_regions_to_train = st.slider("Number of Regions for Training Data", 1, 20, 5)
    days_per_region_to_train = st.slider("Historical Days per Region for Training Data", 30, 730, 365)

    if st.button("Generate Training Data & Train Models", type="primary"):
        with st.spinner("Generating training data and training models..."):
            training_data = st.session_state.mock_generator.generate_training_data(
                num_regions=num_regions_to_train,
                days_per_region=days_per_region_to_train
            )
            
            # Extract features and labels for each hazard type
            X_drought = training_data['drought']['features']
            y_drought = training_data['drought']['labels']
            X_flood = training_data['flood']['features']
            y_flood = training_data['flood']['labels']
            X_earthquake = training_data['earthquake']['features']
            y_earthquake = training_data['earthquake']['labels']

            # Split data for evaluation
            X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_drought, y_drought, test_size=0.2, random_state=42)
            X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)
            X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(X_earthquake, y_earthquake, test_size=0.2, random_state=42)

            # Train the models
            st.session_state.predictor.train(X_d_train, y_d_train, X_f_train, y_f_train, X_e_train, y_e_train)
            st.session_state.predictor.save_models()
            st.success("Models trained and saved successfully!")

            # Evaluate models
            st.subheader("Model Evaluation (after training):")
            drought_metrics = st.session_state.predictor.evaluate_drought_model(X_d_test, y_d_test)
            st.write("**Drought Model Metrics:**", drought_metrics)
            flood_metrics = st.session_state.predictor.evaluate_flood_model(X_f_test, y_f_test)
            st.write("**Flood Model Metrics:**", flood_metrics)
            earthquake_metrics = st.session_state.predictor.evaluate_earthquake_model(X_e_test, y_e_test)
            st.write("**Earthquake Model Metrics:**", earthquake_metrics)

    if st.button("Perform Hyperparameter Tuning & Retrain Models", type="secondary"):
        if not st.session_state.predictor.is_trained:
            st.warning("Please train models first before tuning, or generate data and train.")
        else:
            with st.spinner("Performing hyperparameter tuning and retraining models... (This may take a while)"):
                training_data = st.session_state.mock_generator.generate_training_data(
                    num_regions=num_regions_to_train,
                    days_per_region=days_per_region_to_train
                )
                
                X_drought = training_data['drought']['features']
                y_drought = training_data['drought']['labels']
                X_flood = training_data['flood']['features']
                y_flood = training_data['flood']['labels']
                X_earthquake = training_data['earthquake']['features']
                y_earthquake = training_data['earthquake']['labels']

                # Split data for evaluation
                X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_drought, y_drought, test_size=0.2, random_state=42)
                X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)
                X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(X_earthquake, y_earthquake, test_size=0.2, random_state=42)

                # Tune and retrain each model
                st.session_state.predictor.tune_drought_model(X_d_train, y_d_train)
                st.session_state.predictor.tune_flood_model(X_f_train, y_f_train)
                st.session_state.predictor.tune_earthquake_model(X_e_train, y_e_train)
                
                st.session_state.predictor.save_models()
                st.success("Models retrained with tuned hyperparameters and saved successfully!")

                # Evaluate models after tuning
                st.subheader("Model Evaluation (after tuning):")
                drought_metrics_tuned = st.session_state.predictor.evaluate_drought_model(X_d_test, y_d_test)
                st.write("**Drought Model Metrics (Tuned):**", drought_metrics_tuned)
                flood_metrics_tuned = st.session_state.predictor.evaluate_flood_model(X_f_test, y_f_test)
                st.write("**Flood Model Metrics (Tuned):**", flood_metrics_tuned)
                earthquake_metrics_tuned = st.session_state.predictor.evaluate_earthquake_model(X_e_test, y_e_test)
                st.write("**Earthquake Model Metrics (Tuned):**", earthquake_metrics_tuned)

    st.subheader("Model Status")
    if st.session_state.predictor.is_trained:
        st.success("Models are currently trained and loaded.")
    else:
        st.warning("Models are not trained. Please train them to enable predictions.")

    if st.button("Load Pre-trained Models"):
        with st.spinner("Loading pre-trained models..."):
            st.session_state.predictor.load_models()
            if st.session_state.predictor.is_trained:
                st.success("Pre-trained models loaded successfully!")
            else:
                st.error("Could not load pre-trained models. Please train them.")


if __name__ == "__main__":
    # Load models at startup
    if 'predictor' in st.session_state and not st.session_state.predictor.is_trained:
        st.session_state.predictor.load_models()
    main()
