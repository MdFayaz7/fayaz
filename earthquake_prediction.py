import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.mock_data_generator import MockDataGenerator
from models.enhanced_predictor import EnhancedHazardPredictor # Changed import from MultiHazardPredictor
from visualization.maps import create_region_specific_map

def render_earthquake_page():
    """
    Render the earthquake prediction page.
    """
    st.header("🌋 Earthquake Prediction & Monitoring")
    
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
        ["Ring of Fire - Japan", "San Andreas Fault - California", "East African Rift", 
         "Himalayan Front", "Mediterranean Ridge", "Custom Location"]
    )
    
    if selected_region == "Custom Location":
        custom_lat = st.sidebar.number_input("Latitude", value=35.6762, min_value=-90.0, max_value=90.0)
        custom_lon = st.sidebar.number_input("Longitude", value=139.6503, min_value=-180.0, max_value=180.0)
        region_coords = (custom_lat, custom_lon)
        is_seismic_region = st.sidebar.checkbox("High Seismic Activity Region", value=True)
    else:
        # Predefined coordinates for seismic regions
        region_coords_map = {
            "Ring of Fire - Japan": (35.6762, 139.6503, True),
            "San Andreas Fault - California": (36.7783, -119.4179, True),
            "East African Rift": (-1.2921, 36.8219, True),
            "Himalayan Front": (28.2380, 83.9956, True),
            "Mediterranean Ridge": (35.0000, 25.0000, True)
        }
        region_coords = region_coords_map[selected_region][:2]
        is_seismic_region = region_coords_map[selected_region][2]
    
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
    magnitude_threshold = st.sidebar.slider("Minimum Magnitude", 1.0, 5.0, 2.5, step=0.1)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.7)
    prediction_horizon = st.sidebar.selectbox("Prediction Horizon", ["7 days", "14 days", "30 days", "90 days"])

    # Initialize prediction variables to default values
    magnitude_prediction = 0.0
    risk_probability = 0.0
    confidence = 0.0
    risk_level = "Unknown"
    risk_color = "gray"
    factors = {}
    recent_activity = 0
    total_events = 0
    max_magnitude = 0.0
    mean_magnitude = 0.0
    b_value = 1.0
    overall_risk = "Low"
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Seismic Analysis - {selected_region}")
        
        # Generate seismic data
        with st.spinner("Analyzing seismic data and generating predictions..."):
            # Generate seismic events
            seismic_events = st.session_state.mock_generator.generate_seismic_data(
                days=analysis_days, 
                lat=region_coords[0], 
                is_seismic_region=is_seismic_region
            )
            
            # Convert to DataFrame for easier manipulation
            if seismic_events:
                seismic_df = pd.DataFrame(seismic_events)
                seismic_df = seismic_df[seismic_df['magnitude'] >= magnitude_threshold]
            else:
                seismic_df = pd.DataFrame({
                    'date': pd.Series([], dtype='datetime64[ns]'),
                    'latitude': pd.Series([], dtype='float64'),
                    'longitude': pd.Series([], dtype='float64'),
                    'magnitude': pd.Series([], dtype='float64'),
                    'depth': pd.Series([], dtype='float64')
                })
        
        if not seismic_df.empty:
            # Magnitude distribution over time
            fig_magnitude = px.scatter(
                seismic_df, 
                x='date', 
                y='magnitude',
                color='depth',
                size='magnitude',
                title='Earthquake Magnitude Over Time',
                labels={'magnitude': 'Magnitude', 'date': 'Date', 'depth': 'Depth (km)'},
                color_continuous_scale='Viridis'
            )
            fig_magnitude.add_hline(y=4.0, line_dash="dash", line_color="orange", 
                                   annotation_text="Moderate Earthquake Threshold")
            fig_magnitude.add_hline(y=6.0, line_dash="dash", line_color="red", 
                                   annotation_text="Strong Earthquake Threshold")
            fig_magnitude.update_layout(height=400)
            st.plotly_chart(fig_magnitude, width='stretch')
            
            # Daily seismic activity
            if len(seismic_df) > 0:
                seismic_df['date_only'] = seismic_df['date'].dt.date
            else:
                seismic_df['date_only'] = pd.Series([], dtype='object')
            daily_activity = seismic_df.groupby('date_only').agg({
                'magnitude': ['count', 'max', 'mean'],
                'depth': 'mean'
            }).reset_index()
            daily_activity.columns = ['date', 'event_count', 'max_magnitude', 'mean_magnitude', 'mean_depth']
            
            # Fill missing dates with zeros
            date_range = pd.date_range(start=seismic_df['date'].min().date(), 
                                     end=seismic_df['date'].max().date(), freq='D')
            complete_dates = pd.DataFrame({'date': date_range.date})
            daily_activity = pd.merge(complete_dates, daily_activity, on='date', how='left').fillna(0)
            
            fig_daily = px.bar(
                daily_activity, 
                x='date', 
                y='event_count',
                title='Daily Seismic Activity (Number of Events)',
                labels={'event_count': 'Number of Earthquakes', 'date': 'Date'}
            )
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, width='stretch')
            
            # Depth vs magnitude analysis
            st.subheader("🔍 Magnitude-Depth Analysis")
            
            fig_depth = px.scatter(
                seismic_df,
                x='depth',
                y='magnitude',
                color='magnitude',
                title='Earthquake Magnitude vs Depth',
                labels={'depth': 'Depth (km)', 'magnitude': 'Magnitude'},
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_depth, width='stretch')
        else:
            st.info("No seismic events above the selected magnitude threshold in the analysis period.")
            # Create empty plots for consistency
            fig_empty = go.Figure()
            fig_empty.add_annotation(text="No data available for the selected parameters", 
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig_empty.update_layout(height=400, title="Earthquake Magnitude Over Time")
            st.plotly_chart(fig_empty, width='stretch')
    
    with col2:
        st.subheader("🎯 Current Seismic Assessment")
        
        # Generate earthquake prediction
        if not seismic_df.empty:
            # Ensure we have enough data for feature extraction (e.g., last 10 events)
            num_events_for_prediction = min(10, len(seismic_df))
            if num_events_for_prediction > 0:
                latest_data = {
                    'seismic_activity': seismic_df['magnitude'].tail(num_events_for_prediction).values.tolist(),
                    'depth': seismic_df['depth'].tail(num_events_for_prediction).values.tolist()
                }
            else:
                latest_data = {
                    'seismic_activity': [],
                    'depth': []
                }
        else:
            latest_data = {
                'seismic_activity': [],
                'depth': []
            }
        
        earthquake_prediction = st.session_state.predictor.predict_earthquake_risk(latest_data)
        
        if "error" in earthquake_prediction:
            st.error(f"Prediction Error: {earthquake_prediction['error']}")
            # Re-initialize variables to default values if there's an error
            magnitude_prediction = 0.0
            risk_probability = 0.0
            confidence = 0.0
            risk_level = "Unknown"
            risk_color = "gray"
            factors = {}
        else:
            # Display current risk
            magnitude_prediction = earthquake_prediction['magnitude_prediction']
            risk_probability = earthquake_prediction['probability']
            confidence = earthquake_prediction['confidence']
            factors = earthquake_prediction['factors']

            # Risk level based on predicted magnitude
            if magnitude_prediction >= 6.0:
                risk_level = "High"
                risk_color = "red"
            elif magnitude_prediction >= 4.0:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "Low"
                risk_color = "green"
        
        st.markdown(f"**Current Risk Level:** <span style='color: {risk_color}'>{risk_level}</span>", 
                   unsafe_allow_html=True)
        
        st.metric("Predicted Magnitude", f"{magnitude_prediction:.1f}")
        st.metric("Risk Probability", f"{risk_probability:.1%}")
        st.metric("Model Confidence", f"{confidence:.1%}")
        
        # Risk factors (only display if factors are available and not empty)
        if factors and any(factors.values()):
            st.subheader("📈 Key Risk Factors")
            factor_df = pd.DataFrame([
                {'Factor': 'Seismic Activity Mean', 'Value': f"{factors.get('seismic_mean', 0.0):.2f}", 'Impact': 'High' if factors.get('seismic_mean', 0.0) > 0.6 else 'Normal'},
                {'Factor': 'b-value', 'Value': f"{factors.get('b_value', 1.0):.2f}", 'Impact': 'Decreasing' if factors.get('b_value', 1.0) < 0.8 else 'Normal'},
                {'Factor': 'Recent Events', 'Value': f"{factors.get('seismic_count', 0)}", 'Impact': 'High' if factors.get('seismic_count', 0) > 10 else 'Normal'},
                {'Factor': 'Shallow Events', 'Value': f"{factors.get('shallow_events', 0)}", 'Impact': 'High' if factors.get('shallow_events', 0) > 2 else 'Normal'},
                {'Factor': 'Energy Release', 'Value': f"{factors.get('seismic_energy', 0.0):.2f}", 'Impact': 'High' if factors.get('seismic_energy', 0.0) > 10 else 'Normal'}
            ])
            
            st.dataframe(factor_df, width='stretch')
        else:
            st.info("No risk factors to display (prediction might be unavailable or failed).")
        
        # Historical comparison
        st.subheader("📊 Historical Context")
        
        if not seismic_df.empty:
            # Calculate historical statistics
            recent_activity = len(seismic_df[seismic_df['date'] > (datetime.now() - timedelta(days=7))])
            total_events = len(seismic_df)
            max_magnitude = seismic_df['magnitude'].max()
            mean_magnitude = seismic_df['magnitude'].mean()
            
            # Calculate b-value (Gutenberg-Richter relationship)
            if len(seismic_df) > 0:
                magnitudes = seismic_df['magnitude'].values
            else:
                magnitudes = np.array([])
            mag_bins = np.arange(magnitude_threshold, max_magnitude + 0.5, 0.5)
            hist, _ = np.histogram(magnitudes, bins=mag_bins)
            
            # Calculate b-value using linear regression on log-linear relationship
            if len(hist) > 2 and np.sum(hist > 0) > 1:
                log_counts = np.log10(hist + 1)
                valid_indices = hist > 0
                if np.sum(valid_indices) > 1:
                    b_value = -np.polyfit(mag_bins[:-1][valid_indices], log_counts[valid_indices], 1)[0]
                else:
                    b_value = 1.0
            else:
                b_value = 1.0
        else:
            recent_activity = 0
            total_events = 0
            max_magnitude = 0.0
            mean_magnitude = 0.0
            b_value = 1.0
        
        st.metric("Recent Activity (7 days)", f"{recent_activity} events")
        st.metric("Maximum Recorded Magnitude", f"{max_magnitude:.1f}")
        st.metric("b-value", f"{b_value:.2f}")
    
    # Full width sections
    st.markdown("---")
    
    # Seismic indicators and forecasting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Seismic Indicators")
        
        if not seismic_df.empty:
            # Calculate various seismic indices
            recent_events_7d = len(seismic_df[seismic_df['date'] > (datetime.now() - timedelta(days=7))])
            recent_events_30d = len(seismic_df[seismic_df['date'] > (datetime.now() - timedelta(days=30))])
            
            # Seismic Energy Release
            energy_joules = np.sum(10**(11.8 + 1.5 * seismic_df['magnitude']))  # Energy in Joules
            energy_mt = energy_joules / 4.184e15  # Convert to megatons TNT equivalent
            
            # Seismic Rate Change
            first_half = seismic_df[seismic_df['date'] < (seismic_df['date'].min() + (seismic_df['date'].max() - seismic_df['date'].min())/2)]
            second_half = seismic_df[seismic_df['date'] >= (seismic_df['date'].min() + (seismic_df['date'].max() - seismic_df['date'].min())/2)]
            
            if len(first_half) > 0 and len(second_half) > 0:
                rate_change = (len(second_half) - len(first_half)) / len(first_half) * 100
            else:
                rate_change = 0.0
            
            indicators_df = pd.DataFrame([
                {'Indicator': 'Recent Activity (7 days)', 'Value': f"{recent_events_7d} events", 'Status': 'High' if recent_events_7d > 10 else 'Normal'},
                {'Indicator': 'Monthly Activity Rate', 'Value': f"{recent_events_30d} events", 'Status': 'High' if recent_events_30d > 30 else 'Normal'},
                {'Indicator': 'b-value', 'Value': f"{b_value:.2f}", 'Status': 'Decreasing' if b_value < 0.8 else 'Normal'},
                {'Indicator': 'Energy Release', 'Value': f"{energy_mt:.2e} MT", 'Status': 'High' if energy_mt > 1 else 'Normal'},
                {'Indicator': 'Activity Rate Change', 'Value': f"{rate_change:+.1f}%", 'Status': 'Increasing' if rate_change > 20 else 'Stable'}
            ])
        else:
            indicators_df = pd.DataFrame([
                {'Indicator': 'Recent Activity (7 days)', 'Value': '0 events', 'Status': 'Low'},
                {'Indicator': 'Monthly Activity Rate', 'Value': '0 events', 'Status': 'Low'},
                {'Indicator': 'b-value', 'Value': 'N/A', 'Status': 'No Data'},
                {'Indicator': 'Energy Release', 'Value': '0 MT', 'Status': 'Low'},
                {'Indicator': 'Activity Rate Change', 'Value': '0%', 'Status': 'No Data'}
            ])
        
        st.dataframe(indicators_df, width='stretch')
    
    with col2:
        st.subheader("🔮 Forecast & Probability")
        
        # Generate forecast (probabilistic earthquake forecasting)
        forecast_days = int(prediction_horizon.split()[0])
        
        # Calculate earthquake probabilities using Poisson distribution
        if not seismic_df.empty:
            # Calculate historical rate for different magnitude ranges
            total_period_years = analysis_days / 365.25
            mag_4_rate = len(seismic_df[seismic_df['magnitude'] >= 4.0]) / total_period_years
            mag_5_rate = len(seismic_df[seismic_df['magnitude'] >= 5.0]) / total_period_years
            mag_6_rate = len(seismic_df[seismic_df['magnitude'] >= 6.0]) / total_period_years
            
            forecast_period_years = forecast_days / 365.25
            
            # Poisson probabilities
            prob_4 = 1 - np.exp(-mag_4_rate * forecast_period_years)
            prob_5 = 1 - np.exp(-mag_5_rate * forecast_period_years)
            prob_6 = 1 - np.exp(-mag_6_rate * forecast_period_years)
        else:
            prob_4 = 0.05  # Default low probabilities
            prob_5 = 0.02
            prob_6 = 0.005
        
        # Display probabilities
        prob_df = pd.DataFrame([
            {'Magnitude Range': 'M ≥ 4.0', 'Probability': f"{prob_4:.1%}", 'Risk Level': 'Moderate' if prob_4 > 0.1 else 'Low'},
            {'Magnitude Range': 'M ≥ 5.0', 'Probability': f"{prob_5:.1%}", 'Risk Level': 'High' if prob_5 > 0.1 else 'Moderate' if prob_5 > 0.05 else 'Low'},
            {'Magnitude Range': 'M ≥ 6.0', 'Probability': f"{prob_6:.1%}", 'Risk Level': 'Critical' if prob_6 > 0.05 else 'High' if prob_6 > 0.02 else 'Low'}
        ])
        
        st.dataframe(prob_df, width='stretch')
        
        # Probability visualization
        fig_prob = px.bar(
            prob_df,
            x='Magnitude Range',
            y=[prob_4, prob_5, prob_6],
            title=f'Earthquake Probability - {prediction_horizon}',
            labels={'y': 'Probability', 'x': 'Magnitude Range'}
        )
        fig_prob.update_layout(height=300)
        st.plotly_chart(fig_prob, width='stretch')
        
        # Overall risk assessment
        if prob_6 > 0.05:
            overall_risk = "Critical"
            risk_color = "red"
        elif prob_5 > 0.1:
            overall_risk = "High"
            risk_color = "orange"
        elif prob_4 > 0.2:
            overall_risk = "Moderate"
            risk_color = "orange"
        else:
            overall_risk = "Low"
            risk_color = "green"
        
        st.markdown(f"**{prediction_horizon} Overall Risk:** <span style='color: {risk_color}'>{overall_risk}</span>", 
                   unsafe_allow_html=True)
    
    # Regional map
    st.markdown("---")
    st.subheader("🗺️ Regional Seismic Monitoring")
    
    # Create region-specific map with earthquake data
    historical_events = []
    if not seismic_df.empty:
        # Add recent significant earthquakes to map
        if len(seismic_df) > 0:
            significant_events = seismic_df[seismic_df['magnitude'] >= 4.0].tail(5)
            for _, event in significant_events.iterrows():
                historical_events.append({
                    'lat': float(event['latitude']),
                    'lon': float(event['longitude']),
                    'type': 'earthquake',
                    'date': event['date'].strftime('%Y-%m-%d'),
                    'magnitude': f"M{event['magnitude']:.1f}"
                })
    
    region_data = {
        'monitoring_stations': [
            {'name': 'Seismic Station A', 'lat': region_coords[0] + 0.1, 'lon': region_coords[1] + 0.1, 'type': 'Seismometer'},
            {'name': 'Seismic Station B', 'lat': region_coords[0] - 0.1, 'lon': region_coords[1] - 0.1, 'type': 'Accelerometer'},
            {'name': 'GPS Station C', 'lat': region_coords[0] + 0.05, 'lon': region_coords[1] - 0.05, 'type': 'GPS'}
        ],
        'risk_grid': [
            {'lat': region_coords[0] + i*0.05, 'lon': region_coords[1] + j*0.05, 
             'risk_score': min(1.0, max(0.1, np.random.uniform(0.3, 0.9) * (1.5 if is_seismic_region else 0.5)))}
            for i in range(-2, 3) for j in range(-2, 3)
        ],
        'historical_events': historical_events
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
    
    if overall_risk == "Critical":
        recommendations = [
            "🚨 Issue earthquake alert for M≥6.0 potential",
            "🏗️ Inspect critical infrastructure and buildings",
            "📋 Review and test earthquake response plans",
            "🎒 Advise public on earthquake preparedness",
            "🏥 Alert hospitals and emergency services",
            "📡 Increase seismic monitoring frequency",
            "🔧 Check emergency supply and equipment readiness"
        ]
    elif overall_risk == "High":
        recommendations = [
            "⚠️ Issue earthquake watch advisory",
            "📊 Increase seismic monitoring activities",
            "🏢 Conduct building safety assessments",
            "👥 Alert emergency response teams",
            "📚 Review evacuation procedures",
            "🔍 Monitor for earthquake precursors",
            "📱 Test emergency communication systems"
        ]
    elif overall_risk == "Moderate":
        recommendations = [
            "📊 Continue enhanced monitoring",
            "🏗️ Routine inspection of critical facilities",
            "👥 Maintain emergency preparedness",
            "📋 Update earthquake response procedures",
            "🎓 Conduct public education on earthquake safety"
        ]
    else:
        recommendations = [
            "✅ Continue routine seismic monitoring",
            "📊 Maintain standard observation protocols",
            "🏗️ Perform regular infrastructure maintenance",
            "📚 Update seismic hazard assessments",
            "👥 Conduct periodic emergency drills",
            "🎓 Promote earthquake awareness programs"
        ]
    
    for rec in recommendations:
        st.write(rec)
