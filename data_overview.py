import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.mock_data_generator import MockDataGenerator
from data.data_processor import SatelliteDataProcessor

def render_data_overview():
    """
    Render the data overview page showing data sources, quality, and processing status.
    """
    st.header("📡 Data Sources & Processing Overview")
    
    # Initialize session state components
    if 'mock_generator' not in st.session_state:
        st.session_state.mock_generator = MockDataGenerator()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = SatelliteDataProcessor()
    
    # Data source status overview
    st.subheader("🛰️ Satellite Data Sources Status")
    
    # Mock data source status
    data_sources = [
        {'Source': 'Sentinel-1 (SAR)', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(hours=2), 'Coverage': '95%', 'Quality': 'Excellent'},
        {'Source': 'Sentinel-2 (Optical)', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(hours=6), 'Coverage': '92%', 'Quality': 'Good'},
        {'Source': 'Landsat 8/9', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(hours=12), 'Coverage': '88%', 'Quality': 'Good'},
        {'Source': 'MODIS Terra/Aqua', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(hours=1), 'Coverage': '98%', 'Quality': 'Excellent'},
        {'Source': 'USGS Seismic Network', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(minutes=15), 'Coverage': '85%', 'Quality': 'Good'},
        {'Source': 'Weather Stations', 'Status': 'Degraded', 'Last Update': datetime.now() - timedelta(hours=18), 'Coverage': '76%', 'Quality': 'Fair'},
        {'Source': 'Ocean Buoys', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(hours=3), 'Coverage': '82%', 'Quality': 'Good'},
        {'Source': 'GPS Networks', 'Status': 'Active', 'Last Update': datetime.now() - timedelta(minutes=30), 'Coverage': '91%', 'Quality': 'Excellent'}
    ]
    
    sources_df = pd.DataFrame(data_sources)
    sources_df['Last Update'] = sources_df['Last Update'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Color code status
    def color_status(val):
        if val == 'Active':
            return 'color: green'
        elif val == 'Degraded':
            return 'color: orange'
        else:
            return 'color: red'
    
    styled_df = sources_df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, width='stretch')
    
    # Data processing metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Sources", "8", "1")
    
    with col2:
        st.metric("Active Sources", "7", "0")
    
    with col3:
        st.metric("Average Coverage", "89%", "+2%")
    
    with col4:
        st.metric("Data Quality Score", "4.2/5", "+0.1")
    
    # Data volume and processing
    st.markdown("---")
    st.subheader("📊 Data Volume & Processing Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate mock data volume statistics
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        data_volume = {
            'Date': dates,
            'Sentinel-1 (GB)': np.random.uniform(15, 25, len(dates)),
            'Sentinel-2 (GB)': np.random.uniform(20, 35, len(dates)),
            'Landsat (GB)': np.random.uniform(8, 15, len(dates)),
            'MODIS (GB)': np.random.uniform(5, 12, len(dates)),
            'Ground Data (GB)': np.random.uniform(2, 8, len(dates))
        }
        
        volume_df = pd.DataFrame(data_volume)
        
        # Melt the dataframe for plotting
        volume_melted = volume_df.melt(id_vars=['Date'], var_name='Source', value_name='Volume (GB)')
        
        fig_volume = px.line(
            volume_melted,
            x='Date',
            y='Volume (GB)',
            color='Source',
            title='Daily Data Volume by Source (Last 30 Days)'
        )
        st.plotly_chart(fig_volume, width='stretch')
    
    with col2:
        # Processing time statistics
        processing_stats = {
            'Processing Step': [
                'Data Ingestion', 'Preprocessing', 'Feature Extraction',
                'Model Inference', 'Post-processing', 'Alert Generation'
            ],
            'Avg Time (min)': [5, 12, 8, 3, 4, 1],
            'Success Rate (%)': [99.2, 97.8, 98.5, 99.7, 98.9, 99.8]
        }
        
        processing_df = pd.DataFrame(processing_stats)
        
        fig_processing = px.bar(
            processing_df,
            x='Processing Step',
            y='Avg Time (min)',
            title='Average Processing Time by Step',
            text='Success Rate (%)'
        )
        fig_processing.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_processing.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_processing, width='stretch')
    
    # Data quality assessment
    st.markdown("---")
    st.subheader("🔍 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality metrics by region
        regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
        quality_scores = np.random.uniform(3.5, 5.0, len(regions))
        coverage_scores = np.random.uniform(75, 95, len(regions))
        
        quality_df = pd.DataFrame({
            'Region': regions,
            'Quality Score': quality_scores,
            'Coverage (%)': coverage_scores
        })
        
        fig_quality = px.scatter(
            quality_df,
            x='Coverage (%)',
            y='Quality Score',
            size='Quality Score',
            color='Region',
            title='Data Quality vs Coverage by Region',
            size_max=20
        )
        st.plotly_chart(fig_quality, width='stretch')
    
    with col2:
        # Data freshness
        data_freshness = {
            'Data Type': ['NDVI', 'Soil Moisture', 'Temperature', 'Precipitation', 'Seismic', 'Ground Deformation'],
            'Hours Since Update': [2, 6, 1, 3, 0.25, 12],
            'Freshness Score': [4.8, 4.2, 4.9, 4.6, 5.0, 3.8]
        }
        
        freshness_df = pd.DataFrame(data_freshness)
        
        fig_freshness = px.bar(
            freshness_df,
            x='Data Type',
            y='Freshness Score',
            color='Hours Since Update',
            title='Data Freshness Scores',
            color_continuous_scale='RdYlGn_r'
        )
        fig_freshness.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_freshness, width='stretch')
    
    # Geographic coverage
    st.markdown("---")
    st.subheader("🌍 Geographic Coverage Analysis")
    
    # Generate global coverage data
    global_coverage = st.session_state.mock_generator.generate_global_hazard_data()
    coverage_df = pd.DataFrame(global_coverage)
    
    # Coverage by region type
    coverage_summary = coverage_df.groupby(['region_type', 'hazard_type']).agg({
        'confidence': 'mean',
        'risk_score': 'mean'
    }).reset_index()
    
    fig_coverage = px.sunburst(
        coverage_summary,
        path=['region_type', 'hazard_type'],
        values='confidence',
        title='Data Coverage by Region Type and Hazard'
    )
    st.plotly_chart(fig_coverage, width='stretch')
    
    # Real-time processing status
    st.markdown("---")
    st.subheader("⚡ Real-time Processing Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Current Processing Queue**")
        queue_data = {
            'Job ID': ['DRT_001', 'FLD_023', 'EQK_045', 'DRT_002'],
            'Type': ['Drought Analysis', 'Flood Prediction', 'Seismic Analysis', 'Drought Analysis'],
            'Status': ['Processing', 'Queued', 'Processing', 'Queued'],
            'Progress': ['75%', '0%', '45%', '0%']
        }
        queue_df = pd.DataFrame(queue_data)
        st.dataframe(queue_df, width='stretch')
    
    with col2:
        st.markdown("**System Performance**")
        perf_data = {
            'Metric': ['CPU Usage', 'Memory Usage', 'Storage Usage', 'Network I/O'],
            'Current': ['68%', '45%', '72%', '23%'],
            'Threshold': ['85%', '80%', '90%', '75%']
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, width='stretch')
    
    with col3:
        st.markdown("**Error Log Summary**")
        error_data = {
            'Error Type': ['Connection Timeout', 'Data Corruption', 'Processing Failed', 'Invalid Format'],
            'Count (24h)': [3, 1, 2, 0],
            'Last Occurrence': ['2h ago', '8h ago', '4h ago', 'Never']
        }
        error_df = pd.DataFrame(error_data)
        st.dataframe(error_df, width='stretch')
    
    # Data processing pipeline visualization
    st.markdown("---")
    st.subheader("🔄 Data Processing Pipeline")
    
    # Create a flow diagram using plotly
    fig_pipeline = go.Figure()
    
    # Pipeline stages
    stages = [
        {'name': 'Data\nIngestion', 'x': 1, 'y': 5},
        {'name': 'Quality\nCheck', 'x': 2, 'y': 5},
        {'name': 'Preprocessing', 'x': 3, 'y': 5},
        {'name': 'Feature\nExtraction', 'x': 4, 'y': 6},
        {'name': 'Model\nInference', 'x': 4, 'y': 4},
        {'name': 'Post-\nprocessing', 'x': 5, 'y': 5},
        {'name': 'Alert\nGeneration', 'x': 6, 'y': 5},
        {'name': 'Visualization', 'x': 7, 'y': 5}
    ]
    
    # Add nodes
    for stage in stages:
        fig_pipeline.add_trace(go.Scatter(
            x=[stage['x']],
            y=[stage['y']],
            mode='markers+text',
            marker=dict(size=60, color='lightblue'),
            text=stage['name'],
            textposition='middle center',
            showlegend=False
        ))
    
    # Add connections
    connections = [
        (1, 2), (2, 3), (3, 4), (3, 4), (4, 5), (6, 5), (4, 6), (5, 6), (6, 7), (7, 8)
    ]
    
    for start, end in connections:
        start_stage = stages[start-1]
        end_stage = stages[end-1]
        fig_pipeline.add_trace(go.Scatter(
            x=[start_stage['x'], end_stage['x']],
            y=[start_stage['y'], end_stage['y']],
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False
        ))
    
    fig_pipeline.update_layout(
        title='Data Processing Pipeline Flow',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig_pipeline, width='stretch')
    
    # Data export and API status
    st.markdown("---")
    st.subheader("📤 Data Export & API Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Statistics (Last 30 Days)**")
        export_stats = {
            'Export Format': ['JSON', 'CSV', 'GeoTIFF', 'NetCDF', 'Shapefile'],
            'Downloads': [245, 189, 67, 43, 78],
            'Data Volume (GB)': [12.3, 8.9, 45.2, 23.1, 5.4]
        }
        export_df = pd.DataFrame(export_stats)
        
        fig_export = px.bar(
            export_df,
            x='Export Format',
            y='Downloads',
            title='Data Exports by Format'
        )
        st.plotly_chart(fig_export, width='stretch')
    
    with col2:
        st.markdown("**API Endpoints Status**")
        api_status = {
            'Endpoint': ['/api/drought', '/api/flood', '/api/earthquake', '/api/data', '/api/alerts'],
            'Status': ['Active', 'Active', 'Active', 'Maintenance', 'Active'],
            'Response Time (ms)': [145, 178, 223, 0, 98],
            'Success Rate (%)': [99.7, 99.2, 98.8, 0, 99.9]
        }
        api_df = pd.DataFrame(api_status)
        
        # Apply color coding for status
        def color_api_status(val):
            if val == 'Active':
                return 'color: green'
            elif val == 'Maintenance':
                return 'color: orange'
            else:
                return 'color: red'
        
        styled_api_df = api_df.style.applymap(color_api_status, subset=['Status'])
        st.dataframe(styled_api_df, width='stretch')
    
    # Refresh data button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 Refresh Data Sources", type="primary", width='stretch'):
            with st.spinner("Refreshing data sources..."):
                # Simulate refresh process
                import time
                time.sleep(2)
                st.success("Data sources refreshed successfully!")
                st.rerun()
