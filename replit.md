# Multi-Hazard Prediction System

## Overview

This is a unified AI-powered multi-hazard prediction system that forecasts droughts, floods, and earthquakes using satellite and geophysical data. The system combines machine learning models with real-time data processing to provide early warning capabilities for natural disasters. Built with Streamlit for the web interface, it offers interactive dashboards, geospatial visualizations, and alert management features for disaster preparedness and response.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page navigation
- **Layout**: Wide layout with expandable sidebar for controls and parameters
- **Navigation**: Page-based routing system supporting Dashboard Overview, Data Processing, and individual hazard prediction pages (Drought, Flood, Earthquake)
- **Session State**: Persistent component initialization across page loads for data processors, predictors, and alert systems

### Data Processing Layer
- **SatelliteDataProcessor**: Handles multi-source satellite data including Sentinel-1/2, Landsat 8/9, MODIS, and USGS seismic data
- **Feature Engineering**: Automated calculation of vegetation indices (NDVI), water indices (NDWI), and other Earth observation metrics
- **MockDataGenerator**: Provides realistic synthetic data for demonstration and testing purposes
- **Data Harmonization**: Spatial and temporal resolution standardization across different satellite sources

### Machine Learning Architecture
- **MultiHazardPredictor**: Unified prediction engine combining multiple ML approaches
- **Model Types**: Ensemble methods using Random Forest classifiers/regressors and Multi-Layer Perceptron neural networks
- **Feature Preparation**: Standardized preprocessing pipeline with feature scaling and engineering
- **Multi-Branch Design**: Separate specialized models for drought, flood, and earthquake prediction with shared feature extraction

### Visualization System
- **Interactive Maps**: Folium-based geospatial visualization with multiple tile layers and hazard-specific overlays
- **Plotly Integration**: Time series charts, risk assessment graphs, and interactive data exploration tools
- **Color-Coded Risk Levels**: Visual distinction between Low, Medium, High, and Critical risk levels for each hazard type
- **Region-Specific Views**: Customizable map views for predefined high-risk regions and custom locations

### Alert Management
- **AlertSystem**: Configurable threshold-based alert generation with severity levels
- **Risk Assessment**: Dynamic evaluation combining prediction confidence and risk scores
- **Historical Tracking**: Alert history management and active alert monitoring
- **Priority Weighting**: Severity-based alert prioritization system

### Model Configuration
- **Drought Model**: Random Forest with 100 estimators, max depth 10
- **Flood Model**: MLP with hidden layers (100, 50), 500 max iterations
- **Earthquake Model**: Random Forest with 150 estimators, max depth 15
- **Standardization**: Feature scaling using sklearn StandardScaler for consistent input preprocessing

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for interactive dashboard
- **Pandas/Numpy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning models and preprocessing utilities
- **Plotly**: Interactive data visualization and charting

### Geospatial & Mapping
- **Folium**: Interactive map generation with multiple tile layer support
- **OpenStreetMap**: Base map tiles for geographic visualization
- **Stamen**: Additional terrain and toner map tile options

### Satellite Data Sources (Future Integration)
- **Sentinel-1**: SAR (Synthetic Aperture Radar) data for surface deformation
- **Sentinel-2**: Optical imagery for vegetation and land cover analysis
- **Landsat 8/9**: Long-term Earth observation data
- **MODIS Terra/Aqua**: Daily global coverage for environmental monitoring
- **USGS Seismic Network**: Real-time earthquake and seismic activity data

### Development Tools
- **Joblib**: Model serialization and parallel processing
- **Logging**: Application monitoring and error tracking
- **JSON**: Configuration and data exchange format
- **Datetime**: Temporal data handling and time series processing

### Deployment Considerations
- **Cloud Platforms**: Designed for AWS or Google Earth Engine deployment
- **Real-time Processing**: Architecture supports streaming data ingestion
- **Scalability**: Modular design enables horizontal scaling of prediction components