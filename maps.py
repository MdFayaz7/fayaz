import folium
import pandas as pd
import numpy as np
from folium import plugins
from typing import List, Dict, Tuple
import json

def create_hazard_map(hazard_data: List[Dict], center_lat: float = 20.0, 
                     center_lon: float = 0.0, zoom_start: int = 2) -> folium.Map:
    """
    Create an interactive map showing global hazard predictions.
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add additional tile layers with proper attribution
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        name='Stamen Terrain'
    ).add_to(m)
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        name='Stamen Toner'
    ).add_to(m)
    
    # Color schemes for different hazard types
    hazard_colors = {
        'drought': {'Low': '#FFF3E0', 'Medium': '#FFB74D', 'High': '#FF8F00', 'Critical': '#E65100'},
        'flood': {'Low': '#E3F2FD', 'Medium': '#42A5F5', 'High': '#1976D2', 'Critical': '#0D47A1'},
        'earthquake': {'Low': '#FCE4EC', 'Medium': '#F06292', 'High': '#E91E63', 'Critical': '#AD1457'}
    }
    
    # Create feature groups for each hazard type
    drought_group = folium.FeatureGroup(name='Drought Risk')
    flood_group = folium.FeatureGroup(name='Flood Risk')
    earthquake_group = folium.FeatureGroup(name='Earthquake Risk')
    
    # Add markers for each hazard prediction
    for data_point in hazard_data:
        lat = data_point['latitude']
        lon = data_point['longitude']
        hazard_type = data_point['hazard_type']
        risk_level = data_point['risk_level']
        region = data_point['region']
        confidence = data_point['confidence']
        risk_score = data_point['risk_score']
        
        # Select color based on hazard type and risk level
        color = hazard_colors[hazard_type][risk_level]
        
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h4>{region}</h4>
            <p><strong>Hazard:</strong> {hazard_type.title()}</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p><strong>Risk Score:</strong> {risk_score:.2f}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Last Updated:</strong> {data_point['last_updated'].strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """
        
        # Create marker
        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=8 + (risk_score * 10),  # Size based on risk score
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=f"{region}: {hazard_type.title()} - {risk_level}"
        )
        
        # Add to appropriate group
        if hazard_type == 'drought':
            drought_group.add_child(marker)
        elif hazard_type == 'flood':
            flood_group.add_child(marker)
        elif hazard_type == 'earthquake':
            earthquake_group.add_child(marker)
    
    # Add feature groups to map
    drought_group.add_to(m)
    flood_group.add_to(m)
    earthquake_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a legend
    legend_html = create_legend_html()
    legend_element = folium.Element(legend_html)
    m.get_root().html.add_child(legend_element)
    
    return m

def create_region_specific_map(region_data: Dict, center_lat: float, 
                             center_lon: float, zoom_start: int = 8) -> folium.Map:
    """
    Create a detailed map for a specific region with time series visualization.
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add region boundary if available
    if 'boundary' in region_data:
        folium.GeoJson(
            region_data['boundary'],
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.1
            }
        ).add_to(m)
    
    # Add monitoring stations
    if 'monitoring_stations' in region_data:
        for station in region_data['monitoring_stations']:
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=f"Station: {station['name']}<br>Type: {station['type']}",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
    
    # Add historical event markers
    if 'historical_events' in region_data:
        for event in region_data['historical_events']:
            color_map = {'drought': 'orange', 'flood': 'blue', 'earthquake': 'red'}
            color = color_map.get(event['type'], 'gray')
            
            folium.CircleMarker(
                location=[event['lat'], event['lon']],
                radius=5,
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                popup=f"Event: {event['type']}<br>Date: {event['date']}<br>Magnitude: {event.get('magnitude', 'N/A')}",
                tooltip=f"{event['type']} - {event['date']}"
            ).add_to(m)
    
    # Add heatmap for risk intensity
    if 'risk_grid' in region_data:
        heat_data = []
        for point in region_data['risk_grid']:
            heat_data.append([point['lat'], point['lon'], point['risk_score']])
        
        plugins.HeatMap(heat_data, radius=20, blur=15, max_zoom=1).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def create_temporal_heatmap(time_series_data: pd.DataFrame, 
                          hazard_type: str) -> folium.Map:
    """
    Create a heatmap showing temporal evolution of hazard risk.
    """
    # Calculate center of data
    center_lat = time_series_data['latitude'].mean()
    center_lon = time_series_data['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Group data by time periods
    time_periods = time_series_data['date'].dt.to_period('M').unique()
    
    # Create time series heatmap data
    heat_data = []
    for period in time_periods:
        period_data = time_series_data[time_series_data['date'].dt.to_period('M') == period]
        heat_points = []
        
        for _, row in period_data.iterrows():
            heat_points.append([row['latitude'], row['longitude'], row['risk_score']])
        
        heat_data.append(heat_points)
    
    # Add time series heatmap
    hm = plugins.HeatMapWithTime(
        heat_data,
        index=[str(period) for period in time_periods],
        auto_play=True,
        max_opacity=0.8,
        radius=25,
        blur=15
    )
    
    hm.add_to(m)
    
    return m

def create_legend_html() -> str:
    """
    Create HTML legend for the hazard map.
    """
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 160px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Hazard Risk Levels</h4>
    
    <p><i class="fa fa-circle" style="color:#E65100"></i> Drought - Critical</p>
    <p><i class="fa fa-circle" style="color:#FF8F00"></i> Drought - High</p>
    <p><i class="fa fa-circle" style="color:#FFB74D"></i> Drought - Medium</p>
    <p><i class="fa fa-circle" style="color:#FFF3E0"></i> Drought - Low</p>
    
    <p><i class="fa fa-circle" style="color:#0D47A1"></i> Flood - Critical</p>
    <p><i class="fa fa-circle" style="color:#1976D2"></i> Flood - High</p>
    
    <p><i class="fa fa-circle" style="color:#AD1457"></i> Earthquake - Critical</p>
    <p><i class="fa fa-circle" style="color:#E91E63"></i> Earthquake - High</p>
    </div>
    '''
    return legend_html

def create_prediction_confidence_map(predictions: List[Dict]) -> folium.Map:
    """
    Create a map showing prediction confidence levels.
    """
    # Calculate center
    lats = [p['latitude'] for p in predictions]
    lons = [p['longitude'] for p in predictions]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Color map for confidence levels
    def confidence_color(confidence):
        if confidence >= 0.9:
            return 'green'
        elif confidence >= 0.7:
            return 'orange'
        else:
            return 'red'
    
    # Add confidence markers
    for pred in predictions:
        confidence = pred['confidence']
        color = confidence_color(confidence)
        
        folium.CircleMarker(
            location=[pred['latitude'], pred['longitude']],
            radius=10,
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7,
            popup=f"Confidence: {confidence:.1%}<br>Hazard: {pred['hazard_type']}<br>Risk: {pred['risk_level']}",
            tooltip=f"Confidence: {confidence:.1%}"
        ).add_to(m)
    
    # Add confidence legend
    confidence_legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 100px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Confidence Levels</h4>
    <p><i class="fa fa-circle" style="color:green"></i> High (≥90%)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium (70-90%)</p>
    <p><i class="fa fa-circle" style="color:red"></i> Low (<70%)</p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(confidence_legend_html))
    
    return m

def add_satellite_overlay(map_obj: folium.Map, satellite_data: Dict) -> folium.Map:
    """
    Add satellite imagery overlay to existing map.
    """
    # This would integrate with actual satellite imagery services
    # For demonstration, we'll add a mock overlay
    
    # Add NDVI overlay (mock) - using alternative approach since raster_layers might not be available
    if 'ndvi_bounds' in satellite_data:
        bounds = satellite_data['ndvi_bounds']
        # Alternative: Add as a marker layer instead of image overlay
        folium.Marker(
            location=[(bounds[0][0] + bounds[1][0])/2, (bounds[0][1] + bounds[1][1])/2],
            popup='NDVI Data Available',
            icon=folium.Icon(color='green', icon='leaf')
        ).add_to(map_obj)
    
    # Add soil moisture overlay (mock)
    if 'moisture_bounds' in satellite_data:
        bounds = satellite_data['moisture_bounds']
        # Alternative: Add as a marker layer instead of image overlay
        folium.Marker(
            location=[(bounds[0][0] + bounds[1][0])/2, (bounds[0][1] + bounds[1][1])/2],
            popup='Soil Moisture Data Available',
            icon=folium.Icon(color='blue', icon='tint')
        ).add_to(map_obj)
    
    return map_obj
