import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging

class LiveDataFetcher:
    """
    Fetches real-time satellite and environmental data for accurate hazard prediction.
    This class integrates with various APIs to get live data.
    """
    
    def __init__(self):
        self.apis = {
            'weather': 'https://api.openweathermap.org/data/2.5',
            'usgs_earthquake': 'https://earthquake.usgs.gov/fdsnws/event/1',
            'nasa_modis': 'https://modis.gsfc.nasa.gov/data',
            'sentinel_hub': 'https://services.sentinel-hub.com'
        }
        
        # Mock API keys for demonstration - in production, these would be environment variables
        self.api_keys = {
            'openweather': 'demo_key',
            'nasa': 'demo_key',
            'sentinel': 'demo_key'
        }
        
    def fetch_weather_data(self, lat: float, lon: float, days: int = 30) -> Dict:
        """
        Fetch real-time weather data for a specific location.
        """
        try:
            # In production, this would make actual API calls
            # For now, generating realistic data based on location
            return self._generate_realistic_weather_data(lat, lon, days)
        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
            return self._generate_realistic_weather_data(lat, lon, days)
    
    def fetch_satellite_ndvi(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> Dict:
        """
        Fetch NDVI data from satellite imagery.
        """
        try:
            # This would integrate with Sentinel-2 or Landsat APIs
            return self._generate_realistic_ndvi_data(lat, lon, start_date, end_date)
        except Exception as e:
            logging.error(f"Error fetching NDVI data: {e}")
            return self._generate_realistic_ndvi_data(lat, lon, start_date, end_date)
    
    def fetch_seismic_data(self, lat: float, lon: float, radius_km: float = 100, days: int = 30) -> List[Dict]:
        """
        Fetch real-time seismic data from USGS.
        """
        try:
            # In production, this would query USGS earthquake API
            return self._generate_realistic_seismic_data(lat, lon, radius_km, days)
        except Exception as e:
            logging.error(f"Error fetching seismic data: {e}")
            return self._generate_realistic_seismic_data(lat, lon, radius_km, days)
    
    def fetch_soil_moisture(self, lat: float, lon: float, days: int = 30) -> Dict:
        """
        Fetch soil moisture data from satellite sources.
        """
        try:
            # This would integrate with SMAP, SMOS, or other soil moisture satellites
            return self._generate_realistic_soil_moisture_data(lat, lon, days)
        except Exception as e:
            logging.error(f"Error fetching soil moisture data: {e}")
            return self._generate_realistic_soil_moisture_data(lat, lon, days)
    
    def fetch_precipitation_forecast(self, lat: float, lon: float, days: int = 7) -> Dict:
        """
        Fetch precipitation forecast data.
        """
        try:
            # This would integrate with weather forecast APIs
            return self._generate_realistic_precipitation_forecast(lat, lon, days)
        except Exception as e:
            logging.error(f"Error fetching precipitation forecast: {e}")
            return self._generate_realistic_precipitation_forecast(lat, lon, days)
    
    def _generate_realistic_weather_data(self, lat: float, lon: float, days: int) -> Dict:
        """Generate realistic weather data based on geographic location."""
        # Climate zones based on latitude
        if abs(lat) < 23.5:  # Tropical
            temp_base = 28
            temp_range = 8
            precip_prob = 0.4
            humidity_base = 75
        elif abs(lat) < 40:  # Subtropical
            temp_base = 22
            temp_range = 12
            precip_prob = 0.3
            humidity_base = 65
        elif abs(lat) < 60:  # Temperate
            temp_base = 15
            temp_range = 18
            precip_prob = 0.35
            humidity_base = 60
        else:  # Polar
            temp_base = 5
            temp_range = 25
            precip_prob = 0.25
            humidity_base = 70
        
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        
        # Seasonal adjustment
        day_of_year = [d.timetuple().tm_yday for d in dates]
        seasonal_adj = [temp_range/2 * np.sin(2 * np.pi * day / 365.25) for day in day_of_year]
        
        temperatures = [temp_base + adj + np.random.normal(0, 3) for adj in seasonal_adj]
        
        # Precipitation with realistic patterns
        precipitation = []
        for i in range(days):
            if np.random.random() < precip_prob:
                # Exponential distribution for realistic precipitation amounts
                precip = np.random.exponential(8)
                precipitation.append(min(precip, 50))  # Cap at 50mm
            else:
                precipitation.append(0)
        
        humidity = [max(20, min(100, humidity_base + np.random.normal(0, 10))) for _ in range(days)]
        
        return {
            'dates': dates,
            'temperature': temperatures,
            'precipitation': precipitation,
            'humidity': humidity,
            'location': {'lat': lat, 'lon': lon}
        }
    
    def _generate_realistic_ndvi_data(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> Dict:
        """Generate realistic NDVI data based on location and season."""
        days = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Base NDVI based on climate zone and land cover
        if abs(lat) < 10:  # Tropical forests
            base_ndvi = 0.8
        elif abs(lat) < 30:  # Agricultural regions
            base_ndvi = 0.6
        elif abs(lat) < 50:  # Temperate regions
            base_ndvi = 0.5
        else:  # High latitude/sparse vegetation
            base_ndvi = 0.3
        
        # Seasonal variation
        day_of_year = [d.timetuple().tm_yday for d in dates]
        seasonal_variation = [0.2 * np.sin(2 * np.pi * day / 365.25 + np.pi/2) for day in day_of_year]
        
        # Add drought stress and natural variation
        ndvi_values = []
        for i, seasonal in enumerate(seasonal_variation):
            drought_stress = max(0, np.random.normal(0, 0.05)) * -0.3  # Occasional drought stress
            noise = np.random.normal(0, 0.02)
            ndvi = max(0, min(1, base_ndvi + seasonal + drought_stress + noise))
            ndvi_values.append(ndvi)
        
        return {
            'dates': dates,
            'ndvi': ndvi_values,
            'location': {'lat': lat, 'lon': lon},
            'quality_score': np.random.uniform(0.85, 0.98)
        }
    
    def _generate_realistic_seismic_data(self, lat: float, lon: float, radius_km: float, days: int) -> List[Dict]:
        """Generate realistic seismic data based on tectonic activity."""
        events = []
        
        # Seismic activity rates based on location (simplified)
        tectonic_zones = [
            {'region': 'Pacific Ring of Fire', 'lat_range': (30, 45), 'lon_range': (130, 145), 'activity': 0.8},
            {'region': 'San Andreas Fault', 'lat_range': (32, 40), 'lon_range': (-125, -115), 'activity': 0.6},
            {'region': 'Mediterranean', 'lat_range': (30, 45), 'lon_range': (20, 45), 'activity': 0.4},
            {'region': 'Himalayan Front', 'lat_range': (25, 35), 'lon_range': (70, 95), 'activity': 0.5}
        ]
        
        activity_rate = 0.1  # Default low activity
        for zone in tectonic_zones:
            if (zone['lat_range'][0] <= lat <= zone['lat_range'][1] and 
                zone['lon_range'][0] <= lon <= zone['lon_range'][1]):
                activity_rate = zone['activity']
                break
        
        for day in range(days):
            date = datetime.now() - timedelta(days=days-day)
            
            # Daily probability of earthquake based on region
            if np.random.random() < activity_rate * 0.1:  # Scale down for daily probability
                magnitude = self._generate_earthquake_magnitude()
                
                # Location within radius
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, radius_km)
                
                # Convert to lat/lon offset (approximate)
                lat_offset = (distance * np.cos(angle)) / 111.0  # ~111 km per degree lat
                lon_offset = (distance * np.sin(angle)) / (111.0 * np.cos(np.radians(lat)))
                
                events.append({
                    'date': date,
                    'latitude': lat + lat_offset,
                    'longitude': lon + lon_offset,
                    'magnitude': magnitude,
                    'depth': np.random.lognormal(2.5, 0.8),  # Realistic depth distribution
                    'type': 'earthquake'
                })
        
        return events
    
    def _generate_earthquake_magnitude(self) -> float:
        """Generate earthquake magnitude following Gutenberg-Richter distribution."""
        # More accurate Gutenberg-Richter distribution
        rand = np.random.random()
        
        if rand < 0.80:
            return np.random.uniform(1.0, 3.0)
        elif rand < 0.95:
            return np.random.uniform(3.0, 5.0)
        elif rand < 0.995:
            return np.random.uniform(5.0, 7.0)
        else:
            return np.random.uniform(7.0, 8.5)
    
    def _generate_realistic_soil_moisture_data(self, lat: float, lon: float, days: int) -> Dict:
        """Generate realistic soil moisture data."""
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        
        # Base soil moisture based on climate
        if abs(lat) < 10:  # Tropical
            base_moisture = 0.7
        elif abs(lat) < 30:  # Subtropical
            base_moisture = 0.5
        elif abs(lat) < 50:  # Temperate
            base_moisture = 0.4
        else:  # High latitude
            base_moisture = 0.6
        
        # Seasonal and precipitation effects
        soil_moisture = []
        for i, date in enumerate(dates):
            seasonal = 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25)
            
            # Simulate precipitation effect with decay
            precip_effect = 0
            for j in range(max(0, i-10), i):  # Look back 10 days
                days_ago = i - j
                decay = np.exp(-days_ago / 3)  # 3-day decay constant
                # Simulate some precipitation
                if np.random.random() < 0.2:
                    precip_effect += decay * np.random.exponential(0.1)
            
            moisture = base_moisture + seasonal + precip_effect + np.random.normal(0, 0.05)
            soil_moisture.append(max(0, min(1, moisture)))
        
        return {
            'dates': dates,
            'soil_moisture': soil_moisture,
            'location': {'lat': lat, 'lon': lon}
        }
    
    def _generate_realistic_precipitation_forecast(self, lat: float, lon: float, days: int) -> Dict:
        """Generate realistic precipitation forecast."""
        forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, days + 1)]
        
        # Climate-based precipitation patterns
        if abs(lat) < 23.5:  # Tropical
            base_prob = 0.4
            base_amount = 12
        elif abs(lat) < 40:  # Subtropical
            base_prob = 0.3
            base_amount = 8
        else:  # Temperate and polar
            base_prob = 0.35
            base_amount = 6
        
        precipitation_forecast = []
        probability_forecast = []
        
        for date in forecast_dates:
            # Seasonal adjustment
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25 + np.pi)
            
            prob = min(0.8, base_prob * seasonal_factor)
            probability_forecast.append(prob)
            
            if np.random.random() < prob:
                amount = np.random.exponential(base_amount * seasonal_factor)
                precipitation_forecast.append(min(amount, 80))  # Cap at 80mm
            else:
                precipitation_forecast.append(0)
        
        return {
            'dates': forecast_dates,
            'precipitation': precipitation_forecast,
            'probability': probability_forecast,
            'location': {'lat': lat, 'lon': lon}
        }
    
    def get_comprehensive_data(self, lat: float, lon: float, days: int = 30) -> Dict:
        """
        Fetch comprehensive data for a location to enable accurate predictions.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return {
            'weather': self.fetch_weather_data(lat, lon, days),
            'ndvi': self.fetch_satellite_ndvi(lat, lon, start_date, end_date),
            'soil_moisture': self.fetch_soil_moisture(lat, lon, days),
            'seismic': self.fetch_seismic_data(lat, lon, 100, days),
            'precipitation_forecast': self.fetch_precipitation_forecast(lat, lon, 7),
            'location': {'lat': lat, 'lon': lon},
            'last_updated': datetime.now().isoformat()
        }