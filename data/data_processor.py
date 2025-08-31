import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Tuple, Optional
import logging

class SatelliteDataProcessor:
    """
    Processes satellite and geophysical data for multi-hazard prediction.
    Handles data from multiple sources including Sentinel, Landsat, MODIS, and USGS.
    """
    
    def __init__(self):
        self.supported_sources = [
            'sentinel1', 'sentinel2', 'landsat8', 'landsat9', 
            'modis', 'usgs_seismic', 'weather_stations'
        ]
        self.processing_cache = {}
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        NDVI = (NIR - Red) / (NIR + Red)
        """
        # Avoid division by zero
        denominator = nir_band + red_band
        denominator = np.where(denominator == 0, 1e-8, denominator)
        
        ndvi = (nir_band - red_band) / denominator
        
        # Clip values to valid NDVI range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def calculate_ndwi(self, green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index (NDWI).
        NDWI = (Green - NIR) / (Green + NIR)
        """
        denominator = green_band + nir_band
        denominator = np.where(denominator == 0, 1e-8, denominator)
        
        ndwi = (green_band - nir_band) / denominator
        ndwi = np.clip(ndwi, -1, 1)
        
        return ndwi
    
    def calculate_lst(self, thermal_band: np.ndarray, emissivity: float = 0.95) -> np.ndarray:
        """
        Calculate Land Surface Temperature (LST) from thermal infrared data.
        Simplified Planck's law inversion.
        """
        # Constants for Landsat 8 Band 10
        K1 = 774.89  # W/(m2 sr μm)
        K2 = 1321.08  # K
        
        # Convert DN to radiance (simplified)
        radiance = thermal_band * 0.0003342 + 0.1
        
        # Calculate brightness temperature
        bt = K2 / np.log((K1 / radiance) + 1)
        
        # Convert to surface temperature using emissivity
        lst = bt / (1 + (0.00115 * bt / 1.4388) * np.log(emissivity))
        
        return lst - 273.15  # Convert to Celsius
    
    def extract_soil_moisture(self, sar_data: np.ndarray, incidence_angle: float) -> np.ndarray:
        """
        Extract soil moisture from SAR backscatter data.
        Uses empirical relationship between backscatter and soil moisture.
        """
        # Simplified soil moisture retrieval
        # Normalize backscatter coefficients
        sigma0_db = 10 * np.log10(np.abs(sar_data) + 1e-10)
        
        # Apply incidence angle correction
        sigma0_corrected = sigma0_db + 0.1 * (incidence_angle - 30)
        
        # Empirical soil moisture relationship
        # This is a simplified model - real implementations use more complex algorithms
        soil_moisture = (sigma0_corrected + 20) / 25
        soil_moisture = np.clip(soil_moisture, 0, 1)
        
        return soil_moisture
    
    def detect_water_bodies(self, ndwi: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Detect water bodies using NDWI threshold.
        """
        water_mask = ndwi > threshold
        return water_mask.astype(np.uint8)
    
    def calculate_vegetation_health(self, ndvi_current: np.ndarray, 
                                  ndvi_historical: np.ndarray) -> Dict:
        """
        Calculate vegetation health indicators.
        """
        # Vegetation Condition Index (VCI)
        ndvi_min = np.percentile(ndvi_historical, 5)
        ndvi_max = np.percentile(ndvi_historical, 95)
        
        vci = (ndvi_current - ndvi_min) / (ndvi_max - ndvi_min) * 100
        vci = np.clip(vci, 0, 100)
        
        # Vegetation anomaly
        ndvi_mean = np.mean(ndvi_historical)
        vegetation_anomaly = (ndvi_current - ndvi_mean) / ndvi_mean * 100
        
        return {
            'vci': vci,
            'vegetation_anomaly': vegetation_anomaly,
            'mean_ndvi': np.mean(ndvi_current),
            'std_ndvi': np.std(ndvi_current)
        }
    
    def analyze_precipitation_patterns(self, precipitation_data: List[float], 
                                     timestamps: List[datetime]) -> Dict:
        """
        Analyze precipitation patterns for drought and flood risk.
        """
        precip_array = np.array(precipitation_data)
        
        # Calculate various precipitation indices
        total_precip = np.sum(precip_array)
        mean_daily_precip = np.mean(precip_array)
        max_daily_precip = np.max(precip_array)
        
        # Consecutive dry days
        dry_days = precip_array < 1.0  # Less than 1mm considered dry
        consecutive_dry = self._calculate_consecutive_days(dry_days)
        
        # Heavy precipitation events
        heavy_precip_threshold = np.percentile(precip_array[precip_array > 0], 90)
        heavy_precip_days = np.sum(precip_array > heavy_precip_threshold)
        
        # Standardized Precipitation Index (SPI) - simplified
        precip_std = np.std(precip_array)
        precip_mean = np.mean(precip_array)
        spi = (total_precip - precip_mean) / precip_std if precip_std > 0 else 0
        
        return {
            'total_precipitation': total_precip,
            'mean_daily_precipitation': mean_daily_precip,
            'max_daily_precipitation': max_daily_precip,
            'consecutive_dry_days': consecutive_dry,
            'heavy_precipitation_days': heavy_precip_days,
            'spi': spi,
            'drought_risk': 'High' if consecutive_dry > 30 or spi < -1.5 else 'Low',
            'flood_risk': 'High' if heavy_precip_days > 3 or max_daily_precip > 50 else 'Low'
        }
    
    def process_seismic_data(self, seismic_events: List[Dict]) -> Dict:
        """
        Process seismic data for earthquake prediction.
        """
        if not seismic_events:
            return {'activity_level': 'Low', 'max_magnitude': 0, 'event_count': 0}
        
        magnitudes = [event.get('magnitude', 0) for event in seismic_events]
        times = [event.get('time') for event in seismic_events]
        
        # Calculate seismic activity metrics
        max_magnitude = max(magnitudes) if magnitudes else 0
        mean_magnitude = np.mean(magnitudes) if magnitudes else 0
        event_count = len(seismic_events)
        
        # Gutenberg-Richter relationship analysis
        magnitude_bins = np.arange(1, 8, 0.5)
        hist, _ = np.histogram(magnitudes, bins=magnitude_bins)
        
        # Calculate b-value (simplified)
        log_counts = np.log10(hist + 1)
        b_value = -np.polyfit(magnitude_bins[:-1], log_counts, 1)[0]
        
        # Activity level assessment
        if max_magnitude > 5.0 or event_count > 50:
            activity_level = 'High'
        elif max_magnitude > 3.0 or event_count > 20:
            activity_level = 'Medium'
        else:
            activity_level = 'Low'
        
        return {
            'activity_level': activity_level,
            'max_magnitude': max_magnitude,
            'mean_magnitude': mean_magnitude,
            'event_count': event_count,
            'b_value': b_value,
            'earthquake_risk': 'High' if activity_level == 'High' or b_value < 0.8 else 'Low'
        }
    
    def _calculate_consecutive_days(self, boolean_array: np.ndarray) -> int:
        """
        Calculate maximum consecutive True values in boolean array.
        """
        max_consecutive = 0
        current_consecutive = 0
        
        for value in boolean_array:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def harmonize_spatial_resolution(self, datasets: Dict[str, np.ndarray], 
                                   target_resolution: float) -> Dict[str, np.ndarray]:
        """
        Harmonize spatial resolution across different satellite datasets.
        """
        harmonized_data = {}
        
        for dataset_name, data in datasets.items():
            # This would use proper resampling techniques in a real implementation
            # For now, using simple interpolation
            if data.shape != (100, 100):  # Target shape
                # Simplified resampling
                from scipy.ndimage import zoom
                zoom_factors = (100 / data.shape[0], 100 / data.shape[1])
                harmonized_data[dataset_name] = zoom(data, zoom_factors)
            else:
                harmonized_data[dataset_name] = data
        
        return harmonized_data
    
    def temporal_aggregation(self, time_series_data: Dict[str, List], 
                           aggregation_period: str = 'weekly') -> Dict[str, np.ndarray]:
        """
        Aggregate temporal data to specified periods.
        """
        aggregated_data = {}
        
        for metric, values in time_series_data.items():
            if aggregation_period == 'weekly':
                # Group by weeks and calculate mean
                values_array = np.array(values)
                weeks = len(values_array) // 7
                aggregated = []
                for i in range(weeks):
                    week_data = values_array[i*7:(i+1)*7]
                    aggregated.append(np.mean(week_data))
                aggregated_data[metric] = np.array(aggregated)
            elif aggregation_period == 'monthly':
                # Group by months and calculate mean
                values_array = np.array(values)
                months = len(values_array) // 30
                aggregated = []
                for i in range(months):
                    month_data = values_array[i*30:(i+1)*30]
                    aggregated.append(np.mean(month_data))
                aggregated_data[metric] = np.array(aggregated)
            else:
                aggregated_data[metric] = np.array(values)
        
        return aggregated_data
    
    def process_mock_data(self, spatial_resolution: int, temporal_window: int, 
                         region_size: int) -> Dict:
        """
        Process mock satellite data for demonstration purposes.
        """
        # Generate mock time series data
        dates = [datetime.now() - timedelta(days=i) for i in range(temporal_window, 0, -1)]
        
        # Mock NDVI time series
        ndvi_base = 0.6
        ndvi_trend = -0.001 * np.arange(temporal_window)  # Slight decreasing trend
        ndvi_noise = np.random.normal(0, 0.05, temporal_window)
        ndvi_values = ndvi_base + ndvi_trend + ndvi_noise
        ndvi_values = np.clip(ndvi_values, 0, 1)
        
        # Mock soil moisture time series
        moisture_base = 0.3
        moisture_seasonal = 0.1 * np.sin(2 * np.pi * np.arange(temporal_window) / 365)
        moisture_noise = np.random.normal(0, 0.03, temporal_window)
        moisture_values = moisture_base + moisture_seasonal + moisture_noise
        moisture_values = np.clip(moisture_values, 0, 1)
        
        # Create DataFrames
        ndvi_df = pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values
        })
        
        moisture_df = pd.DataFrame({
            'date': dates,
            'moisture': moisture_values
        })
        
        # Extract features
        features_df = pd.DataFrame({
            'Feature': [
                'Mean NDVI', 'NDVI Trend', 'NDVI Std',
                'Mean Soil Moisture', 'Moisture Trend', 'Moisture Std',
                'Vegetation Health Index', 'Drought Risk Score',
                'Spatial Resolution (m)', 'Temporal Window (days)'
            ],
            'Value': [
                f"{np.mean(ndvi_values):.3f}",
                f"{np.polyfit(range(len(ndvi_values)), ndvi_values, 1)[0]:.6f}",
                f"{np.std(ndvi_values):.3f}",
                f"{np.mean(moisture_values):.3f}",
                f"{np.polyfit(range(len(moisture_values)), moisture_values, 1)[0]:.6f}",
                f"{np.std(moisture_values):.3f}",
                f"{np.mean(ndvi_values) * np.mean(moisture_values):.3f}",
                f"{1 - np.mean(moisture_values):.3f}",
                f"{spatial_resolution}",
                f"{temporal_window}"
            ]
        })
        
        return {
            'ndvi_timeseries': ndvi_df,
            'soil_moisture': moisture_df,
            'features': features_df,
            'processing_metadata': {
                'spatial_resolution': spatial_resolution,
                'temporal_window': temporal_window,
                'region_size': region_size,
                'processing_date': datetime.now().isoformat()
            }
        }
