import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, VotingClassifier, StackingClassifier)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, VotingRegressor, StackingRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from data.live_data_fetcher import LiveDataFetcher
import streamlit as st # Import streamlit for st.write and st.success
from sklearn.exceptions import NotFittedError # Import NotFittedError

class EnhancedHazardPredictor:
    """
    Enhanced multi-hazard prediction system with improved accuracy and real-time capabilities.
    Uses ensemble methods and advanced feature engineering for better predictions.
    """
    
    def __init__(self):
        self.drought_model = None
        self.flood_model = None
        self.earthquake_model = None
        self.feature_scaler = StandardScaler()
        self.live_data_fetcher = LiveDataFetcher()
        self.is_trained = False
        
        # Enhanced model configurations
        self.model_configs = {
            'drought': {
                'rf': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 
                       'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42},
                'gb': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 8, 
                       'min_samples_split': 10, 'subsample': 0.8, 'random_state': 42},
                'et': {'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 3, 
                       'random_state': 42}
            },
            'flood': {
                'mlp': {'hidden_layer_sizes': (200, 100, 50), 'activation': 'relu', 
                        'solver': 'adam', 'alpha': 0.0001, 'learning_rate': 'adaptive',
                        'max_iter': 2000, 'early_stopping': True, 'random_state': 42},
                'rf': {'n_estimators': 250, 'max_depth': 18, 'min_samples_split': 4,
                       'max_features': 'sqrt', 'random_state': 42},
                'gb': {'n_estimators': 180, 'learning_rate': 0.12, 'max_depth': 10,
                       'subsample': 0.85, 'random_state': 42}
            },
            'earthquake': {
                'gb': {'n_estimators': 300, 'learning_rate': 0.08, 'max_depth': 12,
                       'min_samples_split': 8, 'subsample': 0.9, 'random_state': 42},
                'rf': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 3,
                       'max_features': 'sqrt', 'random_state': 42},
                'svr': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
            }
        }
        
        self._build_ensemble_models()

    def train(self, X_drought, y_drought, X_flood, y_flood, X_earthquake, y_earthquake):
        """Train the hazard prediction models."""
        st.write("Re-building ensemble models...")
        self._build_ensemble_models() # Ensure fresh models with no prior feature count are used

        st.write("Fitting feature scaler...")
        # Always fit the scaler with the current combined training data
        self.feature_scaler.fit(pd.concat([X_drought, X_flood, X_earthquake]))

        st.write("Training drought model...") # Use st.write for Streamlit context
        self.drought_model.fit(self.feature_scaler.transform(X_drought), y_drought)
        st.write("Training flood model...")
        self.flood_model.fit(self.feature_scaler.transform(X_flood), y_flood)
        st.write("Training earthquake model...")
        self.earthquake_model.fit(self.feature_scaler.transform(X_earthquake), y_earthquake)
        self.is_trained = True
        st.success("Models trained successfully!")

    def load_models(self):
        """Load pre-trained models and scaler."""
        try:
            self.drought_model = joblib.load('./models/drought_model.joblib')
            self.flood_model = joblib.load('./models/flood_model.joblib')
            self.earthquake_model = joblib.load('./models/earthquake_model.joblib')
            self.feature_scaler = joblib.load('./models/feature_scaler.joblib')
            self.is_trained = True
            logging.info("Models and scaler loaded successfully.")
        except FileNotFoundError:
            logging.warning("Pre-trained models or scaler not found. Models will need to be trained.")
        except Exception as e:
            logging.error(f"Error loading models: {e}")

    def save_models(self):
        """Save trained models and scaler."""
        if self.is_trained:
            joblib.dump(self.drought_model, './models/drought_model.joblib')
            joblib.dump(self.flood_model, './models/flood_model.joblib')
            joblib.dump(self.earthquake_model, './models/earthquake_model.joblib')
            joblib.dump(self.feature_scaler, './models/feature_scaler.joblib')
            logging.info("Models and scaler saved successfully.")
        else:
            logging.warning("Models not trained, cannot save.")
    
    def _build_ensemble_models(self):
        """Build ensemble models for each hazard type."""
        
        # Drought ensemble (classification)
        drought_rf = RandomForestClassifier(**self.model_configs['drought']['rf'])
        drought_gb = GradientBoostingClassifier(**self.model_configs['drought']['gb'])
        drought_et = ExtraTreesClassifier(**self.model_configs['drought']['et'])
        
        self.drought_model = VotingClassifier(
            estimators=[('rf', drought_rf), ('gb', drought_gb), ('et', drought_et)],
            voting='soft'
        )
        
        # Flood ensemble (classification)
        flood_mlp = MLPClassifier(**self.model_configs['flood']['mlp'])
        flood_rf = RandomForestClassifier(**self.model_configs['flood']['rf'])
        flood_gb = GradientBoostingClassifier(**self.model_configs['flood']['gb'])
        
        self.flood_model = VotingClassifier(
            estimators=[('mlp', flood_mlp), ('rf', flood_rf), ('gb', flood_gb)],
            voting='soft'
        )
        
        # Earthquake ensemble (regression)
        earthquake_gb = GradientBoostingRegressor(**self.model_configs['earthquake']['gb'])
        earthquake_rf = RandomForestRegressor(**self.model_configs['earthquake']['rf'])
        earthquake_svr = SVR(**self.model_configs['earthquake']['svr'])
        
        self.earthquake_model = VotingRegressor(
            estimators=[('gb', earthquake_gb), ('rf', earthquake_rf), ('svr', earthquake_svr)]
        )
    
    def _tune_model(self, model, param_grid, X, y, scoring, cv=3):
        """
        Helper method to perform GridSearchCV for hyperparameter tuning.
        """
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_

    def tune_drought_model(self, X_drought, y_drought):
        """
        Tune hyperparameters for the drought prediction model.
        """
        st.write("Tuning drought model hyperparameters...")
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [10, 20],
            'gb__n_estimators': [100, 150],
            'gb__learning_rate': [0.05, 0.1],
            'et__n_estimators': [80, 120]
        }
        best_model, best_params = self._tune_model(self.drought_model, param_grid, X_drought, y_drought, scoring='f1_weighted')
        self.drought_model = best_model
        st.write(f"Best Drought Model Params: {best_params}")
        st.success("Drought model tuning complete!")

    def tune_flood_model(self, X_flood, y_flood):
        """
        Tune hyperparameters for the flood prediction model.
        """
        st.write("Tuning flood model hyperparameters...")
        param_grid = {
            'mlp__hidden_layer_sizes': [(100, 50), (200, 100, 50)],
            'mlp__alpha': [0.0001, 0.001],
            'rf__n_estimators': [150, 250],
            'rf__max_depth': [15, 20],
            'gb__n_estimators': [120, 180]
        }
        best_model, best_params = self._tune_model(self.flood_model, param_grid, X_flood, y_flood, scoring='f1_weighted')
        self.flood_model = best_model
        st.write(f"Best Flood Model Params: {best_params}")
        st.success("Flood model tuning complete!")

    def tune_earthquake_model(self, X_earthquake, y_earthquake):
        """
        Tune hyperparameters for the earthquake prediction model.
        """
        st.write("Tuning earthquake model hyperparameters...")
        param_grid = {
            'gb__n_estimators': [200, 300],
            'gb__learning_rate': [0.05, 0.1],
            'rf__n_estimators': [150, 200],
            'rf__max_depth': [15, 25],
            'svr__C': [0.1, 1.0, 10.0]
        }
        best_model, best_params = self._tune_model(self.earthquake_model, param_grid, X_earthquake, y_earthquake, scoring='neg_mean_squared_error')
        self.earthquake_model = best_model
        st.write(f"Best Earthquake Model Params: {best_params}")
        st.success("Earthquake model tuning complete!")

    def evaluate_drought_model(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the drought prediction model.
        """
        if self.drought_model is None:
            return {"error": "Drought model not trained."}
        
        X_test_scaled = self.feature_scaler.transform(X_test)
        y_pred = self.drought_model.predict(X_test_scaled)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

    def evaluate_flood_model(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the flood prediction model.
        """
        if self.flood_model is None:
            return {"error": "Flood model not trained."}
        
        X_test_scaled = self.feature_scaler.transform(X_test)
        y_pred = self.flood_model.predict(X_test_scaled)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

    def evaluate_earthquake_model(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the earthquake prediction model.
        """
        if self.earthquake_model is None:
            return {"error": "Earthquake model not trained."}
        
        X_test_scaled = self.feature_scaler.transform(X_test)
        y_pred = self.earthquake_model.predict(X_test_scaled)
        
        return {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred)
        }
    
    def extract_advanced_features(self, data: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Extract advanced features from multi-modal data for improved prediction accuracy.
        Handles potential NaN values by providing default numeric values for empty data.
        """
        features = []
        feature_names = []
        
        # Helper to get stats safely
        def safe_stats(arr):
            if len(arr) == 0:
                return 0.0, 0.0, 0.0, 0.0 # mean, std, min, max
            return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)
        
        # NDVI features (vegetation health)
        ndvi_vals = np.array(data.get('ndvi', []))
        mean, std, min_val, max_val = safe_stats(ndvi_vals)
        features.extend([
            mean,
            std,
            min_val,
            max_val,
            np.percentile(ndvi_vals, 25) if len(ndvi_vals) > 0 else 0.0,
            np.percentile(ndvi_vals, 75) if len(ndvi_vals) > 0 else 0.0,
            self._calculate_trend(ndvi_vals),
            self._calculate_variability_index(ndvi_vals)
        ])
        feature_names.extend([
            'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
            'ndvi_q25', 'ndvi_q75', 'ndvi_trend', 'ndvi_variability'
        ])
        
        # Soil moisture features
        moisture_vals = np.array(data.get('soil_moisture', []))
        mean, std, min_val, max_val = safe_stats(moisture_vals)
        features.extend([
            mean,
            std,
            np.percentile(moisture_vals, 10) if len(moisture_vals) > 0 else 0.0,
            np.percentile(moisture_vals, 90) if len(moisture_vals) > 0 else 0.0,
            self._calculate_trend(moisture_vals),
            self._calculate_drought_index(moisture_vals),
            self._calculate_saturation_index(moisture_vals)
        ])
        feature_names.extend([
            'moisture_mean', 'moisture_std', 'moisture_p10', 'moisture_p90',
            'moisture_trend', 'drought_index', 'saturation_index'
        ])
        
        # Temperature features
        temp_vals = np.array(data.get('temperature', []))
        mean, std, min_val, max_val = safe_stats(temp_vals)
        features.extend([
            mean,
            std,
            max_val,
            min_val,
            self._calculate_trend(temp_vals),
            self._calculate_heat_stress_index(temp_vals),
            len(temp_vals[temp_vals > (np.percentile(temp_vals, 90) if len(temp_vals) > 0 else np.inf)])  # Hot days count
        ])
        feature_names.extend([
            'temp_mean', 'temp_std', 'temp_max', 'temp_min',
            'temp_trend', 'heat_stress_index', 'hot_days_count'
        ])
        
        # Precipitation features
        precip_vals = np.array(data.get('precipitation', []))
        mean, std, min_val, max_val = safe_stats(precip_vals)
        features.extend([
            np.sum(precip_vals),
            mean,
            std,
            max_val,
            len(precip_vals[precip_vals > 1]),  # Rainy days
            len(precip_vals[precip_vals > 10]),  # Heavy rain days
            self._calculate_spi(precip_vals),
            self._calculate_consecutive_dry_days(precip_vals)
        ])
        feature_names.extend([
            'precip_total', 'precip_mean', 'precip_std', 'precip_max',
            'rainy_days', 'heavy_rain_days', 'spi', 'consecutive_dry_days'
        ])
        
        # Seismic features
        seismic_vals = np.array(data.get('seismic_activity', []))
        mean, std, min_val, max_val = safe_stats(seismic_vals)
        features.extend([
            mean,
            std,
            max_val,
            float(len(seismic_vals)), # ensure it's a float
            float(len(seismic_vals[seismic_vals > 3.0])) if len(seismic_vals) > 0 else 0.0,  # Significant events
            self._calculate_b_value(seismic_vals),
            self._calculate_seismic_energy(seismic_vals)
        ])
        feature_names.extend([
            'seismic_mean', 'seismic_std', 'seismic_max', 'seismic_count',
            'significant_events', 'b_value', 'seismic_energy'
        ])
        
        # Humidity features (from weather data, if available)
        humidity_vals = np.array(data.get('humidity', []))
        h_mean, h_std, h_min, h_max = safe_stats(humidity_vals)
        features.extend([
            h_mean,
            h_std,
            h_min,
            h_max
        ])
        feature_names.extend([
            'humidity_mean', 'humidity_std', 'humidity_min', 'humidity_max'
        ])

        # Depth features (from seismic data, if available)
        depth_vals = np.array(data.get('depth', []))
        d_mean, d_std, d_min, d_max = safe_stats(depth_vals)
        features.extend([
            d_mean,
            d_std,
            d_min,
            d_max
        ])
        feature_names.extend([
            'depth_mean', 'depth_std', 'depth_min', 'depth_max'
        ])

        # Precipitation Forecast features
        precip_forecast_vals = np.array(data.get('precipitation_forecast', []))
        pf_sum = np.sum(precip_forecast_vals) if len(precip_forecast_vals) > 0 else 0.0
        pf_mean = np.mean(precip_forecast_vals) if len(precip_forecast_vals) > 0 else 0.0
        pf_max = np.max(precip_forecast_vals) if len(precip_forecast_vals) > 0 else 0.0
        features.extend([
            pf_sum,
            pf_mean,
            pf_max
        ])
        feature_names.extend([
            'precip_forecast_sum', 'precip_forecast_mean', 'precip_forecast_max'
        ])

        # Cross-feature interactions
        # Ensure all required features are in feature_names and features is populated
        # Check if the required features are in the `feature_names` list before attempting to access them
        ndvi_mean = features[feature_names.index('ndvi_mean')] if 'ndvi_mean' in feature_names else 0.0
        moisture_mean = features[feature_names.index('moisture_mean')] if 'moisture_mean' in feature_names else 0.0
        features.append(ndvi_mean * moisture_mean)
        feature_names.append('ndvi_moisture_interaction')
        
        temp_mean = features[feature_names.index('temp_mean')] if 'temp_mean' in feature_names else 0.0
        precip_total = features[feature_names.index('precip_total')] if 'precip_total' in feature_names else 0.0
        features.append(temp_mean / (precip_total + 0.01) if (precip_total + 0.01) != 0 else 0.0) # Guard against division by zero
        feature_names.append('temp_precip_ratio')

        # Convert to numpy array and handle any remaining NaN/inf
        final_features = np.nan_to_num(np.array(features).reshape(1, -1), nan=0.0, posinf=1e10, neginf=-1e10)
        return final_features, feature_names
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend in time series."""
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return float(trend)
    
    def _calculate_variability_index(self, values: np.ndarray) -> float:
        """Calculate variability index (coefficient of variation)."""
        if len(values) == 0: return 0.0 # Added explicit check
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        return float(np.std(values) / mean_val)
    
    def _calculate_drought_index(self, moisture_vals: np.ndarray) -> float:
        """Calculate drought severity index based on soil moisture."""
        if len(moisture_vals) == 0: return 0.0 # Added explicit check
        threshold = 0.3  # Drought threshold
        below_threshold = moisture_vals[moisture_vals < threshold]
        if len(moisture_vals) == 0: # This check is redundant due to first line, but safe.
            return 0.0
        return float(len(below_threshold) / len(moisture_vals))
    
    def _calculate_saturation_index(self, moisture_vals: np.ndarray) -> float:
        """Calculate soil saturation index for flood prediction."""
        if len(moisture_vals) == 0: return 0.0 # Added explicit check
        threshold = 0.8  # Saturation threshold
        above_threshold = moisture_vals[moisture_vals > threshold]
        if len(moisture_vals) == 0: # This check is redundant due to first line, but safe.
            return 0.0
        return float(len(above_threshold) / len(moisture_vals))
    
    def _calculate_heat_stress_index(self, temp_vals: np.ndarray) -> float:
        """Calculate heat stress index."""
        if len(temp_vals) == 0: return 0.0 # Added explicit check
        heat_threshold = np.percentile(temp_vals, 85) if len(temp_vals) > 0 else np.inf # Handle empty temp_vals
        heat_days = temp_vals[temp_vals > heat_threshold]
        if len(temp_vals) == 0: # This check is redundant due to first line, but safe.
            return 0.0
        return float(len(heat_days) / len(temp_vals))
    
    def _calculate_spi(self, precip_vals: np.ndarray) -> float:
        """Calculate Standardized Precipitation Index (simplified)."""
        if len(precip_vals) < 5:
            return 0.0
        mean_precip = np.mean(precip_vals)
        std_precip = np.std(precip_vals)
        if len(precip_vals) < 7: # Ensure enough data for recent_precip
            recent_precip = mean_precip # Fallback to mean if not enough recent data
        else:
            recent_precip = np.mean(precip_vals[-7:])  # Last week
        if std_precip == 0:
            return 0.0
        spi = (recent_precip - mean_precip) / std_precip
        return float(np.clip(spi, -3, 3))
    
    def _calculate_consecutive_dry_days(self, precip_vals: np.ndarray) -> float:
        """Calculate maximum consecutive dry days."""
        if len(precip_vals) == 0: return 0.0 # Added explicit check
        dry_threshold = 1.0  # mm
        dry_days = precip_vals < dry_threshold
        max_consecutive = 0
        current_consecutive = 0
        
        for is_dry in dry_days:
            if is_dry:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return float(max_consecutive)
    
    def _calculate_b_value(self, magnitudes: np.ndarray) -> float:
        """Calculate b-value from Gutenberg-Richter law."""
        if len(magnitudes) < 5:
            return 1.0  # Default b-value
        
        # Remove zero magnitudes and create bins
        valid_mags = magnitudes[magnitudes > 0]
        if len(valid_mags) < 3:
            return 1.0
        
        # Ensure there's a range for bins; if all magnitudes are the same, np.arange might create a single bin or no bins
        if np.max(valid_mags) == np.min(valid_mags):
            return 1.0 # Cannot calculate b-value if all magnitudes are identical

        mag_bins = np.arange(np.min(valid_mags), np.max(valid_mags) + 0.5, 0.5)
        if len(mag_bins) < 2: # Ensure at least two bins to form intervals
            return 1.0
        
        hist, _ = np.histogram(valid_mags, bins=mag_bins)
        
        # Calculate b-value using least squares
        non_zero_counts = hist[hist > 0]
        if len(non_zero_counts) < 2:
            return 1.0
        
        log_counts = np.log10(non_zero_counts)
        mags_for_fit = mag_bins[:-1][hist > 0]
        
        # Ensure enough data points for polyfit (at least 2 points for a line)
        if len(mags_for_fit) < 2 or len(log_counts) < 2:
            return 1.0

        try:
            b_value = -np.polyfit(mags_for_fit, log_counts, 1)[0]
            return float(np.clip(b_value, 0.5, 2.0))
        except Exception as e:
            logging.warning(f"Error in b-value calculation: {e}. Returning default.")
            return 1.0
    
    def _calculate_seismic_energy(self, magnitudes: np.ndarray) -> float:
        """Calculate total seismic energy release."""
        if len(magnitudes) == 0:
            return 0.0
        
        # Energy in Joules (simplified Richter scale relationship)
        # Add a small epsilon to prevent log(0) if sum of energies is zero (unlikely but safe)
        energies = 10**(11.8 + 1.5 * magnitudes)
        total_energy = np.sum(energies)
        
        # Convert to log scale for feature use
        return float(np.log10(total_energy + 1e-9)) # Add epsilon here as well
    
    def predict_drought_risk_live(self, lat: float, lon: float, days: int = 30) -> Dict:
        """
        Predict drought risk for a specific location using live data.
        """
        try:
            # Fetch live data
            live_data = self.live_data_fetcher.get_comprehensive_data(lat, lon, days)
            
            # Prepare features
            model_input = {
                'ndvi': live_data['ndvi']['ndvi'],
                'soil_moisture': live_data['soil_moisture']['soil_moisture'],
                'temperature': live_data['weather']['temperature'],
                'precipitation': live_data['weather']['precipitation']
            }
            
            features, feature_names = self.extract_advanced_features(model_input)
            
            # Generate enhanced prediction
            if self.drought_model is None or not self.is_trained:
                return self._enhanced_drought_heuristic(model_input, lat, lon)
            
            # Scale features if scaler is fitted, otherwise use raw features
            try:
                features_scaled = self.feature_scaler.transform(features)
            except NotFittedError:
                logging.warning("Scaler not fitted, using raw features for prediction.")
                features_scaled = features
            except Exception as e:
                logging.error(f"Error scaling features for drought prediction: {e}")
                return {"error": f"Feature scaling error: {e}"}
            
            # Make prediction
            risk_probabilities = self.drought_model.predict_proba(features_scaled)[0]
            risk_level = self.drought_model.predict(features_scaled)[0]
            
            # Calculate confidence based on ensemble agreement
            confidence = self._calculate_prediction_confidence(risk_probabilities)
            
            return {
                'risk_level': risk_level,
                'risk_probability': float(np.max(risk_probabilities)),
                'confidence': confidence,
                'location': {'lat': lat, 'lon': lon},
                'factors': self._analyze_risk_factors(model_input, feature_names, features[0]),
                'recommendation': self._get_drought_recommendations(risk_level, confidence),
                'data_quality': self._assess_data_quality(live_data),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in drought prediction (live): {e}")
            return self._fallback_drought_prediction(lat, lon)
    
    def predict_flood_risk_live(self, lat: float, lon: float, days: int = 30) -> Dict:
        """
        Predict flood risk for a specific location using live data.
        """
        try:
            # Fetch live data
            live_data = self.live_data_fetcher.get_comprehensive_data(lat, lon, days)
            
            # Prepare features including forecast data
            model_input = {
                'precipitation': live_data['weather']['precipitation'],
                'soil_moisture': live_data['soil_moisture']['soil_moisture'],
                'temperature': live_data['weather']['temperature'],
                'precipitation_forecast': live_data['precipitation_forecast']['precipitation']
            }
            
            features, feature_names = self.extract_advanced_features(model_input)
            
            # Generate enhanced prediction
            if self.flood_model is None or not self.is_trained:
                return self._enhanced_flood_heuristic(model_input, lat, lon)
            
            # Scale features and predict
            try:
                features_scaled = self.feature_scaler.transform(features)
            except NotFittedError:
                logging.warning("Scaler not fitted, using raw features for prediction.")
                features_scaled = features
            except Exception as e:
                logging.error(f"Error scaling features for flood prediction: {e}")
                return {"error": f"Feature scaling error: {e}"}

            risk_probabilities = self.flood_model.predict_proba(features_scaled)[0]
            risk_level = self.flood_model.predict(features_scaled)[0]
            confidence = self._calculate_prediction_confidence(risk_probabilities)
            
            return {
                'risk_level': risk_level,
                'risk_probability': float(np.max(risk_probabilities)),
                'confidence': confidence,
                'location': {'lat': lat, 'lon': lon},
                'factors': self._analyze_risk_factors(model_input, feature_names, features[0]),
                'recommendation': self._get_flood_recommendations(risk_level, confidence),
                'forecast_risk': self._assess_forecast_risk(live_data['precipitation_forecast']),
                'data_quality': self._assess_data_quality(live_data),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in flood prediction (live): {e}")
            return self._fallback_flood_prediction(lat, lon)
    
    def predict_earthquake_risk_live(self, lat: float, lon: float, days: int = 30) -> Dict:
        """
        Predict earthquake risk for a specific location using live data.
        """
        try:
            # Fetch live data
            live_data = self.live_data_fetcher.get_comprehensive_data(lat, lon, days)
            
            # Prepare seismic features
            seismic_events = live_data['seismic']
            magnitudes = [event['magnitude'] for event in seismic_events]
            
            model_input = {
                'seismic_activity': magnitudes,
                'depth': [event['depth'] for event in seismic_events]
            }
            
            features, feature_names = self.extract_advanced_features(model_input)
            
            # Generate enhanced prediction
            if self.earthquake_model is None or not self.is_trained:
                return self._enhanced_earthquake_heuristic(model_input, lat, lon)
            
            # Scale features and predict
            try:
                features_scaled = self.feature_scaler.transform(features)
            except NotFittedError:
                logging.warning("Scaler not fitted, using raw features for prediction.")
                features_scaled = features
            except Exception as e:
                logging.error(f"Error scaling features for earthquake prediction: {e}")
                return {"error": f"Feature scaling error: {e}"}

            magnitude_prediction = self.earthquake_model.predict(features_scaled)[0]
            
            # Convert magnitude to probability and risk level
            probability = self._magnitude_to_probability(magnitude_prediction)
            risk_level = self._magnitude_to_risk_level(magnitude_prediction)
            confidence = self._calculate_regression_confidence(magnitude_prediction)
            
            return {
                'magnitude_prediction': float(magnitude_prediction),
                'risk_level': risk_level,
                'probability': probability,
                'confidence': confidence,
                'location': {'lat': lat, 'lon': lon},
                'factors': self._analyze_risk_factors(model_input, feature_names, features[0]),
                'recommendation': self._get_earthquake_recommendations(risk_level, confidence),
                'seismic_activity': self._analyze_seismic_activity(seismic_events),
                'data_quality': self._assess_data_quality(live_data),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in earthquake prediction (live): {e}")
            return self._fallback_earthquake_prediction(lat, lon)
    
    def _enhanced_drought_heuristic(self, data: Dict, lat: float, lon: float) -> Dict:
        """Enhanced drought prediction using multiple indicators."""
        
        # Calculate multiple drought indices
        ndvi_vals = np.array(data.get('ndvi', [0.5]))
        moisture_vals = np.array(data.get('soil_moisture', [0.4]))
        temp_vals = np.array(data.get('temperature', [20]))
        precip_vals = np.array(data.get('precipitation', [5]))
        
        # NDVI-based drought indicator
        ndvi_trend = self._calculate_trend(ndvi_vals)
        current_ndvi = np.mean(ndvi_vals[-7:]) if len(ndvi_vals) >= 7 else np.mean(ndvi_vals)
        historical_ndvi = np.mean(ndvi_vals)
        
        # Soil moisture indicator
        current_moisture = np.mean(moisture_vals[-7:]) if len(moisture_vals) >= 7 else np.mean(moisture_vals)
        drought_threshold = 0.25
        
        # Precipitation indicator
        spi = self._calculate_spi(precip_vals)
        consecutive_dry = self._calculate_consecutive_dry_days(precip_vals)
        
        # Temperature indicator
        temp_anomaly = np.mean(temp_vals[-7:]) - np.mean(temp_vals) if len(temp_vals) >= 14 else 0
        
        # Combine indicators with weights
        indicators = {
            'ndvi_decline': max(0, (historical_ndvi - current_ndvi) / historical_ndvi) if historical_ndvi > 0 else 0,
            'moisture_deficit': max(0, (drought_threshold - current_moisture) / drought_threshold),
            'precipitation_deficit': max(0, -spi / 3.0),  # Normalize SPI
            'consecutive_dry_days': min(1.0, consecutive_dry / 30.0),
            'temperature_excess': max(0, temp_anomaly / 10.0)
        }
        
        # Calculate weighted risk score
        weights = {'ndvi_decline': 0.25, 'moisture_deficit': 0.3, 'precipitation_deficit': 0.25,
                  'consecutive_dry_days': 0.15, 'temperature_excess': 0.05}
        
        risk_score = sum(indicators[key] * weights[key] for key in weights)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'High'
        elif risk_score >= 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        confidence = 0.75 + (len(ndvi_vals) / 100) * 0.2  # Higher confidence with more data
        confidence = min(0.95, confidence)
        
        return {
            'risk_level': risk_level,
            'risk_probability': float(risk_score),
            'confidence': float(confidence),
            'location': {'lat': lat, 'lon': lon},
            'factors': {
                'ndvi_trend': float(ndvi_trend),
                'soil_moisture': float(current_moisture),
                'temperature_anomaly': float(temp_anomaly),
                'consecutive_dry_days': int(consecutive_dry),
                'spi': float(spi)
            },
            'indicators': {k: float(v) for k, v in indicators.items()},
            'recommendation': self._get_drought_recommendations(risk_level, confidence),
            'last_updated': datetime.now().isoformat()
        }
    
    def _enhanced_flood_heuristic(self, data: Dict, lat: float, lon: float) -> Dict:
        """Enhanced flood prediction using multiple indicators."""
        
        precip_vals = np.array(data.get('precipitation', [5]))
        moisture_vals = np.array(data.get('soil_moisture', [0.4]))
        forecast_precip = np.array(data.get('precipitation_forecast', [3]))
        
        # Recent precipitation accumulation
        recent_precip_3d = np.sum(precip_vals[-3:]) if len(precip_vals) >= 3 else np.sum(precip_vals)
        recent_precip_7d = np.sum(precip_vals[-7:]) if len(precip_vals) >= 7 else np.sum(precip_vals)
        
        # Soil saturation
        current_moisture = np.mean(moisture_vals[-3:]) if len(moisture_vals) >= 3 else np.mean(moisture_vals)
        saturation_threshold = 0.8
        
        # Forecast indicators
        forecast_total = np.sum(forecast_precip)
        forecast_max_daily = np.max(forecast_precip) if len(forecast_precip) > 0 else 0
        
        # Calculate flood indicators
        indicators = {
            'recent_rainfall_intensity': min(1.0, recent_precip_3d / 75.0),  # 75mm in 3 days is high
            'cumulative_rainfall': min(1.0, recent_precip_7d / 150.0),  # 150mm in 7 days is very high
            'soil_saturation': max(0, (current_moisture - saturation_threshold) / (1.0 - saturation_threshold)),
            'forecast_risk': min(1.0, forecast_total / 100.0),  # 100mm forecast is concerning
            'extreme_event_risk': min(1.0, forecast_max_daily / 50.0)  # 50mm in one day
        }
        
        # Weighted risk calculation
        weights = {'recent_rainfall_intensity': 0.3, 'cumulative_rainfall': 0.25, 
                  'soil_saturation': 0.2, 'forecast_risk': 0.15, 'extreme_event_risk': 0.1}
        
        risk_score = sum(indicators[key] * weights[key] for key in weights)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Determine risk level
        if risk_score >= 0.6:
            risk_level = 'High'
        elif risk_score >= 0.35:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        confidence = 0.8 + (len(precip_vals) / 50) * 0.15
        confidence = min(0.95, confidence)
        
        return {
            'risk_level': risk_level,
            'risk_probability': float(risk_score),
            'confidence': float(confidence),
            'location': {'lat': lat, 'lon': lon},
            'factors': {
                'recent_precipitation_3d': float(recent_precip_3d),
                'recent_precipitation_7d': float(recent_precip_7d),
                'soil_saturation': float(current_moisture),
                'forecast_precipitation': float(forecast_total),
                'max_forecast_daily': float(forecast_max_daily)
            },
            'indicators': {k: float(v) for k, v in indicators.items()},
            'recommendation': self._get_flood_recommendations(risk_level, confidence),
            'last_updated': datetime.now().isoformat()
        }
    
    def _enhanced_earthquake_heuristic(self, data: Dict, lat: float, lon: float) -> Dict:
        """Enhanced earthquake prediction using seismic indicators."""
        
        magnitudes = np.array(data.get('seismic_activity', [2.0]))
        depths = np.array(data.get('depth', [10.0]))
        
        if len(magnitudes) == 0:
            magnitudes = np.array([2.0])
            depths = np.array([10.0])
        
        # Seismic activity indicators
        recent_events = len(magnitudes)
        max_magnitude = np.max(magnitudes)
        recent_large_events = len(magnitudes[magnitudes >= 4.0])
        
        # Calculate b-value
        b_value = self._calculate_b_value(magnitudes)
        
        # Energy release
        total_energy = self._calculate_seismic_energy(magnitudes)
        
        # Depth analysis
        shallow_events = len(depths[depths < 30]) if len(depths) > 0 else 0
        
        # Risk indicators
        indicators = {
            'activity_level': min(1.0, recent_events / 50.0),  # 50 events is high activity
            'maximum_magnitude': min(1.0, max_magnitude / 8.0),  # Scale to magnitude 8
            'large_event_frequency': min(1.0, recent_large_events / 5.0),  # 5+ M4+ events
            'b_value_decline': max(0, (1.0 - b_value) / 0.5),  # b-value below 1.0 is concerning
            'energy_release': min(1.0, total_energy / 20.0),  # Normalized energy scale
            'shallow_activity': min(1.0, shallow_events / 10.0)  # Shallow events are more dangerous
        }
        
        # Weighted risk calculation
        weights = {'activity_level': 0.2, 'maximum_magnitude': 0.25, 'large_event_frequency': 0.2,
                  'b_value_decline': 0.15, 'energy_release': 0.1, 'shallow_activity': 0.1}
        
        risk_score = sum(indicators[key] * weights[key] for key in weights)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Predict magnitude based on activity
        predicted_magnitude = 2.0 + (risk_score * 5.0)  # Scale between 2.0 and 7.0
        
        # Determine risk level
        if predicted_magnitude >= 6.0:
            risk_level = 'High'
        elif predicted_magnitude >= 4.5:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        probability = risk_score * 0.3  # Convert to probability (max 30%)
        confidence = 0.7 + (recent_events / 100) * 0.25
        confidence = min(0.95, confidence)
        
        return {
            'magnitude_prediction': float(predicted_magnitude),
            'risk_level': risk_level,
            'probability': float(probability),
            'confidence': float(confidence),
            'location': {'lat': lat, 'lon': lon},
            'factors': {
                'seismic_activity': float(max_magnitude),
                'b_value': float(b_value),
                'recent_events': int(recent_events),
                'shallow_events': int(shallow_events),
                'energy_release': float(total_energy)
            },
            'indicators': {k: float(v) for k, v in indicators.items()},
            'recommendation': self._get_earthquake_recommendations(risk_level, confidence),
            'last_updated': datetime.now().isoformat()
        }
    
    def _analyze_risk_factors(self, data: Dict, feature_names: List[str], features: np.ndarray) -> Dict:
        """Analyze and rank risk factors by importance."""
        factors = {}
        
        for i, name in enumerate(feature_names):
            if i < len(features):
                factors[name] = float(features[i])
        
        return factors
    
    def _calculate_prediction_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate prediction confidence based on class probabilities."""
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        
        # Higher confidence when max probability is high and entropy is low
        confidence = (max_prob + (1 - entropy / max_entropy)) / 2
        return float(np.clip(confidence, 0.5, 0.98))
    
    def _calculate_regression_confidence(self, prediction: float) -> float:
        """Calculate confidence for regression predictions."""
        # Simple confidence based on prediction magnitude
        if prediction < 3.0:
            return 0.85
        elif prediction < 5.0:
            return 0.75
        elif prediction < 7.0:
            return 0.65
        else:
            return 0.55
    
    def _magnitude_to_probability(self, magnitude: float) -> float:
        """Convert earthquake magnitude to occurrence probability."""
        # Simplified probability model
        if magnitude < 4.0:
            return 0.1
        elif magnitude < 5.0:
            return 0.05
        elif magnitude < 6.0:
            return 0.02
        elif magnitude < 7.0:
            return 0.005
        else:
            return 0.001
    
    def _magnitude_to_risk_level(self, magnitude: float) -> str:
        """Convert earthquake magnitude to risk level."""
        if magnitude >= 6.0:
            return 'High'
        elif magnitude >= 4.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_data_quality(self, live_data: Dict) -> Dict:
        """Assess the quality of input data."""
        quality_scores = {}
        
        for data_type, data in live_data.items():
            if isinstance(data, dict) and 'quality_score' in data:
                quality_scores[data_type] = data['quality_score']
            else:
                # Default quality assessment
                if data_type == 'weather':
                    quality_scores[data_type] = 0.9
                elif data_type == 'ndvi':
                    quality_scores[data_type] = 0.85
                elif data_type == 'soil_moisture':
                    quality_scores[data_type] = 0.8
                else:
                    quality_scores[data_type] = 0.75
        
        overall_quality = np.mean(list(quality_scores.values()))
        
        return {
            'overall_score': float(overall_quality),
            'individual_scores': quality_scores,
            'data_completeness': len(quality_scores) / 5  # Expected 5 data types
        }
    
    def _assess_forecast_risk(self, forecast_data: Dict) -> Dict:
        """Assess risk based on forecast data."""
        precip_forecast = np.array(forecast_data.get('precipitation', []))
        
        if len(precip_forecast) == 0:
            return {'risk_level': 'Low', 'total_forecast': 0, 'max_daily': 0}
        
        total_forecast = np.sum(precip_forecast)
        max_daily = np.max(precip_forecast)
        
        if total_forecast > 100 or max_daily > 50:
            risk_level = 'High'
        elif total_forecast > 50 or max_daily > 25:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_level': risk_level,
            'total_forecast': float(total_forecast),
            'max_daily': float(max_daily),
            'forecast_days': len(precip_forecast)
        }
    
    def _analyze_seismic_activity(self, seismic_events: List[Dict]) -> Dict:
        """Analyze seismic activity patterns."""
        if not seismic_events:
            return {'activity_level': 'Low', 'recent_events': 0, 'max_magnitude': 0}
        
        magnitudes = [event['magnitude'] for event in seismic_events]
        recent_events = len(seismic_events)
        max_magnitude = max(magnitudes)
        significant_events = len([m for m in magnitudes if m >= 4.0])
        
        if max_magnitude >= 6.0 or significant_events >= 3:
            activity_level = 'High'
        elif max_magnitude >= 4.0 or significant_events >= 1:
            activity_level = 'Medium'
        else:
            activity_level = 'Low'
        
        return {
            'activity_level': activity_level,
            'recent_events': recent_events,
            'max_magnitude': float(max_magnitude),
            'significant_events': significant_events,
            'b_value': float(self._calculate_b_value(np.array(magnitudes)))
        }
    
    def _get_drought_recommendations(self, risk_level: str, confidence: float) -> List[str]:
        """Get drought-specific recommendations."""
        base_recommendations = {
            'High': [
                "Implement emergency water conservation measures",
                "Monitor agricultural impacts and livestock water needs",
                "Consider water rationing for non-essential uses",
                "Activate drought emergency response protocols",
                "Increase monitoring frequency to daily assessments"
            ],
            'Medium': [
                "Increase water conservation awareness",
                "Monitor reservoir and groundwater levels",
                "Prepare drought contingency plans",
                "Advise agricultural sector on water-efficient practices",
                "Issue water use advisories"
            ],
            'Low': [
                "Continue routine monitoring",
                "Maintain drought preparedness plans",
                "Monitor long-term weather forecasts",
                "Ensure water infrastructure maintenance"
            ]
        }
        
        recommendations = base_recommendations.get(risk_level, base_recommendations['Low'])
        
        if confidence < 0.7:
            recommendations.append("Increase data collection for better prediction accuracy")
        
        return recommendations
    
    def _get_flood_recommendations(self, risk_level: str, confidence: float) -> List[str]:
        """Get flood-specific recommendations."""
        base_recommendations = {
            'High': [
                "Issue flood warnings to affected areas",
                "Prepare emergency evacuation plans",
                "Deploy flood barriers and sandbags",
                "Alert emergency services and hospitals",
                "Monitor river levels continuously",
                "Close flood-prone roads if necessary"
            ],
            'Medium': [
                "Issue flood watch advisory",
                "Increase monitoring of water levels",
                "Prepare emergency response teams",
                "Check drainage systems and flood defenses",
                "Advise residents in flood-prone areas"
            ],
            'Low': [
                "Continue routine monitoring",
                "Maintain flood preparedness plans",
                "Monitor weather forecasts",
                "Ensure drainage infrastructure maintenance"
            ]
        }
        
        recommendations = base_recommendations.get(risk_level, base_recommendations['Low'])
        
        if confidence < 0.7:
            recommendations.append("Enhance precipitation monitoring network")
        
        return recommendations
    
    def _get_earthquake_recommendations(self, risk_level: str, confidence: float) -> List[str]:
        """Get earthquake-specific recommendations."""
        base_recommendations = {
            'High': [
                "Issue earthquake preparedness alert",
                "Review building safety and evacuation procedures",
                "Check critical infrastructure resilience",
                "Alert emergency services and hospitals",
                "Increase seismic monitoring frequency",
                "Conduct earthquake preparedness drills"
            ],
            'Medium': [
                "Issue earthquake watch advisory",
                "Monitor seismic activity closely",
                "Review emergency response plans",
                "Check emergency supply kits",
                "Conduct building safety assessments"
            ],
            'Low': [
                "Continue routine seismic monitoring",
                "Maintain earthquake preparedness",
                "Regular emergency drill schedule",
                "Monitor regional seismic trends"
            ]
        }
        
        recommendations = base_recommendations.get(risk_level, base_recommendations['Low'])
        
        if confidence < 0.7:
            recommendations.append("Expand seismic monitoring network coverage")
        
        return recommendations
    
    def _fallback_drought_prediction(self, lat: float, lon: float) -> Dict:
        """Fallback drought prediction when data is unavailable or model not trained. Returns a consistent structure."""
        return {
            'risk_level': 'Unknown',
            'risk_probability': 0.0,
            'confidence': 0.0,
            'location': {'lat': lat, 'lon': lon},
            'factors': {'data_availability': 'Limited', 'temperature_anomaly': 0.0, 'ndvi_trend': 0.0, 'soil_moisture': 0.0},
            'recommendation': ['Improve data collection for better predictions', 'Train models for accurate predictions'],
            'last_updated': datetime.now().isoformat(),
            'note': 'Prediction based on limited data or untrained model'
        }
    
    def _fallback_flood_prediction(self, lat: float, lon: float) -> Dict:
        """Fallback flood prediction when data is unavailable or model not trained. Returns a consistent structure."""
        return {
            'risk_level': 'Unknown',
            'risk_probability': 0.0,
            'confidence': 0.0,
            'location': {'lat': lat, 'lon': lon},
            'factors': {'data_availability': 'Limited', 'recent_rainfall_intensity': 0.0, 'cumulative_rainfall': 0.0, 'soil_saturation': 0.0, 'forecast_risk': 0.0},
            'recommendation': ['Improve data collection for better predictions', 'Train models for accurate predictions'],
            'last_updated': datetime.now().isoformat(),
            'note': 'Prediction based on limited data or untrained model'
        }
    
    def _fallback_earthquake_prediction(self, lat: float, lon: float) -> Dict:
        """
        Fallback earthquake prediction when data is unavailable or model not trained. Returns a consistent structure.
        """
        return {
            'magnitude_prediction': 0.0,
            'risk_level': 'Unknown',
            'probability': 0.0,
            'confidence': 0.0,
            'location': {'lat': lat, 'lon': lon},
            'factors': {'data_availability': 'Limited', 'seismic_mean': 0.0, 'b_value': 0.0, 'seismic_count': 0, 'shallow_events': 0, 'seismic_energy': 0.0},
            'recommendation': ['Improve seismic monitoring for better predictions', 'Train models for accurate predictions'],
            'last_updated': datetime.now().isoformat(),
            'note': 'Prediction based on limited data or untrained model'
        }

    def predict_drought_risk(self, data: Dict) -> Dict:
        """
        Predict drought risk for pre-processed data.
        """
        try:
            features, feature_names = self.extract_advanced_features(data)
            if self.drought_model is None or not self.is_trained:
                return self._enhanced_drought_heuristic(data, 0.0, 0.0) # Use heuristic with dummy lat/lon
            
            try:
                features_scaled = self.feature_scaler.transform(features)
            except NotFittedError:
                logging.warning("Scaler not fitted, using raw features for prediction.")
                features_scaled = features
            except Exception as e:
                logging.error(f"Error scaling features for drought prediction (non-live): {e}")
                return {"error": f"Feature scaling error: {e}"}

            risk_probabilities = self.drought_model.predict_proba(features_scaled)[0]
            risk_level = self.drought_model.predict(features_scaled)[0]
            confidence = self._calculate_prediction_confidence(risk_probabilities)

            return {
                'risk_level': risk_level,
                'risk_probability': float(np.max(risk_probabilities)),
                'confidence': confidence,
                'factors': self._analyze_risk_factors(data, feature_names, features[0]),
                'recommendation': self._get_drought_recommendations(risk_level, confidence),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error in drought prediction (non-live): {e}")
            return {"error": f"Prediction error: {e}", 'risk_level': 'Unknown', 'confidence': 0.0}

    def predict_flood_risk(self, data: Dict) -> Dict:
        """
        Predict flood risk for pre-processed data.
        """
        try:
            features, feature_names = self.extract_advanced_features(data)
            if self.flood_model is None or not self.is_trained:
                return self._enhanced_flood_heuristic(data, 0.0, 0.0) # Use heuristic with dummy lat/lon

            try:
                features_scaled = self.feature_scaler.transform(features)
            except NotFittedError:
                logging.warning("Scaler not fitted, using raw features for prediction.")
                features_scaled = features
            except Exception as e:
                logging.error(f"Error scaling features for flood prediction (non-live): {e}")
                return {"error": f"Feature scaling error: {e}"}

            risk_probabilities = self.flood_model.predict_proba(features_scaled)[0]
            risk_level = self.flood_model.predict(features_scaled)[0]
            confidence = self._calculate_prediction_confidence(risk_probabilities)

            return {
                'risk_level': risk_level,
                'risk_probability': float(np.max(risk_probabilities)),
                'confidence': confidence,
                'factors': self._analyze_risk_factors(data, feature_names, features[0]),
                'recommendation': self._get_flood_recommendations(risk_level, confidence),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error in flood prediction (non-live): {e}")
            return {"error": f"Prediction error: {e}", 'risk_level': 'Unknown', 'confidence': 0.0}

    def predict_earthquake_risk(self, data: Dict) -> Dict:
        """
        Predict earthquake risk for pre-processed data.
        """
        try:
            # Earthquake model expects 'seismic_activity' as list of magnitudes
            seismic_magnitudes = data.get('seismic_activity', [])
            if not seismic_magnitudes:
                 # If no seismic activity in data, use fallback or default for feature extraction
                 model_input = {'seismic_activity': [2.0], 'depth': [10.0]} # Default small event
            else:
                model_input = {'seismic_activity': seismic_magnitudes, 'depth': data.get('depth', [10.0] * len(seismic_magnitudes))}

            features, feature_names = self.extract_advanced_features(model_input)
            if self.earthquake_model is None or not self.is_trained:
                return self._enhanced_earthquake_heuristic(model_input, 0.0, 0.0) # Use heuristic with dummy lat/lon

            try:
                features_scaled = self.feature_scaler.transform(features)
            except NotFittedError:
                logging.warning("Scaler not fitted, using raw features for prediction.")
                features_scaled = features
            except Exception as e:
                logging.error(f"Error scaling features for earthquake prediction (non-live): {e}")
                return {"error": f"Feature scaling error: {e}"}

            magnitude_prediction = self.earthquake_model.predict(features_scaled)[0]
            probability = self._magnitude_to_probability(magnitude_prediction)
            risk_level = self._magnitude_to_risk_level(magnitude_prediction)
            confidence = self._calculate_regression_confidence(magnitude_prediction)

            return {
                'magnitude_prediction': float(magnitude_prediction),
                'risk_level': risk_level,
                'probability': probability,
                'confidence': confidence,
                'factors': self._analyze_risk_factors(model_input, feature_names, features[0]),
                'recommendation': self._get_earthquake_recommendations(risk_level, confidence),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error in earthquake prediction (non-live): {e}")
            return {"error": f"Prediction error: {e}", 'risk_level': 'Unknown', 'confidence': 0.0}