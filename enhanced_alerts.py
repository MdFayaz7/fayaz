import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from models.enhanced_predictor import EnhancedHazardPredictor

@dataclass
class AlertConfig:
    """Configuration for alert thresholds and parameters."""
    drought_thresholds: Dict[str, float]
    flood_thresholds: Dict[str, float]
    earthquake_thresholds: Dict[str, float]
    confidence_requirements: Dict[str, float]
    alert_cooldown_hours: int = 24
    auto_escalation: bool = True

class EnhancedAlertSystem:
    """
    Enhanced alert system with intelligent thresholds, auto-escalation, and real-time monitoring.
    """
    
    def __init__(self):
        self.predictor = EnhancedHazardPredictor()
        self.active_alerts = []
        self.alert_history = []
        self.monitored_locations = []
        
        # Enhanced alert configuration
        self.config = AlertConfig(
            drought_thresholds={
                'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.85
            },
            flood_thresholds={
                'low': 0.35, 'medium': 0.55, 'high': 0.75, 'critical': 0.9
            },
            earthquake_thresholds={
                'low': 0.2, 'medium': 0.4, 'high': 0.65, 'critical': 0.8
            },
            confidence_requirements={
                'critical': 0.8, 'high': 0.7, 'medium': 0.6, 'low': 0.5
            }
        )
        
        # Priority weights for multi-hazard scenarios
        self.hazard_priorities = {
            'earthquake': 3,  # Highest priority due to immediate danger
            'flood': 2,       # Medium-high priority
            'drought': 1      # Lower immediate priority but long-term impact
        }
        
        # Auto-monitoring setup
        self._setup_global_monitoring()
    
    def _setup_global_monitoring(self):
        """Set up automatic monitoring for high-risk global locations."""
        high_risk_locations = [
            {'name': 'Tokyo Bay Area', 'lat': 35.6762, 'lon': 139.6503, 'hazards': ['earthquake', 'flood']},
            {'name': 'San Francisco Bay Area', 'lat': 37.7749, 'lon': -122.4194, 'hazards': ['earthquake', 'drought']},
            {'name': 'Bangladesh Delta', 'lat': 23.6345, 'lon': 90.2934, 'hazards': ['flood', 'drought']},
            {'name': 'Southern California', 'lat': 34.0522, 'lon': -118.2437, 'hazards': ['earthquake', 'drought', 'flood']},
            {'name': 'Turkey - North Anatolian Fault', 'lat': 40.7589, 'lon': 29.9773, 'hazards': ['earthquake']},
            {'name': 'Australian Murray-Darling Basin', 'lat': -34.0, 'lon': 142.0, 'hazards': ['drought']},
            {'name': 'Netherlands Delta Works', 'lat': 51.8069, 'lon': 4.9604, 'hazards': ['flood']},
            {'name': 'Chilean Subduction Zone', 'lat': -33.4489, 'lon': -70.6693, 'hazards': ['earthquake']},
            {'name': 'Horn of Africa', 'lat': 9.1450, 'lon': 40.4897, 'hazards': ['drought']},
            {'name': 'Philippine Islands', 'lat': 14.5995, 'lon': 120.9842, 'hazards': ['earthquake', 'flood']}
        ]
        
        self.monitored_locations = high_risk_locations
    
    def add_monitoring_location(self, name: str, lat: float, lon: float, hazards: List[str]):
        """Add a new location to continuous monitoring."""
        location = {
            'name': name,
            'lat': lat,
            'lon': lon,
            'hazards': hazards,
            'added_date': datetime.now(),
            'last_check': None
        }
        self.monitored_locations.append(location)
        
        logging.info(f"Added monitoring for {name} at ({lat}, {lon}) for hazards: {hazards}")
    
    def run_continuous_monitoring(self) -> List[Dict]:
        """
        Run continuous monitoring for all tracked locations.
        Returns list of new alerts generated.
        """
        new_alerts = []
        
        for location in self.monitored_locations:
            try:
                # Check each hazard type for this location
                for hazard_type in location['hazards']:
                    alert = self._check_location_for_hazard(
                        location['lat'], 
                        location['lon'], 
                        hazard_type, 
                        location['name']
                    )
                    
                    if alert:
                        new_alerts.append(alert)
                
                # Update last check time
                location['last_check'] = datetime.now()
                
            except Exception as e:
                logging.error(f"Error monitoring {location['name']}: {e}")
                continue
        
        return new_alerts
    
    def _check_location_for_hazard(self, lat: float, lon: float, hazard_type: str, location_name: str) -> Optional[Dict]:
        """Check a specific location for a specific hazard type."""
        
        # Get prediction based on hazard type
        if hazard_type == 'drought':
            prediction = self.predictor.predict_drought_risk_live(lat, lon)
        elif hazard_type == 'flood':
            prediction = self.predictor.predict_flood_risk_live(lat, lon)
        elif hazard_type == 'earthquake':
            prediction = self.predictor.predict_earthquake_risk_live(lat, lon)
        else:
            return None
        
        # Evaluate if alert should be generated
        return self._evaluate_prediction_for_alert(prediction, hazard_type, location_name, lat, lon)
    
    def _evaluate_prediction_for_alert(self, prediction: Dict, hazard_type: str, location_name: str, 
                                     lat: float, lon: float) -> Optional[Dict]:
        """Evaluate a prediction to determine if an alert should be generated."""
        
        risk_score = prediction.get('risk_probability', 0)
        confidence = prediction.get('confidence', 0)
        risk_level = prediction.get('risk_level', 'Low')
        
        # Get thresholds for this hazard type
        thresholds = getattr(self.config, f'{hazard_type}_thresholds')
        confidence_req = self.config.confidence_requirements
        
        # Determine alert level
        alert_level = None
        if risk_score >= thresholds['critical'] and confidence >= confidence_req['critical']:
            alert_level = 'Critical'
        elif risk_score >= thresholds['high'] and confidence >= confidence_req['high']:
            alert_level = 'High'
        elif risk_score >= thresholds['medium'] and confidence >= confidence_req['medium']:
            alert_level = 'Medium'
        elif risk_score >= thresholds['low'] and confidence >= confidence_req['low']:
            alert_level = 'Low'
        
        if not alert_level:
            return None
        
        # Check for duplicate recent alerts
        if self._has_recent_alert(location_name, hazard_type, alert_level):
            return None
        
        # Generate alert
        alert = self._create_enhanced_alert(
            prediction, hazard_type, alert_level, location_name, lat, lon
        )
        
        # Add to active alerts and history
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        logging.info(f"Generated {alert_level} {hazard_type} alert for {location_name}")
        
        return alert
    
    def _has_recent_alert(self, location_name: str, hazard_type: str, alert_level: str) -> bool:
        """Check if there's a recent similar alert to avoid spam."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.alert_cooldown_hours)
        
        for alert in self.active_alerts:
            if (alert['location_name'] == location_name and 
                alert['hazard_type'] == hazard_type and
                alert['alert_level'] == alert_level and
                alert['timestamp'] > cutoff_time):
                return True
        
        return False
    
    def _create_enhanced_alert(self, prediction: Dict, hazard_type: str, alert_level: str,
                             location_name: str, lat: float, lon: float) -> Dict:
        """Create an enhanced alert with comprehensive information."""
        
        alert_id = f"ALT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # Calculate alert priority
        base_priority = self.hazard_priorities.get(hazard_type, 1)
        level_multiplier = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        priority = base_priority * level_multiplier.get(alert_level, 1)
        
        # Generate enhanced description
        description = self._generate_enhanced_description(prediction, hazard_type, alert_level, location_name)
        
        # Calculate alert duration
        duration_hours = self._calculate_alert_duration(hazard_type, alert_level, prediction.get('confidence', 0.5))
        
        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'hazard_type': hazard_type,
            'alert_level': alert_level,
            'location_name': location_name,
            'latitude': lat,
            'longitude': lon,
            'priority': priority,
            'risk_score': prediction.get('risk_probability', 0),
            'confidence': prediction.get('confidence', 0),
            'description': description,
            'detailed_prediction': prediction,
            'recommended_actions': prediction.get('recommendation', []),
            'affected_population_estimate': self._estimate_affected_population(lat, lon, hazard_type),
            'economic_impact_estimate': self._estimate_economic_impact(hazard_type, alert_level),
            'expiry_time': datetime.now() + timedelta(hours=duration_hours),
            'status': 'active',
            'escalation_history': [],
            'update_count': 0,
            'data_quality': prediction.get('data_quality', {}),
            'notification_sent': False,
            'acknowledgments': []
        }
        
        return alert
    
    def _generate_enhanced_description(self, prediction: Dict, hazard_type: str, alert_level: str, 
                                     location_name: str) -> str:
        """Generate detailed alert description."""
        
        risk_score = prediction.get('risk_probability', 0)
        confidence = prediction.get('confidence', 0)
        
        base_descriptions = {
            'drought': {
                'Critical': f"CRITICAL DROUGHT ALERT for {location_name}. Severe water shortage conditions detected.",
                'High': f"HIGH DROUGHT RISK in {location_name}. Significant water stress indicators present.",
                'Medium': f"MODERATE DROUGHT CONDITIONS developing in {location_name}.",
                'Low': f"EARLY DROUGHT INDICATORS detected in {location_name}."
            },
            'flood': {
                'Critical': f"CRITICAL FLOOD ALERT for {location_name}. Immediate evacuation may be required.",
                'High': f"HIGH FLOOD RISK in {location_name}. Significant flooding expected.",
                'Medium': f"MODERATE FLOOD CONDITIONS developing in {location_name}.",
                'Low': f"ELEVATED FLOOD RISK detected in {location_name}."
            },
            'earthquake': {
                'Critical': f"CRITICAL SEISMIC ALERT for {location_name}. High earthquake probability detected.",
                'High': f"HIGH EARTHQUAKE RISK in {location_name}. Increased seismic activity.",
                'Medium': f"MODERATE EARTHQUAKE CONDITIONS in {location_name}.",
                'Low': f"ELEVATED SEISMIC ACTIVITY detected in {location_name}."
            }
        }
        
        base_desc = base_descriptions.get(hazard_type, {}).get(alert_level, f"{alert_level} {hazard_type} alert")
        
        # Add specific details
        details = []
        if 'factors' in prediction:
            factors = prediction['factors']
            if hazard_type == 'drought':
                if 'ndvi_trend' in factors and factors['ndvi_trend'] < -0.01:
                    details.append("Vegetation declining rapidly")
                if 'soil_moisture' in factors and factors['soil_moisture'] < 0.25:
                    details.append("Critically low soil moisture")
            elif hazard_type == 'flood':
                if 'recent_precipitation_3d' in factors and factors['recent_precipitation_3d'] > 50:
                    details.append(f"Heavy rainfall: {factors['recent_precipitation_3d']:.1f}mm in 3 days")
                if 'soil_saturation' in factors and factors['soil_saturation'] > 0.8:
                    details.append("Soil near saturation")
            elif hazard_type == 'earthquake':
                if 'recent_events' in factors and factors['recent_events'] > 10:
                    details.append(f"Increased seismic activity: {factors['recent_events']} events")
                if 'max_magnitude' in factors and factors['max_magnitude'] > 4.0:
                    details.append(f"Recent M{factors['max_magnitude']:.1f} earthquake")
        
        # Combine description with details
        full_description = base_desc
        if details:
            full_description += f" Key factors: {', '.join(details)}."
        
        full_description += f" Risk score: {risk_score:.2f}, Confidence: {confidence:.1%}."
        
        return full_description
    
    def _calculate_alert_duration(self, hazard_type: str, alert_level: str, confidence: float) -> int:
        """Calculate how long an alert should remain active."""
        
        base_durations = {
            'drought': {'Critical': 168, 'High': 120, 'Medium': 72, 'Low': 48},  # Hours
            'flood': {'Critical': 48, 'High': 36, 'Medium': 24, 'Low': 12},
            'earthquake': {'Critical': 72, 'High': 48, 'Medium': 24, 'Low': 12}
        }
        
        base_duration = base_durations.get(hazard_type, {}).get(alert_level, 24)
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.8)  # Range: 0.5 to 1.3
        
        return int(base_duration * confidence_multiplier)
    
    def _estimate_affected_population(self, lat: float, lon: float, hazard_type: str) -> Dict:
        """Estimate affected population based on location and hazard type."""
        
        # Simplified population density estimation based on major regions
        region_populations = {
            'tokyo': {'lat_range': (35.0, 36.0), 'lon_range': (139.0, 140.0), 'density': 6000},
            'sf_bay': {'lat_range': (37.0, 38.0), 'lon_range': (-123.0, -122.0), 'density': 1800},
            'bangladesh': {'lat_range': (23.0, 24.0), 'lon_range': (90.0, 91.0), 'density': 1200},
            'socal': {'lat_range': (33.5, 34.5), 'lon_range': (-119.0, -118.0), 'density': 1000},
            'istanbul': {'lat_range': (40.5, 41.0), 'lon_range': (28.5, 29.5), 'density': 2800}
        }
        
        # Default population density
        pop_density = 300  # people per km²
        
        # Check if location matches known high-density areas
        for region, data in region_populations.items():
            if (data['lat_range'][0] <= lat <= data['lat_range'][1] and
                data['lon_range'][0] <= lon <= data['lon_range'][1]):
                pop_density = data['density']
                break
        
        # Estimate affected area based on hazard type
        hazard_impact_radius = {
            'earthquake': 50,  # km radius
            'flood': 30,
            'drought': 100
        }
        
        radius = hazard_impact_radius.get(hazard_type, 50)
        area = np.pi * (radius ** 2)  # km²
        estimated_population = int(area * pop_density)
        
        return {
            'estimated_total': estimated_population,
            'impact_radius_km': radius,
            'population_density': pop_density,
            'uncertainty': 'High' if pop_density == 300 else 'Medium'
        }
    
    def _estimate_economic_impact(self, hazard_type: str, alert_level: str) -> Dict:
        """Estimate potential economic impact."""
        
        # Base economic impact estimates (in millions USD)
        impact_estimates = {
            'drought': {
                'Critical': {'min': 500, 'max': 5000, 'sector': 'Agriculture, Water'},
                'High': {'min': 100, 'max': 1000, 'sector': 'Agriculture'},
                'Medium': {'min': 20, 'max': 200, 'sector': 'Agriculture'},
                'Low': {'min': 5, 'max': 50, 'sector': 'Agriculture'}
            },
            'flood': {
                'Critical': {'min': 1000, 'max': 10000, 'sector': 'Infrastructure, Property'},
                'High': {'min': 200, 'max': 2000, 'sector': 'Property, Transport'},
                'Medium': {'min': 50, 'max': 500, 'sector': 'Transport, Business'},
                'Low': {'min': 10, 'max': 100, 'sector': 'Transport'}
            },
            'earthquake': {
                'Critical': {'min': 5000, 'max': 50000, 'sector': 'Infrastructure, Buildings'},
                'High': {'min': 1000, 'max': 10000, 'sector': 'Buildings, Utilities'},
                'Medium': {'min': 100, 'max': 1000, 'sector': 'Buildings'},
                'Low': {'min': 10, 'max': 100, 'sector': 'Utilities'}
            }
        }
        
        estimates = impact_estimates.get(hazard_type, {}).get(alert_level, 
                                                            {'min': 1, 'max': 10, 'sector': 'General'})
        
        return {
            'estimated_min_usd_millions': estimates['min'],
            'estimated_max_usd_millions': estimates['max'],
            'primary_affected_sectors': estimates['sector'],
            'uncertainty': 'Very High'  # Economic estimates are always uncertain
        }
    
    def get_active_alerts_enhanced(self) -> pd.DataFrame:
        """Get enhanced active alerts with additional information."""
        current_time = datetime.now()
        
        # Filter active, non-expired alerts
        active = [alert for alert in self.active_alerts 
                 if alert['expiry_time'] > current_time and alert['status'] == 'active']
        
        if not active:
            return pd.DataFrame({
                'Alert ID': pd.Series([], dtype='str'),
                'Location': pd.Series([], dtype='str'),
                'Hazard': pd.Series([], dtype='str'),
                'Level': pd.Series([], dtype='str'),
                'Risk Score': pd.Series([], dtype='float'),
                'Confidence': pd.Series([], dtype='float'),
                'Population': pd.Series([], dtype='int'),
                'Time Remaining': pd.Series([], dtype='str'),
                'Priority': pd.Series([], dtype='int')
            })
        
        # Convert to DataFrame with enhanced information
        df_data = []
        for alert in active:
            time_remaining = alert['expiry_time'] - current_time
            hours_remaining = int(time_remaining.total_seconds() / 3600)
            
            df_data.append({
                'Alert ID': alert['alert_id'][-8:],  # Last 8 characters
                'Location': alert['location_name'],
                'Hazard': alert['hazard_type'].title(),
                'Level': alert['alert_level'],
                'Risk Score': f"{alert['risk_score']:.2f}",
                'Confidence': f"{alert['confidence']:.1%}",
                'Population': alert['affected_population_estimate']['estimated_total'],
                'Time Remaining': f"{hours_remaining}h",
                'Priority': alert['priority']
            })
        
        df = pd.DataFrame(df_data)
        return df.sort_values('Priority', ascending=False).reset_index(drop=True)

    def get_alert_history_dataframe(self) -> pd.DataFrame:
        """
        Returns the alert history as a pandas DataFrame.
        """
        if not self.alert_history:
            return pd.DataFrame({
                'Start Time': pd.Series([], dtype='datetime64[ns]'),
                'End Time': pd.Series([], dtype='datetime64[ns]'),
                'Region': pd.Series([], dtype='str'),
                'Hazard Type': pd.Series([], dtype='str'),
                'Severity': pd.Series([], dtype='str')
            })
        
        # Prepare data for DataFrame
        df_data = []
        for alert in self.alert_history:
            df_data.append({
                'Start Time': alert['timestamp'],
                'End Time': alert['expiry_time'],
                'Region': alert['location_name'],
                'Hazard Type': alert['hazard_type'].title(),
                'Severity': alert['alert_level']
            })
        
        df = pd.DataFrame(df_data)
        return df.sort_values('Start Time', ascending=False).reset_index(drop=True)
    
    def get_location_risk_summary(self, lat: float, lon: float, location_name: str = None) -> Dict:
        """Get comprehensive risk summary for a specific location."""
        
        if location_name is None:
            location_name = f"Location ({lat:.2f}, {lon:.2f})"
        
        # Get predictions for all hazard types
        drought_risk = self.predictor.predict_drought_risk_live(lat, lon)
        flood_risk = self.predictor.predict_flood_risk_live(lat, lon)
        earthquake_risk = self.predictor.predict_earthquake_risk_live(lat, lon)
        
        # Calculate overall risk score
        risk_scores = [
            drought_risk.get('risk_probability', 0),
            flood_risk.get('risk_probability', 0),
            earthquake_risk.get('probability', 0)
        ]
        
        overall_risk = max(risk_scores)
        dominant_hazard = ['drought', 'flood', 'earthquake'][np.argmax(risk_scores)]
        
        # Determine overall risk level
        if overall_risk >= 0.7:
            overall_level = 'High'
        elif overall_risk >= 0.4:
            overall_level = 'Medium'
        else:
            overall_level = 'Low'
        
        return {
            'location': {'name': location_name, 'lat': lat, 'lon': lon},
            'overall_risk': {
                'level': overall_level,
                'score': float(overall_risk),
                'dominant_hazard': dominant_hazard
            },
            'individual_risks': {
                'drought': {
                    'level': drought_risk.get('risk_level', 'Low'),
                    'score': drought_risk.get('risk_probability', 0),
                    'confidence': drought_risk.get('confidence', 0)
                },
                'flood': {
                    'level': flood_risk.get('risk_level', 'Low'),
                    'score': flood_risk.get('risk_probability', 0),
                    'confidence': flood_risk.get('confidence', 0)
                },
                'earthquake': {
                    'level': earthquake_risk.get('risk_level', 'Low'),
                    'score': earthquake_risk.get('probability', 0),
                    'confidence': earthquake_risk.get('confidence', 0),
                    'magnitude_prediction': earthquake_risk.get('magnitude_prediction', 0)
                }
            },
            'recommendations': self._consolidate_recommendations([
                drought_risk.get('recommendation', []),
                flood_risk.get('recommendation', []),
                earthquake_risk.get('recommendation', [])
            ]),
            'last_updated': datetime.now().isoformat()
        }
    
    def _consolidate_recommendations(self, recommendation_lists: List[List[str]]) -> List[str]:
        """Consolidate recommendations from multiple hazard types."""
        all_recommendations = []
        for rec_list in recommendation_lists:
            all_recommendations.extend(rec_list)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    def update_alert_thresholds(self, hazard_type: str, level: str, new_threshold: float):
        """Update alert thresholds for a specific hazard and level."""
        threshold_dict = getattr(self.config, f'{hazard_type}_thresholds')
        if level in threshold_dict:
            threshold_dict[level] = new_threshold
            logging.info(f"Updated {hazard_type} {level} threshold to {new_threshold}")
    
    def escalate_alert(self, alert_id: str, reason: str) -> bool:
        """Manually escalate an alert to the next level."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                current_level = alert['alert_level']
                
                escalation_map = {'Low': 'Medium', 'Medium': 'High', 'High': 'Critical'}
                new_level = escalation_map.get(current_level)
                
                if new_level:
                    alert['alert_level'] = new_level
                    alert['update_count'] += 1
                    alert['escalation_history'].append({
                        'timestamp': datetime.now(),
                        'from_level': current_level,
                        'to_level': new_level,
                        'reason': reason
                    })
                    
                    # Update priority
                    level_multiplier = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
                    base_priority = self.hazard_priorities.get(alert['hazard_type'], 1)
                    alert['priority'] = base_priority * level_multiplier.get(new_level, 1)
                    
                    logging.info(f"Escalated alert {alert_id} from {current_level} to {new_level}")
                    return True
        
        return False
    
    def dismiss_alert(self, alert_id: str, reason: str = "Resolved") -> bool:
        """Dismiss an active alert."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'dismissed'
                alert['dismissed_time'] = datetime.now()
                alert['dismissed_reason'] = reason
                alert['update_count'] += 1
                
                logging.info(f"Dismissed alert {alert_id}: {reason}")
                return True
        
        return False
    
    def acknowledge_alert(self, alert_id: str, user: str, note: str = "") -> bool:
        """Acknowledge an alert by a user."""
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                acknowledgment = {
                    'user': user,
                    'timestamp': datetime.now(),
                    'note': note
                }
                alert['acknowledgments'].append(acknowledgment)
                alert['update_count'] += 1
                
                logging.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        
        return False
    
    def get_alert_statistics(self, days: int = 30) -> Dict:
        """Get comprehensive alert statistics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > cutoff_date]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'active_alerts': 0,
                'alerts_by_type': {},
                'alerts_by_level': {},
                'average_confidence': 0,
                'total_affected_population': 0,
                'average_response_time_minutes': 0
            }
        
        # Calculate statistics
        stats = {
            'total_alerts': len(recent_alerts),
            'active_alerts': len([a for a in self.active_alerts if a['status'] == 'active']),
            'alerts_by_type': {},
            'alerts_by_level': {},
            'alerts_by_location': {},
            'average_confidence': np.mean([a['confidence'] for a in recent_alerts]),
            'total_affected_population': sum([a['affected_population_estimate']['estimated_total'] 
                                            for a in recent_alerts]),
            'economic_impact_range': {
                'min': sum([a['economic_impact_estimate']['estimated_min_usd_millions'] 
                           for a in recent_alerts]),
                'max': sum([a['economic_impact_estimate']['estimated_max_usd_millions'] 
                           for a in recent_alerts])
            }
        }
        
        # Count by type, level, and location
        for alert in recent_alerts:
            hazard_type = alert['hazard_type']
            alert_level = alert['alert_level']
            location = alert['location_name']
            
            stats['alerts_by_type'][hazard_type] = stats['alerts_by_type'].get(hazard_type, 0) + 1
            stats['alerts_by_level'][alert_level] = stats['alerts_by_level'].get(alert_level, 0) + 1
            stats['alerts_by_location'][location] = stats['alerts_by_location'].get(location, 0) + 1
        
        return stats