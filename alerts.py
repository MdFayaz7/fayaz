import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import logging

class AlertSystem:
    """
    Manages alert generation and notification system for multi-hazard predictions.
    """
    
    def __init__(self):
        self.alert_thresholds = {
            'drought': {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.9},
            'flood': {'low': 0.4, 'medium': 0.6, 'high': 0.8, 'critical': 0.95},
            'earthquake': {'low': 0.2, 'medium': 0.4, 'high': 0.6, 'critical': 0.8}
        }
        
        self.alert_history = []
        self.active_alerts = []
        
        # Alert priority weights
        self.priority_weights = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        # Generate mock alert history for demonstration
        self._generate_mock_alert_history()
    
    def evaluate_prediction(self, prediction: Dict) -> Dict:
        """
        Evaluate a prediction and determine if an alert should be generated.
        """
        hazard_type = prediction.get('hazard_type', '').lower()
        risk_score = prediction.get('risk_score', 0)
        confidence = prediction.get('confidence', 0)
        region = prediction.get('region', 'Unknown')
        
        # Get thresholds for this hazard type
        thresholds = self.alert_thresholds.get(hazard_type, self.alert_thresholds['drought'])
        
        # Determine alert level
        alert_level = self._calculate_alert_level(risk_score, thresholds)
        
        # Check if alert should be triggered
        should_alert = self._should_trigger_alert(alert_level, confidence, hazard_type, region)
        
        if should_alert:
            alert = self._create_alert(prediction, alert_level)
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            return alert
        
        return {}
    
    def _calculate_alert_level(self, risk_score: float, thresholds: Dict) -> str:
        """
        Calculate alert level based on risk score and thresholds.
        """
        if risk_score >= thresholds['critical']:
            return 'critical'
        elif risk_score >= thresholds['high']:
            return 'high'
        elif risk_score >= thresholds['medium']:
            return 'medium'
        elif risk_score >= thresholds['low']:
            return 'low'
        else:
            return 'none'
    
    def _should_trigger_alert(self, alert_level: str, confidence: float, 
                            hazard_type: str, region: str) -> bool:
        """
        Determine if an alert should be triggered based on various factors.
        """
        # Don't trigger for 'none' level
        if alert_level == 'none':
            return False
        
        # Always trigger for critical alerts
        if alert_level == 'critical':
            return True
        
        # Confidence-based triggering
        confidence_thresholds = {
            'high': 0.6,
            'medium': 0.7,
            'low': 0.8
        }
        
        min_confidence = confidence_thresholds.get(alert_level, 0.7)
        if confidence < min_confidence:
            return False
        
        # Check for duplicate alerts in the same region
        recent_alerts = [a for a in self.active_alerts 
                        if a['region'] == region and 
                        a['hazard_type'] == hazard_type and
                        (datetime.now() - a['timestamp']).total_seconds() < 86400]  # 24 hours
        
        if recent_alerts:
            return False
        
        return True
    
    def _create_alert(self, prediction: Dict, alert_level: str) -> Dict:
        """
        Create an alert object from prediction data.
        """
        alert_id = f"ALT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'hazard_type': prediction.get('hazard_type', 'unknown'),
            'region': prediction.get('region', 'Unknown'),
            'latitude': prediction.get('latitude', 0),
            'longitude': prediction.get('longitude', 0),
            'alert_level': alert_level,
            'risk_score': prediction.get('risk_score', 0),
            'confidence': prediction.get('confidence', 0),
            'description': self._generate_alert_description(prediction, alert_level),
            'recommended_actions': self._get_recommended_actions(prediction['hazard_type'], alert_level),
            'expiry_time': datetime.now() + timedelta(hours=self._get_alert_duration(alert_level)),
            'status': 'active',
            'priority': self.priority_weights[alert_level]
        }
        
        return alert
    
    def _generate_alert_description(self, prediction: Dict, alert_level: str) -> str:
        """
        Generate human-readable alert description.
        """
        hazard_type = prediction.get('hazard_type', 'unknown').title()
        region = prediction.get('region', 'Unknown')
        risk_score = prediction.get('risk_score', 0)
        
        descriptions = {
            'critical': f"CRITICAL {hazard_type} alert for {region}. Immediate action required. Risk score: {risk_score:.2f}",
            'high': f"HIGH {hazard_type} risk detected in {region}. Prepare emergency response. Risk score: {risk_score:.2f}",
            'medium': f"MEDIUM {hazard_type} risk in {region}. Monitor situation closely. Risk score: {risk_score:.2f}",
            'low': f"LOW {hazard_type} risk identified in {region}. Advisory level. Risk score: {risk_score:.2f}"
        }
        
        return descriptions.get(alert_level, f"{hazard_type} alert for {region}")
    
    def _get_recommended_actions(self, hazard_type: str, alert_level: str) -> List[str]:
        """
        Get recommended actions based on hazard type and alert level.
        """
        actions = {
            'drought': {
                'critical': [
                    "Implement emergency water conservation measures",
                    "Activate drought emergency response plan",
                    "Consider water rationing",
                    "Monitor agricultural impacts"
                ],
                'high': [
                    "Increase water conservation efforts",
                    "Monitor reservoir levels",
                    "Issue water use restrictions",
                    "Alert agricultural sector"
                ],
                'medium': [
                    "Monitor precipitation levels",
                    "Prepare water conservation measures",
                    "Review drought contingency plans"
                ],
                'low': [
                    "Continue routine monitoring",
                    "Inform stakeholders of conditions"
                ]
            },
            'flood': {
                'critical': [
                    "Issue immediate evacuation orders",
                    "Activate emergency response teams",
                    "Close flood-prone roads",
                    "Open emergency shelters"
                ],
                'high': [
                    "Issue flood warnings",
                    "Prepare evacuation routes",
                    "Monitor river levels",
                    "Alert emergency services"
                ],
                'medium': [
                    "Issue flood watch",
                    "Monitor weather conditions",
                    "Prepare sandbags and barriers"
                ],
                'low': [
                    "Monitor precipitation forecasts",
                    "Check drainage systems"
                ]
            },
            'earthquake': {
                'critical': [
                    "Issue earthquake alert",
                    "Review building safety protocols",
                    "Prepare emergency response teams",
                    "Check critical infrastructure"
                ],
                'high': [
                    "Monitor seismic activity",
                    "Alert emergency services",
                    "Review evacuation plans"
                ],
                'medium': [
                    "Continue seismic monitoring",
                    "Review preparedness plans"
                ],
                'low': [
                    "Routine seismic monitoring",
                    "Maintain awareness"
                ]
            }
        }
        
        return actions.get(hazard_type, {}).get(alert_level, ["Monitor situation"])
    
    def _get_alert_duration(self, alert_level: str) -> int:
        """
        Get alert duration in hours based on alert level.
        """
        durations = {
            'critical': 72,  # 3 days
            'high': 48,      # 2 days
            'medium': 24,    # 1 day
            'low': 12        # 12 hours
        }
        return durations.get(alert_level, 24)
    
    def get_active_alerts(self) -> pd.DataFrame:
        """
        Get all active alerts as a DataFrame.
        """
        current_time = datetime.now()
        
        # Filter out expired alerts
        active = [alert for alert in self.active_alerts 
                 if alert['expiry_time'] > current_time and alert['status'] == 'active']
        
        if not active:
            # Return empty DataFrame with proper columns
            return pd.DataFrame({
                'Alert ID': pd.Series([], dtype='str'),
                'Hazard Type': pd.Series([], dtype='str'),
                'Region': pd.Series([], dtype='str'),
                'Level': pd.Series([], dtype='str'),
                'Risk Score': pd.Series([], dtype='str'),
                'Timestamp': pd.Series([], dtype='str'),
                'Status': pd.Series([], dtype='str')
            })
        
        # Convert to DataFrame
        df_data = []
        for alert in active:
            df_data.append({
                'Alert ID': alert['alert_id'],
                'Hazard Type': alert['hazard_type'].title(),
                'Region': alert['region'],
                'Level': alert['alert_level'].title(),
                'Risk Score': f"{alert['risk_score']:.2f}",
                'Timestamp': alert['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Status': alert['status'].title()
            })
        
        return pd.DataFrame(df_data)
    
    def get_alert_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get alert history for the specified number of days.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_alerts = [alert for alert in self.alert_history 
                        if alert['timestamp'] > cutoff_date]
        
        if not recent_alerts:
            # Generate mock data for demonstration
            return self._generate_mock_alert_history_df()
        
        # Convert to DataFrame
        df_data = []
        for alert in recent_alerts:
            df_data.append({
                'region': alert['region'],
                'hazard_type': alert['hazard_type'],
                'severity': alert['alert_level'],
                'start_time': alert['timestamp'],
                'end_time': alert['expiry_time'],
                'risk_score': alert['risk_score']
            })
        
        return pd.DataFrame(df_data)
    
    def _generate_mock_alert_history(self):
        """
        Generate mock alert history for demonstration.
        """
        regions = [
            'California Central Valley', 'Amazon Basin', 'Sahel Region',
            'Ganges Delta', 'Ring of Fire - Japan', 'Australian Outback'
        ]
        
        hazard_types = ['drought', 'flood', 'earthquake']
        severity_levels = ['low', 'medium', 'high', 'critical']
        
        # Generate 50 mock alerts over the past 30 days
        for i in range(50):
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            )
            
            hazard_type = np.random.choice(hazard_types)
            severity = np.random.choice(severity_levels, p=[0.5, 0.3, 0.15, 0.05])
            
            alert = {
                'alert_id': f"MOCK_{i:03d}",
                'timestamp': timestamp,
                'hazard_type': hazard_type,
                'region': np.random.choice(regions),
                'latitude': np.random.uniform(-60, 60),
                'longitude': np.random.uniform(-180, 180),
                'alert_level': severity,
                'risk_score': np.random.uniform(0.1, 1.0),
                'confidence': np.random.uniform(0.6, 0.95),
                'expiry_time': timestamp + timedelta(hours=self._get_alert_duration(severity)),
                'status': np.random.choice(['resolved', 'expired', 'active'], p=[0.7, 0.2, 0.1])
            }
            
            self.alert_history.append(alert)
    
    def _generate_mock_alert_history_df(self) -> pd.DataFrame:
        """
        Generate mock alert history DataFrame for demonstration.
        """
        regions = [
            'California Central Valley', 'Amazon Basin', 'Sahel Region',
            'Ganges Delta', 'Ring of Fire - Japan', 'Australian Outback'
        ]
        
        data = []
        for i in range(20):
            start_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
            duration = np.random.randint(6, 72)  # 6 to 72 hours
            
            data.append({
                'region': np.random.choice(regions),
                'hazard_type': np.random.choice(['drought', 'flood', 'earthquake']),
                'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                           p=[0.5, 0.3, 0.15, 0.05]),
                'start_time': start_time,
                'end_time': start_time + timedelta(hours=duration),
                'risk_score': np.random.uniform(0.2, 1.0)
            })
        
        return pd.DataFrame(data)
    
    def update_alert_thresholds(self, hazard_type: str, thresholds: Dict):
        """
        Update alert thresholds for a specific hazard type.
        """
        if hazard_type in self.alert_thresholds:
            self.alert_thresholds[hazard_type].update(thresholds)
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Dismiss an active alert.
        """
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'dismissed'
                alert['dismissed_time'] = datetime.now()
                return True
        return False
    
    def get_alert_statistics(self) -> Dict:
        """
        Get alert statistics for dashboard display.
        """
        if not self.alert_history:
            self._generate_mock_alert_history()
        
        recent_alerts = [a for a in self.alert_history 
                        if (datetime.now() - a['timestamp']).days <= 30]
        
        stats = {
            'total_alerts_30_days': len(recent_alerts),
            'active_alerts': len([a for a in self.active_alerts if a['status'] == 'active']),
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'average_confidence': 0,
            'response_time_hours': np.random.uniform(2, 8)  # Mock response time
        }
        
        if recent_alerts:
            # Count by hazard type
            for alert in recent_alerts:
                hazard_type = alert['hazard_type']
                stats['alerts_by_type'][hazard_type] = stats['alerts_by_type'].get(hazard_type, 0) + 1
            
            # Count by severity
            for alert in recent_alerts:
                severity = alert['alert_level']
                stats['alerts_by_severity'][severity] = stats['alerts_by_severity'].get(severity, 0) + 1
            
            # Average confidence
            confidences = [alert['confidence'] for alert in recent_alerts if 'confidence' in alert]
            stats['average_confidence'] = np.mean(confidences) if confidences else 0.8
        
        return stats
