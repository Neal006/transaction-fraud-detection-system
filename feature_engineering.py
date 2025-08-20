import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from config import FEATURE_PARAMS

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_params = FEATURE_PARAMS
    
    def prepare_features(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Prepare features for a single transaction."""
        try:
            features = {}
            
            # Basic transaction features
            features.update(self._extract_basic_features(transaction))
            
            # Time-based features
            features.update(self._extract_time_features(transaction, historical_data))
            
            # Amount-based features
            features.update(self._extract_amount_features(transaction, historical_data))
            
            # Location-based features
            features.update(self._extract_location_features(transaction, historical_data))
            
            # Device-based features
            features.update(self._extract_device_features(transaction, historical_data))
            
            # Account-based features
            features.update(self._extract_account_features(transaction, historical_data))
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return {}
    
    def _extract_basic_features(self, transaction: Dict) -> Dict:
        """Extract basic transaction features."""
        return {
            'amount': transaction['amount'],
            'currency': transaction['currency'],
            'transaction_type': transaction['transaction_type'],
            'merchant_category': transaction['merchant_category'],
            'is_online': transaction.get('is_online', False),
            'is_international': transaction.get('is_international', False)
        }
    
    def _extract_time_features(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Extract time-based features."""
        timestamp = transaction['timestamp']
        
        # Time of day features
        features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'is_holiday': self._is_holiday(timestamp)
        }
        
        # Time since last transaction
        if not historical_data.empty:
            last_transaction = historical_data['timestamp'].max()
            features['time_since_last_transaction'] = (timestamp - last_transaction).total_seconds()
        
        return features
    
    def _extract_amount_features(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Extract amount-based features."""
        features = {}
        
        if not historical_data.empty:
            # Amount statistics
            amount_stats = historical_data['amount'].describe()
            features.update({
                'amount_mean': amount_stats['mean'],
                'amount_std': amount_stats['std'],
                'amount_max': amount_stats['max'],
                'amount_min': amount_stats['min'],
                'amount_median': amount_stats['50%']
            })
            
            # Amount ratios
            features['amount_to_mean_ratio'] = transaction['amount'] / amount_stats['mean']
            features['amount_to_max_ratio'] = transaction['amount'] / amount_stats['max']
            
            # Rolling statistics
            for window in self.feature_params['amount_windows']:
                rolling_mean = historical_data['amount'].rolling(window=window).mean().iloc[-1]
                rolling_std = historical_data['amount'].rolling(window=window).std().iloc[-1]
                features[f'rolling_mean_{window}'] = rolling_mean
                features[f'rolling_std_{window}'] = rolling_std
                features[f'amount_to_rolling_mean_{window}'] = transaction['amount'] / rolling_mean
        
        return features
    
    def _extract_location_features(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Extract location-based features."""
        features = {}
        
        if not historical_data.empty:
            # Location frequency
            location_counts = historical_data['location'].value_counts()
            features['location_frequency'] = location_counts.get(transaction['location'], 0)
            
            # Distance from last transaction
            if 'latitude' in transaction and 'longitude' in transaction:
                last_transaction = historical_data.iloc[-1]
                if 'latitude' in last_transaction and 'longitude' in last_transaction:
                    distance = self._calculate_distance(
                        transaction['latitude'], transaction['longitude'],
                        last_transaction['latitude'], last_transaction['longitude']
                    )
                    features['distance_from_last_transaction'] = distance
        
        return features
    
    def _extract_device_features(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Extract device-based features."""
        features = {}
        
        if not historical_data.empty:
            # Device frequency
            device_counts = historical_data['device_id'].value_counts()
            features['device_frequency'] = device_counts.get(transaction['device_id'], 0)
            
            # Device count
            features['device_count'] = len(historical_data['device_id'].unique())
            
            # New device flag
            features['is_new_device'] = transaction['device_id'] not in historical_data['device_id'].unique()
        
        return features
    
    def _extract_account_features(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Extract account-based features."""
        features = {}
        
        if not historical_data.empty:
            # Transaction frequency
            for window in self.feature_params['frequency_windows']:
                recent_transactions = historical_data[
                    historical_data['timestamp'] > transaction['timestamp'] - timedelta(hours=window)
                ]
                features[f'transaction_frequency_{window}h'] = len(recent_transactions)
            
            # Merchant diversity
            features['merchant_diversity'] = len(historical_data['merchant_category'].unique())
            
            # Transaction type diversity
            features['transaction_type_diversity'] = len(historical_data['transaction_type'].unique())
        
        return features
    
    def _is_holiday(self, date: datetime) -> bool:
        """Check if a date is a holiday."""
        # Implement holiday checking logic
        # This is a simplified version - you might want to use a proper holiday calendar
        return False
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c 