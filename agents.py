import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from config import MODEL_PARAMS, FEATURE_PARAMS, REGULATORY_RULES

logger = logging.getLogger(__name__)

class AnomalyDetectionAgent:
    def __init__(self):
        """Initialize the anomaly detection models."""
        self.isolation_forest = IsolationForest(**MODEL_PARAMS['isolation_forest'])
        self.pca = PCA(n_components=MODEL_PARAMS['autoencoder']['encoding_dim'])
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, historical_data: pd.DataFrame):
        """Train the anomaly detection models on historical data."""
        try:
            # Prepare features
            features = self._prepare_features(historical_data)
            
            # Train Isolation Forest
            self.isolation_forest.fit(features)
            
            # Train PCA for reconstruction
            self.pca.fit(features)
            
            self.is_trained = True
            logger.info("Anomaly detection models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training anomaly detection models: {str(e)}")
            raise
    
    def detect_anomalies(self, transaction_data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Detect anomalies in transaction data."""
        if not self.is_trained:
            raise ValueError("Models must be trained before detecting anomalies")
        
        try:
            # Prepare features
            features = self._prepare_features(transaction_data)
            
            # Get anomaly scores from Isolation Forest
            iso_scores = self.isolation_forest.score_samples(features)
            
            # Get reconstruction error from PCA
            reconstructed = self.pca.inverse_transform(self.pca.transform(features))
            reconstruction_error = np.mean(np.square(features - reconstructed), axis=1)
            
            # Combine scores
            iso_score_normalized = -iso_scores  # Higher values indicate anomalies
            combined_score = np.mean([iso_score_normalized, reconstruction_error], axis=0)
            normalized_score = (
                np.clip(combined_score / combined_score.max(), 0, 1)
                if combined_score.max() > 0
                else combined_score
            )
            
            # Identify contributing factors
            factors = self._identify_contributing_factors(transaction_data, combined_score)
            
            return float(normalized_score[0]), factors
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return 0.0, []
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare and scale numerical features."""
        numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
        if not hasattr(self.scaler, 'mean_'):
            features = self.scaler.fit_transform(data[numerical_features])
        else:
            features = self.scaler.transform(data[numerical_features])
        return features
    
    def _identify_contributing_factors(self, data: pd.DataFrame, scores: np.ndarray) -> List[str]:
        """Identify features contributing to anomaly scores."""
        factors = []
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            if z_scores.mean() > 2:  # Threshold for significant deviation
                factors.append(f"{col}: {z_scores.mean():.2f}")
        return factors

class RuleBasedAgent:
    def __init__(self):
        self.rules = REGULATORY_RULES
    
    def evaluate_transaction(self, transaction: Dict) -> Tuple[float, List[str]]:
        """Evaluate transaction against rule-based criteria."""
        violations = []
        score = 0.0
        
        try:
            # Check transaction amount
            if transaction['amount'] > self.rules['large_transaction_threshold']:
                violations.append(f"Large transaction amount: ${transaction['amount']:.2f}")
                score += 0.3
            
            # Check transaction timing
            hour = transaction['timestamp'].hour
            if hour >= self.rules['unusual_time_start'] or hour <= self.rules['unusual_time_end']:
                violations.append(f"Unusual transaction time: {hour:02d}:00")
                score += 0.2
            
            # Check device count
            if transaction.get('device_count', 0) > self.rules['max_devices_per_account']:
                violations.append(f"Multiple devices used: {transaction['device_count']}")
                score += 0.2
            
            # Check IP addresses
            if transaction.get('ip_count', 0) > self.rules['max_ip_addresses_per_account']:
                violations.append(f"Multiple IP addresses: {transaction['ip_count']}")
                score += 0.2
            
            # Check rapid transactions
            if transaction.get('time_since_last_transaction', float('inf')) < self.rules['rapid_transaction_window']:
                violations.append("Rapid transaction detected")
                score += 0.1
            
            return min(score, 1.0), violations
            
        except Exception as e:
            logger.error(f"Error in rule-based evaluation: {str(e)}")
            return 0.0, []
    def __init__(self):
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, historical_data: pd.DataFrame):
        """Train the behavioral analysis model."""
        try:
            # Prepare features and labels
            X, y = self._prepare_training_data(historical_data)
            
            # Train model
            self.mlp_model.fit(X, y)
            
            self.is_trained = True
            logger.info("Behavioral analysis model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training behavioral analysis model: {str(e)}")
            raise
    
    def analyze_behavior(self, transaction: Dict, historical_data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Analyze transaction behavior against historical patterns."""
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing behavior")
        
        try:
            # Prepare features for prediction
            X = self._prepare_features(transaction, historical_data)
            
            # Get prediction probability
            score = float(self.mlp_model.predict_proba(X)[0][1])
            
            # Identify behavioral patterns
            patterns = self._identify_patterns(transaction, historical_data)
            
            return score, patterns
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {str(e)}")
            return 0.0, []
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        # Extract relevant features
        features = []
        labels = []
        
        for _, row in data.iterrows():
            feature_vector = self._extract_features(row)
            features.append(feature_vector)
            labels.append(row.get('is_fraud', 0))
        
        # Scale features
        X = self.scaler.fit_transform(np.array(features))
        y = np.array(labels)
        
        return X, y
    

class BehavioralAnalysisAgent:
    def _prepare_features(self, transaction: Dict, historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for a single transaction."""
        feature_vector = self._extract_features(transaction)
        return self.scaler.transform([feature_vector])
    
    def _extract_features(self, data: Dict) -> np.ndarray:
        """Extract relevant features from transaction data."""
        features = []
        
        # Transaction amount
        features.append(float(data.get('amount', 0)))
        
        # Time-based features
        timestamp = data.get('timestamp', datetime.now())
        features.extend([
            timestamp.hour,
            timestamp.weekday(),
            timestamp.month
        ])
        
        # Device and IP features
        features.extend([
            float(data.get('device_count', 0)),
            float(data.get('ip_count', 0))
        ])
        
        return np.array(features)

class EnsembleAgent:
    def __init__(self):
        self.anomaly_agent = AnomalyDetectionAgent()
        self.rule_agent = RuleBasedAgent()
        self.behavioral_agent = BehavioralAnalysisAgent()
        self.weights = {
            'anomaly': 0.4,
            'rule_based': 0.3,
            'behavioral': 0.3
        }
    
    def train(self, historical_data: pd.DataFrame):
        """Train all agents."""
        try:
            self.anomaly_agent.train(historical_data)
            self.behavioral_agent.train(historical_data)
            logger.info("All agents trained successfully")
        except Exception as e:
            logger.error(f"Error training agents: {str(e)}")
            raise
    
    def evaluate_transaction(self, transaction: Dict, historical_data: pd.DataFrame) -> Dict:
        """Evaluate transaction using all agents."""
        try:
            # Get scores from each agent
            anomaly_score, anomaly_factors = self.anomaly_agent.detect_anomalies(
                pd.DataFrame([transaction])
            )
            
            rule_score, rule_violations = self.rule_agent.evaluate_transaction(transaction)
            
            behavioral_score, behavioral_patterns = self.behavioral_agent.analyze_behavior(
                transaction, historical_data
            )
            
            # Calculate ensemble score
            ensemble_score = (
                self.weights['anomaly'] * anomaly_score +
                self.weights['rule_based'] * rule_score +
                self.weights['behavioral'] * behavioral_score
            )
            
            # Combine all factors
            all_factors = {
                'anomaly_factors': anomaly_factors,
                'rule_violations': rule_violations,
                'behavioral_patterns': behavioral_patterns
            }
            
            return {
                'ensemble_score': ensemble_score,
                'anomaly_score': anomaly_score,
                'rule_based_score': rule_score,
                'behavioral_score': behavioral_score,
                'factors': all_factors
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble evaluation: {str(e)}")
            return {
                'ensemble_score': 0.0,
                'anomaly_score': 0.0,
                'rule_based_score': 0.0,
                'behavioral_score': 0.0,
                'factors': {}
            } 