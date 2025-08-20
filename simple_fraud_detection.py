import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging
import traceback

logger = logging.getLogger(__name__)

class SimpleFraudDetector:
    def __init__(self):
        logger.info("Initializing SimpleFraudDetector")
        try:
            self.scaler = StandardScaler()
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.lof = LocalOutlierFactor(contamination=0.1, novelty=False)
            logger.info("SimpleFraudDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SimpleFraudDetector: {str(e)}")
            raise
        
    def preprocess_data(self, df):
        """Simple preprocessing focusing on essential features."""
        logger.info("Starting data preprocessing")
        try:
            # Handle missing values
            df = df.fillna(0)
            logger.info("Filled missing values with 0")
            
            # Create basic features
            try:
                df['hour'] = pd.to_datetime(df['TransactionDate']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['TransactionDate']).dt.dayofweek
                logger.info("Created time-based features")
            except Exception as e:
                logger.error(f"Error creating time features: {str(e)}")
                raise ValueError(f"Error processing TransactionDate: {str(e)}")
            
            try:
                df['amount_zscore'] = (df['TransactionAmount'] - df['TransactionAmount'].mean()) / df['TransactionAmount'].std()
                logger.info("Calculated amount z-score")
            except Exception as e:
                logger.error(f"Error calculating amount z-score: {str(e)}")
                raise ValueError(f"Error processing TransactionAmount: {str(e)}")
            
            # Calculate transaction frequency
            df['transaction_frequency'] = df.groupby('AccountID')['TransactionID'].transform('count')
            
            # Calculate amount statistics
            df['avg_transaction_amount'] = df.groupby('AccountID')['TransactionAmount'].transform('mean')
            df['max_transaction_amount'] = df.groupby('AccountID')['TransactionAmount'].transform('max')
            
            # Device and location features
            df['device_count'] = df.groupby('AccountID')['DeviceID'].transform('nunique')
            df['location_count'] = df.groupby('AccountID')['Location'].transform('nunique')
            
            logger.info("Preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error during preprocessing: {str(e)}")
    
    def extract_features(self, df):
        """Extract essential features for fraud detection."""
        logger.info("Starting feature extraction")
        try:
            feature_columns = [
                'TransactionAmount', 'amount_zscore', 'hour', 'day_of_week',
                'transaction_frequency', 'avg_transaction_amount', 'max_transaction_amount',
                'device_count', 'location_count'
            ]
            
            # Verify all feature columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")
            
            X = df[feature_columns]
            
            # Check for infinite or NaN values
            if X.isnull().any().any() or np.isinf(X).any().any():
                logger.error("Dataset contains NaN or infinite values")
                raise ValueError("Dataset contains NaN or infinite values")
            
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Feature extraction completed successfully")
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error in extract_features: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error during feature extraction: {str(e)}")
    
    def detect_fraud(self, df):
        """Detect fraud using simple ensemble of Isolation Forest and LOF."""
        logger.info("Starting fraud detection")
        try:
            # Preprocess data
            df = self.preprocess_data(df)
            logger.info("Data preprocessing completed")
            
            # Extract features
            X = self.extract_features(df)
            logger.info("Feature extraction completed")
            
            # Get predictions from both models
            try:
                logger.info("Running Isolation Forest")
                iforest_scores = self.isolation_forest.fit_predict(X)
                logger.info("Isolation Forest completed")
                
                logger.info("Running Local Outlier Factor")
                lof_scores = self.lof.fit_predict(X)
                logger.info("Local Outlier Factor completed")
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                raise ValueError(f"Error in anomaly detection models: {str(e)}")
            
            # Combine scores (convert to 0 and 1 instead of 1 and -1)
            df['anomaly_score'] = (
                (iforest_scores == -1).astype(int) + 
                (lof_scores == -1).astype(int)
            ) / 2
            
            # Calculate amount-based risk score
            df['amount_risk'] = df['TransactionAmount'].apply(lambda x: min(x / 1000, 1))  # Normalize amount to 0-1 scale
            
            # Combine anomaly and amount risk scores
            df['combined_risk'] = (df['anomaly_score'] * 0.7 + df['amount_risk'] * 0.3)
            
            # Assign risk labels with more granular thresholds
            df['risk_label'] = 'C'  # Default low risk
            df.loc[df['combined_risk'] >= 0.7, 'risk_label'] = 'A'  # High risk
            df.loc[(df['combined_risk'] >= 0.4) & (df['combined_risk'] < 0.7), 'risk_label'] = 'B'  # Medium risk
            
            # Override risk label for very low amounts
            df.loc[df['TransactionAmount'] < 100, 'risk_label'] = 'C'  # Force low risk for small amounts
            
            # Calculate confidence score based on multiple factors
            df['confidence_score'] = (
                df['combined_risk'] * 0.6 +  # Combined risk score
                (df['amount_zscore'].abs() / 10).clip(0, 1) * 0.2 +  # Amount z-score
                (df['transaction_frequency'] / df['transaction_frequency'].max()).clip(0, 1) * 0.2  # Transaction frequency
            )
            df['confidence_score'] = df['confidence_score'].clip(0, 1)
            
            logger.info(f"Fraud detection completed. Found {len(df[df['risk_label'] == 'A'])} high-risk and {len(df[df['risk_label'] == 'B'])} medium-risk transactions")
            return df
            
        except Exception as e:
            logger.error(f"Error in detect_fraud: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error during fraud detection: {str(e)}")
    
    def get_risk_factors(self, transaction):
        """Get top risk factors for a transaction."""
        factors = []
        
        # Check amount
        if abs(transaction['amount_zscore']) > 2:
            factors.append({
                'feature': 'Transaction Amount',
                'value': transaction['TransactionAmount'],
                'zscore': transaction['amount_zscore']
            })
        
        # Check device count
        if transaction['device_count'] > 1:
            factors.append({
                'feature': 'Multiple Devices',
                'value': transaction['device_count'],
                'zscore': (transaction['device_count'] - 1) * 2
            })
        
        # Check location count
        if transaction['location_count'] > 1:
            factors.append({
                'feature': 'Multiple Locations',
                'value': transaction['location_count'],
                'zscore': (transaction['location_count'] - 1) * 2
            })
        
        return sorted(factors, key=lambda x: abs(x['zscore']), reverse=True)[:3]
    
    def get_transaction_insights(self, df, transaction_id):
        """Get insights for a specific transaction."""
        transaction = df[df['TransactionID'] == transaction_id].iloc[0]
        
        return {
            'transaction_id': transaction_id,
            'timestamp': transaction['TransactionDate'],
            'risk_level': transaction['risk_label'],
            'risk_score': float(transaction['anomaly_score']),
            'confidence_score': float(transaction['confidence_score']),
            'top_contributing_factors': self.get_risk_factors(transaction),
            'regulatory_concerns': self._get_regulatory_concerns(transaction),
            'recommended_actions': self._get_recommended_actions(transaction['risk_label'])
        }
    
    def _get_regulatory_concerns(self, transaction):
        """Get regulatory concerns for a transaction."""
        concerns = []
        
        if abs(transaction['amount_zscore']) > 2:
            concerns.append(f"Large transaction amount: ${transaction['TransactionAmount']:.2f}")
        
        if transaction['device_count'] > 1:
            concerns.append(f"Multiple devices used: {transaction['device_count']}")
        
        if transaction['location_count'] > 1:
            concerns.append(f"Multiple locations: {transaction['location_count']}")
        
        return concerns
    
    def _get_recommended_actions(self, risk_level):
        """Get recommended actions based on risk level."""
        if risk_level == 'A':
            return [
                "Block transaction immediately",
                "Contact customer for verification",
                "Review account for suspicious activity"
            ]
        elif risk_level == 'B':
            return [
                "Flag transaction for review",
                "Monitor account activity"
            ]
        else:
            return ["Normal processing"] 