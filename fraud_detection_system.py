import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from sqlalchemy.orm import Session
from models import Transaction, Alert, TransactionInsight, FraudCase
from agents import EnsembleAgent
from feature_engineering import FeatureEngineer
from config import MODEL_PARAMS

logger = logging.getLogger(__name__)

class FraudDetectionSystem:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.ensemble_agent = EnsembleAgent()
        self.feature_engineer = FeatureEngineer()
        self.is_initialized = False
    
    def initialize(self, historical_data: pd.DataFrame):
        """Initialize the system with historical data."""
        try:
            # Train the ensemble agent
            self.ensemble_agent.train(historical_data)
            self.is_initialized = True
            logger.info("Fraud detection system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing fraud detection system: {str(e)}")
            raise
    
    def process_transaction(self, transaction_data: Dict) -> Dict:
        """Process a new transaction and generate insights."""
        if not self.is_initialized:
            raise ValueError("System must be initialized before processing transactions")
        
        try:
            # Get historical transactions for the account
            historical_data = self._get_historical_data(transaction_data['account_id'])
            
            # Prepare features
            features = self.feature_engineer.prepare_features(transaction_data, historical_data)
            
            # Evaluate transaction using ensemble agent
            evaluation = self.ensemble_agent.evaluate_transaction(transaction_data, historical_data)
            
            # Generate insights
            insights = self._generate_insights(transaction_data, evaluation, features)
            
            # Create alert if necessary
            alert = self._create_alert(transaction_data, evaluation, insights)
            
            # Save to database
            self._save_to_database(transaction_data, evaluation, insights, alert)
            
            return {
                'transaction_id': transaction_data['transaction_id'],
                'risk_score': evaluation['ensemble_score'],
                'insights': insights,
                'alert': alert
            }
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            return {
                'transaction_id': transaction_data['transaction_id'],
                'error': str(e)
            }
    
    def _get_historical_data(self, account_id: str) -> pd.DataFrame:
        """Get historical transactions for an account."""
        try:
            transactions = self.db.query(Transaction).filter(
                Transaction.account_id == account_id
            ).order_by(Transaction.timestamp.desc()).limit(1000).all()
            
            return pd.DataFrame([{
                'transaction_id': t.transaction_id,
                'timestamp': t.timestamp,
                'amount': t.amount,
                'currency': t.currency,
                'transaction_type': t.transaction_type,
                'merchant_category': t.merchant_category,
                'location': t.location,
                'device_id': t.device_id,
                'latitude': t.latitude,
                'longitude': t.longitude,
                'is_online': t.is_online,
                'is_international': t.is_international
            } for t in transactions])
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_insights(self, transaction: Dict, evaluation: Dict, features: Dict) -> Dict:
        """Generate insights about the transaction."""
        risk_level = self._determine_risk_level(evaluation['ensemble_score'])
        
        return {
            'risk_level': risk_level,
            'risk_score': evaluation['ensemble_score'],
            'confidence_score': self._calculate_confidence_score(evaluation),
            'contributing_factors': self._format_contributing_factors(evaluation['factors']),
            'regulatory_concerns': self._check_regulatory_concerns(transaction, evaluation),
            'recommended_actions': self._generate_recommendations(risk_level, evaluation),
            'historical_context': self._generate_historical_context(transaction, features)
        }
    
    def _create_alert(self, transaction: Dict, evaluation: Dict, insights: Dict) -> Optional[Dict]:
        """Create an alert if the transaction is suspicious."""
        if evaluation['ensemble_score'] >= MODEL_PARAMS['alert_threshold']:
            return {
                'transaction_id': transaction['transaction_id'],
                'alert_type': 'high_risk',
                'severity': insights['risk_level'],
                'status': 'new',
                'details': {
                    'risk_score': evaluation['ensemble_score'],
                    'contributing_factors': insights['contributing_factors'],
                    'regulatory_concerns': insights['regulatory_concerns']
                }
            }
        return None
    
    def _save_to_database(self, transaction: Dict, evaluation: Dict, insights: Dict, alert: Optional[Dict]):
        """Save transaction data, evaluation, insights, and alert to database."""
        try:
            # Save transaction
            db_transaction = Transaction(
                transaction_id=transaction['transaction_id'],
                account_id=transaction['account_id'],
                timestamp=transaction['timestamp'],
                amount=transaction['amount'],
                currency=transaction['currency'],
                transaction_type=transaction['transaction_type'],
                merchant_category=transaction['merchant_category'],
                location=transaction['location'],
                device_id=transaction['device_id'],
                latitude=transaction.get('latitude'),
                longitude=transaction.get('longitude'),
                is_online=transaction.get('is_online', False),
                is_international=transaction.get('is_international', False),
                risk_score=evaluation['ensemble_score']
            )
            self.db.add(db_transaction)
            
            # Save insights
            db_insights = TransactionInsight(
                transaction_id=transaction['transaction_id'],
                risk_level=insights['risk_level'],
                confidence_score=insights['confidence_score'],
                contributing_factors=insights['contributing_factors'],
                regulatory_concerns=insights['regulatory_concerns'],
                recommended_actions=insights['recommended_actions'],
                historical_context=insights['historical_context']
            )
            self.db.add(db_insights)
            
            # Save alert if exists
            if alert:
                db_alert = Alert(
                    transaction_id=alert['transaction_id'],
                    alert_type=alert['alert_type'],
                    severity=alert['severity'],
                    status=alert['status'],
                    details=alert['details']
                )
                self.db.add(db_alert)
            
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving to database: {str(e)}")
            raise
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on risk score."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence_score(self, evaluation: Dict) -> float:
        """Calculate confidence score based on model agreement."""
        scores = [
            evaluation['anomaly_score'],
            evaluation['rule_based_score'],
            evaluation['behavioral_score']
        ]
        return 1 - np.std(scores)
    
    def _format_contributing_factors(self, factors: Dict) -> List[str]:
        """Format contributing factors into readable strings."""
        formatted_factors = []
        
        for category, factor_list in factors.items():
            if factor_list:
                formatted_factors.extend(factor_list)
        
        return formatted_factors
    
    def _check_regulatory_concerns(self, transaction: Dict, evaluation: Dict) -> List[str]:
        """Check for regulatory concerns."""
        concerns = []
        
        # Check for large transactions
        if transaction['amount'] > MODEL_PARAMS['large_transaction_threshold']:
            concerns.append(f"Large transaction amount: ${transaction['amount']:.2f}")
        
        # Check for international transactions
        if transaction.get('is_international', False):
            concerns.append("International transaction detected")
        
        # Check for multiple devices
        if evaluation['factors'].get('device_count', 0) > MODEL_PARAMS['max_devices_per_account']:
            concerns.append("Multiple devices used for account")
        
        return concerns
    
    def _generate_recommendations(self, risk_level: str, evaluation: Dict) -> List[str]:
        """Generate recommended actions based on risk level and evaluation."""
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.extend([
                "Block transaction immediately",
                "Contact customer for verification",
                "Review account for other suspicious activity",
                "Consider account suspension"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Flag transaction for review",
                "Request additional verification",
                "Monitor account activity closely"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "Review transaction details",
                "Monitor for similar transactions"
            ])
        
        return recommendations
    
    def _generate_historical_context(self, transaction: Dict, features: Dict) -> Dict:
        """Generate historical context for the transaction."""
        return {
            'transaction_frequency': features.get('transaction_frequency_24h', 0),
            'merchant_diversity': features.get('merchant_diversity', 0),
            'transaction_type_diversity': features.get('transaction_type_diversity', 0),
            'amount_patterns': {
                'mean': features.get('amount_mean', 0),
                'std': features.get('amount_std', 0),
                'max': features.get('amount_max', 0)
            }
        } 