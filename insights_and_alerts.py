import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, jsonify, request
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FraudInsights:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.transaction_history = {}
    
    def generate_insights(self, transaction_id: str) -> Dict:
        try:
            transaction = self.df[self.df['TransactionID'] == transaction_id].iloc[0]
            
            # Get historical context
            historical_context = self._get_historical_context(transaction)
            
            # Generate insights
            insights = {
                'transaction_id': str(transaction_id),
                'timestamp': str(transaction['TransactionDate']),
                'risk_level': str(transaction['fraud_label']),
                'risk_score': float(transaction['ensemble_score']),
                'confidence_score': float(transaction['confidence_score']),
                'top_contributing_factors': self._parse_factors(transaction['top_parameters']),
                'regulatory_concerns': self._get_regulatory_concerns(transaction),
                'recommended_actions': self._get_recommended_actions(transaction),
                'historical_context': historical_context
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights for transaction {transaction_id}: {str(e)}")
            return {
                'transaction_id': str(transaction_id),
                'error': str(e)
            }
    
    def _parse_factors(self, factors_str: str) -> List[Dict]:
        try:
            factors = []
            for factor in factors_str.split(' | '):
                factor = factor.strip()  # Remove leading/trailing spaces
                try:
                    # Split only on the first ': ' to separate feature from rest
                    parts = factor.split(': ', 1)
                    if len(parts) != 2:
                        raise ValueError("Invalid factor format: missing or malformed separator")
                    feature, rest = parts
                    feature = feature.strip()
                    rest = rest.strip()

                    # Split only on the first ' (z-score: ' to separate value from z-score
                    value_parts = rest.split(' (z-score: ', 1)
                    if len(value_parts) != 2:
                        raise ValueError("Invalid z-score format")
                    value_str, zscore_str = value_parts
                    value_str = value_str.strip()
                    zscore_str = zscore_str.rstrip(')').strip()

                    # Convert to floats
                    value = float(value_str)
                    zscore = float(zscore_str)

                    factors.append({
                        'feature': feature,
                        'value': value,
                        'zscore': zscore
                    })
                except Exception as e:
                    logger.error(f"Error parsing factor: {factor}, Error: {str(e)}")
                    continue
            return factors
        except Exception as e:
            logger.error(f"Error parsing factors string: {str(e)}")
            return []
    
    def _get_historical_context(self, transaction: pd.Series) -> Dict:
        try:
            account_transactions = self.df[self.df['AccountID'] == transaction['AccountID']]
            recent_transactions = account_transactions[
                account_transactions['TransactionDate'] <= transaction['TransactionDate']
            ].tail(10)
            
            return {
                'total_transactions': int(len(account_transactions)),
                'previous_fraud_incidents': int(len(account_transactions[account_transactions['fraud_label'] != '0'])),
                'average_transaction_amount': float(account_transactions['TransactionAmount'].mean()),
                'recent_transaction_count': int(len(recent_transactions)),
                'recent_fraud_rate': float((recent_transactions['fraud_label'] != '0').mean())
            }
        except Exception as e:
            logger.error(f"Error getting historical context: {str(e)}")
            return {
                'total_transactions': 0,
                'previous_fraud_incidents': 0,
                'average_transaction_amount': 0.0,
                'recent_transaction_count': 0,
                'recent_fraud_rate': 0.0
            }
    
    def _get_regulatory_concerns(self, transaction: pd.Series) -> List[str]:
        concerns = []
        try:
            if transaction['TransactionAmount'] > 10000:
                concerns.append(f"Large transaction amount: ${transaction['TransactionAmount']:.2f}")
            
            if transaction.get('is_international', False):
                concerns.append("International transaction detected")
            
            if transaction.get('device_count', 0) > 3:
                concerns.append(f"Multiple devices used: {transaction['device_count']}")
            
            if transaction.get('ip_count', 0) > 3:
                concerns.append(f"Multiple IP addresses: {transaction['ip_count']}")
                
        except Exception as e:
            logger.error(f"Error getting regulatory concerns: {str(e)}")
        
        return concerns
    
    def _get_recommended_actions(self, transaction: pd.Series) -> List[str]:
        actions = []
        try:
            risk_level = transaction['fraud_label']
            
            if risk_level == 'A':
                actions.extend([
                    "Block transaction immediately",
                    "Contact customer for verification",
                    "Review account for other suspicious activity",
                    "Consider account suspension"
                ])
            elif risk_level == 'B':
                actions.extend([
                    "Flag transaction for review",
                    "Request additional verification",
                    "Monitor account activity closely"
                ])
            elif risk_level == 'C':
                actions.extend([
                    "Review transaction details",
                    "Monitor for similar transactions"
                ])
                
        except Exception as e:
            logger.error(f"Error getting recommended actions: {str(e)}")
        
        return actions

class AlertSystem:
    def __init__(self, smtp_config: Dict):
        self.smtp_config = smtp_config
    
    def format_alert_message(self, insights: Dict) -> str:
        try:
            message = f"""
            Fraud Alert - Transaction {insights['transaction_id']}
            
            Risk Level: {insights['risk_level']}
            Risk Score: {insights['risk_score']:.2f}
            Confidence Score: {insights['confidence_score']:.2f}
            
            Contributing Factors:
            {chr(10).join(f"- {factor['feature']}: {factor['value']:.2f} (z-score: {factor['zscore']:.2f})" 
                         for factor in insights['top_contributing_factors'])}
            
            Regulatory Concerns:
            {chr(10).join(f"- {concern}" for concern in insights['regulatory_concerns'])}
            
            Recommended Actions:
            {chr(10).join(f"- {action}" for action in insights['recommended_actions'])}
            
            Historical Context:
            - Total Transactions: {insights['historical_context']['total_transactions']}
            - Previous Fraud Incidents: {insights['historical_context']['previous_fraud_incidents']}
            - Average Transaction Amount: ${insights['historical_context']['average_transaction_amount']:.2f}
            """
            return message
        except Exception as e:
            logger.error(f"Error formatting alert message: {str(e)}")
            return "Error formatting alert message"
    
    def send_email_alert(self, to_email: str, subject: str, body: str) -> bool:
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['username']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body to email
            msg.attach(MIMEText(body, 'plain'))
            
            # Create SMTP session
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False

class FraudDashboard:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/risk-distribution')
        def risk_distribution():
            fig = self.create_risk_distribution()
            return jsonify(fig)
            
        @self.app.route('/api/fraud-trends')
        def fraud_trends():
            fig = self.create_fraud_trends()
            return jsonify(fig)
            
        @self.app.route('/api/risk-factors')
        def risk_factors():
            fig = self.create_risk_factors()
            return jsonify(fig)
            
        @self.app.route('/api/recent-alerts')
        def recent_alerts():
            high_risk = self.df[self.df['fraud_label'] == 'A'].tail(10)
            alerts = []
            for _, row in high_risk.iterrows():
                alerts.append({
                    'transaction_id': row['TransactionID'],
                    'timestamp': row['TransactionDate'].isoformat(),
                    'risk_level': 'CRITICAL',
                    'risk_score': float(row['ensemble_score']),
                    'confidence_score': float(row['confidence_score'])
                })
            return jsonify(alerts)
            
        @self.app.route('/api/transaction/<transaction_id>')
        def transaction_details(transaction_id):
            insights_generator = FraudInsights(self.df)
            insights = insights_generator.generate_insights(transaction_id)
            return jsonify(insights)
        
    def create_risk_distribution(self):
        """Create risk score distribution plot."""
        fig = px.histogram(
            self.df,
            x='ensemble_score',
            nbins=50,
            title='Distribution of Risk Scores'
        )
        return fig.to_dict()
        
    def create_fraud_trends(self):
        """Create fraud trends over time plot."""
        daily_fraud = self.df.groupby(self.df['TransactionDate'].dt.date)['fraud_label'].apply(
            lambda x: (x != '0').mean()
        ).reset_index()
        
        fig = px.line(
            daily_fraud,
            x='TransactionDate',
            y='fraud_label',
            title='Daily Fraud Rate Trend'
        )
        return fig.to_dict()
        
    def create_risk_factors(self):
        """Create top risk factors visualization."""
        risk_factors = pd.DataFrame([
            factor for factors in self.df['top_parameters']
            for factor in factors.split(' | ')
        ])
        
        risk_factors['feature'] = risk_factors[0].apply(lambda x: x.split(':')[0])
        risk_factors['zscore'] = risk_factors[0].apply(
            lambda x: float(x.split('z-score:')[1].strip(')'))
        )
        
        top_factors = risk_factors.groupby('feature')['zscore'].mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_factors.index,
            y=top_factors.values,
            title='Top 10 Risk Factors'
        )
        return fig.to_dict()
        
    def run_server(self, debug=True, port=5000):
        """Run the Flask server."""
        self.app.run(debug=debug, port=port) 