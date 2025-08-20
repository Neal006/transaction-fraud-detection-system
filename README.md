# Enhanced Fraud Detection System

A comprehensive fraud detection system that combines machine learning models with contextual insights, real-time alerts, and interactive visualizations.

## Features

- **Advanced Fraud Detection**: Uses ensemble of machine learning models (Random Forest, Gradient Boosting, Neural Networks)
- **Contextual Insights**: Detailed analysis of contributing factors for each transaction
- **Real-time Alerts**: Email notifications for high-risk transactions
- **Interactive Dashboard**: Real-time monitoring of fraud trends and system performance
- **Regulatory Compliance**: Built-in checks for regulatory concerns
- **Explainable AI**: LIME-based explanations for model predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fraud-detection-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure email alerts:
   - Update the SMTP configuration in `insights_and_alerts.py`
   - Set your email credentials and server details

## Usage

1. Prepare your transaction data in CSV format with the following columns:
   - TransactionID
   - TransactionDate
   - TransactionAmount
   - AccountID
   - CustomerAge
   - CustomerOccupation
   - TransactionType
   - Channel
   - Location
   - DeviceID
   - IP Address
   - MerchantID
   - AccountBalance
   - LoginAttempts
   - TransactionDuration
   - PreviousTransactionDate
   - is_fraud (optional, for training)

2. Run the fraud detection system:
```bash
python fraud_detection.py
```

3. Access the dashboard:
   - Open your web browser and navigate to `http://localhost:8050`
   - View real-time fraud trends and insights

## System Components

### 1. Fraud Detection Core
- Feature engineering and preprocessing
- Ensemble model training and prediction
- Risk scoring and classification

### 2. Insights Generation
- Detailed analysis of contributing factors
- Historical context analysis
- Regulatory compliance checks
- Actionable recommendations

### 3. Alert System
- Real-time email notifications
- Risk level classification
- Detailed HTML reports
- Configurable alert thresholds

### 4. Interactive Dashboard
- Real-time risk score distribution
- Fraud trend visualization
- Top risk factors analysis
- Recent alerts table
- Auto-refresh every 5 minutes

## Configuration

### Risk Thresholds
- CRITICAL: > 0.8
- HIGH: > 0.6
- MEDIUM: > 0.4
- LOW: ≤ 0.4

### Regulatory Checks
- Large transactions (> $10,000)
- Rapid transaction patterns
- Unusual time patterns
- Multiple device usage

## Output Files

- `DATASET_with_fraud_scores.csv`: Complete dataset with fraud scores and labels
- `fraud_label_*.csv`: Separate files for each fraud label category
- `fraud_label_summary.csv`: Summary statistics for fraud labels
- `lime_explanation_*.html`: LIME explanations for selected transactions
- `feature_importance_lime.csv`: Feature importance rankings

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 