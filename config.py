import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_PATH = 'fraud_detection.db'
TRANSACTIONS_TABLE = 'transactions'
FRAUD_CASES_TABLE = 'fraud_cases'
REGULATIONS_TABLE = 'regulations'

# API Configuration
API_HOST = 'localhost'
API_PORT = 5000
DEBUG_MODE = True

# Model Configuration
MODEL_PATH = 'models'
ANOMALY_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.7

# Feature Configuration
TIME_WINDOW_HOURS = 24
MIN_TRANSACTIONS_FOR_ANALYSIS = 5
MAX_LOGIN_ATTEMPTS = 3

# Alert Configuration
ALERT_EMAIL = os.getenv('ALERT_EMAIL', 'admin@example.com')
SMTP_CONFIG = {
    'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'port': int(os.getenv('SMTP_PORT', '587')),
    'username': os.getenv('SMTP_USERNAME'),
    'password': os.getenv('SMTP_PASSWORD')
}

# Regulatory Rules
REGULATORY_RULES = {
    'large_transaction_threshold': 10000,
    'rapid_transaction_window': 60,  # seconds
    'unusual_time_start': 22,  # 10 PM
    'unusual_time_end': 6,     # 6 AM
    'max_devices_per_account': 3,
    'max_ip_addresses_per_account': 3
}

# Model Parameters
MODEL_PARAMS = {
    'isolation_forest': {
        'contamination': 0.1,
        'random_state': 42
    },
    'autoencoder': {
        'input_dim': 50,
        'encoding_dim': 25,
        'learning_rate': 0.001,
        'epochs': 100
    },
    'lstm': {
        'sequence_length': 10,
        'lstm_units': 64,
        'dense_units': 32,
        'learning_rate': 0.001,
        'epochs': 100
    }
}

# Feature Engineering Parameters
FEATURE_PARAMS = {
    'time_windows': [1, 3, 6, 12, 24],  # hours
    'amount_windows': [5, 10, 20, 50],  # transactions
    'velocity_windows': [1, 3, 6]        # hours
}

# Risk Scoring Weights
RISK_WEIGHTS = {
    'anomaly_score': 0.4,
    'rule_based_score': 0.3,
    'behavioral_score': 0.3
}

# Logging Configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'fraud_detection.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 