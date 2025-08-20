from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from config import DATABASE_PATH

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, unique=True)
    account_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float)
    currency = Column(String)
    transaction_type = Column(String)
    merchant_id = Column(String)
    location = Column(String)
    device_id = Column(String)
    ip_address = Column(String)
    status = Column(String)
    
    # Fraud detection results
    anomaly_score = Column(Float)
    rule_based_score = Column(Float)
    behavioral_score = Column(Float)
    ensemble_score = Column(Float)
    confidence_score = Column(Float)
    is_fraudulent = Column(Boolean, default=False)
    
    # Additional features
    features = Column(JSON)
    risk_factors = Column(JSON)
    
    # Relationships
    alerts = relationship("Alert", back_populates="transaction")
    insights = relationship("TransactionInsight", back_populates="transaction", uselist=False)

class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String)  # 'anomaly', 'rule_based', 'behavioral', 'ensemble'
    severity = Column(String)    # 'low', 'medium', 'high', 'critical'
    description = Column(String)
    status = Column(String)      # 'new', 'reviewed', 'resolved', 'false_positive'
    reviewed_by = Column(String)
    review_notes = Column(String)
    
    # Relationship
    transaction = relationship("Transaction", back_populates="alerts")

class TransactionInsight(Base):
    __tablename__ = 'transaction_insights'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey('transactions.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Contextual information
    historical_context = Column(JSON)
    behavioral_patterns = Column(JSON)
    regulatory_concerns = Column(JSON)
    
    # Recommendations
    recommended_actions = Column(JSON)
    risk_level = Column(String)
    
    # Relationship
    transaction = relationship("Transaction", back_populates="insights")

class FraudCase(Base):
    __tablename__ = 'fraud_cases'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String)  # 'open', 'investigating', 'resolved', 'closed'
    
    # Case details
    description = Column(String)
    impact_amount = Column(Float)
    resolution_notes = Column(String)
    resolution_date = Column(DateTime)
    
    # Related transactions
    transaction_ids = Column(JSON)
    
    # Investigation details
    investigator = Column(String)
    investigation_notes = Column(JSON)
    evidence = Column(JSON)

class Regulation(Base):
    __tablename__ = 'regulations'
    
    id = Column(Integer, primary_key=True)
    regulation_id = Column(String, unique=True)
    name = Column(String)
    description = Column(String)
    category = Column(String)
    requirements = Column(JSON)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Compliance rules
    rules = Column(JSON)
    thresholds = Column(JSON)

# Create database engine
engine = create_engine(f'sqlite:///{DATABASE_PATH}')

# Create all tables
def init_db():
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    init_db() 