from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import traceback
from simple_fraud_detection import SimpleFraudDetector
import uuid
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.DEBUG)    
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for authentication
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Hardcoded admin credentials (in production, use a secure database)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin@123'

@login_manager.user_loader
def load_user(user_id):
    if user_id == ADMIN_USERNAME:
        return User(user_id)
    return None

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.id != ADMIN_USERNAME:
            flash('You must be logged in as admin to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            user = User(username)
            login_user(user)
            return redirect(url_for('admin'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Global variables
df = None
detector = SimpleFraudDetector()
current_file_info = None  # Track current file information

def process_upload(file_path):
    """Process uploaded file and detect fraud."""
    global df
    try:
        logger.info(f"Starting to process file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False, "File not found"
            
        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        if file_size == 0:
            logger.error("File is empty")
            return False, "The uploaded file is empty"
        
        # Check maximum file size (e.g., 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            logger.error(f"File too large: {file_size} bytes")
            return False, "File size exceeds maximum limit of 10MB"
            
        # Load data
        logger.info("Attempting to read CSV file")
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            logger.error("File encoding error")
            return False, "Invalid file encoding. Please ensure the file is UTF-8 encoded"
        except pd.errors.EmptyDataError:
            logger.error("Empty data error")
            return False, "The uploaded file is empty"
        except pd.errors.ParserError as e:
            logger.error(f"CSV parser error: {str(e)}")
            return False, f"Invalid CSV file format: {str(e)}"
            
        logger.info(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns")
        
        # Check minimum number of rows
        if len(df) < 1:
            logger.error("File contains no data rows")
            return False, "The file contains no data rows"
            
        # Log column names
        logger.info(f"Columns in file: {', '.join(df.columns)}")
        
        # Ensure required columns exist
        required_columns = ['TransactionID', 'TransactionDate', 'TransactionAmount', 'AccountID', 'DeviceID', 'Location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Validate data types and formats
        logger.info("Validating data types")
        try:
            # Check for duplicate TransactionIDs
            if df['TransactionID'].duplicated().any():
                logger.error("Duplicate TransactionIDs found")
                return False, "Duplicate TransactionIDs found in the file"
                
            # Validate TransactionAmount is numeric and positive
            df['TransactionAmount'] = pd.to_numeric(df['TransactionAmount'])
            if (df['TransactionAmount'] <= 0).any():
                logger.error("Invalid transaction amounts found")
                return False, "All transaction amounts must be positive"
                
            # Validate TransactionDate format and range
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
            if df['TransactionDate'].max() > datetime.now():
                logger.error("Future dates found in transactions")
                return False, "Transaction dates cannot be in the future"
                
            # Validate AccountID and DeviceID are not empty
            if df['AccountID'].isna().any() or df['DeviceID'].isna().any():
                logger.error("Missing AccountID or DeviceID values")
                return False, "AccountID and DeviceID cannot be empty"
                
        except Exception as e:
            logger.error(f"Data type validation failed: {str(e)}")
            return False, f"Invalid data format: {str(e)}"
        
        # Detect fraud
        logger.info("Starting fraud detection")
        try:
            df = detector.detect_fraud(df)
            logger.info("Fraud detection completed successfully")
        except Exception as e:
            logger.error(f"Fraud detection failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Error in fraud detection: {str(e)}"
        
        return True, "Data processed successfully"
        
    except Exception as e:
        logger.error(f"Unexpected error in process_upload: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, f"An error occurred while processing the file: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transactions')
def transactions():
    return render_template('transactions.html')

@app.route('/analytics')
@admin_required
def analytics():
    return render_template('analytics.html')

@app.route('/admin')
@admin_required
def admin():
    return render_template('admin.html')

@app.route('/transaction/<transaction_id>')
def transaction_view(transaction_id):
    return render_template('transaction.html', transaction_id=transaction_id)

@app.route('/new_transaction')
def new_transaction():
    return redirect(url_for('transactions'))

@app.route('/upload_data')
def upload_data():
    return redirect(url_for('index'))

@app.route('/reports')
def reports():
    return redirect(url_for('analytics'))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("File upload request received")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith('.csv'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'File must be a CSV'}), 400
            
        # Create temp directory if it doesn't exist
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            logger.info(f"Creating temp directory: {temp_dir}")
            os.makedirs(temp_dir)
            
        # Save the file temporarily with a unique name
        file_path = os.path.join(temp_dir, f'dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        logger.info(f"Saving file to: {file_path}")
        
        try:
            file.save(file_path)
            logger.info("File saved successfully")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
        
        try:
            # Process the file
            success, message = process_upload(file_path)
            
            if success:
                # Store file information
                global current_file_info
                current_file_info = {
                    'filename': file.filename,
                    'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'size': os.path.getsize(file_path),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                logger.info("File processed successfully")
                return jsonify({'message': message})
            else:
                logger.error(f"File processing failed: {message}")
                return jsonify({'error': message}), 400
                
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Temporary file removed: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An unexpected error occurred during file upload: {str(e)}'}), 500

@app.route('/api/dashboard-stats')
@admin_required
def dashboard_stats():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        stats = {
            'total_transactions': len(df),
            'high_risk_count': len(df[df['risk_label'] == 'A']),
            'medium_risk_count': len(df[df['risk_label'] == 'B']),
            'average_risk_score': float(df['anomaly_score'].mean()),
            'file_info': current_file_info  # Include file information in stats
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-dataset')
@admin_required
def download_dataset():
    try:
        if df is None:
            return jsonify({'error': 'No dataset available for download'}), 404
            
        # Create a temporary file for download
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Generate a unique filename
        download_filename = f'dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        file_path = os.path.join(temp_dir, download_filename)
        
        # Save the current dataframe to CSV
        df.to_csv(file_path, index=False)
        
        # Send the file
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up the temporary file after sending
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Temporary download file removed: {file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary download file: {str(e)}")

@app.route('/api/risk-distribution')
@admin_required
def risk_distribution():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        # Calculate risk score distribution with proper binning
        hist_data = df.groupby('risk_label').agg({
            'anomaly_score': ['count', 'mean', 'std'],
            'confidence_score': 'mean'
        }).round(3)
        
        # Create color mapping
        colors = {
            'A': 'rgba(220, 53, 69, 0.5)',  # Red for high risk
            'B': 'rgba(255, 193, 7, 0.5)',  # Yellow for medium risk
            'C': 'rgba(25, 135, 84, 0.5)'   # Green for low risk
        }
        
        # Prepare datasets for visualization
        datasets = []
        for label in ['A', 'B', 'C']:
            if label in hist_data.index:
                count = int(hist_data.loc[label, ('anomaly_score', 'count')])  # Convert np.int64 to int
                avg_score = float(hist_data.loc[label, ('anomaly_score', 'mean')])
                std_score = float(hist_data.loc[label, ('anomaly_score', 'std')])
                
                datasets.append({
                    'label': f'Risk Level {label}',
                    'data': [count],  # Use the converted integer
                    'backgroundColor': colors[label],
                    'borderColor': colors[label].replace('0.5', '1'),
                    'borderWidth': 1,
                    'avgScore': avg_score,
                    'stdScore': std_score
                })
        
        return jsonify({
            'data': {
                'labels': ['Risk Distribution'],
                'datasets': datasets
            },
            'options': {
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': 'Risk Categories'
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': 'Number of Transactions'
                        },
                        'beginAtZero': True
                    }
                },
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Transaction Risk Level Distribution'
                    },
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return [`Count: ${context.raw}`, `Avg Score: ${context.dataset.avgScore.toFixed(3)}`, `Std Dev: ${context.dataset.stdScore.toFixed(3)}`]; }'
                        }
                    }
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in risk_distribution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/high-risk-distribution')
@admin_required
def high_risk_distribution():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        high_risk = df[df['risk_label'] == 'A']
        
        if len(high_risk) == 0:
            return jsonify({
                'data': {
                    'labels': [],
                    'datasets': [{
                        'label': 'High Risk Score Distribution',
                        'data': [],
                        'backgroundColor': 'rgba(220, 53, 69, 0.5)',
                        'borderColor': 'rgba(220, 53, 69, 1)',
                        'borderWidth': 1
                    }]
                }
            })
        
        # Create bins based on anomaly scores
        hist, bin_edges = np.histogram(high_risk['anomaly_score'], bins=10, range=(0, 1))
        
        # Calculate additional statistics for each bin
        bin_stats = []
        for i in range(len(bin_edges)-1):
            mask = (high_risk['anomaly_score'] >= bin_edges[i]) & (high_risk['anomaly_score'] < bin_edges[i+1])
            bin_data = high_risk[mask]
            if len(bin_data) > 0:
                avg_confidence = bin_data['confidence_score'].mean()
                avg_amount = bin_data['TransactionAmount'].mean()
                bin_stats.append({
                    'count': len(bin_data),
                    'avg_confidence': avg_confidence,
                    'avg_amount': avg_amount
                })
            else:
                bin_stats.append({
                    'count': 0,
                    'avg_confidence': 0,
                    'avg_amount': 0
                })
        
        # Create labels with score ranges and statistics
        labels = [
            f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            for i in range(len(bin_edges)-1)
        ]
        
        # Prepare dataset with additional statistics
        dataset = {
            'label': 'Number of High Risk Transactions',
            'data': hist.tolist(),
            'backgroundColor': 'rgba(220, 53, 69, 0.5)',
            'borderColor': 'rgba(220, 53, 69, 1)',
            'borderWidth': 1,
            'avgConfidence': [stat['avg_confidence'] for stat in bin_stats],
            'avgAmounts': [stat['avg_amount'] for stat in bin_stats]
        }
        
        return jsonify({
            'data': {
                'labels': labels,
                'datasets': [dataset]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': 'Risk Score Range'
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': 'Number of Transactions'
                        },
                        'beginAtZero': True
                    }
                },
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'High Risk Transaction Analysis'
                    },
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return [`Count: ${context.raw}`, `Avg Amount: $${context.dataset.avgAmounts[context.dataIndex].toFixed(2)}`, `Confidence: ${context.dataset.avgConfidence[context.dataIndex].toFixed(2)}`]; }'
                        }
                    }
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in high_risk_distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/medium-risk-distribution')
@admin_required
def medium_risk_distribution():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        medium_risk = df[df['risk_label'] == 'B']
        
        if len(medium_risk) == 0:
            return jsonify({
                'data': {
                    'labels': [],
                    'datasets': [{
                        'label': 'Medium Risk Distribution',
                        'data': [],
                        'backgroundColor': 'rgba(255, 193, 7, 0.5)',
                        'borderColor': 'rgba(255, 193, 7, 1)',
                        'borderWidth': 1
                    }]
                }
            })
        
        # Create bins based on anomaly scores
        hist, bin_edges = np.histogram(medium_risk['anomaly_score'], bins=10, range=(0, 1))
        
        # Calculate additional statistics for each bin
        bin_stats = []
        for i in range(len(bin_edges)-1):
            mask = (medium_risk['anomaly_score'] >= bin_edges[i]) & (medium_risk['anomaly_score'] < bin_edges[i+1])
            bin_data = medium_risk[mask]
            if len(bin_data) > 0:
                avg_confidence = bin_data['confidence_score'].mean()
                avg_amount = bin_data['TransactionAmount'].mean()
                bin_stats.append({
                    'count': len(bin_data),
                    'avg_confidence': avg_confidence,
                    'avg_amount': avg_amount
                })
            else:
                bin_stats.append({
                    'count': 0,
                    'avg_confidence': 0,
                    'avg_amount': 0
                })
        
        # Create labels with score ranges
        labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        
        # Prepare dataset with additional statistics
        dataset = {
            'label': 'Medium Risk Distribution',
            'data': hist.tolist(),
            'backgroundColor': 'rgba(255, 193, 7, 0.5)',
            'borderColor': 'rgba(255, 193, 7, 1)',
            'borderWidth': 1,
            'avgConfidence': [stat['avg_confidence'] for stat in bin_stats],
            'avgAmounts': [stat['avg_amount'] for stat in bin_stats]
        }
        
        return jsonify({
            'data': {
                'labels': labels,
                'datasets': [dataset]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': 'Risk Score Range'
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': 'Number of Transactions'
                        },
                        'beginAtZero': True
                    }
                },
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Medium Risk Transaction Analysis'
                    },
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return [`Count: ${context.raw}`, `Avg Amount: $${context.dataset.avgAmounts[context.dataIndex].toFixed(2)}`, `Confidence: ${context.dataset.avgConfidence[context.dataIndex].toFixed(2)}`]; }'
                        }
                    }
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in medium_risk_distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fraud-trends')
@admin_required
def fraud_trends():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        logger.info("Starting fraud trends calculation")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        
        # Convert dates and ensure they're sorted
        try:
            df['date'] = pd.to_datetime(df['TransactionDate'])
            logger.info("Successfully converted TransactionDate to datetime")
        except Exception as e:
            logger.error(f"Error converting dates: {str(e)}")
            logger.error(f"TransactionDate sample: {df['TransactionDate'].head()}")
            return jsonify({'error': f'Date conversion error: {str(e)}'}), 500
        
        # Calculate daily statistics with proper error handling and logging
        try:
            # First, check if risk_label column exists and its contents
            if 'risk_label' not in df.columns:
                logger.error("risk_label column not found in DataFrame")
                logger.debug(f"Available columns: {df.columns.tolist()}")
                return jsonify({'error': 'risk_label column not found'}), 500
                
            logger.debug(f"risk_label unique values: {df['risk_label'].unique()}")
            
            # Group by date with detailed error handling
            daily_stats = df.groupby(df['date'].dt.date).agg({
                'risk_label': lambda x: (len([r for r in x if r in ['A', 'B']]) / len(x) * 100),
                'anomaly_score': 'mean'
            }).reset_index()
            
            daily_stats.columns = ['date', 'fraud_rate', 'avg_risk_score']
            logger.info(f"Successfully calculated daily stats. Shape: {daily_stats.shape}")
            logger.debug(f"Daily stats head: {daily_stats.head().to_dict()}")
            
        except Exception as e:
            logger.error(f"Error calculating daily statistics: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error calculating statistics: {str(e)}'}), 500
        
        try:
            # Sort by date and format for Chart.js
            daily_stats = daily_stats.sort_values('date')
            
            return jsonify({
                'data': {
                    'labels': [d.strftime('%Y-%m-%d') for d in daily_stats['date']],
                    'datasets': [
                        {
                            'label': 'Fraud Rate (%)',
                            'data': daily_stats['fraud_rate'].round(2).tolist(),
                            'borderColor': 'rgba(220, 53, 69, 1)',
                            'backgroundColor': 'rgba(220, 53, 69, 0.1)',
                            'fill': True,
                            'yAxisID': 'y'
                        },
                        {
                            'label': 'Avg Risk Score',
                            'data': daily_stats['avg_risk_score'].round(3).tolist(),
                            'borderColor': 'rgba(255, 193, 7, 1)',
                            'borderDash': [5, 5],
                            'fill': False,
                            'yAxisID': 'y1'
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'interaction': {
                        'mode': 'index',
                        'intersect': False
                    },
                    'scales': {
                        'x': {
                            'type': 'category',
                            'title': {
                                'display': True,
                                'text': 'Date'
                            }
                        },
                        'y': {
                            'type': 'linear',
                            'display': True,
                            'position': 'left',
                            'title': {
                                'display': True,
                                'text': 'Fraud Rate (%)'
                            },
                            'min': 0,
                            'max': 100
                        },
                        'y1': {
                            'type': 'linear',
                            'display': True,
                            'position': 'right',
                            'title': {
                                'display': True,
                                'text': 'Risk Score'
                            },
                            'min': 0,
                            'max': 1,
                            'grid': {
                                'drawOnChartArea': False
                            }
                        }
                    },
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Fraud Trends Analysis'
                        }
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error preparing chart data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error preparing chart data: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in fraud_trends: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'details': {
                'message': 'An unexpected error occurred while generating fraud trends',
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        }), 500

@app.route('/api/risk-factors')
@admin_required
def risk_factors():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        # Get high-risk transactions
        high_risk = df[df['risk_label'] == 'A']
        
        if len(high_risk) == 0:
            return jsonify({
                'data': {
                    'labels': ['No high-risk transactions found'],
                    'datasets': [{
                        'label': 'Risk Factor Impact',
                        'data': [0],
                        'backgroundColor': 'rgba(255, 99, 132, 0.5)',
                        'borderColor': 'rgba(255, 99, 132, 1)',
                        'borderWidth': 1
                    }]
                }
            })
        
        # Calculate risk factors
        risk_factors = {
            'Amount': abs(high_risk['amount_zscore']).mean(),
            'Transaction Frequency': abs(high_risk['transaction_frequency']).mean(),
            'Device Count': high_risk['device_count'].mean(),
            'Location Count': high_risk['location_count'].mean(),
            'Time Pattern': abs(high_risk.get('time_pattern_score', pd.Series([0] * len(high_risk)))).mean()
        }
        
        # Sort factors by impact
        sorted_factors = dict(sorted(risk_factors.items(), key=lambda x: x[1], reverse=True))
        
        # Take top 5 factors
        top_factors = {k: v for k, v in list(sorted_factors.items())[:5]}
        
        return jsonify({
            'data': {
                'labels': list(top_factors.keys()),
                'datasets': [{
                    'label': 'Risk Factor Impact',
                    'data': list(top_factors.values()),
                    'backgroundColor': [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)'
                    ],
                    'borderColor': [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'title': {
                            'display': True,
                            'text': 'Impact Score'
                        }
                    }
                },
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Top Risk Factors in High-Risk Transactions'
                    },
                    'legend': {
                        'display': False
                    }
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in risk_factors: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/high-risk-transactions')
def high_risk_transactions():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        high_risk = df[df['risk_label'] == 'A'].head(10)
        transactions = []
        
        for _, row in high_risk.iterrows():
            transactions.append({
                'transaction_id': str(row['TransactionID']),
                'timestamp': str(row['TransactionDate']),
                'risk_score': float(row['anomaly_score']),
                'confidence_score': float(row['confidence_score'])
            })
        
        return jsonify(transactions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/medium-risk-transactions')
def medium_risk_transactions():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        medium_risk = df[df['risk_label'] == 'B'].head(10)
        transactions = []
        
        for _, row in medium_risk.iterrows():
            transactions.append({
                'transaction_id': str(row['TransactionID']),
                'timestamp': str(row['TransactionDate']),
                'risk_score': float(row['anomaly_score']),
                'confidence_score': float(row['confidence_score'])
            })
        
        return jsonify(transactions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transaction/<transaction_id>')
def transaction_details(transaction_id):
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        # Find the transaction
        transaction = df[df['TransactionID'].astype(str) == str(transaction_id)]
        if len(transaction) == 0:
            return jsonify({'error': 'Transaction not found'}), 404
            
        transaction = transaction.iloc[0]
        
        # Get basic transaction details
        details = {
            'transaction_id': str(transaction_id),
            'timestamp': str(transaction['TransactionDate']),
            'risk_level': transaction['risk_label'],
            'risk_score': float(transaction['anomaly_score']),
            'confidence_score': float(transaction['confidence_score']),
            'amount': float(transaction['TransactionAmount']),
            'account_id': str(transaction['AccountID']),
            'device_id': str(transaction['DeviceID']),
            'location': str(transaction['Location'])
        }
        
        # Get contributing factors
        factors = []
        
        # Check amount anomaly
        if abs(transaction['amount_zscore']) > 2:
            factors.append({
                'feature': 'Unusual Transaction Amount',
                'value': float(transaction['TransactionAmount']),
                'zscore': float(transaction['amount_zscore'])
            })
            
        # Check transaction frequency
        if transaction['transaction_frequency'] > df['transaction_frequency'].mean() + 2 * df['transaction_frequency'].std():
            factors.append({
                'feature': 'High Transaction Frequency',
                'value': float(transaction['transaction_frequency']),
                'zscore': float((transaction['transaction_frequency'] - df['transaction_frequency'].mean()) / df['transaction_frequency'].std())
            })
            
        # Check multiple devices
        if transaction['device_count'] > 1:
            factors.append({
                'feature': 'Multiple Devices Used',
                'value': float(transaction['device_count']),
                'zscore': float((transaction['device_count'] - 1) * 2)
            })
            
        # Check multiple locations
        if transaction['location_count'] > 1:
            factors.append({
                'feature': 'Multiple Locations',
                'value': float(transaction['location_count']),
                'zscore': float((transaction['location_count'] - 1) * 2)
            })
            
        # Sort factors by zscore
        factors = sorted(factors, key=lambda x: abs(x['zscore']), reverse=True)
        
        # Generate regulatory concerns
        concerns = []
        if transaction['risk_label'] == 'A':
            concerns.append(f"High-risk transaction amount: ${transaction['TransactionAmount']:,.2f}")
            if transaction['device_count'] > 1:
                concerns.append(f"Multiple devices used: {transaction['device_count']} devices")
            if transaction['location_count'] > 1:
                concerns.append(f"Multiple locations detected: {transaction['location_count']} locations")
            if transaction['transaction_frequency'] > df['transaction_frequency'].mean() + 2 * df['transaction_frequency'].std():
                concerns.append("Unusually high transaction frequency")
                
        # Generate recommended actions
        actions = []
        if transaction['risk_label'] == 'A':
            actions.extend([
                "Block transaction immediately",
                "Contact customer for verification",
                "Review account for suspicious activity",
                "Flag account for enhanced monitoring"
            ])
        elif transaction['risk_label'] == 'B':
            actions.extend([
                "Review transaction details",
                "Monitor account activity",
                "Consider customer verification"
            ])
        else:
            actions.append("Normal processing")
            
        return jsonify({
            **details,
            'top_contributing_factors': factors,
            'regulatory_concerns': concerns,
            'recommended_actions': actions
        })
        
    except Exception as e:
        logger.error(f"Error in transaction_details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Add new endpoint for admin panel data
@app.route('/api/admin/transactions')
def admin_transactions():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        # Get query parameters for filtering
        risk_level = request.args.get('risk_level')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        # Apply filters
        filtered_df = df.copy()
        if risk_level:
            filtered_df = filtered_df[filtered_df['risk_label'] == risk_level]
        if date_from:
            filtered_df = filtered_df[filtered_df['TransactionDate'] >= date_from]
        if date_to:
            filtered_df = filtered_df[filtered_df['TransactionDate'] <= date_to]
            
        # Sort by risk score and timestamp
        filtered_df = filtered_df.sort_values(['risk_label', 'TransactionDate'], ascending=[True, False])
        
        # Convert to list of dictionaries
        transactions = []
        for _, row in filtered_df.iterrows():
            transactions.append({
                'transaction_id': str(row['TransactionID']),
                'timestamp': str(row['TransactionDate']),
                'amount': float(row['TransactionAmount']),
                'account_id': str(row['AccountID']),
                'risk_level': row['risk_label'],
                'risk_score': float(row['anomaly_score']),
                'confidence_score': float(row['confidence_score']),
                'device_id': str(row['DeviceID']),
                'location': str(row['Location'])
            })
        
        return jsonify({
            'total': len(transactions),
            'transactions': transactions
        })
        
    except Exception as e:
        logger.error(f"Error in admin_transactions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add endpoint for new transaction submission
@app.route('/api/transaction/new', methods=['POST'])
def add_transaction():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['TransactionAmount', 'AccountID', 'DeviceID', 'Location']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
        # Validate field types and values
        try:
            amount = float(data['TransactionAmount'])
            if amount <= 0:
                return jsonify({'error': 'Transaction amount must be positive'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid transaction amount'}), 400
            
        if not str(data['AccountID']).strip():
            return jsonify({'error': 'AccountID cannot be empty'}), 400
            
        if not str(data['DeviceID']).strip():
            return jsonify({'error': 'DeviceID cannot be empty'}), 400
            
        if not str(data['Location']).strip():
            return jsonify({'error': 'Location cannot be empty'}), 400
            
        # Create new transaction record
        new_transaction = {
            'TransactionID': str(uuid.uuid4()),
            'TransactionDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TransactionAmount': amount,
            'AccountID': str(data['AccountID']),
            'DeviceID': str(data['DeviceID']),
            'Location': str(data['Location'])
        }
        
        # Add to dataframe
        global df
        if df is None:
            df = pd.DataFrame([new_transaction])
        else:
            df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)
        
        # Detect fraud for the new transaction
        try:
            df = detector.detect_fraud(df)
        except Exception as e:
            logger.error(f"Fraud detection failed for new transaction: {str(e)}")
            return jsonify({'error': 'Failed to process transaction'}), 500
        
        # Get the fraud detection results for the new transaction
        result = df[df['TransactionID'] == new_transaction['TransactionID']].iloc[0]
        
        return jsonify({
            'transaction_id': new_transaction['TransactionID'],
            'risk_level': result['risk_label'],
            'risk_score': float(result['anomaly_score']),
            'confidence_score': float(result['confidence_score'])
        })
        
    except Exception as e:
        logger.error(f"Error adding new transaction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Add new endpoint to modify transaction labels
@app.route('/api/admin/update-label', methods=['POST'])
@admin_required
def update_transaction_label():
    try:
        if df is None:
            return jsonify({'error': 'Please upload a dataset first'}), 400
            
        data = request.json
        if not data or 'transaction_id' not in data or 'new_label' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        transaction_id = str(data['transaction_id'])
        new_label = data['new_label']
        
        # Validate new label
        if new_label not in ['A', 'B', 'C']:
            return jsonify({'error': 'Invalid label. Must be A, B, or C'}), 400
            
        # Find and update the transaction
        mask = df['TransactionID'].astype(str) == transaction_id
        if not mask.any():
            return jsonify({'error': 'Transaction not found'}), 404
            
        df.loc[mask, 'risk_label'] = new_label
        
        # Return updated transaction details
        updated_transaction = df[mask].iloc[0]
        return jsonify({
            'transaction_id': transaction_id,
            'new_label': new_label,
            'risk_score': float(updated_transaction['anomaly_score']),
            'confidence_score': float(updated_transaction['confidence_score'])
        })
        
    except Exception as e:
        logger.error(f"Error updating transaction label: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True) 