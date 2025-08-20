import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report, make_scorer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def create_features(df):
    
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
    
    df = df.sort_values('TransactionDate')
    
    global_mean = df['TransactionAmount'].mean()
    global_std = df['TransactionAmount'].std()
    df['amount_zscore'] = (df['TransactionAmount'] - global_mean) / global_std
    
    df['hour'] = df['TransactionDate'].dt.hour
    df['day_of_week'] = df['TransactionDate'].dt.dayofweek
    df['month'] = df['TransactionDate'].dt.month
    df['time_diff'] = df['TransactionDate'].diff().dt.total_seconds().fillna(0)
    df['rolling_mean_time'] = df['time_diff'].rolling(window=5, min_periods=1).mean()
    
    df['transactions_per_hour'] = df.groupby(df['TransactionDate'].dt.floor('H'))['TransactionID'].transform('count')
    df['transactions_per_day'] = df.groupby(df['TransactionDate'].dt.date)['TransactionID'].transform('count')
    
    df['rolling_mean_amount'] = df['TransactionAmount'].rolling(window=5, min_periods=1).mean()
    df['rolling_std_amount'] = df['TransactionAmount'].rolling(window=5, min_periods=1).std()
    df['amount_to_mean_ratio'] = df['TransactionAmount'] / df['rolling_mean_amount']
    
    df['balance_to_amount_ratio'] = df['TransactionAmount'] / df['AccountBalance']
    df['balance_change'] = df['AccountBalance'].diff()
    
    age_labels = ['very_young', 'young', 'middle', 'senior', 'elderly']
    df['age_bracket'] = pd.qcut(df['CustomerAge'], q=5, labels=age_labels)
    age_dummies = pd.get_dummies(df['age_bracket'], prefix='age')
    df = pd.concat([df, age_dummies], axis=1)
    
    occupation_dummies = pd.get_dummies(df['CustomerOccupation'], prefix='occupation')
    df = pd.concat([df, occupation_dummies], axis=1)
    
    transaction_type_dummies = pd.get_dummies(df['TransactionType'], prefix='type')
    df = pd.concat([df, transaction_type_dummies], axis=1)
    
    channel_dummies = pd.get_dummies(df['Channel'], prefix='channel')
    df = pd.concat([df, channel_dummies], axis=1)
    
    location_dummies = pd.get_dummies(df['Location'], prefix='location')
    df = pd.concat([df, location_dummies], axis=1)
    
    df['device_count'] = df.groupby('AccountID')['DeviceID'].transform('nunique')
    df['ip_count'] = df.groupby('AccountID')['IP Address'].transform('nunique')
    
    df['login_attempts_zscore'] = (df['LoginAttempts'] - df['LoginAttempts'].mean()) / df['LoginAttempts'].std()
    
    df['duration_zscore'] = (df['TransactionDuration'] - df['TransactionDuration'].mean()) / df['TransactionDuration'].std()
    
    df['days_since_previous'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / (24 * 3600)
    
    df['is_large_transaction'] = df['amount_zscore'] > 3
    df['is_rapid_transaction'] = df['time_diff'] < 60
    df['is_unusual_time'] = ~df['hour'].between(6, 22)
    df['is_high_login_attempts'] = df['login_attempts_zscore'] > 2
    df['is_unusual_duration'] = df['duration_zscore'] > 2
    df['is_new_device'] = df['device_count'] > 1
    df['is_new_ip'] = df['ip_count'] > 1
    df['is_high_balance_change'] = df['balance_change'].abs() > df['balance_change'].std() * 2
    
    df['days_since_last_transaction'] = df.groupby('AccountID')['TransactionDate'].transform(
        lambda x: (x.max() - x).dt.days
    )

    df['transaction_frequency'] = df.groupby('AccountID')['TransactionID'].transform('count')
    df['avg_transactions_per_day'] = df.groupby('AccountID')['TransactionID'].transform('count') / \
        df.groupby('AccountID')['TransactionDate'].transform(lambda x: (x.max() - x.min()).days + 1)

    df['total_transaction_amount'] = df.groupby('AccountID')['TransactionAmount'].transform('sum')
    df['avg_transaction_amount'] = df.groupby('AccountID')['TransactionAmount'].transform('mean')
    df['max_transaction_amount'] = df.groupby('AccountID')['TransactionAmount'].transform('max')

    df['amount_trend'] = df.groupby('AccountID')['TransactionAmount'].transform(
        lambda x: x.diff().rolling(window=3, min_periods=1).mean()
    )
    df['time_pattern_std'] = df.groupby('AccountID')['hour'].transform('std')
    df['time_pattern_mean'] = df.groupby('AccountID')['hour'].transform('mean')
    
    df['merchant_transaction_count'] = df.groupby('MerchantID')['TransactionID'].transform('count')
    df['merchant_unique_customers'] = df.groupby('MerchantID')['AccountID'].transform('nunique')

    df['customer_merchant_frequency'] = df.groupby(['AccountID', 'MerchantID'])['TransactionID'].transform('count')
    
    df['transaction_velocity'] = df['TransactionAmount'] / df['time_diff'].replace(0, 1)

    df['account_utilization'] = df['TransactionAmount'] / df['AccountBalance']
    
    for window in [3, 5, 7]:
        df[f'rolling_std_{window}'] = df.groupby('AccountID')['TransactionAmount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df[f'rolling_mean_{window}'] = df.groupby('AccountID')['TransactionAmount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_skew_{window}'] = df.groupby('AccountID')['TransactionAmount'].transform(
            lambda x: x.rolling(window=window, min_periods=1).skew()
        )

    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['TransactionDate'].apply(lambda x: (x.month == 12 and x.day in [24, 25, 31]) or 
                                                  (x.month == 1 and x.day in [1, 15]) or 
                                                  (x.month == 7 and x.day == 4) or 
                                                  (x.month == 11 and x.day in [11, 24, 25]) or 
                                                  (x.month == 10 and x.day == 31)).astype(int)
    
    columns_to_drop = ['age_bracket', 'PreviousTransactionDate', 'CustomerAge', 
                      'CustomerOccupation', 'TransactionType', 'Channel', 'Location']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    df = df.fillna(0)

    df['is_outlier_amount'] = df['amount_zscore'].abs() > 3
    df['is_sudden_change'] = df['TransactionAmount'].diff().abs() > df['TransactionAmount'].std() * 2
    
    return df

def apply_rule_based_rules(df):
    df['rule_based_score'] = 0
    
    df.loc[df['is_large_transaction'], 'rule_based_score'] += 1
    df.loc[df['balance_to_amount_ratio'] > 0.5, 'rule_based_score'] += 1
    
    df.loc[df['is_rapid_transaction'], 'rule_based_score'] += 1
    df.loc[df['is_unusual_time'], 'rule_based_score'] += 1
    df.loc[df['days_since_previous'] < 0.1, 'rule_based_score'] += 1 
    
    df.loc[df['is_high_login_attempts'], 'rule_based_score'] += 1
    df.loc[df['is_new_device'], 'rule_based_score'] += 1
    df.loc[df['is_new_ip'], 'rule_based_score'] += 1
    
    df.loc[df['is_unusual_duration'], 'rule_based_score'] += 1
    df.loc[df['is_high_balance_change'], 'rule_based_score'] += 1
    
    df['rule_based_score'] = df['rule_based_score'] / df['rule_based_score'].max()
    
    return df

def prepare_features(df):
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X = df[numerical_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, numerical_features, scaler

def tune_models(X_train, y_train):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_uniform('subsample', 0.6, 0.9),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
        
        model = GradientBoostingClassifier(**param, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        return scores.mean()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    return best_params

def train_advanced_models(X_scaled, y=None):
    """Train advanced models including gradient boosting and neural networks."""
    models = {}
    
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        best_params = tune_models(X_train, y_train)
        gb = GradientBoostingClassifier(**best_params, random_state=42)
        gb.fit(X_train, y_train)
        models['gradient_boosting'] = gb
        
        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        nn.fit(X_train, y_train)
        models['neural_network'] = nn
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models['random_forest'] = rf
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        models['logistic_regression'] = lr
        
        estimators = [
            ('gb', gb),
            ('nn', nn),
            ('rf', rf),
            ('lr', lr)
        ]
        final_estimator = LogisticRegression()
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            stack_method='predict_proba'
        )
        stacking.fit(X_train, y_train)
        models['stacking'] = stacking
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_scaled)
    models['lof'] = lof
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_scaled)
    models['isolation_forest'] = iso_forest
    
    return models

def evaluate_models(models, X_scaled, y=None):
    results = {}
    
    for name, model in models.items():
        if name in ['lof', 'isolation_forest']:
            
            predictions = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            results[name] = {
                'predictions': predictions,
                'scores': anomaly_scores
            }
        else:
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            results[name] = {
                'predictions': predictions,
                'probabilities': probabilities
            }
    
    return results

def combine_predictions(results, df):
    df['ensemble_score'] = 0
    model_scores = []
    
    for model_name, model_results in results.items():
        if 'scores' in model_results:
            
            scores = model_results['scores']
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            model_scores.append(normalized_scores)
            df['ensemble_score'] += normalized_scores
        else:
            
            model_scores.append(model_results['probabilities'])
            df['ensemble_score'] += model_results['probabilities']
    
    df['ensemble_score'] = (df['ensemble_score'] - df['ensemble_score'].min()) / \
                          (df['ensemble_score'].max() - df['ensemble_score'].min())
    
    if model_scores:
        scores_array = np.array(model_scores)
        
        agreement_score = np.mean(scores_array == scores_array[0], axis=0)
        
        score_variance = np.var(scores_array, axis=0)
        consistency_score = 1 - (score_variance / 0.25)  
        
        prediction_strength = np.abs(scores_array - 0.5).mean(axis=0) * 2 
        
        df['confidence_score'] = (
            0.4 * agreement_score +      
            0.3 * consistency_score +    
            0.3 * prediction_strength     
        )
    else:
        df['confidence_score'] = 0
    
    df['fraud_label'] = '0'  
    df.loc[df['ensemble_score'] > 0.8, 'fraud_label'] = 'A'  
    df.loc[(df['ensemble_score'] > 0.5) & (df['ensemble_score'] <= 0.8), 'fraud_label'] = 'B' 
    df.loc[(df['ensemble_score'] > 0.2) & (df['ensemble_score'] <= 0.5), 'fraud_label'] = 'C' 
    
    df['fraud_label_with_confidence'] = df['fraud_label'] + '_' + \
        df['confidence_score'].apply(lambda x: 'HIGH' if x > 0.8 else 'MED' if x > 0.5 else 'LOW')
    
    df['top_parameters'] = df.apply(lambda row: identify_top_parameters(row, df), axis=1)
    
    return df

def identify_top_parameters(row, df):
    feature_cols = [col for col in df.columns if col not in 
                   ['ensemble_score', 'confidence_score', 'fraud_label', 
                    'fraud_label_with_confidence', 'top_parameters']]
    
    z_scores = {}
    for col in feature_cols:
        try:
            # Skip if column is not numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:  # Avoid division by zero
                z_scores[col] = abs((row[col] - mean) / std)
            else:
                z_scores[col] = 0
        except Exception as e:
            continue
    
    sorted_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_3 = []
    for feature, z_score in sorted_features[:3]:
        value = row[feature]
        if isinstance(value, (np.float64, np.int64)):
            value = float(value)
        top_3.append(f"{feature}: {value:.2f} (z-score: {z_score:.2f})")
    
    return " | ".join(top_3)

def analyze_lime_explanation(model, X_scaled, feature_names, instance_idx=0):
    try:
        explainer = LimeTabularExplainer(
            X_scaled,
            feature_names=feature_names,
            class_names=['Legitimate', 'Fraud'],
            mode='classification'
        )
        
        explanation = explainer.explain_instance(
            X_scaled[instance_idx],
            model.predict_proba
        )
        
        explanation.save_to_file(f'lime_explanation_{instance_idx}.html')
        
        lime_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': explanation.as_list()
        }).sort_values('importance', key=abs, ascending=False)
        
        lime_importance.to_csv('feature_importance_lime.csv', index=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=lime_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (LIME)')
        plt.tight_layout()
        plt.savefig('lime_feature_importance.png')
        plt.close()
        
        print(f"\nLIME Explanation for instance {instance_idx}:")
        print(f"Predicted class: {'Fraud' if model.predict([X_scaled[instance_idx]])[0] == 1 else 'Legitimate'}")
        print(f"Confidence: {model.predict_proba([X_scaled[instance_idx]])[0].max():.3f}")
        print("\nTop 5 contributing features:")
        print(lime_importance.head())
        
    except Exception as e:
        print(f"LIME analysis failed: {str(e)}")

def save_results(df, output_file):
  
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='ensemble_score', bins=50)
    plt.title('Distribution of Fraud Scores')
    plt.savefig('fraud_score_distribution.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='fraud_label', y='confidence_score')
    plt.title('Confidence Scores by Fraud Label')
    plt.savefig('confidence_scores.png')
    plt.close()
    
    for label in ['A', 'B', 'C', '0']:
        df_label = df[df['fraud_label'] == label]
        df_label.to_csv(f'fraud_label_{label}.csv', index=False)
    
    print("\nFraud Label Distribution:")
    print(f"Total transactions: {len(df)}")
    for label in ['A', 'B', 'C', '0']:
        df_label = df[df['fraud_label'] == label]
        print(f"\nLabel {label}: {len(df_label)} ({len(df_label)/len(df)*100:.2f}%)")
        if len(df_label) > 0:
            print(f"  Average confidence: {df_label['confidence_score'].mean():.3f}")
            print(f"  High confidence cases: {len(df_label[df_label['confidence_score'] > 0.7])}")
            print(f"  Medium confidence cases: {len(df_label[(df_label['confidence_score'] > 0.4) & (df_label['confidence_score'] <= 0.7)])}")
            print(f"  Low confidence cases: {len(df_label[df_label['confidence_score'] <= 0.4])}")
            
            print("\n  Example cases with top parameters:")
            for _, row in df_label.head(3).iterrows():
                print(f"    Score: {row['ensemble_score']:.3f}, Confidence: {row['confidence_score']:.3f}")
                print(f"    Top parameters: {row['top_parameters']}")
    
    fraud_summary = pd.DataFrame({
        'Label': ['A', 'B', 'C', '0'],
        'Count': [len(df[df['fraud_label'] == label]) for label in ['A', 'B', 'C', '0']],
        'Percentage': [len(df[df['fraud_label'] == label])/len(df)*100 for label in ['A', 'B', 'C', '0']],
        'Avg_Confidence': [df[df['fraud_label'] == label]['confidence_score'].mean() for label in ['A', 'B', 'C', '0']],
        'High_Confidence_Cases': [len(df[(df['fraud_label'] == label) & (df['confidence_score'] > 0.7)]) for label in ['A', 'B', 'C', '0']]
    })
    fraud_summary.to_csv('fraud_label_summary.csv', index=False)

def main():
    # Load and preprocess data
    df = load_data('DATASET.csv')
    df = preprocess_data(df)
    df = create_features(df)
    df = apply_rule_based_rules(df)
    
    # Prepare features and train models
    X_scaled, feature_names, scaler = prepare_features(df)
    y = df['is_fraud'] if 'is_fraud' in df.columns else None
    models = train_advanced_models(X_scaled, y)
    
    # Evaluate models and combine predictions
    results = evaluate_models(models, X_scaled, y)
    df = combine_predictions(results, df)
    
    # Generate LIME explanations
    if 'random_forest' in models:
        analyze_lime_explanation(models['random_forest'], X_scaled, feature_names)
        high_risk_idx = df[df['fraud_label'] == 'A'].index[0]
        analyze_lime_explanation(models['random_forest'], X_scaled, feature_names, high_risk_idx)
    
    # Save results
    save_results(df, 'DATASET_with_fraud_scores.csv')
    
    # Initialize insights and alerts system
    from insights_and_alerts import FraudInsights, AlertSystem, FraudDashboard
    
    insights_generator = FraudInsights(df)
    
    # Configure email settings from environment variables
    smtp_config = {
        'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'port': int(os.getenv('SMTP_PORT', '587')),
        'username': os.getenv('SMTP_USERNAME'),
        'password': os.getenv('SMTP_PASSWORD')
    }
    
    # Only initialize alert system if credentials are available
    if smtp_config['username'] and smtp_config['password']:
        alert_system = AlertSystem(smtp_config)
        
        # Generate insights for high-risk transactions
        high_risk_transactions = df[df['fraud_label'] == 'A']['TransactionID'].tolist()
        for transaction_id in high_risk_transactions:
            insights = insights_generator.generate_insights(transaction_id)
            
            # Send email alert for critical cases
            if insights['risk_level'] == 'CRITICAL':
                try:
                    alert_message = alert_system.format_alert_message(insights)
                    alert_system.send_email_alert(
                        to_email=os.getenv('ALERT_EMAIL', 'nealdaftary0405@gmail.com'),
                        subject=f'CRITICAL: Fraud Alert - Transaction {transaction_id}',
                        body=alert_message
                    )
                except Exception as e:
                    print(f"Failed to send email alert for transaction {transaction_id}: {str(e)}")
    else:
        print("Email alerts disabled: SMTP credentials not configured")
    
    # Initialize and run dashboard
    dashboard = FraudDashboard(df)
    dashboard.run_server(debug=True, port=8050)
    

if __name__ == "__main__":
    main()