import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

class AviationDelayPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.xgb_model = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load data and create engineered features"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert datetime columns
        self.df['scheduled_arrival'] = pd.to_datetime(self.df['scheduled_arrival'])
        self.df['actual_arrival'] = pd.to_datetime(self.df['actual_arrival'])
        self.df['scheduled_departure'] = pd.to_datetime(self.df['scheduled_departure'])
        self.df['actual_departure'] = pd.to_datetime(self.df['actual_departure'])
        
        print(f"Loaded {len(self.df)} records")
        print(f"Delay rate: {self.df['is_delayed'].mean():.2%}")
        
    def engineer_features(self):
        """Create advanced features for modeling"""
        print("\nEngineering features...")
        
        # Time-based features
        self.df['hour_of_day'] = self.df['scheduled_arrival'].dt.hour
        self.df['day_of_week'] = self.df['scheduled_arrival'].dt.dayofweek
        self.df['month'] = self.df['scheduled_arrival'].dt.month
        
        # Buffer times
        self.df['cleaning_buffer'] = self.df['cleaning_time'] - 20  # Deviation from mean
        self.df['fueling_buffer'] = self.df['fueling_time'] - 15
        self.df['boarding_buffer'] = self.df['boarding_time'] - 25
        
        # Congestion level (flights per hour)
        self.df['flights_per_hour'] = self.df.groupby(
            self.df['scheduled_arrival'].dt.floor('H')
        )['flight_id'].transform('count')
        
        # Weather index (numeric encoding)
        weather_severity = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
        self.df['weather_index'] = self.df['weather_condition'].map(weather_severity)
        
        # Delay propagation indicator
        self.df['arrival_delay'] = (
            self.df['actual_arrival'] - self.df['scheduled_arrival']
        ).dt.total_seconds() / 60
        
        # Critical path time (max of parallel operations)
        self.df['critical_path_time'] = self.df[[
            'cleaning_time', 'fueling_time', 'baggage_time'
        ]].max(axis=1)
        
        # Total ground time needed
        self.df['total_ground_time'] = (
            self.df['critical_path_time'] + 
            self.df['boarding_time'] + 
            self.df['gate_delay'] + 
            self.df['weather_delay']
        )
        
        print("Feature engineering complete!")
        
    def prepare_model_data(self):
        """Prepare features and target for modeling"""
        print("\nPreparing model data...")
        
        # Select features for modeling
        feature_cols = [
            'cleaning_time', 'fueling_time', 'boarding_time', 'baggage_time',
            'gate_available', 'crew_ready_offset', 'gate_delay', 'weather_delay',
            'hour_of_day', 'day_of_week', 'month',
            'cleaning_buffer', 'fueling_buffer', 'boarding_buffer',
            'flights_per_hour', 'weather_index', 'arrival_delay',
            'critical_path_time', 'total_ground_time'
        ]
        
        X = self.df[feature_cols]
        y = self.df['is_delayed']
        
        self.feature_names = feature_cols
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
    def train_models(self):
        """Train Random Forest and XGBoost models"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate RF
        rf_pred = self.rf_model.predict(self.X_test)
        rf_prob = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        print("\nRandom Forest Results:")
        print(classification_report(self.y_test, rf_pred))
        print(f"ROC-AUC: {roc_auc_score(self.y_test, rf_prob):.4f}")
        
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgb_model.fit(self.X_train, self.y_train)
        
        # Evaluate XGBoost
        xgb_pred = self.xgb_model.predict(self.X_test)
        xgb_prob = self.xgb_model.predict_proba(self.X_test)[:, 1]
        
        print("\nXGBoost Results:")
        print(classification_report(self.y_test, xgb_pred))
        print(f"ROC-AUC: {roc_auc_score(self.y_test, xgb_prob):.4f}")
        
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        import os
        print("\n" + "="*50)
        print("Analyzing Feature Importance...")
        print("="*50)
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features (Random Forest):")
        print(rf_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=rf_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        output_path = os.path.join(project_dir, 'data', 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to {output_path}")
        
    def generate_shap_analysis(self):
        """Generate SHAP values for model interpretability"""
        import os
        print("\n" + "="*50)
        print("Generating SHAP Analysis...")
        print("="*50)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(self.X_test)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        output_path = os.path.join(project_dir, 'data', 'shap_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {output_path}")
        
    def save_models(self):
        """Save trained models"""
        import os
        print("\nSaving models...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(project_dir, 'data')
        
        with open(os.path.join(data_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)
        with open(os.path.join(data_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.xgb_model, f)
        print("Models saved successfully!")
        
    def run_full_pipeline(self):
        """Execute the complete modeling pipeline"""
        self.load_and_prepare_data()
        self.engineer_features()
        self.prepare_model_data()
        self.train_models()
        self.analyze_feature_importance()
        self.generate_shap_analysis()
        self.save_models()
        print("\n" + "="*50)
        print("Pipeline Complete!")
        print("="*50)

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'aviation_turnaround_data.csv')
    
    predictor = AviationDelayPredictor(data_path)
    predictor.run_full_pipeline()
