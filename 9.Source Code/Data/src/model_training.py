import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Handles model training and evaluation for crop recommendation"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.best_model = None
        
    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        """Train a Random Forest Classifier"""
        print("\n=== Training Random Forest Classifier ===")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Random Forest model trained successfully")
        
        return self.model
    
    def evaluate_model(self, model, X_test, y_test, label_encoder):
        """Evaluate the model performance"""
        print("\n=== Model Evaluation ===")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=label_encoder.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy, y_pred, cm
    
    def cross_validate(self, model, X_train, y_train, cv=5):
        """Perform cross-validation"""
        print("\n=== Cross-Validation ===")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("\n=== Hyperparameter Tuning ===")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        return self.best_model, grid_search.best_params_
    
    def feature_importance(self, model, feature_names):
        """Get and display feature importance"""
        print("\n=== Feature Importance ===")
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance_df)
        
        return feature_importance_df
    
    def save_model(self, model, filename='crop_recommendation_model.pkl'):
        """Save the trained model"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(model, model_path)
        
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, filename='crop_recommendation_model.pkl'):
        """Load a trained model"""
        model_path = os.path.join(self.model_dir, filename)
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return self.model
        else:
            print(f"Model file not found at {model_path}")
            return None
    
    def training_pipeline(self, X_train, X_test, y_train, y_test, label_encoder, feature_names):
        """Run the complete training pipeline"""
        print("=== Starting Model Training Pipeline ===")
        
        # Train Random Forest
        model = self.train_random_forest(X_train, y_train)
        
        # Evaluate model
        accuracy, y_pred, cm = self.evaluate_model(model, X_test, y_test, label_encoder)
        
        # Cross-validation
        cv_scores = self.cross_validate(model, X_train, y_train)
        
        # Feature importance
        feature_importance = self.feature_importance(model, feature_names)
        
        # Save model
        model_path = self.save_model(model)
        
        print("\n=== Model Training Pipeline Complete ===")
        print(f"Final Model Accuracy: {accuracy:.4f}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        
        return model, accuracy, feature_importance


if __name__ == "__main__":
    import pandas as pd
    from data_preprocessing import DataPreprocessor
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'crop_recommendation.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    # Preprocess data
    preprocessor = DataPreprocessor(data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocessor.preprocess_pipeline()
    
    # Save preprocessors
    preprocessor.save_preprocessors(model_dir)
    
    # Train model
    trainer = ModelTrainer(model_dir)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    model, accuracy, feature_importance = trainer.training_pipeline(
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, feature_names
    )
