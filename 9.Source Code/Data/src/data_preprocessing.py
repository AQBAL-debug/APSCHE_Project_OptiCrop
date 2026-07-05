import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class DataPreprocessor:
    """Handles data preprocessing for crop recommendation system"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load the crop recommendation dataset"""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Explore the dataset"""
        print("\n=== Dataset Info ===")
        print(self.df.info())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        print("\n=== Statistical Summary ===")
        print(self.df.describe())
        print("\n=== Crop Types ===")
        print(self.df['label'].value_counts())
        print(f"\nNumber of unique crops: {self.df['label'].nunique()}")
        
    def clean_data(self):
        """Clean the dataset"""
        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.df = self.df.dropna()
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows...")
            self.df = self.df.drop_duplicates()
        
        print(f"Data cleaned. Final shape: {self.df.shape}")
        return self.df
    
    def prepare_features(self):
        """Prepare features and target variables"""
        # Features (N, P, K, temperature, humidity, ph, rainfall)
        self.X = self.df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        
        # Target (crop label)
        self.y = self.label_encoder.fit_transform(self.df['label'])
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"Training set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale the features using StandardScaler"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled successfully")
        return self.X_train_scaled, self.X_test_scaled
    
    def save_preprocessors(self, model_dir):
        """Save the scaler and label encoder for later use"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        
        print(f"Preprocessors saved to {model_dir}")
    
    def get_crop_names(self):
        """Get the list of crop names"""
        return self.label_encoder.classes_
    
    def preprocess_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("=== Starting Data Preprocessing ===")
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Clean data
        self.clean_data()
        
        # Prepare features
        self.prepare_features()
        
        # Split data
        self.split_data()
        
        # Scale features
        self.scale_features()
        
        print("=== Data Preprocessing Complete ===")
        
        return (self.X_train_scaled, self.X_test_scaled, 
                self.y_train, self.y_test, 
                self.scaler, self.label_encoder)


if __name__ == "__main__":
    # Test the preprocessor
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'data', 'crop_recommendation.csv')
    preprocessor = DataPreprocessor(data_path)
    preprocessor.preprocess_pipeline()
