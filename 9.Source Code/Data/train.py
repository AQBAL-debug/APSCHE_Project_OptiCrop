"""
Training script for OptiCrop model
This script trains the crop recommendation model and saves it to the models directory
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

def main():
    print("=" * 60)
    print("OptiCrop Model Training Pipeline")
    print("=" * 60)
    
    # Setup paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'crop_recommendation.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the dataset is in the data directory.")
        return
    
    # Step 1: Data Preprocessing
    print("\n[Step 1/3] Data Preprocessing")
    print("-" * 60)
    preprocessor = DataPreprocessor(data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = preprocessor.preprocess_pipeline()
    
    # Save preprocessors
    preprocessor.save_preprocessors(model_dir)
    
    # Step 2: Model Training
    print("\n[Step 2/3] Model Training")
    print("-" * 60)
    trainer = ModelTrainer(model_dir)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    model, accuracy, feature_importance = trainer.training_pipeline(
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, feature_names
    )
    
    # Step 3: Verification
    print("\n[Step 3/3] Model Verification")
    print("-" * 60)
    
    # Test with sample data
    from crop_recommendation import CropRecommendationSystem
    
    try:
        system = CropRecommendationSystem(model_dir)
        
        # Test prediction
        print("\nTesting prediction with sample data...")
        test_result = system.predict_crop(
            N=90, P=42, K=43, 
            temperature=20.88, humidity=82.00, 
            ph=6.50, rainfall=202.94
        )
        print(f"✓ Recommended Crop: {test_result['recommended_crop']}")
        print(f"✓ Confidence: {test_result['confidence']:.2%}")
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"\nModel accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_dir}")
        print("\nYou can now run the application with: streamlit run app.py")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return

if __name__ == "__main__":
    main()
