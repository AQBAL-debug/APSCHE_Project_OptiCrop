import numpy as np
import joblib
import os
import pandas as pd

class CropRecommendationSystem:
    """Main class for crop recommendation system"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Load all components
        self.load_components()
    
    def load_components(self):
        """Load the trained model and preprocessors"""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'crop_recommendation_model.pkl')
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            
            # Load label encoder
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            self.label_encoder = joblib.load(encoder_path)
            
            print("All components loaded successfully")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            raise
    
    def predict_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Predict the best crop based on soil and environmental parameters
        
        Parameters:
        - N: Nitrogen content ratio in soil
        - P: Phosphorous content ratio in soil
        - K: Potassium content ratio in soil
        - temperature: Soil temperature in °C
        - humidity: Relative humidity in %
        - ph: Soil pH value
        - rainfall: Rainfall in mm
        
        Returns:
        - Dictionary with recommended crop and confidence scores
        """
        # Prepare input features as DataFrame with proper column names
        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability scores for all crops
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decode the predicted crop
        recommended_crop = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get top 3 recommendations with probabilities
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = self.label_encoder.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        
        result = {
            'recommended_crop': recommended_crop,
            'confidence': float(probabilities[prediction]),
            'top_3_recommendations': [
                {'crop': crop, 'probability': float(prob)} 
                for crop, prob in zip(top_3_crops, top_3_probs)
            ]
        }
        
        return result
    
    def assess_crop_suitability(self, N, P, K, temperature, humidity, ph, rainfall, target_crop):
        """
        Assess suitability of a specific crop for given conditions
        
        Parameters:
        - N, P, K, temperature, humidity, ph, rainfall: Environmental parameters
        - target_crop: The crop to assess suitability for
        
        Returns:
        - Dictionary with suitability score and assessment
        """
        # Prepare input features as DataFrame with proper column names
        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability scores for all crops
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Find the target crop index
        try:
            target_crop_encoded = self.label_encoder.transform([target_crop])[0]
            suitability_score = float(probabilities[target_crop_encoded])
        except ValueError:
            return {
                'error': f'Crop "{target_crop}" not found in the dataset',
                'available_crops': list(self.label_encoder.classes_)
            }
        
        # Determine suitability level
        if suitability_score >= 0.7:
            suitability_level = "Highly Suitable"
        elif suitability_score >= 0.5:
            suitability_level = "Moderately Suitable"
        elif suitability_score >= 0.3:
            suitability_level = "Less Suitable"
        else:
            suitability_level = "Not Suitable"
        
        result = {
            'target_crop': target_crop,
            'suitability_score': suitability_score,
            'suitability_level': suitability_level,
            'recommendation': f"The conditions are {suitability_level.lower()} for {target_crop}."
        }
        
        return result
    
    def get_all_crops(self):
        """Get list of all crops in the dataset"""
        return list(self.label_encoder.classes_)
    
    def analyze_environmental_factors(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Analyze environmental factors and provide insights
        
        Returns:
        - Dictionary with analysis of each factor
        """
        analysis = {
            'nitrogen': {
                'value': N,
                'status': self._analyze_nitrogen(N)
            },
            'phosphorus': {
                'value': P,
                'status': self._analyze_phosphorus(P)
            },
            'potassium': {
                'value': K,
                'status': self._analyze_potassium(K)
            },
            'temperature': {
                'value': temperature,
                'status': self._analyze_temperature(temperature)
            },
            'humidity': {
                'value': humidity,
                'status': self._analyze_humidity(humidity)
            },
            'ph': {
                'value': ph,
                'status': self._analyze_ph(ph)
            },
            'rainfall': {
                'value': rainfall,
                'status': self._analyze_rainfall(rainfall)
            }
        }
        
        return analysis
    
    def _analyze_nitrogen(self, N):
        """Analyze nitrogen level"""
        if N < 50:
            return "Low - Consider adding nitrogen-rich fertilizers"
        elif N < 100:
            return "Moderate - Suitable for most crops"
        else:
            return "High - Monitor for nitrogen toxicity"
    
    def _analyze_phosphorus(self, P):
        """Analyze phosphorus level"""
        if P < 30:
            return "Low - Add phosphorus fertilizers"
        elif P < 70:
            return "Moderate - Good for most crops"
        else:
            return "High - Adequate for phosphorus-loving crops"
    
    def _analyze_potassium(self, K):
        """Analyze potassium level"""
        if K < 40:
            return "Low - Add potassium fertilizers"
        elif K < 80:
            return "Moderate - Suitable for most crops"
        else:
            return "High - Good for potassium-demanding crops"
    
    def _analyze_temperature(self, temp):
        """Analyze temperature"""
        if temp < 15:
            return "Cool - Suitable for cool-season crops"
        elif temp < 30:
            return "Moderate - Ideal for most crops"
        else:
            return "High - Suitable for heat-tolerant crops"
    
    def _analyze_humidity(self, humidity):
        """Analyze humidity"""
        if humidity < 50:
            return "Low - May require irrigation"
        elif humidity < 80:
            return "Moderate - Good for most crops"
        else:
            return "High - Risk of fungal diseases"
    
    def _analyze_ph(self, ph):
        """Analyze soil pH"""
        if ph < 5.5:
            return "Acidic - Consider adding lime"
        elif ph < 7.5:
            return "Neutral - Ideal for most crops"
        else:
            return "Alkaline - Consider adding sulfur"
    
    def _analyze_rainfall(self, rainfall):
        """Analyze rainfall"""
        if rainfall < 50:
            return "Low - Drought conditions, irrigation needed"
        elif rainfall < 150:
            return "Moderate - Good for most crops"
        else:
            return "High - Risk of waterlogging"


if __name__ == "__main__":
    # Test the recommendation system
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'models')
    
    # Initialize system
    system = CropRecommendationSystem(model_dir)
    
    # Test prediction
    print("\n=== Test Crop Recommendation ===")
    result = system.predict_crop(
        N=90, P=42, K=43, 
        temperature=20.88, humidity=82.00, 
        ph=6.50, rainfall=202.94
    )
    print(f"Recommended Crop: {result['recommended_crop']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 3 Recommendations:")
    for i, rec in enumerate(result['top_3_recommendations'], 1):
        print(f"{i}. {rec['crop']}: {rec['probability']:.2%}")
    
    # Test suitability assessment
    print("\n=== Test Crop Suitability Assessment ===")
    suitability = system.assess_crop_suitability(
        N=90, P=42, K=43, 
        temperature=20.88, humidity=82.00, 
        ph=6.50, rainfall=202.94,
        target_crop='rice'
    )
    print(f"Suitability for rice: {suitability['suitability_level']}")
    print(f"Score: {suitability['suitability_score']:.2%}")
    
    # Test environmental analysis
    print("\n=== Environmental Factor Analysis ===")
    analysis = system.analyze_environmental_factors(
        N=90, P=42, K=43, 
        temperature=20.88, humidity=82.00, 
        ph=6.50, rainfall=202.94
    )
    for factor, data in analysis.items():
        print(f"{factor.capitalize()}: {data['value']} - {data['status']}")
