import numpy as np
import joblib
import os
import pandas as pd
try:
    from crop_profiles import CROP_PROFILES
except ImportError:
    from .crop_profiles import CROP_PROFILES

class CropRecommendationSystem:
    """Main class for crop recommendation system"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.crop_profiles = CROP_PROFILES
        
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
        - N: Nitrogen content ratio in soil (0 - 140)
        - P: Phosphorous content ratio in soil (5 - 145)
        - K: Potassium content ratio in soil (5 - 205)
        - temperature: Soil temperature in °C (0 - 45)
        - humidity: Relative humidity in % (10 - 100)
        - ph: Soil pH value (3.5 - 9.5)
        - rainfall: Rainfall in mm (20 - 300)
        
        Returns:
        - Dictionary with recommended crop and confidence scores
        """
        # Prepare input features as DataFrame with proper column names
        features = pd.DataFrame([[float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]], 
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
        Assess suitability of a specific crop for given conditions.
        Combines model probability with environmental range compatibility.
        """
        target_crop_clean = target_crop.strip().lower()
        
        # Find target crop index
        try:
            target_crop_encoded = self.label_encoder.transform([target_crop_clean])[0]
        except ValueError:
            return {
                'error': f'Crop "{target_crop}" not found in the dataset',
                'available_crops': list(self.label_encoder.classes_)
            }
        
        # Prepare features for ML model
        features = pd.DataFrame([[float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]], 
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        features_scaled = self.scaler.transform(features)
        ml_prob = float(self.model.predict_proba(features_scaled)[0][target_crop_encoded])
        
        # Calculate environmental envelope suitability
        profile = self.crop_profiles.get(target_crop_clean)
        inputs = {
            'N': float(N), 'P': float(P), 'K': float(K),
            'temperature': float(temperature), 'humidity': float(humidity),
            'ph': float(ph), 'rainfall': float(rainfall)
        }
        
        param_labels = {
            'N': 'Nitrogen (N)', 'P': 'Phosphorus (P)', 'K': 'Potassium (K)',
            'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)',
            'ph': 'Soil pH', 'rainfall': 'Rainfall (mm)'
        }
        
        param_analysis = []
        param_scores = []
        
        if profile:
            for p, val in inputs.items():
                p_min = profile[p]['min']
                p_max = profile[p]['max']
                p_mean = profile[p]['mean']
                
                # Tolerance range with 15% margin
                tolerance = max((p_max - p_min) * 0.15, 1.0)
                
                if val < (p_min - tolerance):
                    status = "Low / Deficient"
                    score = max(0.0, 1.0 - (p_min - val) / (p_mean if p_mean else 1.0))
                    advice = f"Increase {param_labels[p]} towards optimal range ({p_min:.1f} - {p_max:.1f})."
                elif val > (p_max + tolerance):
                    status = "High / Excessive"
                    score = max(0.0, 1.0 - (val - p_max) / (p_mean if p_mean else 1.0))
                    advice = f"Reduce or manage {param_labels[p]} towards optimal range ({p_min:.1f} - {p_max:.1f})."
                else:
                    status = "Optimal"
                    score = 1.0
                    advice = f"Ideal range for {target_crop_clean.capitalize()}."
                
                param_scores.append(score)
                param_analysis.append({
                    'param': p,
                    'label': param_labels[p],
                    'user_value': round(val, 2),
                    'min': p_min,
                    'mean': p_mean,
                    'max': p_max,
                    'status': status,
                    'advice': advice
                })
        
        env_score = float(np.mean(param_scores)) if param_scores else 0.5
        
        # Blended final suitability score (60% environmental tolerance + 40% ML probability)
        # If ML probability is very high, it boosts the score; if environmental conditions are within tolerance, it maintains high suitability.
        blended_score = round(float(0.65 * env_score + 0.35 * ml_prob), 4)
        blended_score = min(max(blended_score, 0.05), 1.0)
        
        # Determine suitability level
        if blended_score >= 0.75:
            suitability_level = "Highly Suitable"
            rec_text = f"Excellent growing conditions! Your soil and climate parameters align closely with the natural requirements for {target_crop_clean.capitalize()}."
        elif blended_score >= 0.55:
            suitability_level = "Moderately Suitable"
            rec_text = f"Favorable conditions with minor deviations. With minor fertilizer or irrigation adjustments, {target_crop_clean.capitalize()} can produce good yield."
        elif blended_score >= 0.35:
            suitability_level = "Marginally Suitable"
            rec_text = f"Challenging environmental conditions for {target_crop_clean.capitalize()}. Significant soil amendments or controlled irrigation will be required."
        else:
            suitability_level = "Not Suitable"
            rec_text = f"Conditions are unfavorable for {target_crop_clean.capitalize()}. Consider growing alternative recommended crops to avoid crop failure."
        
        return {
            'target_crop': target_crop_clean,
            'suitability_score': blended_score,
            'suitability_level': suitability_level,
            'ml_probability': round(ml_prob, 4),
            'env_compatibility': round(env_score, 4),
            'recommendation': rec_text,
            'param_analysis': param_analysis
        }
    
    def get_all_crops(self):
        """Get list of all crops in the dataset"""
        return sorted(list(self.label_encoder.classes_))
    
    def analyze_environmental_factors(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Analyze environmental factors and provide insights
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
        if N < 50:
            return "Low - Consider applying Urea or organic compost"
        elif N <= 100:
            return "Optimal - Balanced nitrogen level for healthy vegetative growth"
        else:
            return "High - Ample nitrogen; monitor to avoid excessive foliage"
    
    def _analyze_phosphorus(self, P):
        if P < 30:
            return "Low - Apply DAP or Single Super Phosphate (SSP) for root growth"
        elif P <= 70:
            return "Optimal - Adequate phosphorus for root and flowering stage"
        else:
            return "High - Elevated phosphorus; suitable for high phosphorus-demanding crops"
    
    def _analyze_potassium(self, K):
        if K < 40:
            return "Low - Apply Muriate of Potash (MOP) to boost pest resistance"
        elif K <= 80:
            return "Optimal - Strong potassium reserves for disease resistance"
        else:
            return "High - Rich potassium levels ideal for tubers and fruit crops"
    
    def _analyze_temperature(self, temp):
        if temp < 18:
            return "Cool - Best for cool-season crops (e.g. chickpea, lentil)"
        elif temp <= 32:
            return "Moderate/Warm - Ideal tropical & sub-tropical conditions"
        else:
            return "Hot - Requires adequate irrigation and heat-tolerant varieties"
    
    def _analyze_humidity(self, humidity):
        if humidity < 40:
            return "Arid / Low - May require frequent irrigation or mulch"
        elif humidity <= 80:
            return "Moderate - Ideal for most cereal, pulse and cash crops"
        else:
            return "High - Humid climate; monitor for fungal pathogens"
    
    def _analyze_ph(self, ph):
        if ph < 5.5:
            return "Acidic - Apply agricultural lime (calcium carbonate) to neutralize"
        elif ph <= 7.5:
            return "Neutral / Ideal - Perfect nutrient availability zone"
        else:
            return "Alkaline - Apply agricultural gypsum or organic matter to lower pH"
    
    def _analyze_rainfall(self, rainfall):
        if rainfall < 60:
            return "Low Rainfall - Dryland farming, supplemental irrigation required"
        elif rainfall <= 180:
            return "Moderate Rainfall - Balanced precipitation for general cultivation"
        else:
            return "Heavy Rainfall - High water availability; ensure proper field drainage"

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'models')
    system = CropRecommendationSystem(model_dir)
    res = system.predict_crop(90, 42, 43, 20.88, 82.00, 6.50, 202.94)
    print("Test recommendation:", res)
    suit = system.assess_crop_suitability(90, 42, 43, 20.88, 82.00, 6.50, 202.94, 'rice')
    print("Test suitability:", suit['suitability_level'], suit['suitability_score'])
