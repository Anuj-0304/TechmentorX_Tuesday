"""
Keras Model Integration for Crop Recommendation
This script shows how to integrate your .keras model files with the AgriTech platform
"""

import numpy as np
from tensorflow import keras
import pandas as pd

class CropPredictionModel:
    """
    Wrapper class for crop prediction using Keras models
    """
    
    def __init__(self, model_path='crop_model.keras'):
        """
        Initialize the model
        
        Args:
            model_path: Path to your .keras model file
        """
        self.model = None
        self.model_path = model_path
        self.crop_labels = [
            'Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 
            'Pulses', 'Vegetables', 'Fruits', 'Groundnut', 'Soybean'
        ]
        
    def load_model(self):
        """Load the trained Keras model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, nitrogen, phosphorus, potassium, 
                        temperature, humidity, ph, rainfall):
        """
        Preprocess input features for the model
        
        Args:
            nitrogen: Nitrogen content in soil (kg/ha)
            phosphorus: Phosphorus content in soil (kg/ha)
            potassium: Potassium content in soil (kg/ha)
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            ph: Soil pH value
            rainfall: Rainfall in mm
            
        Returns:
            Preprocessed numpy array ready for prediction
        """
        # Create feature array
        features = np.array([[nitrogen, phosphorus, potassium, 
                            temperature, humidity, ph, rainfall]])
        
        # Normalize features (adjust based on your training normalization)
        feature_ranges = {
            'nitrogen': (0, 200),
            'phosphorus': (0, 100),
            'potassium': (0, 100),
            'temperature': (10, 45),
            'humidity': (30, 100),
            'ph': (4, 9),
            'rainfall': (0, 300)
        }
        
        normalized_features = []
        for i, (feat_name, (min_val, max_val)) in enumerate(feature_ranges.items()):
            normalized = (features[0][i] - min_val) / (max_val - min_val)
            normalized_features.append(normalized)
        
        return np.array([normalized_features])
    
    def predict_crop(self, nitrogen, phosphorus, potassium, 
                    temperature, humidity, ph, rainfall):
        """
        Predict the best crop based on soil and environmental conditions
        
        Returns:
            tuple: (crop_name, confidence_score)
        """
        if self.model is None:
            print("Model not loaded. Using rule-based prediction.")
            return self._rule_based_prediction(ph, rainfall, temperature, humidity)
        
        # Preprocess input
        input_features = self.preprocess_input(
            nitrogen, phosphorus, potassium, 
            temperature, humidity, ph, rainfall
        )
        
        # Make prediction
        predictions = self.model.predict(input_features, verbose=0)
        
        # Get the crop with highest probability
        crop_index = np.argmax(predictions[0])
        confidence = predictions[0][crop_index]
        
        crop_name = self.crop_labels[crop_index]
        
        return crop_name, float(confidence)
    
    def get_top_n_crops(self, nitrogen, phosphorus, potassium, 
                       temperature, humidity, ph, rainfall, n=3):
        """
        Get top N crop recommendations
        
        Returns:
            list: List of tuples (crop_name, confidence_score)
        """
        if self.model is None:
            return [self._rule_based_prediction(ph, rainfall, temperature, humidity)]
        
        input_features = self.preprocess_input(
            nitrogen, phosphorus, potassium, 
            temperature, humidity, ph, rainfall
        )
        
        predictions = self.model.predict(input_features, verbose=0)
        
        # Get top N predictions
        top_indices = np.argsort(predictions[0])[-n:][::-1]
        
        results = []
        for idx in top_indices:
            crop_name = self.crop_labels[idx]
            confidence = predictions[0][idx]
            results.append((crop_name, float(confidence)))
        
        return results
    
    def _rule_based_prediction(self, ph, rainfall, temperature, humidity):
        """
        Fallback rule-based prediction when model is not available
        """
        if ph < 6.5:
            if rainfall > 150:
                return "Rice", 0.85
            else:
                return "Wheat", 0.80
        elif ph > 7.5:
            if temperature > 25:
                return "Cotton", 0.82
            else:
                return "Sugarcane", 0.78
        else:
            if humidity > 70:
                return "Maize", 0.88
            else:
                return "Pulses", 0.75


# Example usage
if __name__ == "__main__":
    # Initialize model
    predictor = CropPredictionModel('crop_model.keras')
    
    # Try to load the model
    model_loaded = predictor.load_model()
    
    # Example prediction
    print("\n=== Crop Prediction Example ===")
    print("\nInput Parameters:")
    params = {
        'nitrogen': 90,
        'phosphorus': 42,
        'potassium': 43,
        'temperature': 20.8,
        'humidity': 82,
        'ph': 6.5,
        'rainfall': 202
    }
    
    for key, value in params.items():
        print(f"{key.capitalize()}: {value}")
    
    # Get prediction
    crop, confidence = predictor.predict_crop(**params)
    
    print(f"\n=== Prediction Results ===")
    print(f"Recommended Crop: {crop}")
    print(f"Confidence: {confidence:.2%}")
    
    # Get top 3 recommendations
    print(f"\n=== Top 3 Recommendations ===")
    top_crops = predictor.get_top_n_crops(**params, n=3)
    for i, (crop_name, conf) in enumerate(top_crops, 1):
        print(f"{i}. {crop_name}: {conf:.2%}")
