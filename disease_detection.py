"""
CNN-based Crop Disease Detection Module
Supports image-based disease identification using Keras models
"""

import numpy as np
from tensorflow import keras
from PIL import Image
import cv2

class DiseaseDetectionModel:
    """
    CNN model for detecting crop diseases from images
    """
    
    def __init__(self, model_path='disease_model.keras'):
        """
        Initialize the disease detection model
        
        Args:
            model_path: Path to your .keras CNN model file
        """
        self.model = None
        self.model_path = model_path
        
        # Common crop diseases - expand based on your training data
        self.disease_labels = {
            0: 'Healthy',
            1: 'Bacterial Blight',
            2: 'Brown Spot',
            3: 'Leaf Smut',
            4: 'Powdery Mildew',
            5: 'Rust',
            6: 'Yellow Leaf Disease',
            7: 'Leaf Blight',
            8: 'Mosaic Virus',
            9: 'Anthracnose',
            10: 'Cercospora Leaf Spot',
            11: 'Common Leaf Spot',
            12: 'Early Blight',
            13: 'Late Blight',
            14: 'Septoria Leaf Spot',
            15: 'Target Spot'
        }
        
        # Treatment recommendations for each disease
        self.treatments = {
            'Healthy': {
                'severity': 'None',
                'treatment': 'No treatment needed. Continue regular care.',
                'prevention': 'Maintain good agricultural practices.',
                'organic': 'N/A',
                'chemical': 'N/A'
            },
            'Bacterial Blight': {
                'severity': 'High',
                'treatment': 'Remove infected parts, apply copper-based bactericide',
                'prevention': 'Use disease-free seeds, avoid overhead irrigation',
                'organic': 'Neem oil spray, copper sulfate solution',
                'chemical': 'Streptomycin sulfate, Copper oxychloride'
            },
            'Brown Spot': {
                'severity': 'Medium',
                'treatment': 'Apply fungicide, improve drainage',
                'prevention': 'Balanced fertilization, avoid water stress',
                'organic': 'Trichoderma viride, Pseudomonas fluorescens',
                'chemical': 'Mancozeb, Carbendazim'
            },
            'Leaf Smut': {
                'severity': 'Medium',
                'treatment': 'Remove infected leaves, apply systemic fungicide',
                'prevention': 'Crop rotation, resistant varieties',
                'organic': 'Bordeaux mixture, Sulfur dust',
                'chemical': 'Propiconazole, Tebuconazole'
            },
            'Powdery Mildew': {
                'severity': 'Medium',
                'treatment': 'Apply sulfur-based fungicide, improve air circulation',
                'prevention': 'Avoid overcrowding, reduce humidity',
                'organic': 'Baking soda solution, Milk spray (1:9 ratio)',
                'chemical': 'Sulfur, Myclobutanil'
            },
            'Rust': {
                'severity': 'High',
                'treatment': 'Apply fungicide immediately, remove infected parts',
                'prevention': 'Plant resistant varieties, proper spacing',
                'organic': 'Neem oil, Garlic extract',
                'chemical': 'Propiconazole, Mancozeb'
            },
            'Yellow Leaf Disease': {
                'severity': 'High',
                'treatment': 'No cure, remove infected plants to prevent spread',
                'prevention': 'Control insect vectors, use healthy planting material',
                'organic': 'Remove and destroy infected plants',
                'chemical': 'Insecticides for vector control'
            },
            'Leaf Blight': {
                'severity': 'High',
                'treatment': 'Apply systemic fungicide, improve drainage',
                'prevention': 'Crop rotation, avoid excessive nitrogen',
                'organic': 'Copper fungicide, Bacillus subtilis',
                'chemical': 'Chlorothalonil, Mancozeb'
            },
            'Mosaic Virus': {
                'severity': 'High',
                'treatment': 'Remove infected plants, control aphid vectors',
                'prevention': 'Use virus-free seeds, control weeds',
                'organic': 'Neem oil for aphid control',
                'chemical': 'Systemic insecticides for aphid control'
            },
            'Anthracnose': {
                'severity': 'Medium',
                'treatment': 'Apply fungicide, remove infected debris',
                'prevention': 'Crop rotation, avoid overhead watering',
                'organic': 'Copper fungicide, Bacillus subtilis',
                'chemical': 'Chlorothalonil, Azoxystrobin'
            },
            'Cercospora Leaf Spot': {
                'severity': 'Medium',
                'treatment': 'Apply fungicide, improve air circulation',
                'prevention': 'Crop rotation, remove crop debris',
                'organic': 'Copper fungicide, Trichoderma',
                'chemical': 'Mancozeb, Chlorothalonil'
            },
            'Common Leaf Spot': {
                'severity': 'Low',
                'treatment': 'Apply fungicide if severe, remove infected leaves',
                'prevention': 'Proper spacing, avoid wet foliage',
                'organic': 'Neem oil, Copper fungicide',
                'chemical': 'Chlorothalonil, Mancozeb'
            },
            'Early Blight': {
                'severity': 'Medium',
                'treatment': 'Apply fungicide early, remove lower leaves',
                'prevention': 'Crop rotation, mulching, resistant varieties',
                'organic': 'Copper fungicide, Bacillus subtilis',
                'chemical': 'Chlorothalonil, Mancozeb'
            },
            'Late Blight': {
                'severity': 'High',
                'treatment': 'Apply fungicide immediately, remove infected plants',
                'prevention': 'Resistant varieties, avoid overhead irrigation',
                'organic': 'Copper fungicide applied preventively',
                'chemical': 'Metalaxyl, Chlorothalonil'
            },
            'Septoria Leaf Spot': {
                'severity': 'Medium',
                'treatment': 'Apply fungicide, remove infected leaves',
                'prevention': 'Crop rotation, avoid overhead watering',
                'organic': 'Copper fungicide, Neem oil',
                'chemical': 'Chlorothalonil, Mancozeb'
            },
            'Target Spot': {
                'severity': 'Medium',
                'treatment': 'Apply fungicide, improve air circulation',
                'prevention': 'Crop rotation, remove debris',
                'organic': 'Copper fungicide, Trichoderma',
                'chemical': 'Azoxystrobin, Chlorothalonil'
            }
        }
        
    def load_model(self):
        """Load the trained CNN model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Disease detection model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Model not found. Using rule-based detection: {str(e)}")
            return False
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess image for CNN model
        
        Args:
            image: PIL Image or numpy array
            target_size: Target size for the model (width, height)
            
        Returns:
            Preprocessed numpy array
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def analyze_image_features(self, image):
        """
        Rule-based image analysis for disease detection
        Used as fallback when CNN model is not available
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to HSV for better color analysis
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        else:
            return 'Healthy', 0.5
        
        # Analyze color distribution
        h, s, v = cv2.split(hsv)
        
        # Calculate mean values
        mean_hue = np.mean(h)
        mean_sat = np.mean(s)
        mean_val = np.mean(v)
        
        # Calculate standard deviation (variation indicates spots/lesions)
        std_hue = np.std(h)
        std_val = np.std(v)
        
        # Rule-based classification
        # Yellow/brown spots
        if mean_hue > 20 and mean_hue < 40 and std_val > 40:
            return 'Brown Spot', 0.75
        
        # White powdery appearance
        elif mean_val > 200 and mean_sat < 50:
            return 'Powdery Mildew', 0.72
        
        # Dark/black spots
        elif mean_val < 100 and std_val > 50:
            return 'Leaf Blight', 0.70
        
        # Orange/rust colored
        elif mean_hue > 5 and mean_hue < 20 and mean_sat > 100:
            return 'Rust', 0.73
        
        # Yellow discoloration
        elif mean_hue > 40 and mean_hue < 60 and std_hue < 20:
            return 'Yellow Leaf Disease', 0.68
        
        # Mosaic pattern (high variation)
        elif std_hue > 30 and std_val > 40:
            return 'Mosaic Virus', 0.65
        
        # Healthy - uniform green
        elif mean_hue > 60 and mean_hue < 90 and std_hue < 15 and std_val < 35:
            return 'Healthy', 0.88
        
        # Default to common leaf spot if uncertain
        else:
            return 'Common Leaf Spot', 0.60
    
    def detect_disease(self, image):
        """
        Detect disease from crop image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (disease_name, confidence, treatment_info)
        """
        if self.model is not None:
            # Use CNN model
            preprocessed = self.preprocess_image(image)
            predictions = self.model.predict(preprocessed, verbose=0)
            
            # Get top prediction
            disease_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][disease_idx])
            disease_name = self.disease_labels.get(disease_idx, 'Unknown Disease')
            
        else:
            # Use rule-based detection
            disease_name, confidence = self.analyze_image_features(image)
        
        # Get treatment information
        treatment_info = self.treatments.get(disease_name, {
            'severity': 'Unknown',
            'treatment': 'Consult agricultural expert',
            'prevention': 'Practice good agricultural hygiene',
            'organic': 'Contact local agricultural extension office',
            'chemical': 'Consult with agronomist'
        })
        
        return disease_name, confidence, treatment_info
    
    def get_multiple_predictions(self, image, top_n=3):
        """
        Get top N disease predictions
        
        Returns:
            list: List of tuples (disease_name, confidence, treatment_info)
        """
        if self.model is not None:
            preprocessed = self.preprocess_image(image)
            predictions = self.model.predict(preprocessed, verbose=0)
            
            # Get top N predictions
            top_indices = np.argsort(predictions[0])[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                disease_name = self.disease_labels.get(idx, 'Unknown')
                confidence = float(predictions[0][idx])
                treatment = self.treatments.get(disease_name, {})
                results.append((disease_name, confidence, treatment))
            
            return results
        else:
            # Return single rule-based prediction
            disease, conf, treatment = self.detect_disease(image)
            return [(disease, conf, treatment)]
    
    def calculate_disease_severity(self, image):
        """
        Calculate percentage of leaf area affected
        
        Returns:
            float: Percentage of affected area (0-100)
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if len(img_array.shape) != 3:
            return 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define range for healthy green leaves
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask for healthy areas
        healthy_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentages
        total_pixels = img_array.shape[0] * img_array.shape[1]
        healthy_pixels = np.sum(healthy_mask > 0)
        
        affected_percentage = ((total_pixels - healthy_pixels) / total_pixels) * 100
        
        return min(100.0, max(0.0, affected_percentage))


# Example usage
if __name__ == "__main__":
    detector = DiseaseDetectionModel('disease_model.keras')
    detector.load_model()
    
    print("Disease Detection Model Ready")
    print(f"Supported diseases: {len(detector.disease_labels)}")
