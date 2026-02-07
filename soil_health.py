"""
Soil Health Analysis Module
Comprehensive soil health assessment and recommendations
"""

import numpy as np
import pandas as pd

class SoilHealthAnalyzer:
    """
    Analyzes soil health based on multiple parameters
    """
    
    def __init__(self):
        self.optimal_ranges = {
            'ph': {
                'very_acidic': (0, 5.5),
                'acidic': (5.5, 6.5),
                'neutral': (6.5, 7.5),
                'alkaline': (7.5, 8.5),
                'very_alkaline': (8.5, 14)
            },
            'nitrogen': {
                'low': (0, 50),
                'medium': (50, 100),
                'high': (100, 200)
            },
            'phosphorus': {
                'low': (0, 20),
                'medium': (20, 50),
                'high': (50, 100)
            },
            'potassium': {
                'low': (0, 30),
                'medium': (30, 70),
                'high': (70, 100)
            },
            'organic_matter': {
                'very_low': (0, 1),
                'low': (1, 2),
                'medium': (2, 4),
                'good': (4, 6),
                'excellent': (6, 100)
            },
            'electrical_conductivity': {
                'non_saline': (0, 2),
                'slightly_saline': (2, 4),
                'moderately_saline': (4, 8),
                'strongly_saline': (8, 16),
                'very_strongly_saline': (16, 100)
            }
        }
        
        self.crop_soil_preferences = {
            'Rice': {'ph': (5.5, 7.0), 'nitrogen': 'high', 'phosphorus': 'medium'},
            'Wheat': {'ph': (6.0, 7.5), 'nitrogen': 'high', 'phosphorus': 'medium'},
            'Cotton': {'ph': (6.0, 7.5), 'nitrogen': 'medium', 'phosphorus': 'medium'},
            'Sugarcane': {'ph': (6.0, 7.5), 'nitrogen': 'high', 'phosphorus': 'high'},
            'Maize': {'ph': (5.8, 7.0), 'nitrogen': 'high', 'phosphorus': 'medium'},
            'Pulses': {'ph': (6.0, 7.5), 'nitrogen': 'low', 'phosphorus': 'medium'},
            'Vegetables': {'ph': (6.0, 7.0), 'nitrogen': 'high', 'phosphorus': 'high'},
            'Fruits': {'ph': (5.5, 6.5), 'nitrogen': 'medium', 'phosphorus': 'medium'},
        }
    
    def classify_parameter(self, value, param_type):
        """
        Classify a parameter value into its range category
        
        Args:
            value: Parameter value
            param_type: Type of parameter (ph, nitrogen, etc.)
            
        Returns:
            Category name
        """
        ranges = self.optimal_ranges.get(param_type, {})
        
        for category, (min_val, max_val) in ranges.items():
            if min_val <= value < max_val:
                return category
        
        return 'unknown'
    
    def calculate_health_score(self, ph, nitrogen, phosphorus, potassium, 
                              organic_matter, electrical_conductivity):
        """
        Calculate overall soil health score (0-100)
        
        Args:
            ph: Soil pH value
            nitrogen: Nitrogen content (kg/ha)
            phosphorus: Phosphorus content (kg/ha)
            potassium: Potassium content (kg/ha)
            organic_matter: Organic matter percentage
            electrical_conductivity: EC in dS/m
            
        Returns:
            tuple: (score, grade, components_scores)
        """
        scores = {}
        
        # pH score (optimal range 6.5-7.5)
        if 6.5 <= ph <= 7.5:
            scores['ph'] = 100
        elif 6.0 <= ph < 6.5 or 7.5 < ph <= 8.0:
            scores['ph'] = 80
        elif 5.5 <= ph < 6.0 or 8.0 < ph <= 8.5:
            scores['ph'] = 60
        elif 5.0 <= ph < 5.5 or 8.5 < ph <= 9.0:
            scores['ph'] = 40
        else:
            scores['ph'] = 20
        
        # Nitrogen score
        if nitrogen >= 100:
            scores['nitrogen'] = 100
        elif nitrogen >= 75:
            scores['nitrogen'] = 85
        elif nitrogen >= 50:
            scores['nitrogen'] = 70
        elif nitrogen >= 30:
            scores['nitrogen'] = 50
        else:
            scores['nitrogen'] = 30
        
        # Phosphorus score
        if phosphorus >= 50:
            scores['phosphorus'] = 100
        elif phosphorus >= 35:
            scores['phosphorus'] = 85
        elif phosphorus >= 20:
            scores['phosphorus'] = 70
        elif phosphorus >= 10:
            scores['phosphorus'] = 50
        else:
            scores['phosphorus'] = 30
        
        # Potassium score
        if potassium >= 70:
            scores['potassium'] = 100
        elif potassium >= 50:
            scores['potassium'] = 85
        elif potassium >= 30:
            scores['potassium'] = 70
        elif potassium >= 15:
            scores['potassium'] = 50
        else:
            scores['potassium'] = 30
        
        # Organic matter score
        if organic_matter >= 4:
            scores['organic_matter'] = 100
        elif organic_matter >= 3:
            scores['organic_matter'] = 85
        elif organic_matter >= 2:
            scores['organic_matter'] = 70
        elif organic_matter >= 1:
            scores['organic_matter'] = 50
        else:
            scores['organic_matter'] = 30
        
        # EC score (lower is better for most crops)
        if electrical_conductivity <= 2:
            scores['ec'] = 100
        elif electrical_conductivity <= 4:
            scores['ec'] = 75
        elif electrical_conductivity <= 8:
            scores['ec'] = 50
        elif electrical_conductivity <= 16:
            scores['ec'] = 25
        else:
            scores['ec'] = 10
        
        # Calculate weighted average
        weights = {
            'ph': 0.20,
            'nitrogen': 0.20,
            'phosphorus': 0.15,
            'potassium': 0.15,
            'organic_matter': 0.20,
            'ec': 0.10
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        # Determine grade
        if total_score >= 90:
            grade = 'Excellent'
        elif total_score >= 75:
            grade = 'Good'
        elif total_score >= 60:
            grade = 'Fair'
        elif total_score >= 40:
            grade = 'Poor'
        else:
            grade = 'Very Poor'
        
        return total_score, grade, scores
    
    def generate_recommendations(self, ph, nitrogen, phosphorus, potassium,
                                organic_matter, electrical_conductivity, crop=None):
        """
        Generate soil improvement recommendations
        
        Returns:
            dict: Recommendations for different aspects
        """
        recommendations = {
            'immediate_actions': [],
            'nutrient_management': [],
            'soil_amendments': [],
            'long_term_practices': [],
            'warnings': []
        }
        
        # pH recommendations
        if ph < 5.5:
            recommendations['immediate_actions'].append(
                "🔴 CRITICAL: Apply lime (CaCO3) to raise pH. For every 0.5 unit increase needed, apply 500-700 kg/ha of lime."
            )
            recommendations['soil_amendments'].append(
                "Add dolomitic limestone for both pH correction and magnesium supply"
            )
        elif ph < 6.0:
            recommendations['immediate_actions'].append(
                "⚠️ Apply agricultural lime: 300-500 kg/ha to optimize pH"
            )
        elif ph > 8.5:
            recommendations['immediate_actions'].append(
                "🔴 CRITICAL: Apply sulfur or gypsum to lower pH. Use 100-150 kg/ha of elemental sulfur."
            )
            recommendations['soil_amendments'].append(
                "Incorporate acidifying organic materials like pine needles or peat moss"
            )
        elif ph > 8.0:
            recommendations['immediate_actions'].append(
                "⚠️ Apply gypsum (CaSO4): 200-300 kg/ha to gradually reduce pH"
            )
        
        # Nitrogen recommendations
        n_status = self.classify_parameter(nitrogen, 'nitrogen')
        if n_status == 'low':
            recommendations['nutrient_management'].append(
                "Add nitrogen fertilizers: Urea (100-120 kg/ha) or Ammonium Sulfate (120-150 kg/ha)"
            )
            recommendations['long_term_practices'].append(
                "Incorporate leguminous crops in rotation to improve nitrogen naturally"
            )
        elif n_status == 'high':
            recommendations['nutrient_management'].append(
                "Nitrogen levels are adequate. Avoid excessive application to prevent leaching"
            )
        
        # Phosphorus recommendations
        p_status = self.classify_parameter(phosphorus, 'phosphorus')
        if p_status == 'low':
            recommendations['nutrient_management'].append(
                "Apply phosphorus: Single Super Phosphate (150-200 kg/ha) or DAP (75-100 kg/ha)"
            )
            recommendations['soil_amendments'].append(
                "Rock phosphate for long-term P supply: 300-400 kg/ha"
            )
        
        # Potassium recommendations
        k_status = self.classify_parameter(potassium, 'potassium')
        if k_status == 'low':
            recommendations['nutrient_management'].append(
                "Apply potassium: Muriate of Potash (MOP) 50-75 kg/ha or Sulfate of Potash (SOP) 60-80 kg/ha"
            )
        
        # Organic matter recommendations
        om_status = self.classify_parameter(organic_matter, 'organic_matter')
        if om_status in ['very_low', 'low']:
            recommendations['immediate_actions'].append(
                "🌱 Add organic matter: 5-10 tonnes/ha of well-decomposed FYM or compost"
            )
            recommendations['long_term_practices'].append(
                "Practice green manuring with crops like Dhaincha, Sunhemp, or Cowpea"
            )
            recommendations['long_term_practices'].append(
                "Incorporate crop residues instead of burning them"
            )
            recommendations['soil_amendments'].append(
                "Apply vermicompost: 2-3 tonnes/ha for nutrient-rich organic matter"
            )
        
        # Electrical conductivity (salinity) recommendations
        ec_status = self.classify_parameter(electrical_conductivity, 'electrical_conductivity')
        if ec_status in ['moderately_saline', 'strongly_saline', 'very_strongly_saline']:
            recommendations['immediate_actions'].append(
                "🔴 CRITICAL: High salinity detected. Install drainage system and practice leaching irrigation"
            )
            recommendations['soil_amendments'].append(
                "Apply gypsum: 2-5 tonnes/ha to displace sodium and improve soil structure"
            )
            recommendations['warnings'].append(
                "Avoid salt-sensitive crops. Consider salt-tolerant varieties"
            )
            recommendations['long_term_practices'].append(
                "Practice raised bed cultivation to improve drainage"
            )
        elif ec_status == 'slightly_saline':
            recommendations['warnings'].append(
                "Monitor salinity levels. Ensure adequate drainage"
            )
        
        # Crop-specific recommendations
        if crop and crop in self.crop_soil_preferences:
            prefs = self.crop_soil_preferences[crop]
            ph_range = prefs['ph']
            
            if not (ph_range[0] <= ph <= ph_range[1]):
                recommendations['warnings'].append(
                    f"⚠️ Soil pH ({ph:.1f}) is not optimal for {crop}. Optimal range: {ph_range[0]}-{ph_range[1]}"
                )
        
        # General best practices
        recommendations['long_term_practices'].extend([
            "Practice crop rotation to maintain soil fertility",
            "Use mulching to conserve moisture and add organic matter",
            "Conduct soil testing every 2-3 years to monitor changes"
        ])
        
        return recommendations
    
    def get_fertilizer_schedule(self, crop, area_hectares):
        """
        Generate fertilizer application schedule
        
        Args:
            crop: Crop type
            area_hectares: Area in hectares
            
        Returns:
            DataFrame with fertilizer schedule
        """
        # Fertilizer requirements per hectare (NPK in kg)
        crop_requirements = {
            'Rice': {'N': 120, 'P': 60, 'K': 40},
            'Wheat': {'N': 120, 'P': 60, 'K': 40},
            'Cotton': {'N': 120, 'P': 60, 'K': 60},
            'Sugarcane': {'N': 200, 'P': 80, 'K': 80},
            'Maize': {'N': 120, 'P': 60, 'K': 40},
            'Pulses': {'N': 25, 'P': 60, 'K': 30},
            'Vegetables': {'N': 150, 'P': 75, 'K': 75},
            'Fruits': {'N': 100, 'P': 50, 'K': 100},
        }
        
        if crop not in crop_requirements:
            crop = 'Rice'  # Default
        
        req = crop_requirements[crop]
        
        # Calculate total requirements
        total_n = req['N'] * area_hectares
        total_p = req['P'] * area_hectares
        total_k = req['K'] * area_hectares
        
        # Create schedule (split application)
        schedule = []
        
        # Basal dose (at sowing/planting)
        schedule.append({
            'Stage': 'Basal (At Sowing)',
            'N (kg)': round(total_n * 0.3, 1),
            'P (kg)': round(total_p * 1.0, 1),  # All P at basal
            'K (kg)': round(total_k * 0.5, 1),
            'Timing': 'Day 0',
            'Method': 'Broadcast and incorporate'
        })
        
        # First top dressing
        schedule.append({
            'Stage': '1st Top Dressing',
            'N (kg)': round(total_n * 0.35, 1),
            'P (kg)': 0,
            'K (kg)': round(total_k * 0.25, 1),
            'Timing': '20-30 days after sowing',
            'Method': 'Side placement or fertigation'
        })
        
        # Second top dressing
        schedule.append({
            'Stage': '2nd Top Dressing',
            'N (kg)': round(total_n * 0.35, 1),
            'P (kg)': 0,
            'K (kg)': round(total_k * 0.25, 1),
            'Timing': '45-60 days after sowing',
            'Method': 'Side placement or fertigation'
        })
        
        return pd.DataFrame(schedule)
    
    def assess_micronutrients(self, zinc, iron, manganese, copper, boron):
        """
        Assess micronutrient status
        
        Returns:
            dict: Micronutrient status and recommendations
        """
        status = {}
        recommendations = []
        
        # Zinc (optimal range: 0.6-1.0 ppm)
        if zinc < 0.6:
            status['Zinc'] = 'Deficient'
            recommendations.append("Apply Zinc Sulfate: 25 kg/ha or spray 0.5% ZnSO4 solution")
        else:
            status['Zinc'] = 'Adequate'
        
        # Iron (optimal range: 4.5-10 ppm)
        if iron < 4.5:
            status['Iron'] = 'Deficient'
            recommendations.append("Apply Ferrous Sulfate: 20-25 kg/ha or foliar spray of 0.5% FeSO4")
        else:
            status['Iron'] = 'Adequate'
        
        # Manganese (optimal range: 1.0-5.0 ppm)
        if manganese < 1.0:
            status['Manganese'] = 'Deficient'
            recommendations.append("Apply Manganese Sulfate: 20 kg/ha")
        else:
            status['Manganese'] = 'Adequate'
        
        # Copper (optimal range: 0.2-0.5 ppm)
        if copper < 0.2:
            status['Copper'] = 'Deficient'
            recommendations.append("Apply Copper Sulfate: 10-15 kg/ha")
        else:
            status['Copper'] = 'Adequate'
        
        # Boron (optimal range: 0.5-1.0 ppm)
        if boron < 0.5:
            status['Boron'] = 'Deficient'
            recommendations.append("Apply Borax: 10 kg/ha or spray 0.2% boric acid solution")
        else:
            status['Boron'] = 'Adequate'
        
        return {
            'status': status,
            'recommendations': recommendations
        }


# Example usage
if __name__ == "__main__":
    analyzer = SoilHealthAnalyzer()
    
    # Example soil data
    score, grade, components = analyzer.calculate_health_score(
        ph=6.5,
        nitrogen=85,
        phosphorus=35,
        potassium=50,
        organic_matter=3.2,
        electrical_conductivity=1.5
    )
    
    print(f"Soil Health Score: {score:.1f}/100")
    print(f"Grade: {grade}")
    print(f"Component Scores: {components}")
