from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .serializers import CoffeePredictionInputSerializer, CoffeePredictionOutputSerializer
import joblib
import pandas as pd
import numpy as np
import os

@method_decorator(csrf_exempt, name='dispatch')
class PredictionView(APIView):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_logic')
        self.model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
        self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
    
    def post(self, request):
        input_serializer = CoffeePredictionInputSerializer(data=request.data)
        
        if not input_serializer.is_valid():
            return Response(
                {'error': 'Invalid input', 'details': input_serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user_data = input_serializer.validated_data
        
        try:
            features = self.prepare_features(user_data)
            prediction = self.model.predict(features)[0]
            
            tree_predictions = np.array([tree.predict(features)[0] for tree in self.model.estimators_])
            prediction_std = np.std(tree_predictions)
            confidence = max(0, min(100, 100 - (prediction_std * 10)))
            
            stress_category = self.get_stress_category(prediction)
            recommendation = self.generate_recommendation(prediction, user_data)
            
            output_data = {
                'predicted_stress_level': round(prediction, 2),
                'stress_category': stress_category,
                'confidence_score': round(confidence, 2),
                'recommendation': recommendation
            }
            
            output_serializer = CoffeePredictionOutputSerializer(data=output_data)
            output_serializer.is_valid(raise_exception=True)
            
            return Response(output_serializer.data, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response(
                {'error': 'Prediction failed', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def prepare_features(self, user_data):
        df = pd.DataFrame([{
            'Age': user_data['age'],
            'Gender': user_data['gender'],
            'Country': user_data['country'],
            'Coffee_Intake': user_data['coffee_intake'],
            'Caffeine_mg': user_data['caffeine_mg'],
            'Sleep_Hours': user_data['sleep_hours'],
            'Sleep_Quality': user_data['sleep_quality'],
            'BMI': user_data['bmi'],
            'Heart_Rate': user_data['heart_rate'],
            'Physical_Activity_Hours': user_data['physical_activity_hours'],
            'Health_Issues': user_data['health_issues'],
            'Occupation': user_data['occupation'],
            'Smoking': user_data['smoking'],
            'Alcohol_Consumption': user_data['alcohol_consumption']
        }])
        
        estimated_weight = df['BMI'] * (1.7 ** 2)
        df['caffeine_per_kg'] = df['Caffeine_mg'] / estimated_weight
        df['coffee_intensity'] = df['Coffee_Intake'] * df['Caffeine_mg']
        df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
        df['age_group'] = df['age_group'].fillna(2).astype(int)
        df['sleep_deficit'] = (8 - df['Sleep_Hours']).clip(lower=0)
        df['health_risk_score'] = df['Smoking'] + df['Alcohol_Consumption']
        df['activity_level'] = pd.cut(df['Physical_Activity_Hours'], bins=[-0.1, 3, 7, 12, 20], labels=[0, 1, 2, 3])
        df['activity_level'] = df['activity_level'].fillna(1).astype(int)
        
        df['Gender'] = self.label_encoders['Gender'].transform([df['Gender'].iloc[0]])[0]
        
        if df['Country'].iloc[0] in self.label_encoders['Country'].classes_:
            df['Country'] = self.label_encoders['Country'].transform([df['Country'].iloc[0]])[0]
        else:
            df['Country'] = 0
        
        df['Sleep_Quality'] = self.label_encoders['Sleep_Quality'].transform([df['Sleep_Quality'].iloc[0]])[0]
        
        health_value = df['Health_Issues'].iloc[0]
        if health_value == 'None':
            health_value = 'Unknown'
        
        if health_value in self.label_encoders['Health_Issues'].classes_:
            df['Health_Issues'] = self.label_encoders['Health_Issues'].transform([health_value])[0]
        else:
            df['Health_Issues'] = 0
        
        df['Occupation'] = self.label_encoders['Occupation'].transform([df['Occupation'].iloc[0]])[0]
        
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        
        # Keep as DataFrame instead of converting to numpy array
        features_scaled = self.scaler.transform(df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_columns)
        
        return features_scaled_df
    def get_stress_category(self, prediction):
        if prediction < 1.5:
            return 'Low'
        elif prediction < 2.5:
            return 'Medium'
        else:
            return 'High'
    
    def generate_recommendation(self, stress_level, user_data):
        if stress_level > 2.5:
            if user_data['coffee_intake'] > 4:
                return "High stress detected. Consider reducing coffee to 3 cups or less daily."
            elif user_data['sleep_hours'] < 6:
                return "High stress detected. Aim for 7-8 hours of sleep per night."
            else:
                return "High stress. Try meditation, exercise, or stress management techniques."
        elif stress_level > 1.5:
            return "Moderate stress. Monitor your habits and maintain balance."
        else:
            return "Low stress. Your current lifestyle supports good health."

def index_view(request):
    return render(request, 'index.html')
