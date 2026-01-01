from rest_framework import serializers

class CoffeePredictionInputSerializer(serializers.Serializer):
    """
    Validates user input data before passing to the ML model.
    Each field corresponds to features expected by the trained model.
    """
    age = serializers.IntegerField(min_value=18, max_value=80)
    gender = serializers.ChoiceField(choices=['Male', 'Female', 'Other'])
    country = serializers.CharField(max_length=100)
    coffee_intake = serializers.FloatField(min_value=0.0, max_value=10.0)
    caffeine_mg = serializers.FloatField(min_value=0.0, max_value=800.0)
    sleep_hours = serializers.FloatField(min_value=0.0, max_value=24.0)
    sleep_quality = serializers.ChoiceField(choices=['Poor', 'Fair', 'Good', 'Excellent'])
    bmi = serializers.FloatField(min_value=15.0, max_value=50.0)
    heart_rate = serializers.IntegerField(min_value=50, max_value=120)
    physical_activity_hours = serializers.FloatField(min_value=0.0, max_value=20.0)
    health_issues = serializers.ChoiceField(choices=['None', 'Mild', 'Severe'])
    occupation = serializers.ChoiceField(choices=['Student', 'Office', 'Service', 'Healthcare', 'Other'])
    smoking = serializers.IntegerField(min_value=0, max_value=1)
    alcohol_consumption = serializers.IntegerField(min_value=0, max_value=1)

class CoffeePredictionOutputSerializer(serializers.Serializer):
    """
    Structures the prediction response sent to the client.
    """
    predicted_stress_level = serializers.FloatField()
    stress_category = serializers.CharField()
    confidence_score = serializers.FloatField()
    recommendation = serializers.CharField()
