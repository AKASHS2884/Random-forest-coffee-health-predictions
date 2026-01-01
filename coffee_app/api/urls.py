from django.urls import path
from .views import PredictionView, index_view

urlpatterns = [
    path('', index_view, name='index'),
    path('api/predict/', PredictionView.as_view(), name='predict'),
]
