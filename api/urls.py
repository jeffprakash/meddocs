from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.BreastCancerPredictionAPIView.as_view(), name='breast-cancer-prediction'),
]
