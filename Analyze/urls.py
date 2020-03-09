from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.analyze_data, name='analyze'),
]