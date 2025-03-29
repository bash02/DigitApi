# serializers.py
from rest_framework import serializers
from .models import DigitPrediction, AlphabetPrediction, OCRPrediction

class DigitPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DigitPrediction
        fields = ['id', 'image', 'predicted_digit', 'uploaded_at']
        read_only_fields = ['predicted_digit', 'uploaded_at']


class AlphabetPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlphabetPrediction
        fields = ['id', 'image', 'predicted_letter', 'uploaded_at']
        read_only_fields = ['predicted_letter', 'uploaded_at']


class OCRPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = OCRPrediction
        fields = ['id', 'image', 'predicted_text', 'uploaded_at']
        read_only_fields = ['predicted_text', 'uploaded_at']