# models.py
from django.db import models

class DigitPrediction(models.Model):
    image = models.ImageField(upload_to='digit_images/')
    predicted_digit = models.IntegerField(null=True, blank=True)  # will be filled after prediction
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: {self.predicted_digit}"


class AlphabetPrediction(models.Model):
    image = models.ImageField(upload_to='alphabets/')
    predicted_letter = models.CharField(max_length=1, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)



class OCRPrediction(models.Model):
    image = models.ImageField(upload_to='ocr_images/')
    predicted_text = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"OCR Result ({self.id}) - {self.predicted_text[:30]}"