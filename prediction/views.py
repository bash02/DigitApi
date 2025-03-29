# views.py
from PIL import Image
import torch
from rest_framework.viewsets import ModelViewSet
from rest_framework.parsers import MultiPartParser, FormParser
from torchvision import transforms
import string

from .models import DigitPrediction, AlphabetPrediction, OCRPrediction
from .serializers import DigitPredictionSerializer, AlphabetPredictionSerializer, OCRPredictionSerializer
from .digit_model.model import model as digit_model,  device as digit_device 
from .alphabet_model.model import model as alphabe_model, device as alphabet_device
from .ocr_model.ocr_utils import extract_text

# Preprocessing
digit_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

alphabet_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class DigitPredictionViewSet(ModelViewSet):
    queryset = DigitPrediction.objects.all()
    serializer_class = DigitPredictionSerializer
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        instance = serializer.save()

        # Open image and predict
        image_path = instance.image.path
        try:
            image = Image.open(image_path).convert("L")
        except Exception as e:
            instance.predicted_digit = -1  # or some error flag
            instance.save()
            return

        # Preprocess and predict
        input_tensor = digit_preprocess(image).unsqueeze(0).to(digit_device)
        with torch.no_grad():
            outputs = digit_model(input_tensor)
            _, predicted = torch.max(outputs, 1)

            instance.predicted_digit = predicted.item()
            instance.save()



class AlphabetPredictionViewSet(ModelViewSet):
    queryset = AlphabetPrediction.objects.all()
    serializer_class = AlphabetPredictionSerializer
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        instance = serializer.save()
        image = Image.open(instance.image.path).convert("L")
        input_tensor = alphabet_preprocess(image).unsqueeze(0).to(alphabet_device)

        with torch.no_grad():
            output = alphabe_model(input_tensor)
            _, predicted = torch.max(output, 1)
            letter = string.ascii_uppercase[predicted.item()]
            instance.predicted_letter = letter
            instance.save()


class OCRPredictionViewSet(ModelViewSet):
    queryset = OCRPrediction.objects.all()
    serializer_class = OCRPredictionSerializer
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        instance = serializer.save()

        # Extract text using TrOCR AI model
        try:
            text = extract_text(instance.image.path)
        except Exception as e:
            text = "Error during OCR: " + str(e)

        instance.predicted_text = text
        instance.save()