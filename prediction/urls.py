from django.urls import path
from django.urls.conf import include
from rest_framework_nested import routers
from . import views

router = routers.DefaultRouter()
router.register('predict-digit', views.DigitPredictionViewSet, basename='digit')
router.register('predict-alphabet', views.AlphabetPredictionViewSet, basename='alphabet')
router.register('predict-text', views.OCRPredictionViewSet, basename='text')
# URLConf
urlpatterns = router.urls