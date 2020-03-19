from abc import ABC, abstractmethod

from models.utils.BlurDetector import BlurChecker
from models.ResponseModel import Response
from models.classifier.Classifier import Classifier


class Controller(ABC):
    def __init__(self):
        self.classifier = self.getClassifier()

    @abstractmethod
    def getClassifier(self) -> Classifier:
        pass

    @abstractmethod
    def getPrediction(self, image, userId, imageUrl) -> Response:
        pass

    @abstractmethod
    def convertImageToArray(self, image):
        pass

    @staticmethod
    def checkBlurLevel(imageUrl):
        return BlurChecker.variance_of_laplacian(imageUrl)
