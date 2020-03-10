from abc import ABC, abstractmethod
from enum import Enum


class Classifier(ABC):
    def __init__(self):
        super().__init__()
        self.model = self.initialize()

    @abstractmethod
    def initialize(self):
        pass

    # Should return prediction type and percentage prediction
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def getOutputType(self, prediction):
        pass

    @abstractmethod
    def convertToPercentageAndGetType(self, result):
        pass


class PredictionClass(str, Enum):
    MALIGNANT = 'malignant'
    BENIGN = 'benign'
    UNKNOWN = 'unknown'
