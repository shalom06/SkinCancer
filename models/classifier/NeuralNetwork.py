import numpy as np
from tensorflow.keras.models import load_model

from models.classifier.Classifier import Classifier, PredictionClass


class NeuralNetwork(Classifier):
    def convertToPercentageAndGetType(self, result):
        predictionType = NeuralNetwork.getType(result)
        percentage = NeuralNetwork.getPercentage(result, predictionType)
        return predictionType, percentage

    def getOutputType(self, prediction):
        maximum = np.max(prediction[0])
        if np.where(prediction[0] == maximum) == 0:
            return PredictionClass.BENIGN
        else:
            return PredictionClass.MALIGNANT

    def predict(self, image):
        result = self.model.predict(image)
        return self.convertToPercentageAndGetType(result[0][0])

    def initialize(self):
        return load_model('final.h5')
        # return load_model('/Users/shalommathews/PycharmProjects/SkinCancer/final.h5')

    @staticmethod
    def convertToPercentageBenign(result):
        return '{0:.2f}'.format(((result - 0.5) / (0.0 - 0.5)) * 100)

    @staticmethod
    def convertToPercentage(result):
        return '{0:.2f}'.format(((result - 0.5) / (1.0 - 0.5)) * 100)

    @staticmethod
    def getPercentage(result, predictionType: PredictionClass):
        if predictionType == PredictionClass.UNKNOWN:
            return 0
        elif predictionType == PredictionClass.BENIGN:
            return NeuralNetwork.convertToPercentageBenign(result)
        elif predictionType == PredictionClass.MALIGNANT:
            return NeuralNetwork.convertToPercentage(result)

    @staticmethod
    def getType(result) -> PredictionClass:
        if result < 0.5:
            return PredictionClass.BENIGN
        elif 0.5 < result <= 1.0:
            return PredictionClass.MALIGNANT
        else:
            return PredictionClass.UNKNOWN
