from models.classifier.Classifier import Classifier
from models.classifier.NeuralNetwork import NeuralNetwork


class ClassifierFactory:
    @staticmethod
    def getClassifier(classifierType) -> Classifier:
        if classifierType == "NEURAL_NETWORK":
            return NeuralNetwork()
