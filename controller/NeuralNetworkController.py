import io
import urllib.request

import numpy as np
from PIL import Image

from controller.Controller import Controller
from models.classifier.Classifier import Classifier
from models.factory.ClassifierFactory import ClassifierFactory
from views.ResponseModel import Response


class NeuralNetworkController(Controller):

    def convertImageToArray(self, imageUrl):
        with urllib.request.urlopen(imageUrl) as url:
            image = io.BytesIO(url.read())
        openedImage = np.asarray((Image.open(image).resize((224, 224), Image.ANTIALIAS).convert("RGB")))
        imageToArray = np.array([openedImage], dtype='uint8') / 255
        return imageToArray

    def getPrediction(self, image, userId, imageUrl) -> Response:
        predictionType, percentage = self.classifier.predict(image)
        isAcceptable = self.checkIfValid(imageUrl)
        return Response(userId, predictionType, percentage, isAcceptable)

    def getClassifier(self) -> Classifier:
        return ClassifierFactory.createClassifier("NEURAL_NETWORK")
