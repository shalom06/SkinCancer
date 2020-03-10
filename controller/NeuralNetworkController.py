import io
import urllib.request

import numpy as np
from PIL import Image

from controller.Controller import Controller
from models.ResponseModel import Response
from models.classifier.Classifier import Classifier
from models.classifier.NeuralNetwork import NeuralNetwork


class NeuralNetworkController(Controller):

    def convertImageToArray(self, imageUrl):
        with urllib.request.urlopen(imageUrl) as url:
            image = io.BytesIO(url.read())
        openedImage = np.asarray((Image.open(image).resize((224, 224), Image.ANTIALIAS).convert("RGB")))
        imageToArray = np.array([openedImage], dtype='uint8') / 255
        return imageToArray

    def getPrediction(self, image, userId, imageUrl) -> Response:
        predictionType, percentage = self.classifier.predict(image)
        isBlurry = self.checkBlurLevel(imageUrl)
        return Response(userId, predictionType, percentage, isBlurry)

    def getClassifier(self) -> Classifier:
        return NeuralNetwork()
