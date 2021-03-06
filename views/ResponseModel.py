from flask import json

from models.classifier.Classifier import PredictionClass


class Response:
    def __init__(self, user, prediction: PredictionClass, percentage, isAcceptable):
        self.userId = user
        self.predictionType = prediction
        self.percentageAccuracy = percentage
        self.isAcceptable = isAcceptable

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
