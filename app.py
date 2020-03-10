# app.py
import os

from flask import Flask, request  # import flask
from gevent.pywsgi import WSGIServer

from controller.NeuralNetworkController import NeuralNetworkController

app = Flask(__name__)  # create an app instance


@app.route("/skin-cancer/get-prediction", methods=['POST'])  # at the end point /
def getPrediction():
    imageUrl = request.form.get('image-url')
    userId = request.form.get('user-id')
    image = nn.convertImageToArray(imageUrl)
    prediction = nn.getPrediction(image, userId, imageUrl)
    return prediction.toJSON()


if __name__ == "__main__":  # on running python app.py
    nn = NeuralNetworkController()
    port = int(os.environ.get("PORT", 8080))
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()
