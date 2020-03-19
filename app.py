# app.py
import os

from flask import Flask, request  # import flask
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from controller.NeuralNetworkController import NeuralNetworkController

app = Flask(__name__)  # create an app instance


@app.route("/skin-cancer/get-prediction", methods=['GET', 'OPTIONS'])  # at the end point /
def getPrediction():
    imageUrl = request.args.get('image-url') + "&token=" + request.args.get('token')
    userId = request.args.get('user-id')
    image = nn.convertImageToArray(imageUrl)
    prediction = nn.getPrediction(image, userId,
                                  imageUrl)
    return prediction.toJSON()


if __name__ == "__main__":  # on running python app.py

    nn = NeuralNetworkController()
    port = int(os.environ.get("PORT", 8080))
    http_server = WSGIServer(('', port), app)
    CORS(app)
    http_server.serve_forever()
