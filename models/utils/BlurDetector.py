import cv2
from skimage import io


class ImageQualityChecker:

    @staticmethod
    def checkIfImageAcceptable(image):
        threshold = 50
        imageDecoded = io.imread(image)
        shape = imageDecoded.shape
        score = cv2.Laplacian(imageDecoded, cv2.CV_64F).var()
        return True if score > threshold and shape[0] > 300 and shape[1] > 300 else False
