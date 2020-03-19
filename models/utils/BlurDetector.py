import cv2
from skimage import io


class BlurChecker:

    @staticmethod
    def variance_of_laplacian(image):
        threshold = 50
        imageDecoded = io.imread(image)
        score = cv2.Laplacian(imageDecoded, cv2.CV_64F).var()
        return False if score > threshold else True
