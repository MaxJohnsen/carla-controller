"""
TODO: Write docstring
"""
from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model
import cv2
import numpy as np


class ModelInterface(ABC):
    """
    TODO: Write docstring
    """

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def get_prediction(self, images, info):
        pass


class DriveModelKeras(ModelInterface):
    """
    TODO: Write docstring
    """

    def __init__(self):
        self._model = None

    def load_model(self, path):
        self._model = load_model(path)

    def get_prediction(self, images, info):
        if self._model is None:
            return False
        img = cv2.cvtColor(images["rgb_center"], cv2.COLOR_BGR2LAB)
        steer = self._model.predict(np.array([img]))
        throttle = None
        brake = None
        return (steer, throttle, brake)
