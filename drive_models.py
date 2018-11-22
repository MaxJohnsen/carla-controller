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
        self._one_hot_hlc = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def load_model(self, path):
        self._model = load_model(path)

    def get_prediction(self, images, info):
        if self._model is None:
            return False
        img_input = cv2.cvtColor(images["rgb_center"], cv2.COLOR_BGR2LAB)
        info_input = [
            info["speed"] / 100,
            info["speed_limit"] / 100,
            1 if info["traffic_light"] == 2 else 0,
        ]
        hlc_input = self._one_hot_hlc[int(info["hlc"])]
        print(hlc_input)
        prediction = self._model.predict(
            [np.array([img_input]), np.array([info_input]), np.array([hlc_input])]
        )
        prediction = prediction[0]
        steer = prediction[0]
        throttle = prediction[1]
        brake = prediction[2]
        return (steer, throttle, brake)
