"""
TODO: Write docstring
"""
from enum import Enum


class GameState(Enum):
    """ TODO: Write Docstring """

    NOT_RECORDING = 0
    RECORDING = 1
    WRITING = 2


class HighLevelCommand(Enum):
    """ TODO: Write Docstring """

    FOLLOW_ROAD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STRAIGHT_AHEAD = 3


class TrafficLight(Enum):
    """ TODO: Write Docstring """

    GREEN = 0
    YELLOW = 1
    RED = 2
    NONE = 3

