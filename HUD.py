"""TODO: write module docstring """
from enum import Enum
import pygame


class VerticalAlign(Enum):
    """TODO: write docstring"""

    TOP = 0
    CENTER = 1
    BOTTOM = 2


class HUD:
    """TODO: write docstring"""

    def __init__(
        self,
        window_width,
        window_height,
        screen,
        vehicle_in_reverse,
        autopilot_enabled,
        game_state,
    ):
        # Styling of box
        self._background_color = (200, 200, 200, 200)
        self._text_color = (0, 0, 0)
        self._antialias = False
        self._font = pygame.font.SysFont("Tahoma", int(window_width * 0.014), bold=True)

        # Size of a box in the HUD
        self._rectangle_width = window_width * 0.33
        self._rectangle_height = window_height * 0.13

        # Starting position for the boxes
        self._left_rectangle_x = window_width * 0.03
        self._left_rectangle_y = window_height * 0.04

        self._right_rectangle_x = window_width * 0.385
        self._right_rectangle_y = window_height * 0.04

        # Screen to draw HUD on
        self._screen = screen

        # CarlaController states
        self._vehicle_in_reverse = vehicle_in_reverse
        self._autopilot_enabled = autopilot_enabled
        self._game_state = game_state

    def _render_HUD_text(self, label, value, rectangle_x, vertical_align):

        """
        Renders text onto the hood. Right aligns labels (left column) and
        right aligns values (right column)
        Args:
            label (str):                    Description of displayed value
            value (str):                    Value to display in the hood
            x_pos (int):                    Starting point of text for value
                                            display text in
            veritcal_align (VerticalAlign): Enum that is either TOP, CENTER or BOTTOM
                                            Used to decide which row to place the text
        """

        label_text = self._font.render(label, self._antialias, self._text_color)
        label_pos = label_text.get_rect()

        value_text = self._font.render(value, self._antialias, self._text_color)
        value_pos = value_text.get_rect()
        value_pos.x = rectangle_x + self._rectangle_width * 0.6

        line_gap = self._rectangle_height * 0.35
        if vertical_align == VerticalAlign.TOP:
            label_pos.topright = (rectangle_x + self._rectangle_width * 0.4, line_gap)
            value_pos.y = line_gap
        elif vertical_align == VerticalAlign.CENTER:
            label_pos.topright = (
                rectangle_x + self._rectangle_width * 0.4,
                line_gap * 2,
            )
            value_pos.y = line_gap * 2
        else:
            label_pos.topright = (
                rectangle_x + self._rectangle_width * 0.4,
                line_gap * 3,
            )
            value_pos.y = line_gap * 3

        self._screen.blit(label_text, label_pos)
        self._screen.blit(value_text, value_pos)

    def render_HUD(
        self,
        autopilot_status,
        reverse_status,
        speed_value,
        speed_limit_value,
        traffic_light_value,
    ):

        """
        TODO: write proper docstring
        Renders the hud of the simulator at the top of the window:
            - Left rectangle displays autopilot, reverse and game state)
            - Right rectangle displays speed, speed limit and traffic lights

        The size of the hud is responsive, but assumes approximately 100:75 relationship
        between width and height
        """
        # Draw left rectangle
        driving_state_surface = pygame.Surface(
            (self._rectangle_width, self._rectangle_height), pygame.SRCALPHA
        )
        driving_state_surface.fill(self._background_color)
        self._screen.blit(
            driving_state_surface, (self._left_rectangle_x, self._left_rectangle_y)
        )

        # Draw right rectangle
        game_state_surface = pygame.Surface(
            (self._rectangle_width, self._rectangle_height), pygame.SRCALPHA
        )
        game_state_surface.fill(self._background_color)
        self._screen.blit(
            game_state_surface, (self._right_rectangle_x, self._right_rectangle_y)
        )

        # Render autopilot state
        self._render_HUD_text(
            "Autopilot: ", autopilot_status, self._left_rectangle_x, VerticalAlign.TOP
        )
        # Render reverse state
        self._render_HUD_text(
            "Reverse: ", reverse_status, self._left_rectangle_x, VerticalAlign.CENTER
        )
        # Render game state
        self._render_HUD_text(
            "Game state: ",
            self._game_state.name,
            self._left_rectangle_x,
            VerticalAlign.BOTTOM,
        )
        # Render speed
        self._render_HUD_text(
            "Speed: ", speed_value, self._right_rectangle_x, VerticalAlign.TOP
        )

        # Render speed limit
        self._render_HUD_text(
            "Speed limit: ",
            speed_limit_value,
            self._right_rectangle_x,
            VerticalAlign.CENTER,
        )
        # Render traffic light
        self._render_HUD_text(
            "Traffic light: ",
            traffic_light_value,
            self._right_rectangle_x,
            VerticalAlign.BOTTOM,
        )
