from __future__ import print_function

import argparse
import logging
import random
import numpy as np
import pygame
import os
import time
import pandas as pd

from timer import Timer
from pathlib import Path

from enum import Enum

from pygame.locals import *

from carla.client import make_carla_client, VehicleControl
from carla import sensor
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla import image_converter as ic

from matplotlib import pyplot as plt

from disk_writer import DiskWriter

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

OUTPUT_IMGAGE_WIDTH = 300
OUTPUT_IMAGE_HEIGHT = 180


class GameState(Enum):
    NOT_RECORDING = 0
    RECORDING = 1
    WRITING = 2


class VerticalAlign(Enum):
    TOP = 0
    CENTER = 1
    BOTTOM = 2


class CarlaController:
    def __init__(self, carla_client, args):
        self.client = carla_client

        # Stores the latest recieved game image
        self._game_image = None
        # Stores the latest received measurement
        self._measurements = None

        self._image_history = None
        self._driving_history = None

        self._pygame_display = None
        self._carla_settings = None

        self._weather_id = 1
        self._number_of_vehicles = 50
        self._number_of_pedastrians = 30
        self._quality_level = "Epic"

        self._timer = Timer()

        self._output_path = args.output_path

        self._game_state = GameState.NOT_RECORDING

        self._new_episode_flag = False
        self._vehicle_in_reverse = False
        self._autopilot_enabled = False
        self._joystick_enabled = args.joystick
        self._joystick = None

        self._disk_writer_thread = None

    def _initialize_pygame(self):
        self._pygame_display = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        if self._joystick_enabled:
            pygame.joystick.init()
            self._joystick = pygame.joystick.Joystick(0)
            print(pygame.joystick.get_count())
            self._joystick.init()
            logging.info("Use steering wheel to control vehicle")
        else:
            logging.info("Use keyboard to control vehicle")
        self._on_new_episode()
        pygame.font.init()

        logging.debug("pygame initialized")

    def _initialize_carla(self):
        # Initialize settings
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            NumberOfVehicles=self._number_of_vehicles,
            NumberOfPedestrians=self._number_of_pedastrians,
            WeatherId=self._weather_id,
            QualityLevel=self._quality_level,
            SendNonPlayerAgentsInfo=True,
        )
        settings.randomize_seeds()

        # Add a game camera
        game_camera = sensor.Camera("GameCamera")
        game_camera.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        game_camera.set_position(2.0, 0.0, 1.4)
        game_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(game_camera)

        # Add RGB center camera
        rgb_camera_center = sensor.Camera("RGBCameraCenter")
        rgb_camera_center.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        rgb_camera_center.set_position(2.0, 0.0, 1.4)
        rgb_camera_center.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_center)

        # Add RGB left camera
        rgb_camera_left = sensor.Camera("RGBCameraLeft")
        rgb_camera_left.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        rgb_camera_left.set_position(2.0, -1, 1.4)
        rgb_camera_left.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_left)

        # Add RGB right camera
        rgb_camera_right = sensor.Camera("RGBCameraRight")
        rgb_camera_right.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        rgb_camera_right.set_position(2.0, 1, 1.4)
        rgb_camera_right.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_right)

        # Add depth camera
        depth_camera = sensor.Camera("DepthCamera", PostProcessing="Depth")
        depth_camera.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        depth_camera.set_position(2.0, 0.0, 1.4)
        depth_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(depth_camera)

        # Add semantic segmentation camera
        sem_seg_camera = sensor.Camera(
            "SemSegCamera", PostProcessing="SemanticSegmentation"
        )
        sem_seg_camera.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        sem_seg_camera.set_position(2.0, 0.0, 1.4)
        sem_seg_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(sem_seg_camera)

        self._carla_settings = settings

        logging.debug("Carla initialized")

    def _initialize_history(self):
        self._driving_history = pd.DataFrame(
            columns=["Location", "ForwardSpeed", "PlayerControl", "AutopilotControls"]
        )
        self._image_history = []

    def _on_new_episode(self):
        scene = self.client.load_settings(self._carla_settings)
        number_of_start_positions = len(scene.player_start_spots)
        start_postition = np.random.randint(number_of_start_positions)
        self.client.start_episode(start_postition)
        self._timer.new_episode()
        self._new_episode_flag = False
        self._disk_writer_thread = None
        self._image_history = []
        self._initialize_history()

    def _get_keyboard_control(self, keys):
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        control.reverse = self._vehicle_in_reverse

        return control

    def _get_joystick_control(self, keys):
        control = VehicleControl()
        control.steer = self._joystick.get_axis(0)
        control.throttle = max(self._joystick.get_axis(1) * -1, 0)
        control.brake = max(self._joystick.get_axis(1), 0)

        return control

    def _handle_keydown_event(self, key):
        if self._game_state is not GameState.WRITING:
            if key == K_p:
                self._autopilot_enabled = not self._autopilot_enabled
            elif key == K_q:
                self._vehicle_in_reverse = not self._vehicle_in_reverse
            elif key == K_e:
                if self._game_state == GameState.RECORDING:
                    self._game_state = GameState.WRITING
                    self._write_history_to_disk()
                self._new_episode_flag = True
            elif key == K_r:
                if (
                    self._game_state == GameState.NOT_RECORDING
                    and self._output_path is not None
                ):
                    self._game_state = GameState.RECORDING
                elif self._game_state == GameState.RECORDING:
                    self._game_state = GameState.WRITING
                    self._write_history_to_disk()

    def _render_HUD_text(
        self,
        screen,
        width,
        height,
        x,
        font,
        label,
        value,
        vertical_align,
        text_color=(0, 0, 0),
        antialias=False,
    ):

        label_text = font.render(label, antialias, text_color)
        label_pos = label_text.get_rect()

        value_text = font.render(value, antialias, text_color)
        value_pos = value_text.get_rect()
        value_pos.x = x + width * 0.6

        line_gap = height * 0.35
        if vertical_align == VerticalAlign.TOP:
            label_pos.topright = (x + width * 0.4, line_gap)
            value_pos.y = line_gap
        elif vertical_align == VerticalAlign.CENTER:
            label_pos.topright = (x + width * 0.4, line_gap * 2)
            value_pos.y = line_gap * 2
        else:
            label_pos.topright = (x + width * 0.4, line_gap * 3)
            value_pos.y = line_gap * 3

        screen.blit(label_text, label_pos)
        screen.blit(value_text, value_pos)

    def _render_HUD(self):

        """ 
        TODO: write proper docstring 
        TODO: get speed limit data
        TODO: get traffic ligfht data
        
        Renders the hud of the simulator at the top of the window:
            - Left rectangle displays autopilot, reverse and game state)
            - Right rectangle displays speed, speed limit and traffic lights

        The size of the hud is responsive, but assumes approximatelyu 100:75 relationship 
        between width and height
        """
        font = pygame.font.SysFont("Tahoma", int(WINDOW_WIDTH * 0.014), bold=True)

        # Size parameters
        rectangle_width = WINDOW_WIDTH * 0.33
        rectangle_height = WINDOW_HEIGHT * 0.13

        rectangle_y = WINDOW_HEIGHT * 0.04
        left_rectangle_x = WINDOW_WIDTH * 0.03
        right_rectangle_x = WINDOW_WIDTH * 0.385

        # Draw left rectangle
        driving_state_surface = pygame.Surface(
            (rectangle_width, rectangle_height), pygame.SRCALPHA
        )
        driving_state_surface.fill((200, 200, 200, 200))
        self._pygame_display.blit(
            driving_state_surface, (left_rectangle_x, rectangle_y)
        )

        # Draw right rectangle
        game_state_surface = pygame.Surface(
            (rectangle_width, rectangle_height), pygame.SRCALPHA
        )
        game_state_surface.fill((200, 200, 200, 200))
        self._pygame_display.blit(game_state_surface, (right_rectangle_x, rectangle_y))

        # Render autopilot state
        autopilot_status = "Enabled" if self._autopilot_enabled else "Disabled"
        self._render_HUD_text(
            self._pygame_display,
            rectangle_width,
            rectangle_height,
            left_rectangle_x,
            font,
            "Autopilot: ",
            autopilot_status,
            VerticalAlign.TOP,
        )
        # Render reverse state
        reverse_status = "Enabled" if self._vehicle_in_reverse else "Disabled"
        self._render_HUD_text(
            self._pygame_display,
            rectangle_width,
            rectangle_height,
            left_rectangle_x,
            font,
            "Reverse: ",
            reverse_status,
            VerticalAlign.CENTER,
        )

        # Render game state
        self._render_HUD_text(
            self._pygame_display,
            rectangle_width,
            rectangle_height,
            left_rectangle_x,
            font,
            "Game state: ",
            self._game_state.name,
            VerticalAlign.BOTTOM,
        )

        # Render speed
        speed_value = "{:.0f} km/h".format(
            self._measurements.player_measurements.forward_speed * 3.6
        )
        self._render_HUD_text(
            self._pygame_display,
            rectangle_width,
            rectangle_height,
            right_rectangle_x,
            font,
            "Speed: ",
            speed_value,
            VerticalAlign.TOP,
        )

        # Render speed limit
        speed_limit_value = "TODO"
        self._render_HUD_text(
            self._pygame_display,
            rectangle_width,
            rectangle_height,
            right_rectangle_x,
            font,
            "Speed limit: ",
            speed_limit_value,
            VerticalAlign.CENTER,
        )
        # Render traffic light
        traffic_light_value = "TODO"
        self._render_HUD_text(
            self._pygame_display,
            rectangle_width,
            rectangle_height,
            right_rectangle_x,
            font,
            "Traffic light: ",
            traffic_light_value,
            VerticalAlign.BOTTOM,
        )

    def _render_pygame(self):
        if self._game_image is not None:
            array = ic.to_rgb_array(self._game_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            if self._game_state == GameState.WRITING:
                self._render_progressbar(
                    surface, 300, 40, self._disk_writer_thread.progress
                )

            self._pygame_display.blit(surface, (0, 0))

        self._render_HUD()

        pygame.display.flip()

    def _render_progressbar(self, surface, width, height, progress):
        left = (WINDOW_WIDTH / 2) - (width / 2)
        top = (WINDOW_HEIGHT / 2) - (height / 2)
        pygame.draw.rect(
            surface, (255, 255, 255), pygame.Rect(left, top, width * progress, height)
        )
        pygame.draw.rect(
            surface, (128, 128, 128), pygame.Rect(left, top, width, height), 1
        )
        self._pygame_display.blit(surface, (0, 0))

    def _save_to_history(self, sensor_data, measurements, control):
        frame = self._timer.episode_frame

        self._image_history.append(
            (
                f"{frame}_rgb_center.png",
                ic.to_bgra_array(sensor_data.get("RGBCameraCenter", None)),
            )
        )
        self._image_history.append(
            (
                f"{frame}_rgb_left.png",
                ic.to_bgra_array(sensor_data.get("RGBCameraLeft", None)),
            )
        )
        self._image_history.append(
            (
                f"{frame}_rgb_right.png",
                ic.to_bgra_array(sensor_data.get("RGBCameraRight", None)),
            )
        )
        self._image_history.append(
            (
                f"{frame}_depth.png",
                ic.depth_to_logarithmic_grayscale(sensor_data.get("DepthCamera", None)),
            )
        )
        self._image_history.append(
            (
                f"{frame}_sem_seg.png",
                ic.labels_to_cityscapes_palette(sensor_data.get("SemSegCamera", None)),
            )
        )

        loc = measurements.player_measurements.transform.location
        speed = measurements.player_measurements.forward_speed
        autopilot = measurements.player_measurements.autopilot_control
        self._driving_history = self._driving_history.append(
            pd.Series(
                [
                    (loc.x, loc.y),
                    speed,
                    (control.steer, control.throttle, control.brake, control.reverse),
                    (
                        autopilot.steer,
                        autopilot.throttle,
                        autopilot.brake,
                        autopilot.reverse,
                    ),
                ],
                index=self._driving_history.columns,
            ),
            ignore_index=True,
        )

    def _write_history_to_disk(self):
        path = Path(f"{self._output_path}/{self._timer.episode_timestamp_str}")
        self._disk_writer_thread = DiskWriter(
            path,
            self._image_history,
            self._driving_history,
            on_complete=self._write_complete,
        )
        self._disk_writer_thread.start()

    def _write_complete(self):
        self._game_state = GameState.NOT_RECORDING
        self._initialize_history()

    def _on_loop(self):

        if self._game_state is not GameState.WRITING:
            self._timer.tick()

            measurements, sensor_data = self.client.read_data()
            self._measurements = measurements

            self._game_image = sensor_data.get("GameCamera", None)

            if self._joystick_enabled:
                control = self._get_joystick_control(pygame.key.get_pressed())
            else:
                control = self._get_keyboard_control(pygame.key.get_pressed())

            if self._new_episode_flag:
                self._on_new_episode()
            elif self._autopilot_enabled:
                self.client.send_control(
                    measurements.player_measurements.autopilot_control
                )
            else:
                self.client.send_control(control)

            if self._game_state == GameState.RECORDING:
                self._save_to_history(sensor_data, measurements, control)

        self._render_pygame()

    def execute(self):
        pygame.init()

        self._initialize_carla()
        self._initialize_pygame()
        if self._output_path is not None:
            logging.info(f"Recorded data will be saved to: {self._output_path}")
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == KEYDOWN:
                        self._handle_keydown_event(event.key)

                self._on_loop()
        finally:
            pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="localhost",
        help="IP of the host server (default: localhost)",
    )
    argparser.add_argument(
        "-j",
        "--joystick",
        action="store_true",
        help="control vehicle with an external steering wheel",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        dest="output_path",
        default=None,
        help="Recorded data will be saved to this path",
    )
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("Listening to server %s:%s", args.host, args.port)

    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaController(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")

