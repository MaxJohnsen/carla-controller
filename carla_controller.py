from __future__ import print_function

import argparse
import logging
import random
import numpy as np
import pygame
import os
import time

from timer import Timer
from pathlib import Path

from pygame.locals import *

from carla.client import make_carla_client, VehicleControl
from carla import sensor
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla import image_converter as ic

from matplotlib import pyplot as plt

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

OUTPUT_IMGAGE_WIDTH = 300
OUTPUT_IMAGE_HEIGHT = 180

class CarlaController:
    def __init__(self, carla_client, args):
        self.client = carla_client

        #Stores the latest recieved images
        self._game_image = None

        self._rgb_image_center = None
        self._rgb_image_left = None
        self._rgb_image_right = None
        self._depth_image = None
        self._sem_seg_image = None

        self._pygame_display = None
        self._carla_settings = None

        self._weather_id = 1
        self._number_of_vehicles = 50
        self._number_of_pedastrians = 30
        self._quality_level = 'Epic'

        self._timer = Timer()

        self._output_path = args.output_path
        self._episode_path_created = False
        self._recording_enabled = False

        self._new_episode_flag = False
        self._vehicle_in_reverse = False
        self._autopilot_enabled = False

    def _initialize_pygame(self):
        self._pygame_display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT),pygame.HWSURFACE | pygame.DOUBLEBUF)
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
            QualityLevel = self._quality_level
        )
        settings.randomize_seeds()
        
        # Add a game camera
        game_camera = sensor.Camera('GameCamera')
        game_camera.set_image_size(WINDOW_WIDTH,WINDOW_HEIGHT)
        game_camera.set_position(2.0, 0.0, 1.4)
        game_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(game_camera)

        # Add RGB center camera
        rgb_camera_center = sensor.Camera('RGBCameraCenter')
        rgb_camera_center.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        rgb_camera_center.set_position(2.0, 0.0, 1.4)
        rgb_camera_center.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_center)

        # Add RGB left camera
        rgb_camera_left = sensor.Camera('RGBCameraLeft')
        rgb_camera_left.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        rgb_camera_left.set_position(2.0, -1, 1.4)
        rgb_camera_left.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_left)

        # Add RGB right camera
        rgb_camera_right = sensor.Camera('RGBCameraRight')
        rgb_camera_right.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        rgb_camera_right.set_position(2.0, 1, 1.4)
        rgb_camera_right.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_right)

        # Add depth camera
        depth_camera = sensor.Camera('DepthCamera', PostProcessing='Depth')
        depth_camera.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        depth_camera.set_position(2.0, 0.0, 1.4)
        depth_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(depth_camera)

        # Add semantic segmentation camera
        sem_seg_camera = sensor.Camera('SemSegCamera', PostProcessing='SemanticSegmentation')
        sem_seg_camera.set_image_size(OUTPUT_IMGAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)
        sem_seg_camera.set_position(2.0, 0.0, 1.4)
        sem_seg_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(sem_seg_camera)

        self._carla_settings = settings

        logging.debug("Carla initialized")

    def _on_new_episode(self):
        scene = self.client.load_settings(self._carla_settings)
        number_of_start_positions = len(scene.player_start_spots)
        start_postition = np.random.randint(number_of_start_positions)
        self.client.start_episode(start_postition)
        self._timer.lap()
        self._new_episode_flag = False
        self._episode_path_created = False

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

    def _handle_keydown_event(self, key):
        if key == K_p:
            self._autopilot_enabled = not self._autopilot_enabled
        elif key == K_q:
            self._vehicle_in_reverse = not self._vehicle_in_reverse
        elif key == K_e:
            self._new_episode_flag = True
        elif key == K_r:
            self._recording_enabled = not self._recording_enabled

    def _render_pygame(self):

        if self._rgb_image_center is not None:
            array = ic.to_rgb_array(self._game_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._pygame_display.blit(surface, (0, 0))
        
        pygame.display.flip()

    def _save_data_to_disk(self):
        episode_path = Path(f"{self._output_path}/{self._timer.get_lap_timestamp()}")
        image_path = episode_path / "imgs"
        step = self._timer.step

        if not self._episode_path_created:
            image_path.mkdir(parents=True, exist_ok=True)
            self._episode_path_created = True
        
        rgb_center = ic.to_rgb_array(self._rgb_image_center)
        rgb_left = ic.to_rgb_array(self._rgb_image_left)
        rgb_right = ic.to_rgb_array(self._rgb_image_right)
        depth = np.array(ic.depth_to_logarithmic_grayscale(self._depth_image)).astype(int)
        sem_seg = np.array(ic.labels_to_cityscapes_palette(self._sem_seg_image)).astype(int)

        plt.imsave(image_path / f"{step}_rgb_center.png", rgb_center)
        plt.imsave(image_path / f"{step}_rgb_left.png",  rgb_left)
        plt.imsave(image_path / f"{step}_rgb_right.png", rgb_right)
        plt.imsave(image_path / f"{step}_depth.png", depth)
        plt.imsave(image_path / f"{step}_sem_seg.png", sem_seg)

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self._game_image = sensor_data.get("GameCamera", None)
        self._rgb_image_center = sensor_data.get('RGBCameraCenter', None)
        self._rgb_image_left = sensor_data.get('RGBCameraLeft', None)
        self._rgb_image_right = sensor_data.get('RGBCameraRight', None)

        self._depth_image = sensor_data.get('DepthCamera', None)
        self._sem_seg_image = sensor_data.get('SemSegCamera', None)

        control = self._get_keyboard_control(pygame.key.get_pressed())

        if self._new_episode_flag:
            self._on_new_episode()
        elif self._autopilot_enabled:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

        self._render_pygame()

        if self._recording_enabled and self._output_path is not None:
            self._save_data_to_disk()

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
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-o', '--output',
        metavar='PATH',
        dest='output_path',
        default=None,
        help='Recorded data will be saved to this path')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaController(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
