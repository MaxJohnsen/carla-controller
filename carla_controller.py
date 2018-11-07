from __future__ import print_function

import argparse
import logging
import random
import numpy as np
import pygame

from timer import Timer

from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_SPACE
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_q
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_r
from pygame.locals import K_p


from carla.client import make_carla_client, VehicleControl
from carla import sensor
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla import image_converter

from matplotlib import pyplot as plt

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class CarlaController:
    def __init__(self, carla_client, args):
        self.client = carla_client

        #Stores the latest recieved images
        self._rgb_image_center = None
        self._rgb_image_left = None
        self._rgb_image_right = None
        self._depth_image_center = None
        self._semantic_seg_image = None

        self._pygame_display = None
        self._carla_settings = None

        self._weather_id = 1
        self._number_of_vehicles = 50
        self._number_of_pedastrians = 30
        self._quality_level = 'Epic'

        self._timer = None

        self._new_episode_flag = False
        
        self._vehicle_in_reverse = False
        self._autopilot_enabled = False

    def _initialize_pygame(self):
        self._pygame_display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT),pygame.HWSURFACE | pygame.DOUBLEBUF)


        self._on_new_episode()

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
        
        # Add RGB center camera
        camera_rgb_center = sensor.Camera('CameraRgbCenter')
        camera_rgb_center.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        camera_rgb_center.set_position(2.0, 0.0, 1.4)
        camera_rgb_center.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera_rgb_center)

        # Add depth camera
        camera_depth = sensor.Camera('CameraDepth')
        camera_depth.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        camera_depth.set_position(2.0, 0.0, 1.4)
        camera_depth.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera_depth)

        
        # Add semantic segmentation camera
        camera_sematic_seg = sensor.Camera('CameraSemanticSeg')
        camera_sematic_seg.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        camera_sematic_seg.set_position(2.0, 0.0, 1.4)
        camera_sematic_seg.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera_sematic_seg)

        self._carla_settings = settings

    def _on_new_episode(self):
        scene = self.client.load_settings(self._carla_settings)
        number_of_start_positions = len(scene.player_start_spots)
        start_postition = np.random.randint(number_of_start_positions)
        self.client.start_episode(start_postition)
        self._timer = Timer()
        self._new_episode_flag = False

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
        if keys[K_p]:
            self._autopilot_enabled = not self._autopilot_enabled
        if keys[K_q]:
            self._vehicle_in_reverse = not self._vehicle_in_reverse
        if keys[K_r]:
            self._new_episode_flag = True
        control.reverse = self._vehicle_in_reverse
        
        return control

    def _render_pygame(self):

        if self._rgb_image_center is not None:
            array = image_converter.to_rgb_array(self._rgb_image_center)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._pygame_display.blit(surface, (0, 0))

        pygame.display.flip()

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self._rgb_image_center = sensor_data.get('CameraRgbCenter', None)
        self._depth_image_center = sensor_data.get('CameraDepth', None)
        self._semantic_seg_image = sensor_data.get('CameraSemanticSeg', None)

        control = self._get_keyboard_control(pygame.key.get_pressed())

        if self._new_episode_flag:
            self._on_new_episode()
        elif self._autopilot_enabled:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

        self._render_pygame()

    def execute(self):
        pygame.init()
        self._initialize_carla()
        self._initialize_pygame()

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
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
 
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

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
