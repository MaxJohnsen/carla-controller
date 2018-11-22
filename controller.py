"""
TODO: Write Docstring
"""
from __future__ import print_function
import time
import argparse
import logging
import configparser
from pathlib import Path
import pandas as pd
import pygame
import pygame.locals as pl
import numpy as np
from carla.client import make_carla_client, VehicleControl
from carla import sensor
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla import image_converter as ic
from timer import Timer

from disk_writer import DiskWriter
from HUD import InfoBox
from enums import GameState, HighLevelCommand, TrafficLight
from non_player_objects import NonPlayerObjects
from drive_models import DriveModelKeras


class CarlaController:
    """ TODO: Write Docstring """

    def __init__(self, carla_client, args, settings):
        self.client = carla_client
        self._game_image = None
        self._measurements = None
        self._sensor_data = None
        self._image_history = None
        self._driving_history = None
        self._frame_history = None
        self._pygame_display = None
        self._carla_settings = None
        self._settings = self._initialize_settings(settings)
        self._timer = Timer()
        self._output_path = args.output_path
        self._game_state = GameState.NOT_RECORDING
        self._new_episode_flag = False
        self._exit_flag = False
        self._vehicle_in_reverse = False
        self._autopilot_enabled = False
        self._drive_model_enabled = False
        self._joystick_enabled = args.joystick
        self._joystick = None
        self._drive_model_path = args.drive_model_path
        self._drive_model = None
        self._disk_writer_thread = None
        self._bottom_left_hud = InfoBox((200, 75))
        self._bottom_right_hud = InfoBox((250, 75))
        self._current_traffic_light = None
        self._current_speed_limit = None
        self._current_hlc = None
        self._traffic_lights = NonPlayerObjects("traffic_light")
        self._speed_limits = NonPlayerObjects("speed_limit_sign")

    def _initialize_settings(self, f):
        s = {}
        s["quality_level"] = f.get("Carla", "QualityLevel", fallback="Epic")
        s["weather_id"] = int(f.get("Carla", "WeatherId", fallback=1))
        s["number_of_vehicles"] = int(f.get("Carla", "NumberOfVehicles", fallback=50))
        s["number_of_pedastrians"] = int(
            f.get("Carla", "NumberOfPedestrians", fallback=30)
        )
        s["autopilot_steer_noise"] = float(f.get("AutoPilot", "SteerNoise", fallback=0))
        s["autopilot_throttle_noise"] = float(
            f.get("AutoPilot", "ThrottleNoise", fallback=0)
        )

        s["window_width"] = int(f.get("Pygame", "WindowWidth", fallback=1024))
        s["window_height"] = int(f.get("Pygame", "WindowHeight", fallback=768))
        s["output_image_width"] = int(
            f.get("Pygame", "OutputImageWidth", fallback=1024)
        )
        s["output_image_height"] = int(
            f.get("Pygame", "OutputImageHeight", fallback=768)
        )
        s["randomize_weather"] = f.getboolean(
            "Carla", "RandomizeWeather", fallback=False
        )
        s["autostart_recording"] = f.getboolean(
            "Controller", "AutoStartRecording", fallback=False
        )
        s["frame_limit"] = int(f.get("Controller", "FrameLimit", fallback=0))
        s["episode_limit"] = int(f.get("Controller", "EpisodeLimit", fallback=0))
        s["drive_model_steer"] = f.getboolean(
            "DriveModel", "ControlSteer", fallback=False
        )
        s["drive_model_throttle"] = f.getboolean(
            "DriveModel", "ControlThrottle", fallback=False
        )
        s["drive_model_brake"] = f.getboolean(
            "DriveModel", "ControlBrake", fallback=False
        )
        return s

    def _initialize_pygame(self):
        self._pygame_display = pygame.display.set_mode(
            (self._settings["window_width"], self._settings["window_height"]),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
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

        logging.debug("pygame initialized")

    def _initialize_carla(self):
        # Initialize settings
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            NumberOfVehicles=self._settings["number_of_vehicles"],
            NumberOfPedestrians=self._settings["number_of_vehicles"],
            WeatherId=self._settings["weather_id"],
            QualityLevel=self._settings["quality_level"],
            SendNonPlayerAgentsInfo=True,
        )
        settings.randomize_seeds()

        output_image_width = self._settings["output_image_width"]
        output_image_height = self._settings["output_image_height"]

        # Add a game camera
        game_camera = sensor.Camera("GameCamera")
        game_camera.set_image_size(
            self._settings["window_width"], self._settings["window_height"]
        )
        game_camera.set_position(2.0, 0.0, 1.4)
        game_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(game_camera)

        # Add RGB center camera
        rgb_camera_center = sensor.Camera("RGBCameraCenter")
        rgb_camera_center.set_image_size(output_image_width, output_image_height)
        rgb_camera_center.set_position(2.0, 0.0, 1.4)
        rgb_camera_center.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_center)

        # Add RGB left camera
        rgb_camera_left = sensor.Camera("RGBCameraLeft")
        rgb_camera_left.set_image_size(output_image_width, output_image_height)
        rgb_camera_left.set_position(2.0, -1, 1.4)
        rgb_camera_left.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_left)

        # Add RGB right camera
        rgb_camera_right = sensor.Camera("RGBCameraRight")
        rgb_camera_right.set_image_size(output_image_width, output_image_height)
        rgb_camera_right.set_position(2.0, 1, 1.4)
        rgb_camera_right.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(rgb_camera_right)

        # Add depth camera
        depth_camera = sensor.Camera("DepthCamera", PostProcessing="Depth")
        depth_camera.set_image_size(output_image_width, output_image_height)
        depth_camera.set_position(2.0, 0.0, 1.4)
        depth_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(depth_camera)

        # Add semantic segmentation camera
        sem_seg_camera = sensor.Camera(
            "SemSegCamera", PostProcessing="SemanticSegmentation"
        )
        sem_seg_camera.set_image_size(output_image_width, output_image_height)
        sem_seg_camera.set_position(2.0, 0.0, 1.4)
        sem_seg_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(sem_seg_camera)

        self._carla_settings = settings

        logging.debug("Carla initialized")

    def _initialize_drive_model(self):
        if self._drive_model_path:
            self._drive_model = DriveModelKeras()
            logging.info("Loading drive model from: %s", self._drive_model_path)
            self._drive_model.load_model(self._drive_model_path)

    def _initialize_history(self):
        self._driving_history = pd.DataFrame(
            columns=[
                "CenterRGB",
                "LeftRGB",
                "RightRGB",
                "Depth",
                "SemSeg",
                "Location",
                "Speed",
                "Controls",
                "APControls",
                "HLC",
                "SpeedLimit",
                "TrafficLight",
                "AutoPilotEnabled",
                "WeatherID",
            ]
        )
        self._image_history = []
        self._frame_history = []

    def _on_new_episode(self):
        self._timer.new_episode()
        if self._settings["episode_limit"] != 0:
            if self._settings["episode_limit"] < self._timer.episode_num:
                self._exit_flag = True
                return
        scene = self.client.load_settings(self._carla_settings)
        number_of_start_positions = len(scene.player_start_spots)
        start_postition = np.random.randint(number_of_start_positions)
        self.client.start_episode(start_postition)
        self._new_episode_flag = False
        self._disk_writer_thread = None
        self._initialize_history()
        self._current_speed_limit = 30
        self._current_traffic_light = (TrafficLight.NONE, 15)
        if self._settings["randomize_weather"]:
            self._carla_settings.set(WeatherId=np.random.randint(0, 15))
        if self._drive_model:
            self._current_hlc = HighLevelCommand.FOLLOW_ROAD

    def _get_keyboard_control(self, keys):
        control = VehicleControl()
        if keys[pl.K_LEFT] or keys[pl.K_a]:
            control.steer = -1.0
        if keys[pl.K_RIGHT] or keys[pl.K_d]:
            control.steer = 1.0
        if keys[pl.K_UP] or keys[pl.K_w]:
            control.throttle = 1.0
        if keys[pl.K_DOWN] or keys[pl.K_s]:
            control.brake = 1.0
        if keys[pl.K_SPACE]:
            control.hand_brake = True
        control.reverse = self._vehicle_in_reverse

        return control

    def _get_joystick_control(self):
        control = VehicleControl()
        control.steer = self._joystick.get_axis(0)
        control.throttle = max(self._joystick.get_axis(1) * -1, 0)
        control.brake = max(self._joystick.get_axis(1), 0)

        return control

    def _get_autopilot_control(self):
        speed = int(self._measurements.player_measurements.forward_speed * 3.6)
        autopilot = self._measurements.player_measurements.autopilot_control
        control = VehicleControl()
        steer_noise = self._settings["autopilot_steer_noise"]
        if steer_noise != 0:
            steer_noise = np.random.uniform(-steer_noise, steer_noise)
        control.steer = autopilot.steer + steer_noise

        throttle_noise = self._settings["autopilot_throttle_noise"] if speed > 10 else 0

        if throttle_noise != 0:
            throttle_noise = np.random.uniform(-throttle_noise, throttle_noise)
        control.throttle = autopilot.throttle + throttle_noise
        control.brake = autopilot.brake
        return control

    def _get_drive_model_control(self, control):
        images = self._get_camera_images()
        info = {
            "speed": self._measurements.player_measurements.forward_speed * 3.6,
            "speed_limit": self._current_speed_limit,
            "traffic_light": self._current_traffic_light[0].value,
            "hlc": self._current_hlc.value,
        }
        steer, throttle, brake = self._drive_model.get_prediction(images, info)

        if self._settings["drive_model_steer"]:
            control.steer = steer
        if self._settings["drive_model_throttle"]:
            control.throttle = throttle
        if self._settings["drive_model_brake"]:
            if brake > 0.3:
                control.brake = brake
        return control

    def _writeback_hlc_to_history(self, command):
        look_back = 70
        for i, row in self._driving_history.iterrows():
            if int(row["HLC"]) == 0:
                if i >= len(self._driving_history.index) - look_back:
                    self._driving_history.at[i, "HLC"] = command.value

    def _handle_keydown_event(self, key):
        if self._game_state is not GameState.WRITING:
            if key == pl.K_p:
                self._autopilot_enabled = not self._autopilot_enabled
            elif key == pl.K_m:
                if self._drive_model:
                    self._drive_model_enabled = not self._drive_model_enabled
                    self._current_hlc = HighLevelCommand.FOLLOW_ROAD
            elif key == pl.K_q:
                self._vehicle_in_reverse = not self._vehicle_in_reverse
            elif key == pl.K_e:
                if self._game_state == GameState.RECORDING:
                    self._game_state = GameState.WRITING
                    self._write_history_to_disk()
                self._new_episode_flag = True
            elif key == pl.K_r:
                if (
                    self._game_state == GameState.NOT_RECORDING
                    and self._output_path is not None
                ):
                    self._game_state = GameState.RECORDING
                elif self._game_state == GameState.RECORDING:
                    self._game_state = GameState.WRITING
                    self._write_history_to_disk()
        if self._game_state == GameState.RECORDING:
            if key == pl.K_KP8:
                self._writeback_hlc_to_history(HighLevelCommand.STRAIGHT_AHEAD)
            elif key == pl.K_KP4:
                self._writeback_hlc_to_history(HighLevelCommand.TURN_LEFT)
            elif key == pl.K_KP6:
                self._writeback_hlc_to_history(HighLevelCommand.TURN_RIGHT)
        if self._drive_model_enabled:
            if key == pl.K_KP8:
                self._current_hlc = HighLevelCommand.STRAIGHT_AHEAD
            elif key == pl.K_KP4:
                self._current_hlc = HighLevelCommand.TURN_LEFT
            elif key == pl.K_KP6:
                self._current_hlc = HighLevelCommand.TURN_RIGHT
            elif key == pl.K_KP5:
                self._current_hlc = HighLevelCommand.FOLLOW_ROAD

    def _render_HUD(self):
        speed = int(self._measurements.player_measurements.forward_speed * 3.6)
        autopilot_status = "Enabled" if self._autopilot_enabled else "Disabled"
        drive_model_status = "Enabled" if self._drive_model_enabled else "Disabled"
        reverse_status = "Enabled" if self._vehicle_in_reverse else "Disabled"
        speed_value = "{} km/h".format(speed)
        speed_limit = "{} km/h".format(self._current_speed_limit)
        traffic_light = self._current_traffic_light[0].name
        current_hlc = (
            self._current_hlc.name if self._drive_model_enabled else "Disabled"
        )
        self._bottom_left_hud.update_content(
            [
                ("Speed", speed_value),
                ("Speed Limit", speed_limit),
                ("Reverse", reverse_status),
                ("Traffic Light", traffic_light),
            ]
        )
        self._bottom_right_hud.update_content(
            [
                ("Autopilot", autopilot_status),
                ("Recording State", self._game_state.name),
                ("Drive Model", drive_model_status),
                ("Drive Model HLC", current_hlc),
            ]
        )

        sw_x = 20
        sw_y = self._settings["window_height"] - self._bottom_left_hud.size[1] - 20
        se_x = self._settings["window_width"] - self._bottom_right_hud.size[0] - 20
        se_y = self._settings["window_height"] - self._bottom_right_hud.size[1] - 20

        self._pygame_display.blit(self._bottom_left_hud.render_surface(), (sw_x, sw_y))
        self._pygame_display.blit(self._bottom_right_hud.render_surface(), (se_x, se_y))

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
        left = (self._settings["window_width"] / 2) - (width / 2)
        top = (self._settings["window_height"] / 2) - (height / 2)
        pygame.draw.rect(
            surface, (255, 255, 255), pygame.Rect(left, top, width * progress, height)
        )
        pygame.draw.rect(
            surface, (128, 128, 128), pygame.Rect(left, top, width, height), 1
        )
        self._pygame_display.blit(surface, (0, 0))

    def _get_camera_images(self):
        sensor_data = self._sensor_data

        image_object = {
            "rgb_center": ic.to_bgra_array(sensor_data.get("RGBCameraCenter", None)),
            "rgb_left": ic.to_bgra_array(sensor_data.get("RGBCameraLeft", None)),
            "rgb_right": ic.to_bgra_array(sensor_data.get("RGBCameraRight", None)),
            "depth": ic.depth_to_logarithmic_grayscale(
                sensor_data.get("DepthCamera", None)
            ),
            "sem_seg": ic.labels_to_cityscapes_palette(
                sensor_data.get("SemSegCamera", None)
            ),
        }
        return image_object

    def _save_to_history(self, control):
        measurements = self._measurements

        frame = self._timer.episode_frame

        self._image_history.append(self._get_camera_images())
        self._frame_history.append(frame)

        loc = measurements.player_measurements.transform.location
        speed = measurements.player_measurements.forward_speed * 3.6
        autopilot = measurements.player_measurements.autopilot_control

        self._driving_history = self._driving_history.append(
            pd.Series(
                [
                    f"imgs/{frame}_rgb_center.png",
                    f"imgs/{frame}_rgb_left.png",
                    f"imgs/{frame}_rgb_right.png",
                    f"imgs/{frame}_depth.png",
                    f"imgs/{frame}_sem_seg.png",
                    (loc.x, loc.y),
                    speed,
                    (
                        control.steer,
                        control.throttle,
                        control.brake,
                        int(control.reverse),
                    ),
                    (
                        autopilot.steer,
                        autopilot.throttle,
                        autopilot.brake,
                        int(autopilot.reverse),
                    ),
                    0,
                    self._current_speed_limit,
                    self._current_traffic_light[0].value,
                    int(self._autopilot_enabled),
                    self._settings["weather_id"],
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
            self._frame_history,
            on_complete=self._write_complete,
        )
        self._disk_writer_thread.start()

    def _write_complete(self):
        self._game_state = GameState.NOT_RECORDING
        self._initialize_history()

    def _update_current_traffic_light(self):
        old_state, old_dist = self._current_traffic_light
        agent, new_dist = self._traffic_lights.get_closest_with_rotation(
            self._measurements.player_measurements.transform, 12, -90, 15
        )
        if agent is not None:
            new_state = agent.state

            if new_dist <= old_dist:
                self._current_traffic_light = (TrafficLight(new_state), new_dist)
            else:
                self._current_traffic_light = (TrafficLight(old_state), new_dist)
        else:
            self._current_traffic_light = (TrafficLight.NONE, 15)

    def _update_current_speed_limit(self):
        agent = self._speed_limits.get_closest_with_rotation(
            self._measurements.player_measurements.transform, 12, -90, 20
        )[0]
        if agent is not None:
            self._current_speed_limit = int(agent.speed_limit * 3.6)

    def _on_loop(self):

        if (
            self._game_state is GameState.NOT_RECORDING
            and self._settings["autostart_recording"]
        ):
            if self._timer.episode_frame is 40:
                self._game_state = GameState.RECORDING

        if self._game_state is not GameState.WRITING:
            self._timer.tick()

            if self._new_episode_flag:
                self._on_new_episode()

                if self._exit_flag:
                    return False

            measurements, sensor_data = self.client.read_data()
            self._measurements = measurements
            self._sensor_data = sensor_data

            self._game_image = sensor_data.get("GameCamera", None)

            self._traffic_lights.update_agents(measurements.non_player_agents)

            if not self._traffic_lights.valid:
                self._traffic_lights.initialize_KD_tree()
            else:
                self._update_current_traffic_light()

            if not self._speed_limits.valid:
                self._speed_limits.update_agents(measurements.non_player_agents)
                self._speed_limits.initialize_KD_tree()
            else:
                self._update_current_speed_limit()

            if not self._autopilot_enabled:
                if self._joystick_enabled:
                    control = self._get_joystick_control()
                else:
                    control = self._get_keyboard_control(pygame.key.get_pressed())
            else:
                control = self._get_autopilot_control()

            if self._drive_model and self._drive_model_enabled:
                control = self._get_drive_model_control(control)

            self.client.send_control(control)

            if self._game_state == GameState.RECORDING:
                self._save_to_history(control)

        if self._settings["frame_limit"] != 0:
            if self._settings["frame_limit"] < self._timer.episode_frame:
                if self._game_state == GameState.RECORDING:
                    self._game_state = GameState.WRITING
                    self._write_history_to_disk()
                self._new_episode_flag = True

        self._render_pygame()

    def execute(self):
        """ TODO: Write docstring """
        pygame.init()

        self._initialize_carla()
        self._initialize_pygame()
        self._initialize_drive_model()
        if self._output_path is not None:
            logging.info("Recorded data will be saved to: %s", self._output_path)
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pl.KEYDOWN:
                        self._handle_keydown_event(event.key)

                if self._on_loop() is False:
                    break
        finally:
            pygame.quit()


def main():
    """ TODO: Write docstring """
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
        help="recorded data will be saved to this path",
    )
    argparser.add_argument(
        "-m",
        "--model",
        metavar="M",
        dest="drive_model_path",
        default=None,
        help="path to drive model",
    )
    args = argparser.parse_args()

    settings = configparser.ConfigParser()
    settings.read("settings.ini")

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("Listening to server %s:%s", args.host, args.port)

    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaController(client, args, settings)
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
