"""
TODO: Write Docstring
"""
from threading import Thread
import os
import cv2


class DiskWriter(Thread):
    """ TODO: Write Docstring """

    def __init__(self, episode_path, images, driving_log, on_complete=None):
        Thread.__init__(self)
        self.progress = 0.0
        self._images = images
        self._driving_log = driving_log
        self._episode_path = episode_path
        self._on_complete = on_complete

    def run(self):
        image_path = self._episode_path / "imgs"
        image_path.mkdir(parents=True, exist_ok=True)

        for i in range(len(self._images)):
            filename, image = self._images[i]
            path = image_path / filename
            cv2.imwrite(str(path), image)
            self.progress = (i + 1) / len(self._images)

        csv_path = f"{str(self._episode_path)}/driving_log.csv"
        if not os.path.isfile(csv_path):
            self._driving_log.to_csv(csv_path)
        else:
            self._driving_log.to_csv(csv_path, mode="a", header=False)

        if self._on_complete is not None:
            self._on_complete()
