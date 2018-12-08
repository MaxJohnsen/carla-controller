"""
TODO: Write Docstring
"""
from threading import Thread
import os
import cv2


class ImageWriter(Thread):
    """ TODO: Write Docstring """

    def __init__(self, episode_path, images, driving_log, frames, on_complete=None):
        Thread.__init__(self)
        self.progress = 0.0
        self._images = images
        self._driving_log = driving_log
        self._frames = frames
        self._episode_path = episode_path
        self._on_complete = on_complete

    def run(self):
        image_path = self._episode_path / "imgs"
        image_path.mkdir(parents=True, exist_ok=True)

        for i in range(len(self._images)):
            for key, value in self._images[i].items():
                image = value
                filename = f"{self._frames[i]}_{key}.png"
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


class VideoWriter(Thread):
    """ TODO: Write Docstring """

    def __init__(self, episode_path, images, info, fps=30, on_complete=None):
        Thread.__init__(self)
        self.progress = 0.0
        self._images = images
        self._episode_path = episode_path
        self._on_complete = on_complete
        self._fps = fps
        self._info = info

    def _draw_info(self, img, info):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        speed, speed_limit, traffic_light, hlc = info
        cv2.putText(img, "Speed:", (132, 30), font, fontScale, fontColor, lineType)
        cv2.putText(img, "Speed Limit:", (41, 65), font, fontScale, fontColor, lineType)
        cv2.putText(
            img, "Traffic Light:", (37, 100), font, fontScale, fontColor, lineType
        )
        cv2.putText(
            img, "Activated HLC:", (10, 135), font, fontScale, fontColor, lineType
        )
        cv2.putText(img, speed, (270, 30), font, fontScale, fontColor, lineType)
        cv2.putText(img, speed_limit, (270, 65), font, fontScale, fontColor, lineType)
        cv2.putText(
            img, traffic_light, (270, 100), font, fontScale, fontColor, lineType
        )
        cv2.putText(img, hlc, (270, 135), font, fontScale, fontColor, lineType)
        return img

    def run(self):
        video_path = self._episode_path / "videos"
        video_path.mkdir(parents=True, exist_ok=True)

        for i in range(len(self._images)):
            imgs = self._images[i]
            shape = imgs[0].shape[:2]
            video_size = (shape[1], shape[0])
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            path = video_path / f"camera{i}.avi"
            out = cv2.VideoWriter(str(path), fourcc, self._fps, video_size)

            for f in range(len(imgs)):
                img = imgs[f]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = self._draw_info(img, self._info[f])
                out.write(img)
                self.progress = (len(imgs) * i + f + 1) / (
                    len(self._images) * len(imgs)
                )
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        if self._on_complete is not None:
            self._on_complete()
