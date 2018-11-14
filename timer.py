import time


class Timer(object):
    def __init__(self):
        self._timestamp = time.time()
        self._episode_timestamp = time.time()

        self.episode_num = 0
        self.frame = 0
        self.timestamp_str = self._get_timestamp_str(self._timestamp)

        self.episode_frame = 0
        self.episode_timestamp_str = self._get_timestamp_str(self._episode_timestamp)

    def tick(self):
        self.frame += 1
        self.episode_frame += 1

    def new_episode(self):
        self.episode_frame = 0
        self._episode_timestamp = time.time()
        self.episode_timestamp_str = self._get_timestamp_str(self._episode_timestamp)
        self.episode_num += 1

    def _get_timestamp_str(self, timestamp):
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp))
