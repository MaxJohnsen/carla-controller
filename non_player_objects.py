"""
TODO: Write docstring
"""

from scipy import spatial


class NonPlayerObjects:
    def __init__(self, agent_type):
        self._agents = None
        self._KD_tree = None
        self._agent_type = agent_type

        self.valid = False

    def _is_valid_rot(self, rot1, rot2, rot_dif, thresh):
        yaw_diff = ((rot1 - rot2 + 180) % 360) - 180 + rot_dif
        return -thresh <= yaw_diff <= thresh

    def update_agents(self, all_agents):
        filtered = list(
            filter(lambda agent: hasattr(agent, self._agent_type), all_agents)
        )
        self._agents = list(map(lambda a: getattr(a, self._agent_type), filtered))

    def initialize_KD_tree(self):
        locations = []
        for agent in self._agents:
            loc = agent.transform.location
            locations.append([loc.x, loc.y, loc.z])

        if locations:
            self._KD_tree = spatial.KDTree(locations)
            self.valid = True

    def get_closest_with_rotation(self, player_trans, radius, rot_dif, thresh):
        if not self.valid:
            return None, None

        car_loc = [
            player_trans.location.x,
            player_trans.location.y,
            player_trans.location.z,
        ]
        car_rot = player_trans.rotation.yaw

        distance, index = self._KD_tree.query(car_loc)

        if distance < radius:
            closest_agent = self._agents[index]
            agent_rot = closest_agent.transform.rotation.yaw

            if self._is_valid_rot(agent_rot, car_rot, rot_dif, thresh):
                return closest_agent, distance

        return None, None

