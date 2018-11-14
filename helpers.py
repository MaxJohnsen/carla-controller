from scipy import spatial
from enum import Enum
import time
import numpy as np


class TrafficLight(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2
    ERROR = 3
    NONE = 4


def get_agent(agent_id, agents):
    """
    Given a list of agents and a ID it returns the agent with the corresponding id
    If the agent does not exist in the list it returns None
    """
    agents = list(filter(lambda agent: agent.id == agent_id, agents))
    return agents[0] if agents else None


def is_valid_yaw(car_yaw, agent_yaw):
    yaw_min = -10.0
    yaw_max = 10.0
    car_yaw += 180
    agent_yaw += 180
    yaw_diff = ((car_yaw - agent_yaw + 180) % 360) - 90
    if yaw_diff < yaw_max and yaw_diff > yaw_min:
        return True
    return False


def get_agents(non_player_agents, agent_type):
    "Filters out specific agent type from non_player_agents"
    return list(filter(lambda agent: agent.HasField(agent_type), non_player_agents))


def get_distance(loc1, loc2):
    """ Calculates the distance between two vectors"""
    return abs(np.linalg.norm(loc2 - loc1))


def get_KDtree(non_player_agents, agent_type):
    """
    Create a KD-tree for one agent type
    Return the KD-tree and corresponding agent IDs
    Args:
        non_player_agents (list obj): List of vehicles, pedestrian, traffic lights and speed limit signs
        agent_type (str): Type of non player agent to create a KD-tree for 
    Returns:
        (KDTree): A KDTree of all the agents of given type
        (list int): List of agent IDs corresponding to the tree nodes.

        returns None, None if the KD-tree is not given any points 
    """
    # Filter out agents that is of type agent_type
    filtered_agents = get_agents(non_player_agents, agent_type)

    # List of locations to feed to the KD-tree
    agents_loaction = []
    # List of IDs corresponding to tree node index
    agent_ids = []
    for agent in filtered_agents:
        if agent_type == "traffic_light":
            agent_location = [
                agent.traffic_light.transform.location.x,
                agent.traffic_light.transform.location.y,
                agent.traffic_light.transform.location.z,
            ]
            agents_loaction.append(agent_location)
            agent_ids.append(agent.id)

        elif agent_type == "speed_limit":
            agent_location = [
                agent.traffic_light.transform.location.x,
                agent.traffic_light.transform.location.y,
                agent.traffic_light.transform.location.z,
            ]
            agents_loaction.append(agent_location)
            agent_ids.append(agent.id)

    if agents_loaction:
        return spatial.KDTree(agents_loaction), agent_ids
    return None, None


def find_current_traffic_light(
    KDTree, traffic_lights_ID, non_player_agents, car_transform, raduis=12.5
):
    """
    Uses a KDTree to determine closest traffic light to the car and checks if the rotation is correct
    Returns the state of the closest traffic light and the distance to the car
    """
    # Return error if there are no traffic lights in the simulator
    if KDTree is None:
        return TrafficLight(4), None

    # Location of car
    car_location = [
        car_transform.location.x,
        car_transform.location.y,
        car_transform.location.z,
    ]

    # Rotation of car
    car_yaw = car_transform.rotation.yaw

    # List of all traffic lights in the simulator
    all_traffic_lights = get_agents(non_player_agents, "traffic_light")

    # List of indecies and distances of all traffic lights within the radius
    filtered_indices = KDTree.query_ball_point(car_location, raduis)
    if filtered_indices:
        # List if IDs of all traffic lights within the radius
        filtered_traffic_lights_IDs = list(
            map(lambda x: traffic_lights_ID[x], filtered_indices)
        )
        # List of traffic light agents within the radius
        filtered_traffic_lights = []
        for i in range(len(filtered_traffic_lights_IDs)):
            traffic_light_id = filtered_traffic_lights_IDs[i]
            traffic_light = get_agent(traffic_light_id, all_traffic_lights)

            if traffic_light is not None:
                filtered_traffic_lights.append(traffic_light)
        # Return traffic light state GREEN if there are no traffic lights within the radius
        if not filtered_traffic_lights:
            return TrafficLight.GREEN, None

        # KDTree of traffic lights within the radius
        filtered_KDTree, filtered_traffic_lights_IDs = get_KDtree(
            filtered_traffic_lights, "traffic_light"
        )

        # Find closest traffic light and return the state if it has correct rotation
        closest_query = filtered_KDTree.query(car_location)
        closest_traffic_light_index = closest_query[1]
        closest_traffic_light_distance = closest_query[0]

        closest_traffic_light_id = filtered_traffic_lights_IDs[
            closest_traffic_light_index
        ]
        closest_traffic_light = get_agent(closest_traffic_light_id, all_traffic_lights)
        closest_traffic_light_yaw = (
            closest_traffic_light.traffic_light.transform.rotation.yaw
        )

        # Only choose if rotation yaw has approperiate value and the traffic light is closer than last frame
        if closest_traffic_light is not None and is_valid_yaw(
            car_yaw, closest_traffic_light_yaw
        ):

            return (
                TrafficLight(closest_traffic_light.traffic_light.state),
                closest_traffic_light_distance,
            )
    return TrafficLight.GREEN, None
