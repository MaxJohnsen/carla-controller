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
    """Checks wheter the rotation between two objects is valied, e.g. if the car and traffic light is facing each other"""
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

        elif agent_type == "speed_limit_sign":
            agent_location = [
                agent.speed_limit_sign.transform.location.x,
                agent.speed_limit_sign.transform.location.y,
                agent.speed_limit_sign.transform.location.z,
            ]
            agents_loaction.append(agent_location)
            agent_ids.append(agent.id)

    if agents_loaction:
        return spatial.KDTree(agents_loaction), agent_ids
    return None, None


def find_current_agent(
    KDTree, agent_type, agents_IDs, non_player_agents, car_transform, raduis=12.5
):
    """
    Uses a KDTree to determine closest traffic light/speed limit sign to the car and checks if the rotation is correct
    In the code "agent" refers to either traffic light or speed limit signs 
    Returns the state of the closest traffic light/speed limit sign and the distance to the car

    Args: 
        KDTree (KDTree):            A KDTree with all the agents, used to look up closest neighbours
        agent_type (str):           Indicates if it is a traffic light or speed limit sign 
        agentn_IDs (list <int>):    List of agent ID with corresponding to the tree nodes in the KD-tree
        non_player_agents (obj):    List of all non player agents in the simulator
        car_transform (obj):        Transformation of the car (location, rotation and orientation)
        radiu (float):              Radius from the car it will look for traffic lights/speed limit signs 

    """
    # Return error if there are no agent in the simulator
    if KDTree is None:
        if agent_type == "traffic_light":
            return TrafficLight(4), None
        return None, None

    # Location of car
    car_location = [
        car_transform.location.x,
        car_transform.location.y,
        car_transform.location.z,
    ]

    # Rotation of car
    car_yaw = car_transform.rotation.yaw

    # List of all agent in the simulator
    all_agents = get_agents(non_player_agents, agent_type)

    # List of indecies and distances of all agent within the radius
    filtered_indices = KDTree.query_ball_point(car_location, raduis)
    if filtered_indices:
        # List if IDs of all agent within the radius
        filtered_agents_IDs = list(map(lambda x: agents_IDs[x], filtered_indices))
        # List of traffic light agents within the radius
        filtered_agents = []
        for i in range(len(filtered_agents_IDs)):
            agent_id = filtered_agents_IDs[i]
            agent = get_agent(agent_id, all_agents)

            if agent is not None:
                filtered_agents.append(agent)

        # If there are no agents within the radius
        if not filtered_agents:
            if agent_type == "traffic_light":
                # Return traffic light state GREEN if there are no agent within the radius
                return TrafficLight.GREEN, None
            else:
                # Return default speed limit if it cant find a speed limit sign
                return None, None

        # KDTree of agent within the radius
        filtered_KDTree, filtered_agents_IDs = get_KDtree(filtered_agents, agent_type)

        # Find closest traffic light and return the state if it has correct rotation
        closest_query = filtered_KDTree.query(car_location)
        closest_agent_index = closest_query[1]
        closest_agent_distance = closest_query[0]

        closest_agent_id = filtered_agents_IDs[closest_agent_index]
        closest_agent = get_agent(closest_agent_id, all_agents)
        if agent_type == "traffic_light":
            closest_agent_yaw = closest_agent.traffic_light.transform.rotation.yaw
        else:
            closest_agent_yaw = closest_agent.speed_limit_sign.transform.rotation.yaw

        # Only choose if rotation yaw has approperiate value and the traffic light is closer than last frame
        if closest_agent is not None and is_valid_yaw(car_yaw, closest_agent_yaw):
            if agent_type == "traffic_light":
                return (
                    TrafficLight(closest_agent.traffic_light.state),
                    closest_agent_distance,
                )
            else:  # TODO: make speed limit enum match with server
                return (
                    closest_agent.speed_limit_sign.speed_limit,
                    closest_agent_distance,
                )
    if agent_type == "traffic_light":
        return TrafficLight.GREEN, None
    return None, None

