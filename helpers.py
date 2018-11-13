from scipy import spatial
from enum import Enum


class TrafficLight(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2
    ERROR = 3
    NONE = 4


def get_agent(agent_id, agents):
    """
    Given a list of agents and a ID it returns the agent with the corresponding id
    If the agent does not exist in the list it returns none 
    """
    for agent in agents:
        if agent.id == agent_id:
            return agent

    return None


def is_valid_yaw(car_yaw, agent_yaw):
    car_yaw += 180
    agent_yaw += 180
    yaw_diff = ((car_yaw - agent_yaw + 180) % 360) - 90
    if yaw_diff < 5.0 and yaw_diff > -0.5:
        return True
    return False


def get_agents(non_player_agents, agent_type):
    "Filters out specific agent type from non_player_agents"
    agents = []
    for agent in non_player_agents:
        if agent.HasField(agent_type):
            agents.append(agent)

    return agents


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
    KDTree, traffic_lights_ID, non_player_agents, car_transform, raduis=12.0
):
    """
    Uses a KDTree to determine closest traffic light to the car and checks if the rotation is correct
    Returns the state of the closest traffic light 
    """
    # Return error if there are no traffic lights in the simulator
    if KDTree is None:
        return TrafficLight(4)

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

    # List of indecies of all traffic lights within the radius
    filtered_traffic_lights_indices = KDTree.query_ball_point(car_location, raduis)
    # List if IDs of all traffic lights within the radius
    filtered_traffic_lights_IDs = list(
        map(lambda x: traffic_lights_ID[x], filtered_traffic_lights_indices)
    )
    # List of traffic light agents within the radius
    filtered_traffic_lights = []
    for traffic_light_id in filtered_traffic_lights_IDs:
        traffic_light = get_agent(traffic_light_id, all_traffic_lights)
        if traffic_light is not None:
            filtered_traffic_lights.append(traffic_light)
    # Return traffic light state GREEN if there are no traffic lights within the radius
    if not filtered_traffic_lights:
        return TrafficLight.GREEN

    # KDTree of traffic lights within the radius
    filtered_KDTree, filtered_traffic_lights_IDs = get_KDtree(
        filtered_traffic_lights, "traffic_light"
    )

    # Find closest traffic light and return the state if it has correct rotation
    closest_traffic_light_index = filtered_KDTree.query(car_location)
    closest_traffic_light_id = filtered_traffic_lights_IDs[
        closest_traffic_light_index[1]
    ]
    closest_traffic_light = get_agent(closest_traffic_light_id, all_traffic_lights)

    # Only choose if rotation yaw has approperiate value
    if closest_traffic_light is not None and is_valid_yaw(
        car_yaw, closest_traffic_light.traffic_light.transform.rotation.yaw
    ):
        return TrafficLight(closest_traffic_light.traffic_light.state)
    return TrafficLight.GREEN
