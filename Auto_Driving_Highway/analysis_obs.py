from customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe
)
def available_action(toolModels):

    available_action_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableActions)), None)
    # Use tools to analyze the situation
    available_action = {}
    ego_vehicle_id = 'ego'
    available_lanes_analysis = available_action_tool.inference(ego_vehicle_id)
    available_action[available_action_tool] = available_lanes_analysis

    return available_action

def get_available_lanes(toolModels):

    available_lanes_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableLanes)), None)
    # Use tools to analyze the situation
    situation_analysis = {}
    ego_vehicle_id = 'ego'
    available_lanes_analysis = available_lanes_tool.inference(ego_vehicle_id)
    situation_analysis[available_lanes_tool] = available_lanes_analysis

    return situation_analysis

def get_involved_cars(toolModels):
    lane_cars_info = {}
    lane_involved_car_tool = next((tool for tool in toolModels if isinstance(tool, getLaneInvolvedCar)), None)
    lane_ids=['lane_0', 'lane_1', 'lane_2']
    for lane_id in lane_ids:
            cars_in_lane_info = lane_involved_car_tool.inference(lane_id)
            lane_cars_info[lane_id] = cars_in_lane_info

    return lane_cars_info

# function get info from available lanes
def extract_lanes_info(available_lanes_info):
    lanes = {
        'current': None,
        'left': None,
        'right': None
    }

    parts = available_lanes_info.split(". ")
    for part in parts:
        if "is the current lane" in part:
            lanes['current'] = part.split("`")[1]  # Extract the current lane
        elif "to the left of the current lane" in part:
            lanes['left'] = part.split("`")[1]  # Extract the left adjacent lane
        elif "to the right of the current lane" in part:
            lanes['right'] = part.split("`")[1]  # Extract the right adjacent lane

    return lanes

# These two functions get info from get_involved_cars

def extract_car_id_from_info(lane_info):
    # Extracts the car ID from the lane information string
    if "is driving" in lane_info:
        parts = lane_info.split()
        car_id_index = parts.index("is") - 1
        return parts[car_id_index]
    return None

def extract_lane_and_car_ids(lanes_info, lane_cars_info):
    lane_car_ids = {
        'current_lane': {'lane_id': None, 'car_id': None},
        'left_lane': {'lane_id': None, 'car_id': None},
        'right_lane': {'lane_id': None, 'car_id': None}
    }
    current_lane_id = lanes_info['current']
    left_lane_id = lanes_info['left']
    right_lane_id = lanes_info['right']

    # Extract car ID for the left adjacent lane, if it exists
    if current_lane_id and current_lane_id in lane_cars_info:
        current_lane_info = lane_cars_info[current_lane_id]
        current_car_id = extract_car_id_from_info(current_lane_info)
        lane_car_ids['current_lane'] = {'lane_id': current_lane_id, 'car_id': current_car_id}

    # Extract car ID for the left adjacent lane, if it exists
    if left_lane_id and left_lane_id in lane_cars_info:
        left_lane_info = lane_cars_info[left_lane_id]
        left_car_id = extract_car_id_from_info(left_lane_info)
        lane_car_ids['left_lane'] = {'lane_id': left_lane_id, 'car_id': left_car_id}

    # Extract car ID for the right adjacent lane, if it exists
    if right_lane_id and right_lane_id in lane_cars_info:
        right_lane_info = lane_cars_info[right_lane_id]
        right_car_id = extract_car_id_from_info(right_lane_info)
        lane_car_ids['right_lane'] = {'lane_id': right_lane_id, 'car_id': right_car_id}

    return lane_car_ids

# F
def assess_lane_change_safety(toolModels, lane_car_ids):
    lane_change_tool = next((tool for tool in toolModels if isinstance(tool, isChangeLaneConflictWithCar)), None)
    safety_assessment = {
        'left_lane_change_safe': True,
        'right_lane_change_safe': True
    }

    # Check if changing to the left lane is safe
    if lane_car_ids['left_lane']['lane_id'] and lane_car_ids['left_lane']['car_id']:
        left_lane_id = lane_car_ids['left_lane']['lane_id']
        left_car_id = lane_car_ids['left_lane']['car_id']
        input_str = f"{left_lane_id},{left_car_id}"
        left_lane_safety = lane_change_tool.inference(input_str)
        safety_assessment['left_lane_change_safe'] = 'safe' in left_lane_safety
    else:
        # If no car is in the left lane, consider it safe to change
        safety_assessment['left_lane_change_safe'] = True

    # Check if changing to the right lane is safe
    if lane_car_ids['right_lane']['lane_id'] and lane_car_ids['right_lane']['car_id']:
        right_lane_id = lane_car_ids['right_lane']['lane_id']
        right_car_id = lane_car_ids['right_lane']['car_id']
        input_str = f"{right_lane_id},{right_car_id}"
        right_lane_safety = lane_change_tool.inference(input_str)
        safety_assessment['right_lane_change_safe'] = 'safe' in right_lane_safety
    else:
        # If no car is in the right lane, consider it safe to change
        safety_assessment['right_lane_change_safe'] = True

    return safety_assessment


def check_safety_in_current_lane(toolModels, lane_and_car_ids):
    safety_analysis = {
        'acceleration_conflict': None,
        'keep_speed_conflict': None,
        'deceleration_conflict': None
    }

    # Extract tools from toolModels
    acceleration_tool = next((tool for tool in toolModels if isinstance(tool, isAccelerationConflictWithCar)), None)
    keep_speed_tool = next((tool for tool in toolModels if isinstance(tool, isKeepSpeedConflictWithCar)), None)
    deceleration_tool = next((tool for tool in toolModels if isinstance(tool, isDecelerationSafe)), None)

    current_lane_car_id = lane_and_car_ids['current_lane']['car_id']

    if current_lane_car_id:
        # Check for conflicts if there is a car in the current lane
        if acceleration_tool:
            safety_analysis['acceleration_conflict'] = acceleration_tool.inference(current_lane_car_id)
        if keep_speed_tool:
            safety_analysis['keep_speed_conflict'] = keep_speed_tool.inference(current_lane_car_id)
        if deceleration_tool:
            safety_analysis['deceleration_conflict'] = deceleration_tool.inference(current_lane_car_id)

    return safety_analysis

def get_current_speed(toolModels):
    lane_cars_info = {}
    lane_involved_car_tool = next((tool for tool in toolModels if isinstance(tool, getLaneInvolvedCar)), None)
    ego_speed = round(lane_involved_car_tool.sce.vehicles['ego'].speed, 1)
    speed_info = f"Your current speed is {ego_speed} m/s.\n\n"
    return speed_info

def format_training_info(available_actions_msg, lanes_info_msg, speed_info, all_lane_info_msg, lanes_adjacent_info, cars_near_lane, lane_change_safety, current_lane_safety):
    formatted_message = ""

    # Add available actions information
    formatted_message += "Available Actions:\n"
    for tool, action_info in available_actions_msg.items():
        formatted_message += f"- {action_info}\n"

    # Add information about lanes
    formatted_message += "\nLane Information:\n"
    formatted_message += f"- Current Lane: {lanes_adjacent_info['current']}\n"
    formatted_message += f"- Left Adjacent Lane: {lanes_adjacent_info['left'] or 'None'}\n"
    formatted_message += f"- Right Adjacent Lane: {lanes_adjacent_info['right'] or 'None'}\n"

    # Add details about vehicles in each lane
    formatted_message += f"{speed_info}\n"
    formatted_message += "\nOther Vehicles in all the Lanes:\n"
    for lane_id, car_info in all_lane_info_msg.items():
        formatted_message += f"- {lane_id}: {car_info}\n"

    # Safety assessment for lane changes
    formatted_message += "\nSafety Assessment for Lane Changes:\n"
    formatted_message += f"- Left Lane Change: {'Safe' if lane_change_safety['left_lane_change_safe'] else 'Not Safe'}\n"
    formatted_message += f"- Right Lane Change: {'Safe' if lane_change_safety['right_lane_change_safe'] else 'Not Safe'}\n"

    # Safety assessment in the current lane
    formatted_message += "\nSafety Assessment in Current Lane:\n"
    for action, safety in current_lane_safety.items():
        formatted_message += f"- {action.capitalize().replace('_', ' ')}: {safety}\n"

    return formatted_message