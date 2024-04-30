from openai import OpenAI
import pre_prompt
from logging import lastResort

with open('MY_KEY.txt', 'r') as f:
    api_key = f.read()
API_KEY = api_key


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'change lane to the left of the current lane,',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle'
}

def send_to_chatgpt(last_action, current_scenario, sce):
    client = OpenAI(api_key=API_KEY)
    print("=========================",type(last_action),"=========================")
    action_id = int(last_action)  # Convert to integer
    message_prefix = pre_prompt.SYSTEM_MESSAGE_PREFIX
    traffic_rules = pre_prompt.get_traffic_rules()
    decision_cautions = pre_prompt.get_decision_cautions()
    action_name = ACTIONS_ALL.get(action_id, "Unknown Action")
    action_description = ACTIONS_DESCRIPTION.get(action_id, "No description available")

    prompt = (f"{message_prefix}"
              f"You, the 'ego' car, are now driving on a highway. You have already driven for {sce.frame} seconds.\n"
              "There are several rules you need to follow when you drive on a highway:\n"
              f"{traffic_rules}\n\n"
              "Here are your attention points:\n"
              f"{decision_cautions}\n\n"
              "Once you make a final decision, output it in the following format:\n"
              "```\n"
              "Final Answer: \n"
              "    \"decision\": {\"<ego car's decision, ONE of the available actions>\"},\n"
              "```\n")
    user_prompt = (f"The decision made by the agent LAST time step was `{action_name}` ({action_description}).\n\n"
                   "Here is the current scenario:\n"
                   f"{current_scenario}\n\n")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return completion.choices[0].message