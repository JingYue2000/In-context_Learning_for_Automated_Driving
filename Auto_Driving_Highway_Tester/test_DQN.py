import highway_env
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv
import random
from agent_train import MyHighwayEnv
from scenario import Scenario
from customTools import getAvailableActions, getAvailableLanes, getLaneInvolvedCar,isChangeLaneConflictWithCar,isAccelerationConflictWithCar,isKeepSpeedConflictWithCar,isDecelerationSafe
from analysis_obs import available_action, get_available_lanes, get_involved_cars, extract_lanes_info, extract_lane_and_car_ids, assess_lane_change_safety, check_safety_in_current_lane, format_training_info
from customTools import ACTIONS_ALL
import ask_llm

def main():
    env = MyHighwayEnv(vehicleCount=5)
    observation = env.reset()
    print("Initial Observation:", observation)
    print("Observation space:", env.observation_space)
    # print("Action space:", env.action_space)

    # Wrap the environment in a DummyVecEnv for SB3
    env = DummyVecEnv([lambda: env])  # Add this line
    available_actions = env.envs[0].get_available_actions()
    model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            train_freq=2,
            learning_starts=20,
            exploration_fraction=0.5,
            learning_rate=0.0001,
        )
    # Initialize scenario and tools
    sce = Scenario(vehicleCount=5)
    toolModels = [
        getAvailableActions(env.envs[0]),
        getAvailableLanes(sce),
        getLaneInvolvedCar(sce),
        isChangeLaneConflictWithCar(sce),
        isAccelerationConflictWithCar(sce),
        isKeepSpeedConflictWithCar(sce),
        isDecelerationSafe(sce),
        # isActionSafe()
    ]
    frame = 0
    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            sce.updateVehicles(obs, frame)
            # Observation translation
            msg0 = available_action(toolModels)
            msg1 = get_available_lanes(toolModels)
            msg2 = get_involved_cars((toolModels))
            msg1_info = next(iter(msg1.values()))
            lanes_info = extract_lanes_info(msg1_info)

            lane_car_ids = extract_lane_and_car_ids(lanes_info, msg2)
            safety_assessment = assess_lane_change_safety(toolModels, lane_car_ids)
            safety_msg = check_safety_in_current_lane(toolModels, lane_car_ids)
            formatted_info = format_training_info(msg0, msg1, msg2, lanes_info, lane_car_ids, safety_assessment,
                                                  safety_msg)

            action, _ = model.predict(obs)
            action_id = int(action[0])
            action_name = ACTIONS_ALL.get(action_id, "Unknown Action")
            print(f"DQN action: {action_name}")

            llm_response = ask_llm.send_to_chatgpt(action, formatted_info, sce)
            decision_content = llm_response.content
            print(llm_response)
            llm_suggested_action = extract_decision(decision_content)
            print(f"llm action: {llm_suggested_action}")

            env.env_method('set_llm_suggested_action', llm_suggested_action)
            # print(f"Action: {action}")
            # print(f"Observation: {next_obs}")

            obs, custom_reward, done, info = env.step(action)
            print(f"Reward: {custom_reward}\n")
            frame += 1
            model.learn(total_timesteps=10, reset_num_timesteps=False)

    obs = env.reset()
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        print(f"Reward: {rewards}\n")

    env.close()

# utils.py
def extract_decision(response_content):
    try:
        start = response_content.find('"decision": {') + len('"decision": {')
        end = response_content.find('}', start)
        decision = response_content[start:end].strip('"')
        return decision
    except Exception as e:
        print(f"Error in extracting decision: {e}")
        return None




if __name__ == "__main__":
    main()
