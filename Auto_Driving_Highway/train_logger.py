from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt

class TrainingLogger(BaseCallback):
    def __init__(self, check_freq, verbose=0):
        super(TrainingLogger, self).__init__(verbose)
        self.check_freq = check_freq
        # Metrics for collision and speed
        self.collision_count = 0
        self.total_speed = 0
        self.steps = 0
        self.collision_probabilities = []
        self.average_speeds = []
        # Metrics for matching percentage
        self.match_count = 0
        self.total_decisions = 0  # Adjusted to total_decisions for clarity
        self.match_percentages = []

    def _on_step(self) -> bool:
        self.steps += 1
        self.total_decisions += 1  # Increment total decisions

        # Extract collision, speed, and LLM match from 'info'
        collision = self.locals['infos'][0].get('crashed', False)
        average_speed = self.locals['infos'][0].get('speed', 0)
        llm_reward = self.locals['infos'][0].get('llm_reward', 0)

        # Update metrics
        self.collision_count += int(collision)
        self.total_speed += average_speed
        self.match_count += int(llm_reward == 1)

        if self.steps % self.check_freq == 0:
            # Calculate and store metrics
            collision_probability = self.collision_count / self.check_freq
            avg_speed = self.total_speed / self.check_freq
            match_percentage = (self.match_count / self.total_decisions) * 100
            self.collision_probabilities.append(collision_probability)
            self.average_speeds.append(avg_speed)
            self.match_percentages.append(match_percentage)
            print(f"Step: {self.steps}, Match Percentage: {match_percentage}%, Collision Probability: {collision_probability}, Average Speed: {avg_speed}")
            # Reset counters
            self.collision_count = 0
            self.total_speed = 0
            self.match_count = 0
            self.total_decisions = 0

        return True

    def _on_training_end(self):
        # Plot all metrics
        plt.figure(figsize=(18, 6))

        # Subplot for collision probabilities
        plt.subplot(1, 3, 1)
        plt.plot(self.collision_probabilities, label='Collision Probability', color='red')
        plt.xlabel('Checkpoints')
        plt.ylabel('Collision Probability')
        plt.title('Collision Probability Over Time')

        # Subplot for average speeds
        plt.subplot(1, 3, 2)
        plt.plot(self.average_speeds, label='Average Speed', color='orange')
        plt.xlabel('Checkpoints')
        plt.ylabel('Average Speed')
        plt.title('Average Speed Over Time')

        # Subplot for matching percentages
        plt.subplot(1, 3, 3)
        plt.plot(self.match_percentages, label='Matching Percentage', color='blue')
        plt.xlabel('Checkpoints')
        plt.ylabel('Matching Percentage (%)')
        plt.title('Matching Percentage Over Time')

        plt.tight_layout()
        plt.legend()
        plt.show()
