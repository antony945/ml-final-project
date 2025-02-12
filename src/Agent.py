import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt
import os

class Agent:
    def __init__(
        self,
        env: gym.Env,
        n_episodes: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        load_from_file: bool = False,
    ):
        # Initialize agent
        self.env = env
        self.n_episodes = n_episodes
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        _, self.info = env.reset()
        self.training_error = []

        # If obs space is continuous we must discretize it
        self.obs_spaces_size = -1
        self.obs_spaces = []
        self.divisions = 5

        # Handle Discrete obs space (e.g. FrozenLake)
        if (isinstance(self.env.observation_space, gym.spaces.Discrete)):
            self.obs_spaces_size = self.env.observation_space.n
            
        # Handle other cases (e.g. FlappyBird, ..)
        else:
            self._handle_box_space(self.env.observation_space, self.obs_spaces, self.divisions)

        # Create filename for the q-table based on the env
        self.filename = self._create_filename(self.env)

        # Create q_table or load it from a file if specified
        self.q_table = None

        # Load from model
        if load_from_file:
            with open(self.filename, "rb") as f:
                # Load dict table
                self.q_table = pickle.load(f)
            
        if self.q_table is not None:
            # Transform dict table into default dict
            self.q_table = defaultdict(self._create_numpy_entry, self.q_table) # can't use lambda due to pickle not able to "pickle" an anon function
        else:
            # Create new default dict
            # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
            self.q_table = defaultdict(self._create_numpy_entry) # can't use lambda due to pickle not able to "pickle" an anon function

    def _create_filename(self, env: gym.Env):
        MODELS_DIRECTORY = "models"

        self.env_basename = f"{env.spec.name}"
        self.env_fullname = self.env_basename

        # Create different filename for different models 
        for k,v in env.spec.kwargs.items():
            if k != "render_mode":
                self.env_fullname += f"_{v}"

        return f"{MODELS_DIRECTORY}/{self.env_fullname}.pk1"

    def _handle_box_space(self, obs_space: gym.spaces.Box, obs_spaces: list, divisions = 10):
        # Flatten the box
        original_obs_space = gym.spaces.flatten_space(obs_space)

        # print(self.env.observation_space.shape)
        for i in range(0, original_obs_space.shape[0]):
            # When an obs will be discretized it could have values between 0 and divisions included (divisions+1 total values)
            space = np.linspace(original_obs_space.low[i], original_obs_space.high[i], divisions)
            obs_spaces.append(space)

        # self.divisions+1 * obs_space.shape spaces
        # q_table will have obs_spaces_size x action_spaces
        # print(f"How many obs spaces: {len(self.obs_spaces)}")
        self.obs_spaces_size = 1
        for i, s in enumerate(self.obs_spaces):
            # print(f"{i}) length: {len(s)}")
            # print(f"{i}) s: {s}")
            self.obs_spaces_size *= (len(s)+1)
            # print(f"size = {self.obs_spaces_size}")

    def _create_numpy_entry(self):
            return np.zeros(self.env.action_space.n)

    def get_action(self, obs, is_training=True) -> int:
        if is_training and np.random.random() < self.epsilon:
            # Exploration -> Choose random action
            return self.env.action_space.sample()
        else:
            # Exploitation -> Follow Q-table
            obs = self._discretize_obs(obs)
            return int(np.argmax(self.q_table[obs]))

    def _discretize_obs(self, obs):
        # Don't discretize if obs is only one integer
        if (isinstance(obs, int)):
            return obs
        # Index where to insert obs
        obs_idx = sum([np.digitize(obs[i], self.obs_spaces[i]) * (len(self.obs_spaces[i]) ** i) for i in range(len(obs))])
        return obs_idx

    def update(self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
        is_training=True
    ):
        if not is_training: return

        obs = self._discretize_obs(obs)
        next_obs = self._discretize_obs(next_obs)

        # Update Q-table after having performed action
        future_q_value = (not terminated) * np.max(self.q_table[next_obs])
        current_q_value = self.q_table[obs][action]
        temporal_difference = reward + self.discount_factor * future_q_value - current_q_value
        self.q_table[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        # Linear decay
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_table(self):
        print(f"MAX Q_TABLE SIZE: {self.obs_spaces_size} x {self.env.action_space.n}")
        print(f"ACTUAL Q_TABLE SIZE: {len(self.q_table)} x {self.env.action_space.n}")

        with open(self.filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def plot_results(self, mean_rewards, epsilon_values, training = False, integrated = False):
        RESULTS_DIRECTORY = "results"
        
        if (integrated):
            # print(f'Episode time taken: {env.time_queue}')
            # print(f'Episode total rewards: {env.return_queue}')
            # print(f'Episode lengths: {env.length_queue}')

            # visualize the episode rewards, episode length and training error in one figure
            fig, axs = plt.subplots(1, 3, figsize=(20, 8))

            # np.convolve will compute the rolling mean for 100 episodes
            axs[0].plot(np.convolve(self.env.return_queue, np.ones(100)))
            axs[0].set_title("Episode Rewards")
            axs[0].set_xlabel("Episode")
            axs[0].set_ylabel("Reward")

            axs[1].plot(np.convolve(self.env.length_queue, np.ones(100)))
            axs[1].set_title("Episode Lengths")
            axs[1].set_xlabel("Episode")
            axs[1].set_ylabel("Length")

            if len(self.training_error) > 0:
                axs[2].plot(np.convolve(self.training_error, np.ones(100)))
                axs[2].set_title("Training Error")
                axs[2].set_xlabel("Episode")
                axs[2].set_ylabel("Temporal Difference")
        else:
            fig, ax1 = plt.subplots()
            lines = []

            # Plot mean_rewards on the primary y-axis
            line1, = ax1.plot(mean_rewards, label="Mean Reward", color="b")
            lines.append(line1)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.tick_params(axis="y")

            if epsilon_values is not None:
                # Create a second y-axis
                ax2 = ax1.twinx()
                line2, = ax2.plot(epsilon_values, label="Epsilon Decay", color="r", linestyle="dashed")
                lines.append(line2)
                ax2.set_ylabel("Epsilon")
                ax2.tick_params(axis="y")

        # Combine legends
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")  # Single legend in the best position

        plt.title(f"N={len(mean_rewards)}, LR={self.learning_rate}, E_DECAY={self.epsilon_decay}, DISCOUNT={self.discount_factor}, DIV={self.divisions}")
        
        if training:
            filename = f"{RESULTS_DIRECTORY}/{self.env_fullname}_{len(mean_rewards)}_training_results.png"
        else:
            filename = f"{RESULTS_DIRECTORY}/{self.env_fullname}_{len(mean_rewards)}_test_results.png"

        plt.savefig(filename)
        plt.tight_layout()
        plt.show()