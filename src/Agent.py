import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt
import os
import re
import abc
import torch
from torch import nn
import torch.nn.functional as F
from ReplayMemory import ReplayMemory
from tqdm import tqdm
import itertools
from datetime import datetime
import time
import csv

class Agent:
    """
    Abstract base class for reinforcement learning agents.

    This class defines the common structure for agents interacting with an OpenAI Gym environment.
    It includes methods for selecting actions, updating the agent's knowledge, and saving models.
    Subclasses must implement the abstract methods `get_action`, `update`, and `save_model`.

    Methods:
        get_action(obs, is_training=True) -> int:
            Selects an action based on the given observation.
        
        update(obs, action, reward, terminated, next_obs, is_training=True):
            Updates the agent's knowledge based on the experience tuple.
        
        save_model():
            Saves the trained model to a file.
    """

    __metaclass__ = abc.ABCMeta
    MODELS_DIRECTORY = "models"
    RESULTS_DIRECTORY = "results"
    LOGS_DIRECTORY = "logs"
    RUNS_DIRECTORY = "runs"
    TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
    
    def initialize(self,
        env: gym.Env,
        hyperparameters: dict,
    ):
        # Initialize agent
        self.env = env

        # Create different filename for different models 
        self.env_basename = f"{self.env.spec.name}"
        self.env_fullname = self.env_basename
        for k,v in env.spec.kwargs.items():
            if k != "render_mode":
                self.env_fullname += f"_{v}"

        # Create model filename
        self.filename = f"{self.env_basename}"

        # Initialize hyperparameters
        self.hyperparameters = hyperparameters
        self.learning_rate =        self.hyperparameters.get('learning_rate', 0.001)
        self.epsilon =              self.hyperparameters.get('epsilon_init', 1)
        self.epsilon_decay =        self.hyperparameters.get('epsilon_decay', 0.99995)
        self.final_epsilon =        self.hyperparameters.get('epsilon_final', 0.01)
        self.discount_factor =      self.hyperparameters.get('discount_factor', None)
        self.stop_on_reward =       self.hyperparameters.get('stop_on_reward', None)
        self.enable_ER =            self.hyperparameters.get('enable_ER', False)
        self.enable_PER =           self.hyperparameters.get('enable_PER', False)
        self.epsilon_PER =          self.hyperparameters.get('epsilon_PER', 0.001)
        self.alpha_PER =            self.hyperparameters.get('alpha_PER', 0.6)
        self.beta_init_PER =        self.hyperparameters.get('beta_init_PER', 0.4)
        self.mini_batch_size =      self.hyperparameters.get('mini_batch_size', 128)
        self.min_memory_size =      self.hyperparameters.get('min_memory_size', 128)
        self.max_memory_size =      self.hyperparameters.get('max_memory_size', 131_072)
        self.lazy_update =          self.hyperparameters.get('lazy_update', False)

        self.beta_PER = self.beta_init_PER

        # Initialize loss values
        self.loss_per_episode = []
        # Useful for DQN_Agent
        self.synced = False
        self.add_parameters = {}

    def decay_epsilon(self, is_training):
        if is_training:
            self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)

    def get_largest_model_name(self, filename: str, extension: str, n_episodes: int | None = None):
        if n_episodes is not None:
            return os.path.join(Agent.MODELS_DIRECTORY, f"{filename}_{n_episodes}.{extension}")
        else:
            # Get the list of all files that match the base filename pattern
            pattern = rf"{filename}_(\d+)\.{extension}"
            files = [f for f in os.listdir(Agent.MODELS_DIRECTORY) if re.match(pattern, f)]
            
            # Find the largest episode number
            largest_episode = -1
            largest_model = None
            for file in files:
                match = re.search(pattern, file)
                if match:
                    episode_number = int(match.group(1))
                    if episode_number > largest_episode:
                        largest_episode = episode_number
                        largest_model = file

            # TODO: Handle case when largest model is still None            
            return os.path.join(Agent.MODELS_DIRECTORY, largest_model)

    def create_new_name(self, training: bool, n_episodes: int | None, additional_parameters: dict | None = None) -> str:
        agent_type = "Q" if isinstance(self, Q_Agent) else "DQN"
        lr = f"LR={self.learning_rate}"
        df = f"DF={self.discount_factor}"
        eps_decay = f"eDECAY={self.epsilon_decay}"
        eps_final = f"eFIN={self.final_epsilon}"

        if self.enable_PER:
            memory = "PER"
            mem = f"MEM={memory}_BATCH={self.mini_batch_size}"
        elif self.enable_ER:
            memory = "ER"
            mem = f"MEM={memory}_BATCH={self.mini_batch_size}"
        else:
            memory = "None"
            mem = f"MEM={memory}"

        # Compose filename
        filename = self.env_basename

        if training:
            filename += "_training"
        else:
            filename += "_test"

        if n_episodes is None:
            filename += "_temp"
            n = "_N=-1"
        else:
            n = f"_N={n_episodes}"

        filename += f"_{agent_type}_{lr}_{df}_{eps_decay}_{eps_final}_{mem}"

        if additional_parameters is not None:
            for k,v in additional_parameters.items():
                filename += f"_{k}={v}"

        filename += n

        return filename

    def create_new_modelfile_name(self, extension: str, n_episodes: int | None, additional_parameters: dict | None = None) -> str:
        filename = self.create_new_name(training=True, n_episodes=n_episodes, additional_parameters=additional_parameters)
        filename += f".{extension}"
        return os.path.join(Agent.MODELS_DIRECTORY, filename)

    def log_hyperparameters(self, hyperparameters: dict):
        log_msg = f"{datetime.now().strftime(Agent.TIME_FORMAT)}: HYPERPARAMETERS"
        print(log_msg)

        for k,v in hyperparameters.items():
            log_msg = f"{datetime.now().strftime(Agent.TIME_FORMAT)}: {k} = {v}"
            print(log_msg)

    def create_new_runfile_name(self, training: bool, additional_parameters: dict | None = None) -> str:
        name = f"{datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")}_{self.create_new_name(training=training, n_episodes=-1, additional_parameters=additional_parameters)}"
        return os.path.join(Agent.RUNS_DIRECTORY, f"{name}.csv")

    def initialize_runfile(self, training: bool):
        # Create runfile
        self.runfile = self.create_new_runfile_name(training=training, additional_parameters=self.add_parameters)
        with open(self.runfile, mode="w", newline="") as file:
            writer = csv.writer(file)            
            writer.writerow(["timestamp", "episode", "epsilon", "frames", "avg_frames", "reward", "avg_reward", "best_reward"])

    def log_episode_to_csv(self, episode, epsilon, frames, avg_frames, reward, avg_reward, best_reward):
        timestamp = datetime.now().isoformat()  # Current timestamp
        row = [timestamp, episode, epsilon, frames, avg_frames, reward, avg_reward, best_reward]

        # Append to CSV file
        with open(self.runfile, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def _internal_plot(self, all_rewards, mean_rewards, epsilon_values, training = False, temp = False, show = True, integrated = False, additional_parameters: dict | None = None):
        if (integrated):
            # visualize the episode length and training error in one figure
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

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
            # Create a second subplot for loss and epsilon decay
            if training:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                ax2.grid(True)
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
                
            ax1.grid(True)

            colors = ["b", "r", "g"]

            # Plot rewards and epsilon decay on the first subplot
            lines = []
            line0, = ax1.plot(all_rewards, label="Episode Reward", color=colors[0], alpha=0.2)
            lines.append(line0)
            line1, = ax1.plot(mean_rewards, label="Mean Reward", color=colors[0])
            lines.append(line1)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.tick_params(axis="y")

            if epsilon_values is not None:
                ax3 = ax1.twinx()
                line2, = ax3.plot(epsilon_values, label="Epsilon Decay", color=colors[1], linestyle="dashed")
                lines.append(line2)
                ax3.set_ylabel("Epsilon")
                ax3.tick_params(axis="y")

            # Combine legends
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="best")  # Single legend in the best position

            # Plot loss and epsilon decay on the second subplot
            if training:
                # Calulate mean loss for the last 100 episodes
                mean_loss_100 = [np.mean(self.loss_per_episode[max(0, i-100):i+1]) for i in range(len(self.loss_per_episode))]
                lines = []
                
                line3, = ax2.plot(self.loss_per_episode, label="MSE Loss", color=colors[2], alpha=0.2)
                lines.append(line3)
                line4, = ax2.plot(mean_loss_100, label="Mean Loss", color=colors[2])
                lines.append(line4)

                ax2.set_xlabel("Episode")
                ax2.set_ylabel("Loss")
                ax2.tick_params(axis="y")

                if epsilon_values is not None:
                    ax4 = ax2.twinx()
                    line5, = ax4.plot(epsilon_values, label="Epsilon Decay", color=colors[1], linestyle="dashed")
                    lines.append(line5)
                    ax4.set_ylabel("Epsilon")
                    ax4.tick_params(axis="y")

                # Combine legends
                labels = [line.get_label() for line in lines]
                ax2.legend(lines, labels, loc="best")  # Single legend in the best position

        # Create title with additional parameters
        title = self.create_new_name(n_episodes=len(mean_rewards), training=training, additional_parameters=additional_parameters)
        plt.title(title)

        if temp:
            n = None
        elif training:
            n = len(mean_rewards)
        else:
            n = self.hyperparameters.get("n_episodes", None)

        filename = self.create_new_name(n_episodes=n, training=training, additional_parameters=additional_parameters) + ".png"
        filename = os.path.join(Agent.RESULTS_DIRECTORY, filename) 

        plt.savefig(filename)
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

    def store_and_sample_memory(self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
        is_training=True,
    ):
        if not is_training: return None

        # Current observation
        current_row = (obs, action, reward, terminated, next_obs)
        
        # Depending on memory enabled or not, handle the update
        if self.enable_ER or self.enable_PER:
            # Append memory
            self.memory.append(current_row)

            # If lazy_update (update after every episode, not every step), stop here
            if self.lazy_update: return None

            # Otherwise check if sampling from memory is possible
            return self.sample_memory()
        else:
            # If experience replay not enabled, process just current observation
            return ([current_row], None, None)

    def sample_memory(self):
        """
        :return: (batch, importance weights, indices)
        """

        # Check if we collected enough experience, if not don't update
        if len(self.memory) <= self.min_memory_size: return None

        # If so, sample batch size from memory to reduce correlation between consecutive experiences 
        mini_batch, weights, indices = self.memory.sample(self.mini_batch_size)

        return mini_batch, weights, indices

    def optimize_update(self, mini_batch, indices):
        if (mini_batch is None): return

        td_errors, loss = self.optimize(mini_batch)
        # Sum the loss for the current episode
        if len(self.loss_per_episode) == self.episode+1:
            self.loss_per_episode[self.episode] += loss
        else:
            self.loss_per_episode.append(loss)

        if (self.enable_PER):
            self.memory.update_priorities(indices, np.abs(td_errors))

    @abc.abstractmethod
    def get_action(self, obs, is_training=True) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
        is_training=True,
    ):
        raise NotImplementedError
    
    @abc.abstractmethod
    def optimize(self, minibatch):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, model_name: str):    
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, n_episodes: int | None):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, n_episodes: int | None, is_training = True, show_plots = True, verbose = True, model_name = None, seed = None):
        if is_training:
            # Handle experience replay if enabled    
            if self.enable_ER or self.enable_PER:
                # Create the buffer for experience replay
                self.memory = ReplayMemory(
                    capacity=self.max_memory_size,
                    use_priority=self.enable_PER,
                    alpha=self.alpha_PER,
                    beta=self.beta_init_PER,
                    epsilon=self.epsilon_PER,
                    seed=seed)
            
            start_time = datetime.now()

            # Debug
            self.log_hyperparameters(self.hyperparameters)
            log_msg = f"{start_time.strftime(Agent.TIME_FORMAT)}: STARTED TRAINING"
            print(log_msg)

        # Initialize run file
        self.initialize_runfile(training=is_training)

        # Reset environment
        _, self.info = self.env.reset(seed=seed)

        # Track rewards during episodes
        rewards_per_episode = []
        avg_rewards_per_episode = []
        best_reward = -999999

        # Track epsilon values during episodes
        epsilon_values = []

        # Track frames (no. of steps) during episodes
        frames_per_episode = []
        avg_frames_per_episode = []

        if n_episodes is not None:
            # Fixed number of episodes
            iterable = range(n_episodes)
            if not verbose:
                iterable = tqdm(iterable)
        else:
            # "Infinite" number of episodes
            iterable = itertools.count()

        try:
            # Loop over episodes
            for self.episode in iterable:
                obs, info = self.env.reset(seed=seed)
                done = False
                rewards = 0
                frames = 0
                self.synced = False

                # Play one episode
                while not done:
                    # Choose action based using e-greedy strategy
                    action = self.get_action(obs, is_training=is_training)                    

                    # Perform action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)

                    # Update the agent model
                    self.update(obs, action, reward, terminated, next_obs, is_training=is_training)

                    # Update environment and collect reward
                    done = terminated or truncated
                    obs = next_obs
                    rewards += reward
                    frames += 1

                rewards_per_episode.append(rewards)
                avg_reward = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
                avg_rewards_per_episode.append(avg_reward)
                epsilon_values.append(self.epsilon)
                
                frames_per_episode.append(frames)
                avg_frames = np.mean(frames_per_episode[len(frames_per_episode)-100:])
                avg_frames_per_episode.append(avg_frames)

                # ONLY IN TRAINING
                # At the end of episode handle rewards
                if rewards > best_reward:
                    log_msg = f"{datetime.now().strftime(Agent.TIME_FORMAT)}: Episode {self.episode}) Steps: {frames}, New Best Reward {rewards:.2f} ({((rewards-best_reward)/abs(best_reward)*100):+.1f}%)"

                    if verbose:
                        print(log_msg)

                    best_reward = rewards
                                       
                    if is_training:
                        # Save also model but without specifying n_episodes (so it will override everytime)
                        self.save_model(n_episodes = None)

                        # Exit loop on a particular reward
                        if self.stop_on_reward is not None and best_reward >= self.stop_on_reward: 
                            break

                # Append episode row to log
                self.log_episode_to_csv(self.episode, self.epsilon, frames, avg_frames, rewards, avg_reward, best_reward)

                # If is not training continue to next episode
                if not is_training:
                    continue

                # Debug every 100 episodes
                if self.episode%100==0:
                    log_msg = f"{datetime.now().strftime(Agent.TIME_FORMAT)}: Episode {self.episode}) Epsilon: {self.epsilon:0.3f}, Mean Steps: {avg_frames}, Reward: {rewards:.2f}, Best Reward: {best_reward:.2f}, Mean Reward {avg_reward:.2f}"
                    if self.enable_PER:
                        log_msg += f", Beta: {self.memory.beta:.3f}"
                
                    if verbose:
                        print(log_msg)
                    
                # Save graph every 1000 episodes
                if self.episode%1000==0:
                    self.plot_results(rewards_per_episode, avg_rewards_per_episode, epsilon_values, training=is_training, temp=True, show=False, integrated=False)

                # Decay epsilon and perform update here if lazy update is enabled
                self.decay_epsilon(is_training)

                # Decide if to perform update at the end of episode
                if self.lazy_update and (self.enable_ER or self.enable_PER):
                    tuple_returned = self.sample_memory()
                    if tuple_returned is not None:
                        mini_batch, weights, indices = tuple_returned
                        self.optimize_update(mini_batch, indices)

        except KeyboardInterrupt:
            print("\nCTRL+C pressed.\n")
        finally:
            # Close the env
            self.env.close()

        # Save model on file specifying also n_iterations
        if is_training:
            end_time = datetime.now()
            log_msg = f"{end_time.strftime(Agent.TIME_FORMAT)}: ENDED TRAINING. Training took {str(end_time-start_time).split('.')[0]}"
            print(log_msg)

            self.save_model(n_episodes=len(avg_rewards_per_episode))

        # Display plots
        if show_plots:
            epsilon_values = None if not is_training else epsilon_values
            self.plot_results(rewards_per_episode, avg_rewards_per_episode, epsilon_values, training=is_training, temp=False, show=show_plots, integrated=False)

    @abc.abstractmethod
    def plot_results(self, all_rewards, mean_rewards, epsilon_values, training = False, temp=False, show = True, integrated = False):
        raise NotImplementedError

class Q_Agent(Agent):
    """
    Reinforcement learning agent using the Q-learning algorithm.

    This agent learns an optimal policy by maintaining a Q-table, which stores the expected rewards 
    for state-action pairs. It updates the Q-values and follows an 
    exploration-exploitation strategy to balance learning and performance.
    """

    MODEL_EXTENSION = "pk1"

    def __init__(self,
        env: gym.Env,
        hyperparameters: dict,
    ):
        super().initialize(env, hyperparameters)
        self.in_batch = True

        # For deciding how to discretize continuous observations
        self.divisions = self.hyperparameters.get("divisions", 10)
        self.training_error = []

        # If obs space is continuous we must discretize it
        self.obs_spaces_size = -1
        self.obs_spaces = []

        # Handle Discrete obs space (e.g. FrozenLake)
        if (isinstance(self.env.observation_space, gym.spaces.Discrete)):
            self.obs_spaces_size = self.env.observation_space.n
            
        # Handle other cases (e.g. FlappyBird, ..)
        else:
            self._handle_box_space(self.env.observation_space, self.obs_spaces, self.divisions)

        # Used for stored q_table
        self.q_table = None

        self.add_parameters = {
            "DIV": self.divisions
        }

    def _handle_box_space(self, obs_space: gym.spaces.Box, obs_spaces: list, divisions = 10):
        # Flatten the box
        original_obs_space = gym.spaces.flatten_space(obs_space)

        for i in range(0, original_obs_space.shape[0]):
            # When an obs will be discretized it could have values between 0 and divisions included (divisions+1 total values)
            space = np.linspace(original_obs_space.low[i], original_obs_space.high[i], divisions)
            obs_spaces.append(space)

        # Q_table will have obs_spaces_size x action_spaces
        self.obs_spaces_size = 1
        for i, s in enumerate(self.obs_spaces):
            self.obs_spaces_size *= (len(s)+1)

    def _create_numpy_entry(self):
            return np.zeros(self.env.action_space.n)
    
    def _discretize_obs(self, obs):
        # Don't discretize if obs is only one integer
        if (isinstance(obs, int)):
            return obs
        # Index where to insert obs
        # Finds where the observation obs[i] falls among predefined bins
        # Sum all the indexes to get a unique index for the whole observation
        obs_idx = sum([np.digitize(obs[i], self.obs_spaces[i]) * (len(self.obs_spaces[i]) ** i) for i in range(len(obs))])
        return obs_idx

    # ==================================== Superclass methods
    
    def get_action(self, obs, is_training=True) -> int:
        if is_training and np.random.random() < self.epsilon:
            # Exploration -> Choose random action
            return self.env.action_space.sample()
        else:
            # Exploitation -> Follow Q-table
            obs = self._discretize_obs(obs)
            return int(np.argmax(self.q_table[obs]))

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

        # Call parent update to get mini batch to process
        tuple_returned = super().store_and_sample_memory(obs, action, reward, terminated, next_obs, is_training)
        if tuple_returned is None or self.lazy_update: return

        mini_batch, weights, indices = tuple_returned

        # Update Q-table after having performed action
        self.optimize_update(mini_batch, indices)

    def optimize(self, mini_batch):
        # Process mini batch all at once to reduce time
        observations, actions, rewards, terminations, next_observations = zip(*mini_batch)

        # Convert tuples to NumPy arrays where possible for efficient batch operations (q_table as defaultdict will not accept numpy array as keys)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminations = np.array(terminations, dtype=int)

        # Compute max Q-values for next states (set to 0 if terminated)
        future_q_values = (1-terminations) * np.array([max(self.q_table[next_obs]) for next_obs in next_observations])

        # Compute target Q-values
        q_targets = rewards + self.discount_factor * future_q_values

        # Compute temporal differences
        current_q_values = np.array([self.q_table[obs][action] for obs, action in zip(observations, actions)])
        td_errors = q_targets - current_q_values

        # Update Q-table using NumPy arrays (vectorized update)
        updates = self.learning_rate * td_errors
        for obs, action, update in zip(observations, actions, updates):
            self.q_table[obs][action] += update

        # Implement loss as the mean squared error
        total_loss = np.mean((q_targets - current_q_values) ** 2)
        return td_errors, total_loss

    def load_model(self, model_name: str):
        self.model_name = model_name
        print(f"Loading Q-Table from: '{self.model_name}'..")

        with open(self.model_name, "rb") as f:
            # Load dict representation of q-table
            self.q_table = pickle.load(f)

    def save_model(self, n_episodes: int | None):
        """
        Save trained model. If n_episodes is None it's meant to be a temporary version.
        """
        self.model_name = self.create_new_modelfile_name(Q_Agent.MODEL_EXTENSION, n_episodes, self.add_parameters)

        # Print only when saving non-temporary models
        if n_episodes is not None:
            print(f"Saving Q-Table to: '{self.model_name}'..")
            print(f"Q-Table size: {len(self.q_table)} x {self.env.action_space.n}")

        with open(self.model_name, "wb") as f:
            # Save dict representation of q-table 
            pickle.dump(dict(self.q_table), f)

    def run(self, n_episodes: int | None, is_training = True, show_plots = True, verbose = True, model_name = None, seed = None):
        # Load from model if is not training
        if not is_training:
            self.load_model(model_name=model_name)
            
        if self.q_table is not None:
            # Transform dict table into default dict
            self.q_table = defaultdict(self._create_numpy_entry, self.q_table) # can't use lambda due to pickle not able to "pickle" an anon function
            print(f"Q-Table size: {len(self.q_table)} x {self.env.action_space.n} (instead of {self.obs_spaces_size} x {self.env.action_space.n})")
        else:
            # Create new default dict
            # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
            self.q_table = defaultdict(self._create_numpy_entry) # can't use lambda due to pickle not able to "pickle" an anon function

        super().run(n_episodes, is_training, show_plots, verbose, model_name, seed)

    def plot_results(self, all_rewards, mean_rewards, epsilon_values, training = False, temp = False, show = True, integrated = False):

        self._internal_plot(all_rewards, mean_rewards, epsilon_values, training, temp, show, integrated, self.add_parameters)

class DQN_Agent(Agent):
    """
    Deep Q-Network (DQN) reinforcement learning agent.

    This agent utilizes a neural network to approximate Q-values instead of using a traditional Q-table.
    It learns an optimal policy by training a deep neural network on state-action-reward transitions,
    improving over time with experience replay and target networks.
    """

    MODEL_EXTENSION = "pt"

    class DQN(nn.Module):
        def __init__(self,
            state_dim,
            action_dim,
            hidden_dim = 128,
            dueling_dqn = False
        ):
            # nn.Module init method
            super().__init__()

            self.dueling_dqn = dueling_dqn

            # Create layers: #states x #hidden x #actions
            self.fc1 = nn.Linear(state_dim, hidden_dim)

            if self.dueling_dqn:
                # Value stream
                self.fc2_val = nn.Linear(hidden_dim, hidden_dim)
                self.val = nn.Linear(hidden_dim, 1)
                # Advantage stream
                self.fc2_adv = nn.Linear(hidden_dim, hidden_dim)
                self.adv = nn.Linear(hidden_dim, action_dim)
            else:
                self.fc2 = nn.Linear(hidden_dim, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))

            if self.dueling_dqn:
                v = F.relu(self.fc2_val(x))
                V = self.val(v)
                a = F.relu(self.fc2_adv(x))
                A = self.adv(a)
                q = V + A - torch.mean(A, dim=1, keepdim=True)
            else:
                q = self.fc2(x)

            return q

    def __init__(self,
        env: gym.Env,
        hyperparameters: dict,
    ):
        # Agent initialize method
        super().initialize(env, hyperparameters)

        self.double_dqn =           self.hyperparameters.get('double_dqn', False)
        self.dueling_dqn =          self.hyperparameters.get('dueling_dqn', False)
        self.network_sync_rate =    self.hyperparameters.get('network_sync_rate', 100)
        self.hidden_dim =           self.hyperparameters.get('fc1_nodes', 128)
        self.device =               self.hyperparameters.get('device', "cpu")

        self.add_parameters = {
            "dDQN": self.double_dqn,
            "duelDQN": self.dueling_dqn,
            "HID": self.hidden_dim,
            "DEV": self.device
        }
        
        # Reset to cpu if cuda is not available
        if not torch.cuda.is_available():
            self.device = 'cpu'

        # Create DQN
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy_dqn = self.DQN(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.dueling_dqn
        ).to(self.device)

        # NN loss function, MSE = Mean Squared Error
        self.loss_fn = nn.MSELoss()

        # Policy network optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_dqn.parameters(),
            lr=self.learning_rate
        )

        print(self.policy_dqn)

    # ==================================== Superclass methods

    def get_action(self, obs, is_training=True) -> int:
        if is_training and np.random.random() < self.epsilon:
            # Exploration -> Choose random action
            action = self.env.action_space.sample()
        else:
            # Convert obs to tensor in device
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
            # Exploitation -> Follow DQN
            with torch.no_grad():
                # tensor([1,2,3,..]) --> tensor([[1,2,3,..]])
                action = self.policy_dqn(obs.unsqueeze(dim=0)).squeeze().argmax().item()
        
        return action

    def update(self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
        is_training = True,
    ):
        if not is_training: return

        # Convert obs, action, reward and next_obs to tensor in device
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=self.device)
        
        # Call parent update to get mini batch to process
        tuple_returned = super().store_and_sample_memory(obs, action, reward, terminated, next_obs, is_training)
        if tuple_returned is None or self.lazy_update: return

        mini_batch, weights, indices = tuple_returned

        # Optimize target and policy dqn
        self.optimize_update(mini_batch, indices)

    def optimize(self, mini_batch):
        # Process mini batch all at once to reduce time
        observations, actions, rewards, terminations, next_observations = zip(*mini_batch)

        # Stack tensors to create batch tensors, tensor([[1,2,3,..]])
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_observations = torch.stack(next_observations)
        # Convert true/false in 1.0/0.0
        terminations = torch.tensor(terminations).float().to(self.device)

        # Calculate target q values (expected returns)
        with torch.no_grad():    
            if self.double_dqn:
                # Select best action based on next observations in policy dqn, not target dqn
                best_actions = self.policy_dqn(next_observations).argmax(dim=1)
                # Then apply those actions to target dqn
                target_q = rewards + (1-terminations) * self.discount_factor * self.target_dqn(next_observations).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
            else:
                # Here simply select the max q-value from policy dqn
                target_q = rewards + (1-terminations) * self.discount_factor * self.policy_dqn(next_observations).max(dim=1)[0]
                '''
                    policy_dqn(next_observations)   --> tensor([[.2, .8], [.3, .6], [.1, .4]])
                        .max(dim=1)                 --> torch.return_types.max(values=tensor([.8,.6,.4]), indices=tensor([1,1,1]))
                            [0]                     --> tensor([.8,.6,.4])
                '''

        # Calculate q values from current policy (use actions done as index in the tensors)
        current_q = self.policy_dqn(observations).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            Supposing we have actions = tensor([1, 0, 1])

            policy_dqn(observations)                            --> tensor([[.2, .8], [.3, .6], [.1, .4]])
                .gather(dim=1, index=actions.unsqueeze(dim=1))  --> tensor([[.8], [.3], [.4]]) values correspondent to actions
                    .squeeze()                                  --> tensor([.8, .3, .4])
        '''

        # Compute loss for the whole mini batch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients (backpropagation)
        self.optimizer.step()       # Update network parameters (weight and biases)

        # Copy policy network to target network after a certain number of epsiodes
        if self.double_dqn and self.episode % self.network_sync_rate == 0 and not self.synced:
            print(F"Syncing policy and target networks at episode {self.episode}..")
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
            self.synced = True

        # Compute TD errors and convert to NumPy array
        td_errors = (target_q - current_q).detach().cpu().numpy()
        return td_errors, loss.item()
    
    def load_model(self, model_name: str):
        self.model_name = model_name
        print(f"Loading DQN from: '{self.model_name}'..")

        # Load policy from file
        self.policy_dqn.load_state_dict(torch.load(self.model_name))

    def save_model(self, n_episodes: int | None):
        self.model_name = self.create_new_modelfile_name(DQN_Agent.MODEL_EXTENSION, n_episodes, self.add_parameters)

        # Print only when saving non-temprary models
        if n_episodes is not None:
            print(f"Saving DQN to: '{self.model_name}'..")
        
        # Save policy to file
        torch.save(self.policy_dqn.state_dict(), self.model_name)

    def run(self, n_episodes: int | None, is_training = True, show_plots = True, verbose = True, model_name = None, seed = None):
        if is_training:
            # Create target dqn used while training
            if self.double_dqn:
                self.target_dqn = self.DQN(self.state_dim, self.action_dim, self.hidden_dim, self.dueling_dqn).to(self.device)
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        else:
            # Load learned policy
            self.load_model(model_name=model_name)

            # Switch model to evaluation mode
            self.policy_dqn.eval()

        super().run(n_episodes, is_training, show_plots, verbose, model_name, seed)

    def plot_results(self, all_rewards, mean_rewards, epsilon_values, training = False, temp = False, show = True, integrated = False):

        self._internal_plot(all_rewards, mean_rewards, epsilon_values, training, temp, show, integrated, self.add_parameters)