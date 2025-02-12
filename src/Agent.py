import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt
import os
import abc
import torch
from torch import nn
import torch.nn.functional as F
from ReplayMemory import ReplayMemory
from tqdm import tqdm
import itertools
import yaml
from datetime import datetime

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
    TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
    
    def initialize(self,
        env: gym.Env,
        hyperparameters: dict,
        discount_factor: float = 0.95
    ):
        # Initialize agent
        self.env = env

        # Create different filename for different models 
        self.env_basename = f"{self.env.spec.name}"
        self.env_fullname = self.env_basename
        for k,v in env.spec.kwargs.items():
            if k != "render_mode":
                self.env_fullname += f"_{v}"

        # Create logfile
        self.logfile = os.path.join(Agent.LOGS_DIRECTORY, f"{self.env_fullname}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log")

        # Initialize hyperparameters
        self.hyperparameters = hyperparameters
        self.learning_rate =    self.hyperparameters['learning_rate']
        self.epsilon =          self.hyperparameters['epsilon_init']
        self.epsilon_decay =    self.hyperparameters['epsilon_decay']
        self.final_epsilon =    self.hyperparameters['epsilon_min']
        self.discount_factor =  discount_factor

    def decay_epsilon(self, is_training):
        # Linear decay
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)

    def plot_results(self, mean_rewards, epsilon_values, training = False, integrated = False):
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

        plt.title(f"N={len(mean_rewards)}, LR={self.learning_rate}, E_DECAY={self.epsilon_decay}, DISCOUNT={self.discount_factor}")
        
        if training:
            filename = os.path.join(Agent.RESULTS_DIRECTORY, f"{self.env_fullname}_{len(mean_rewards)}_training.png")
        else:
            filename = os.path.join(Agent.RESULTS_DIRECTORY, f"{self.env_fullname}_{len(mean_rewards)}_test.png")

        plt.savefig(filename)
        plt.tight_layout()
        plt.show()

    @abc.abstractmethod
    def get_action(self, obs, is_training=True) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
        is_training=True
    ):
        raise NotImplementedError
    
    @abc.abstractmethod
    def save_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, n_episodes: int | None, is_training = True, show_plots = True, verbose = True, seed = None):
        # Initialize log file
        if is_training:
            start_time = datetime.now()
            log_msg = f"{start_time.strftime(Agent.TIME_FORMAT)}: Training starting..."
            print(log_msg)
            with open(self.logfile, 'w') as file:
                file.write(log_msg + "\n")

        # Reset environment
        _, self.info = self.env.reset(seed=seed)

        # Track rewards during episodes
        rewards_per_episode = []
        mean_rewards = []
        best_reward = -999999

        # Track epsilon values during episodes
        epsilon_values = []

        if n_episodes is not None:
            # Fixed number of episodes
            iterable = range(n_episodes)
            if not verbose:
                iterable = tqdm(iterable)
        else:
            # "Infinite" number of episodes
            iterable = itertools.count()

        try:
            for episode in iterable:
                obs, info = self.env.reset(seed=seed)
                done = False
                rewards = 0

                # Play one episode
                while not done:
                    # Choose action
                    action = self.get_action(obs, is_training=is_training)                    
                    # print(f"{episode} - Agent choose action: {action}")

                    # Perform action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)

                    # Update the agent
                    self.update(obs, action, reward, terminated, next_obs, is_training=is_training)

                    # Update environment and collect reward
                    done = terminated or truncated
                    obs = next_obs
                    rewards += reward

                # At the end of episode handle rewards      
                if is_training and rewards > best_reward:
                    log_msg = f"{datetime.now().strftime(Agent.TIME_FORMAT)}: Episode {episode}) New Best Reward {rewards:.2f} ({((rewards-best_reward)/abs(best_reward)*100):+.1f}%), saving model..."
                    if verbose:
                        print(log_msg)
                    with open(self.logfile, 'a') as file:
                        file.write(log_msg + "\n")
                    
                    # Save also model
                    self.save_model()
                    best_reward = rewards
                
                rewards_per_episode.append(rewards)
                mean_reward = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
                mean_rewards.append(mean_reward)
                epsilon_values.append(self.epsilon)

                # Debug every 100 episodes
                if is_training and episode%100==0:
                    log_msg = f"{datetime.now().strftime(Agent.TIME_FORMAT)}: Episode {episode}) Epsilon: {self.epsilon:0.2f}, Reward: {rewards:.3f}, Best Reward: {best_reward:.2f}, Mean Reward {mean_reward:.2f}"
                    with open(self.logfile, 'a') as file:
                        file.write(log_msg + "\n")

                    if verbose:
                        print(log_msg)
                    
                    # Update also graph in that case
                    # self.plot_results(mean_rewards, epsilon_values, training=is_training, integrated=False)

                # Decay epsilon
                self.decay_epsilon(is_training)
        except KeyboardInterrupt:
            print("\nCTRL+C pressed.\n")
        finally:
            # Close the env
            self.env.close()

        # Save on file
        if is_training:
            self.save_model()

        # Display plots
        if show_plots:
            epsilon_values = None if not is_training else epsilon_values
            self.plot_results(mean_rewards, epsilon_values, training=is_training, integrated=False)

class Q_Agent(Agent):
    """
    Reinforcement learning agent using the Q-learning algorithm.

    This agent learns an optimal policy by maintaining a Q-table, which stores the expected rewards 
    for state-action pairs. It updates the Q-values and follows an 
    exploration-exploitation strategy to balance learning and performance.
    """

    def __init__(self,
        env: gym.Env,
        hyperparameters: dict,
        is_training: bool = True,
    ):
        super().initialize(env, hyperparameters)
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

        # Create q_table or load it from a file if specified
        self.q_table = None

        # Create filename
        self.filename = os.path.join(Agent.MODELS_DIRECTORY, f"{self.env_fullname}.pk1")

        # Load from model if is not training
        if not is_training:
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
    
    def _discretize_obs(self, obs):
        # Don't discretize if obs is only one integer
        if (isinstance(obs, int)):
            return obs
        # Index where to insert obs
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

        # Update Q-table after having performed action
        future_q_value = (not terminated) * np.max(self.q_table[next_obs])
        current_q_value = self.q_table[obs][action]
        temporal_difference = reward + self.discount_factor * future_q_value - current_q_value
        self.q_table[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    def save_model(self):
        print(f"MAX Q_TABLE SIZE: {self.obs_spaces_size} x {self.env.action_space.n}")
        print(f"ACTUAL Q_TABLE SIZE: {len(self.q_table)} x {self.env.action_space.n}")

        with open(self.filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def run(self, n_episodes: int | None, is_training = True, show_plots = True, verbose = True, seed = None):
        super().run(n_episodes, is_training, show_plots, verbose, seed)

class DQN(nn.Module):
    def __init__(self,
        state_dim,
        action_dim,
        hidden_dim = 256
    ):
        # nn.Module init method
        super().__init__()

        # Create layers: #states x #hidden x #actions
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN_Agent(Agent):
    """
    Deep Q-Network (DQN) reinforcement learning agent.

    This agent utilizes a neural network to approximate Q-values instead of using a traditional Q-table.
    It learns an optimal policy by training a deep neural network on state-action-reward transitions,
    improving over time with experience replay and target networks.
    """

    def __init__(self,
        env: gym.Env,
        hyperparameters: dict,
        hidden_dim = 256
    ):
        # Agent initialize method
        super().initialize(env, hyperparameters)
        # Create filename
        self.filename = os.path.join(Agent.MODELS_DIRECTORY, f"{self.env_fullname}.pt")

        self.replay_memory_size =   self.hyperparameters['replay_memory_size']
        self.mini_batch_size =      self.hyperparameters['mini_batch_size']
        self.network_sync_rate =    self.hyperparameters['network_sync_rate']

        # NN loss function, MSE = Mean Squared Error
        self.loss_fn = nn.MSELoss()
        # NN optimizer to initialize later
        self.optimizer = None

        # Create DQN
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.hidden_dim = hidden_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_dqn = DQN(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        print(self.policy_dqn)

    def _optimize(self, mini_batch, policy_dqn, target_dqn):
        # Process mini batch all at once to reduce time
        observations, actions, rewards, terminations, next_observations = zip(*mini_batch)

        # Stack tensors to create batch tensors, tensor([[1,2,3,..]])
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_observations = torch.stack(next_observations)
        # Convert true/false in 1.0/0.0
        terminations = torch.tensor(terminations).float().to(self.device)

        with torch.no_grad():
            # Calculate target q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(next_observations).max(dim=1)[0]
            '''
                target_dqn(next_observations)   --> tensor([[.2, .8], [.3, .6], [.1, .4]])
                    .max(dim=1)                 --> torch.return_types.max(values=tensor([.8,.6,.4]), indices=tensor([1,1,1]))
                        [0]                     --> tensor([.8,.6,.4])
            '''

        # Calculate q values from current policy (use actions done as index in the tensors)
        current_q = policy_dqn(observations).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
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

    # ==================================== Superclass methods

    def decay_epsilon(self, is_training):
        super().decay_epsilon(is_training)
    
        # Also update experience
        # If we collected enough experience
        if is_training and len(self.memory) > self.mini_batch_size:
            # Sample from memory
            mini_batch = self.memory.sample(self.mini_batch_size)

            # Optimize target and policy dqn
            self._optimize(mini_batch, self.policy_dqn, self.target_dqn)

            # Copy policy network to target network after a certain number of steps
            if self.step_count > self.network_sync_rate:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                self.step_count = 0

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
        is_training = True
    ):
        if not is_training: return

        # Convert obs, action, reward and next_obs to tensor in device
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=self.device)
        
        # Append memory if during training
        if is_training:
            self.memory.append((obs, action, reward, terminated, next_obs))
            self.step_count += 1
    
    def save_model(self):
        torch.save(self.policy_dqn.state_dict(), self.filename)

    def run(self, n_episodes: int | None, is_training = True, show_plots = True, verbose = True, seed = None):
        # If is training create the deque for experience replay
        if is_training:
            self.memory = ReplayMemory(self.replay_memory_size, seed)
            # TODO: Train for infinite episodes regardless of n_episodes parameter
            n_episodes = None

            # Create target dqn used while training
            self.target_dqn = DQN(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            # Track number of step taken
            self.step_count = 0

            # Policy network optimizer
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
        else:
            # Load learned policy
            self.policy_dqn.load_state_dict(torch.load(self.filename))

            # Switch model to evaluation mode
            self.policy_dqn.eval()

        super().run(n_episodes, is_training, show_plots, verbose, seed)