import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt

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
        use_dict: bool = False
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
        self.use_dict = use_dict

        # If obs space is continuous we must discretize it
        self.obs_spaces_size = -1
        self.obs_spaces = []
        self.divisions = 10
        if (not isinstance(self.env.observation_space, gym.spaces.Discrete)):
            # print(self.env.observation_space.shape)
            for i in range(0, self.env.observation_space.shape[0]):
                # When an obs will be discretized it could have values between 0 and self.divisions included (self.divisions+1 total values)
                space = np.linspace(env.observation_space.low[i], env.observation_space.high[i], self.divisions)
                self.obs_spaces.append(space)
                
            # self.divisions+1 * obs_space.shape spaces
            # q_table will have obs_spaces_size x action_spaces
            self.obs_spaces_size = np.prod([len(s) + 1 for s in self.obs_spaces])
        else:
            self.obs_spaces_size = self.env.observation_space.n
        
        # Create filename for the q-table based on the env
        self.filename = self._create_filename(self.env)

        # Create q_table or load it from a file if specified
        if load_from_file:
            with open(self.filename, "rb") as f:
                self.q_table = pickle.load(f)
        else:
            if self.use_dict:
                # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
                self.q_table = defaultdict(self._create_numpy_entry) # can't use lambda due to pickle not able to "pickle" an anon function
            else:
                self.q_table = np.zeros((self.obs_spaces_size, self.env.action_space.n))

    def _create_filename(self, env: gym.Env):
        MODELS_DIRECTORY = "models"

        self.envname = f"{env.spec.name}"
        
        # Create different filename for different models 
        for k,v in env.spec.kwargs.items():
            if k != "render_mode":
                self.envname += f"_{v}"

        return f"{MODELS_DIRECTORY}/{self.envname}.pk1"

    def _create_numpy_entry(self):
            return np.zeros(self.env.action_space.n)

    def get_action(self, obs, is_training=True) -> int:
        if is_training and np.random.random() < self.epsilon:
            # Exploration -> Choose random action
            return self.env.action_space.sample()
        else:
            # Exploitation -> Follow Q-table
            obs = self._discretize_obs(obs)
            if self.use_dict:
                return int(np.argmax(self.q_table[obs]))
            else:
                return int(np.argmax(self.q_table[obs, :]))

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

        # print(obs)
        # print(next_obs)
        # print(f"MAX SIZE: {self.obs_spaces_size}")
        # input("...")

        # Update Q-table after having performed action
        # Q(s, a) = r + max Q(s', a')
        if self.use_dict:
            future_q_value = (not terminated) * np.max(self.q_table[next_obs])
            current_q_value = self.q_table[obs][action]
            temporal_difference = reward + self.discount_factor * future_q_value - current_q_value
            self.q_table[obs][action] += self.learning_rate * temporal_difference
        else:
            future_q_value = (not terminated) * np.max(self.q_table[next_obs, :])
            current_q_value = self.q_table[obs, action]
            temporal_difference = reward + self.discount_factor * future_q_value - current_q_value
            self.q_table[obs, action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        # Linear decay
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_table(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def plot_results(self, mean_rewards, training = False, integrated = False):
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
            plt.plot(mean_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")

        plt.title(f"N={len(mean_rewards)}, LR={self.learning_rate}, E_DECAY={self.epsilon_decay}, DISCOUNT={self.discount_factor}")
        
        if training:
            filename = f"results/{self.envname}_{len(mean_rewards)}_training_results.png"
        else:
            filename = f"results/{self.envname}_{len(mean_rewards)}_test_results.png"

        plt.savefig(filename)
        plt.tight_layout()
        plt.show()