import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle

class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        load_from_file: bool = False
    ):
        # Initialize agent
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        _, self.info = env.reset()
        self.training_error = []

        # Create filename for the q-table based on the env
        self.filename = f"{self.env.spec.name}"
        for k,v in self.env.spec.kwargs.items():
            if k != "render_mode":
                self.filename += f"_{v}"
        self.filename += ".pk1"

        # Create q_table or load it from a file if specified
        if load_from_file:
            with open(self.filename, "rb") as f:
                self.q_table = pickle.load(f)
        else:
            # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
            self.q_table = defaultdict(self.create_numpy_entry) # can't use lambda due to pickle not able to "pickle" an anon function

    def create_numpy_entry(self):
            return np.zeros(self.env.action_space.n)

    def get_action(self, obs, is_training=True) -> int:
        if is_training and np.random.random() < self.epsilon:
            # Exploration -> Choose random action
            return self.env.action_space.sample()
        else:
            # Exploitation -> Follow Q-table
            return int(np.argmax(self.q_table[obs]))

    def update(self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
        is_training=True
    ):
        if not is_training: return

        # Update Q-table after having performed action
        # Q(s, a) = r + max Q(s', a')
        future_q_value = (not terminated) * np.max(self.q_table[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_table[obs][action]

        self.q_table[obs][action] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_table(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.q_table, f)