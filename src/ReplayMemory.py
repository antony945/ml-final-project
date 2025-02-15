from collections import deque
import random
from SumTree import SumTree
import numpy as np

class ReplayMemory():
    def __init__(self,
            capacity: int,
            use_priority: bool = False,
            alpha = 0.6,
            epsilon = 0.01, # or 1e-5
            seed=None):
        
        self.capacity = capacity
        self.use_priority = use_priority

        if self.use_priority:
            # Create the SumTree and setup hyperparameters
            self.memory = SumTree(self.capacity)
            self.epsilon = epsilon # Small constant to avoid some experiences to have zero probability of being chosen 
            self.alpha = alpha # Tradeoff between taking only exp with high priority and sampling randomly
            # self.beta = beta_init # Importance-sampling, from initial value increasing to 1
        else:
            self.memory = deque([], maxlen=self.capacity)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """
        Add a new experience with max priority.
        :param transition: (obs, action, reward, next_obs, terminated)
        """
        if self.use_priority:
            # Get highest priority leaf
            max_priority = np.max(self.memory.tree[-self.memory.capacity:])
            # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
            # So we use a minimum priority
            if max_priority == 0:
                max_priority = 1
            
            self.memory.add(max_priority, transition)
        else:
            self.memory.append(transition)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch based on priority.
        :param batch_size: Number of samples
        :param beta: Importance-sampling correction factor
        :return: (batch, importance weights, indices)
        """
        # Randomly sample from deque for normal ER
        if not self.use_priority:
            return random.sample(self.memory, batch_size), None, None
        
        # Use SumTree for PER
        batch = []
        indices = []
        weights = np.zeros(batch_size)

        # Calculate the priority segment
        # Here we divide the Range[0, ptotal] into batch_size ranges
        segment = self.memory.total_priority / batch_size

        for i in range(batch_size):
            # A value is uniformly sample from each range
            sample_val = random.uniform(segment * i, segment * (i + 1))
            # Experience that correspond to each value is retrieved
            idx, priority, data = self.memory.get(sample_val)

            # Compute probability of being selected
            prob = priority / self.memory.total_priority
            # Importance sampling
            weights[i] = (1.0 / (self.capacity * prob)) ** beta
            batch.append(data)
            indices.append(idx)

        # Normalize weights
        weights /= np.max(weights)
        return batch, weights, indices

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        :param indices: List of sampled indices
        :param td_errors: List of TD errors
        """
        if not self.use_priority:
            return

        # Avoid zero priority
        abs_td_errors = (np.abs(td_errors) + self.epsilon)
        # Clip if more than 1
        clipped_errors = np.minimum(abs_td_errors, 1)
        new_priorities = np.power(clipped_errors, self.alpha)

        for idx, priority in zip(indices, new_priorities):
            self.memory.update(idx, priority)

    def __len__(self):
        return len(self.memory)