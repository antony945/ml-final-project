from collections import deque
import random
import numpy as np

# Implementation of a SumTree for Prioritized Experience Replay
# NOT USED AT THE END IN THE FINAL PROJECT FOR LACK OF TIME
class SumTree:
    def __init__(self, capacity):
        """
        Initialize a sum tree.
        :param capacity: Max number of elements to store (must be power of 2 for efficiency)
        """
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Stores transitions (leaf)
        self.data = np.empty(capacity, dtype=object)

        # Index for the current data
        self.data_pointer = 0

        # Variable for counting n_entries in the tree
        self.n_entries = 0

    def add(self, priority, data):
        """
        Add a new experience with a given priority.
        :param priority: The priority of the experience
        :param data: The transition (obs, action, reward, terminated, next_obs)
        """
        # Leaf node index
        idx = self.data_pointer + self.capacity - 1
        # Store experience
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(idx, priority)
        # Circular buffer (data_pointer always between 0 and self.capacity excluded)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        # Keep count of elements
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """
        Update the priority of an experience.
        :param idx: Index in the tree
        :param priority: New priority value
        """
        # Change = new priority score - former priority score
        delta = priority - self.tree[idx]
        # Update priority of leaf node
        self.tree[idx] = priority
        # Propagate the change through tree
        self._propagate(idx, delta)

    def _propagate(self, idx, delta):
        """
        Update the sum tree.
        :param idx: Index in the tree
        :param delta: Change in priority value
        """
        # parent = (idx - 1) // 2
        # self.tree[parent] += delta  # Update parent
        # if parent != 0:
        #     self._propagate(parent, delta)

        # this method is faster than the recursive loop
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    @property
    def total_priority(self):
        """
        Get the total sum of priorities.
        :return: Sum of all priority values in the tree
        """
        return self.tree[0]

    def get(self, sample_val):
        """
        Get an experience based on priority sampling.
        :param sample_val: A value in range [0, total_priority]
        :return: (index, priority, data)
        """
        # Find leaf node
        idx = self._retrieve(0, sample_val)
        # Convert to data index
        data_idx = idx - self.capacity + 1
        # Return idx, priority, data
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, parent_idx, sample_val):
        """
        Retrieve a priority-based sample.
        :param parent_idx: Current node index
        :param sample_val: The sampled value
        :return: Leaf node index
        """
        # left = 2 * parent_idx + 1
        # right = left + 1

        # if left >= len(self.tree):  # Leaf node reached
        #     return parent_idx

        # if sample_val <= self.tree[left]:  # Traverse left
        #     return self._retrieve(left, sample_val)
        # else:  # Traverse right
        #     return self._retrieve(right, sample_val - self.tree[left])
        
        while True:
            left_child_index = 2 * parent_idx + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_idx
                break
            else: # downward search, always search for a higher priority node
                if sample_val <= self.tree[left_child_index]:
                    parent_idx = left_child_index
                else:
                    sample_val -= self.tree[left_child_index]
                    parent_idx = right_child_index

        return leaf_index

    def __len__(self):
        return self.n_entries  # Return the number of elements stored

class ReplayMemory():
    def __init__(self,
            capacity: int,
            use_priority: bool = False,
            alpha = 0.6,
            beta = 0.4,
            epsilon = 1e-5, # or 1e-5
            seed=None):
        
        self.capacity = capacity
        self.use_priority = use_priority

        if self.use_priority:
            # Create the SumTree and setup hyperparameters
            self.memory = SumTree(self.capacity)
            self.epsilon = epsilon # Small constant to avoid some experiences to have zero probability of being chosen 
            self.alpha = alpha # Tradeoff between taking only exp with high priority and sampling randomly
            self.beta = beta # Importance-sampling, from initial value increasing to 1
            self.beta_increment_per_sampling = 1e-4
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

    def sample(self, batch_size):
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
            weights[i] = (1.0 / (self.capacity * prob)) ** self.beta
            batch.append(data)
            indices.append(idx)

        # Normalize weights
        weights /= np.max(weights)
        self.beta = min(self.beta+self.beta_increment_per_sampling, 1)
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