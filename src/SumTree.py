import numpy as np

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