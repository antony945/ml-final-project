import gymnasium as gym
from Agent import Agent
from tqdm import tqdm
import numpy as np
import logging
from matplotlib import pyplot as plt

def run(env_name: str, n_episodes: int, agent: Agent, show_plots = False):
    # Initialise the environment
    env = gym.make(env_name, map_name="8x8", is_slippery=True, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs, is_training=False)
            # print(f"{episode} - Agent choose action: {action}")
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
        
        # Decay epsilon at the end of every episode
        agent.decay_epsilon()

    
    if show_plots:
        # print(f'Episode time taken: {env.time_queue}')
        # print(f'Episode total rewards: {env.return_queue}')
        # print(f'Episode lengths: {env.length_queue}')

        # visualize the episode rewards, episode length and training error in one figure
        fig, axs = plt.subplots(1, 3, figsize=(14, 8))

        # np.convolve will compute the rolling mean for 100 episodes
        axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
        axs[0].set_title("Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")

        axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Length")

        axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
        axs[2].set_title("Training Error")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Temporal Difference")

        plt.tight_layout()
        plt.show()

    return agent

def train(env_name: str, n_episodes: int, show_plots = False):
    # hyperparameters
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1
    
    # Initialise the environment
    env = gym.make(env_name, map_name="8x8", is_slippery=True, render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = Agent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs, is_training=True)
            # print(f"{episode} - Agent choose action: {action}")

            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
    
    
    if show_plots:
        # print(f'Episode time taken: {env.time_queue}')
        print(f'Episode total rewards: {env.return_queue}')
        # print(f'Episode lengths: {env.length_queue}')

        # visualize the episode rewards, episode length and training error in one figure
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        # np.convolve will compute the rolling mean for 100 episodes
        axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
        axs[0].set_title("Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")

        axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Length")

        axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
        axs[2].set_title("Training Error")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Temporal Difference")

        plt.tight_layout()
        plt.show()

    return agent

if __name__ == '__main__':
    # env_name = "Taxi-v3"
    env_name = "FrozenLake-v1"
    n_episodes = 100_000

    # TRAINING
    my_agent = train(env_name, n_episodes=n_episodes, show_plots=True)
    
    # RUN
    run(env_name, n_episodes=5, agent=my_agent, show_plots=False)