import gymnasium as gym
from Agent import Agent
from tqdm import tqdm
import numpy as np
import logging
from matplotlib import pyplot as plt
import pickle

def run(env_args: dict, n_episodes: int, is_training = False, show_plots = False, show_render = False):
    # Initialise the environment
    env = gym.make(**env_args, render_mode=None if not show_render else "human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # hyperparameters
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2) # reduce the exploration over time
    final_epsilon = 0.1

    agent = Agent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        load_from_file=not is_training
    )
    
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs, is_training=is_training)
            # print(f"{episode} - Agent choose action: {action}")

            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs, is_training=is_training)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
    
    # Save on file
    if is_training:
        agent.save_table()

    # Display plots
    if show_plots:
        # print(f'Episode time taken: {env.time_queue}')
        # print(f'Episode total rewards: {env.return_queue}')
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

        if len(agent.training_error) > 0:
            axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
            axs[2].set_title("Training Error")
            axs[2].set_xlabel("Episode")
            axs[2].set_ylabel("Temporal Difference")

        plt.tight_layout()
        plt.show()

    return agent

def main():
    env_args = {
        "id": "FrozenLake-v1",
        "map_name": "8x8",
        "is_slippery": True
    }

    # env_args = {
    #     "id": "Taxi-v3",
    # }

    # env_args = {
    #     "id": "LunarLander-v3",
    #     "continuous": False,
    #     "gravity": -10.0,
    #     "enable_wind": False,
    #     "wind_power": 15.0,
    #     "turbulence_power": 1.5
    # }

    env_args = {
        "id": 'Blackjack-v1',
        "natural": False,
        "sab": False
    }

    # Initialise the environment
    n_episodes = 100_000

    # TRAINING
    run(env_args, n_episodes, is_training=True, show_plots=True, show_render=False)

    # RUN
    run(env_args, 1000, is_training=False, show_plots=True, show_render=False)

if __name__ == '__main__':
    main()