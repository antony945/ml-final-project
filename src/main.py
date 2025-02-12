import gymnasium as gym
from Agent import Agent
from tqdm import tqdm
import numpy as np
from tetris_gymnasium.envs import Tetris
import flappy_bird_gymnasium
import cv2
import sys

def test(env_args: dict, n_episodes, show_render = True):
    # Initialise the environment
    env = gym.make(**env_args, render_mode=None if not show_render else "human")
    
    print(f"Action space: {env.action_space}")
    print(f"Obs space: {env.observation_space}")

    for episode in range(n_episodes):
        done = False
        obs, info = env.reset(seed=42)

        # play one episode
        i = 0
        while not done:
            # Render the current state of the game
            env.render()

            action = env.action_space.sample()
            # print(f"{episode} - {i}) Agent choose action: {action}")

            next_obs, reward, terminated, truncated, info = env.step(action)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

            key = cv2.waitKey(20) # timeout to see the movement

            i = i+1

    env.close()
    return

def run(env_args: dict, n_episodes: int, learning_rate: float, is_training = False, show_plots = False, show_render = False, verbose = False, record_video = False, max_episode_steps = None):
    # Initialise the environment
    env = gym.make(**env_args, render_mode="rgb_array" if not show_render else "human") # or "none"

    training_period = 5000
    if is_training and record_video:
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="training", episode_trigger=lambda x: x % training_period == 0)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # hyperparameters
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes/2) # reduce the exploration over time
    # epsilon_decay = 5e-4
    final_epsilon = 0.05

    agent = Agent(
        env=env,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        load_from_file=not is_training,
    )
    
    rewards_per_episode = []
    mean_rewards = []
    epsilon_values = []
    best_reward = -99999

    iterable = range(n_episodes)
    if not verbose:
        iterable = tqdm(iterable)

    try:
        for episode in iterable:
        # for episode in range(n_episodes):
            obs, info = env.reset()
            done = False

            rewards = 0

            # play one episode
            i = 0
            while not done:
                action = agent.get_action(obs, is_training=is_training)
                # print(f"{episode} - Agent choose action: {action}")

                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the agent
                agent.update(obs, action, reward, terminated, next_obs, is_training=is_training)

                # update if the environment is done and the current obs
                if max_episode_steps is not None:
                    done = terminated or i > max_episode_steps
                else:
                    done = terminated or truncated
                
                obs = next_obs
                rewards += reward
                
                i=i+1
            
            if rewards > best_reward:
                best_reward = rewards

            rewards_per_episode.append(rewards)
            mean_reward = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
            mean_rewards.append(mean_reward)
            epsilon_values.append(agent.epsilon)

            if verbose and is_training and episode%100==0:
                print(f"Episode: {episode}, Epsilon: {agent.epsilon:0.2f}, Reward: {rewards:.3f}, Best Reward: {best_reward:.3f}, Mean Reward {mean_reward:0.3f}")

            agent.decay_epsilon()
        
    except KeyboardInterrupt:
        print("\nCTRL+C pressed.\n")
    
    env.close()

    # Save on file
    if is_training:
        agent.save_table()

    # Display plots
    if show_plots:
        epsilon_values = None if not is_training else epsilon_values
        agent.plot_results(mean_rewards, epsilon_values, training=is_training, integrated=False)

    return agent

def main():
    max_episode_steps = None

    # 100_000 and 0.001
    # env_args = {
    #     "id": "FrozenLake-v1",
    #     "map_name": "8x8",
    #     "is_slippery": True
    # }

    # 100_000 and 0.001
    # env_args = {
    #     "id": "Taxi-v3",
    # }

    # 5_000 and 0.001
    # env_args = {
    #     "id": "LunarLander-v3",
    #     "continuous": False,
    # }

    # 5_000 and 0.9 and 1000 steps
    # env_args = {
    #     "id": "MountainCar-v0",
    # }
    # max_episode_steps = 1000

    # env_args = {
    #     "id": "tetris_gymnasium/Tetris"    
    # }

    # 200_000 and 0.001
    env_args = {
        "id": "FlappyBird-v0",
        "use_lidar": False    
    }

    # TEST
    # test(env_args, 3, show_render=True)
    # return

    # Initialise the environment
    n_episodes = 200000
    learning_rate = 0.001

    # TRAINING
    run(env_args, n_episodes, learning_rate, is_training=True, show_plots=True, show_render=False, verbose=True, record_video=False, max_episode_steps=max_episode_steps)

    # RUN
    run(env_args, 1000, learning_rate, is_training=False, show_plots=True, show_render=False, verbose=False, record_video=False)

    # RENDER
    run(env_args, 5, learning_rate, is_training=False, show_plots=False, show_render=True, verbose=False, record_video=False)

if __name__ == '__main__':
    main()