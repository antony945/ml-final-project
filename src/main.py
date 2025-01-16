import gymnasium as gym
from Agent import Agent
from tqdm import tqdm
import numpy as np
import logging
from matplotlib import pyplot as plt
import pickle

def plot_results(env, agent, mean_rewards, integrated = False):
    if (integrated):
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
    else:
        plt.plot(mean_rewards)
        
    plt.savefig(f"results/{agent.envname}_results_{len(mean_rewards)}.png")
    plt.tight_layout()
    plt.show()

def test(env_args: dict, show_render = True):
    # Initialise the environment
    env = gym.make(**env_args, render_mode=None if not show_render else "human")
    
    print(f"Action space: {env.action_space}")
    print(f"Obs space: {env.observation_space}")

    obs, info = env.reset()
    done = False

    # play one episode
    i = 0
    while not done:
        action = env.action_space.sample()
        # print(f"{episode} - {i}) Agent choose action: {action}")

        next_obs, reward, terminated, truncated, info = env.step(action)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

        i = i+1

    return

def run(env_args: dict, n_episodes: int, learning_rate: float, is_training = False, show_plots = False, show_render = False, debug = False, record_video = False):
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
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        load_from_file=not is_training,
        use_dict=True
    )
    
    rewards_per_episode = []
    mean_rewards = []
    best_reward = -99999

    iterable = range(n_episodes)
    if not debug:
        iterable = tqdm(iterable)

    try:
        for episode in iterable:
        # for episode in range(n_episodes):
            obs, info = env.reset()
            done = False

            rewards = 0

            # play one episode
            while not done:
                action = agent.get_action(obs, is_training=is_training)
                # print(f"{episode} - Agent choose action: {action}")

                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the agent
                agent.update(obs, action, reward, terminated, next_obs, is_training=is_training)

                # update if the environment is done and the current obs
                done = terminated or truncated
                # done = terminated or rewards < -1000
                obs = next_obs
                rewards += reward
            
            if rewards > best_reward:
                best_reward = rewards

            rewards_per_episode.append(rewards)
            mean_reward = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
            mean_rewards.append(mean_reward)

            if debug and is_training and episode%100==0:
                print(f"Episode: {episode}, Epsilon: {agent.epsilon:0.2f}, Reward: {rewards:.3f}, Best Reward: {best_reward:.3f}, Mean Reward {mean_reward:0.3f}")

            agent.decay_epsilon()
        
    except KeyboardInterrupt:
        print("\nCTRL+C pressed.\n")

    # Save on file
    if is_training:
        agent.save_table()

    # Display plots
    if show_plots:
        plot_results(env, agent, mean_rewards, integrated=False)

    env.close()
    return agent

def main():
    # env_args = {
    #     "id": "FrozenLake-v1",
    #     "map_name": "8x8",
    #     "is_slippery": True
    # }

    # env_args = {
    #     "id": "Taxi-v3",
    # }

    env_args = {
        "id": "LunarLander-v3",
        "continuous": False,
    }

    # env_args = {
    #     "id": 'Blackjack-v1',
    #     "natural": False,
    #     "sab": False
    # }

    # env_args = {
    #     "id": "MountainCar-v0",
    # }

    # TEST
    # test(env_args, n_episodes)

    # Initialise the environment
    n_episodes = 50_000
    learning_rate = 0.001

    # TRAINING
    run(env_args, n_episodes, learning_rate, is_training=True, show_plots=True, show_render=False, debug=True, record_video=False)

    # RUN
    run(env_args, 1000, learning_rate, is_training=False, show_plots=True, show_render=False, debug=False, record_video=False)

    # RENDER
    run(env_args, 5, learning_rate, is_training=False, show_plots=True, show_render=True, debug=False, record_video=False)

if __name__ == '__main__':
    main()