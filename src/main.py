import gymnasium as gym
from Agent import Agent, Q_Agent, DQN_Agent
from tqdm import tqdm
import numpy as np
import flappy_bird_gymnasium
import yaml

def test(hyperparameters: dict, n_episodes, show_render = True):
    # Initialise the environment
    env = gym.make(**hyperparameters['env_args'], render_mode=None if not show_render else "human")
    
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

            i = i+1

    env.close()
    return

def run(hyperparameters: dict, n_episodes, dqn = False, is_training = False, show_plots = False, show_render = False, verbose = False, record_video = False, seed = None):
    # Initialise the environment
    env = gym.make(**hyperparameters['env_args'], render_mode="rgb_array" if not show_render else "human") # or "none"
    # env.metadata['render_fps'] = 120

    if is_training:
        training_period = 5000
        name_prefix = "training"
    else:
        training_period = 100
        name_prefix = "test"

    if record_video:
        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=f"{name_prefix}_lr_{hyperparameters['learning_rate']}", episode_trigger=lambda x: x % training_period == 1)
    # env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=hyperparameters['n_episodes'])

    if dqn:
        agent = DQN_Agent(
            env,
            hyperparameters
        )
    else:
        agent = Q_Agent(
            env,
            hyperparameters,
        )
    
    agent.run(n_episodes, is_training, show_plots, verbose, seed=seed)
    return agent

def main(hyperparameter_set: str):
    # Load hyperparameters set
    with open('hyperparameters.yml', 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameter_set]

    # TEST
    # test(hyperparameters, 3, show_render=True)

    seed = hyperparameters.get('seed', None)
    DQN = hyperparameters.get('dqn', True)

    # TRAINING
    run(hyperparameters, hyperparameters.get('n_episodes', None), dqn=DQN, is_training=True, show_plots=True, show_render=False, verbose=True, record_video=False, seed=seed)

    # RUN
    run(hyperparameters, 1000, dqn=DQN, is_training=False, show_plots=True, show_render=False, verbose=False, record_video=False, seed=seed)

    # RENDER
    run(hyperparameters, 5, dqn=DQN, is_training=False, show_plots=False, show_render=True, verbose=False, record_video=False, seed=seed)

if __name__ == '__main__':
    hyperparameter_set = "flappybird_dqn"
    # hyperparameter_set = "flappybird_q"
    
    main(hyperparameter_set)