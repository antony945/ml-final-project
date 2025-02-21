import gymnasium as gym
from Agent import Agent, Q_Agent, DQN_Agent
from tqdm import tqdm
import numpy as np
import yaml
import flappy_bird_gymnasium
import argparse

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

def run(hyperparameters: dict, n_episodes, dqn = False, is_training = False, show_plots = False, show_render = False, verbose = False, record_video = False, model_name = None, seed = None):
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
    
    agent.run(n_episodes, is_training, show_plots, verbose, model_name, seed=seed)
    return agent.model_name

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
    model_name = run(hyperparameters, hyperparameters.get('n_episodes', None), dqn=DQN, is_training=True, show_plots=True, show_render=False, verbose=True, record_video=False, model_name=None, seed=seed)

    # model_name = "models/FlappyBird_LR=0.001_DF=0.95_EPS=0.03_MEM=None__N=200000.pk1"
    # model_name = "FlappyBird_LR=0.001_DF=0.95_EPS=0.03_MEM=PER, BATCH=64, ALPHA=0.6, FIXEDBETA=0.4, EPS=0.001_LAZY_N=200000.pk1"
    # model_name = "FlappyBird_temp_LR=0.001_DF=0.95_EPS=0.03_MEM=PER, BATCH=64, ALPHA=0.6, BETA=0.4, EPS=0.001_.pk1"
    # model_name = "FlappyBird_temp_LR=0.001_DF=0.95_EPS=0.03_MEM=ER, BATCH=64_LAZY.pk1"
    
    # model_name = "FlappyBird_LR=0.001_DF=0.95_EPS=0.03_MEM=ER, BATCH=64_LAZY_N=185021.pk1"
    # model_name = "FlappyBird_temp_LR=0.001_DF=0.95_EPS=0.03_MEM=ER, BATCH=64_LAZY.pk1"
    # model_name = "FlappyBird_temp_LR=0.001_DF=0.95_EPS=0.01_MEM=None_.pk1"
    # model_name = "FlappyBird_temp_LR=0.0001_DF=0.95_EPS=0.01_MEM=None_.pk1"
    # model_name = "FlappyBird_LR=0.0001_DF=0.95_EPS=0.01_MEM=None__N=1000000.pk1" # avg reward in test: 7
    # model_name = "FlappyBird_LR=0.001_DF=0.95_EPS=0.01_MEM=None__N=1000000.pk1" # avg reward in test: 5

    # model_name = "FlappyBird_False_temp.pt"         # avg reward in test: 20
    # model_name = "FlappyBird_173742.pt"             # avg reward in test: 3.90
    # model_name = "FlappyBird_195776.pt"             # avg reward in test: 20
    # model_name = "FlappyBird_231320.pt"             # avg reward in test: 13
    # model_name = "FlappyBird_850005.pt"             # avg reward in test: 4.8
    # model_name = "FlappyBird_1036177_best_model.pt" # avg reward in test: 50
    # model_name = "FlappyBird_temp_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=64_LAZY.pt" # avg reward in test: 10
    # model_name = "FlappyBird_temp_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=32_LAZY.pt" # avg reward in test: 5
    # model_name = "FlappyBird_temp_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=64_.pt" # avg reward at 10k: 15, at 20k: 100
    # model_name = "FlappyBird_temp_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=128_.pt" # avg reward at 17.5k: 100, at 20k: 105
    # model_name = "FlappyBird_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=128__N=27052.pt" # avg reward at 27k: 500
    # model_name = "FlappyBird_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=64__N=50874.pt" # avg reward at 50k: 200
    # model_name = "FlappyBird_LR=0.0001_DF=0.95_EPS=0.01_MEM=ER, BATCH=64_tDQN_N=50000.pt" # avg reward at 50k: 250
    # model_name = "models/FlappyBird_training_temp_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_N=-1.pt"

    # model_name = f"oldmodels/{model_name}"
    # RUN
    run(hyperparameters, 1000, dqn=DQN, is_training=False, show_plots=True, show_render=False, verbose=False, record_video=False, model_name=model_name, seed=seed)

    # RENDER
    run(hyperparameters, 5, dqn=DQN, is_training=False, show_plots=False, show_render=True, verbose=False, record_video=False, model_name=model_name, seed=seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the RL agent with specified parameters.')
    parser.add_argument('--q', action='store_true', help='Use Q-learning agent')
    parser.add_argument('--dqn', action='store_true', help='Use DQN agent')
    parser.add_argument('--er', action='store_true', help='Use Experience Replay')
    parser.add_argument('--per', action='store_true', help='Use Prioritized Experience Replay')
    parser.add_argument('--double', action='store_true', help='Use DoubleDQN')
    parser.add_argument('--dueling', action='store_true', help='Use DuelingDQN')
    args = parser.parse_args()

    if args.q:
        agent_type = 'q'
    elif args.dqn:
        agent_type = 'dqn'
        if args.double:
            agent_type += '_double'
        if args.dueling:
            agent_type += '_dueling'
    else:
        raise ValueError("You must specify either --q or --dqn")

    if args.er:
        replay_type = 'er'
    elif args.per:
        replay_type = 'per'
    else:
        replay_type = 'none'

    hyperparameter_set = f"flappybird_{agent_type}_{replay_type}"
    
    main(hyperparameter_set)